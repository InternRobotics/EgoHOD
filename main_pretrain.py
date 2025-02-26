# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import util.dist_utils as dist_utils

from engine_pretrain import train_one_epoch,validate_ek100_mir_zeroshot,build_transform
from util.config import get_config
from dataset.egodataset import EgoExoDataset
from dataset.ek100dataset import EK100Dataset
from model.clip import *
import torch
import torch.cuda.amp as amp
from model import loss
def get_args_parser():
    parser = argparse.ArgumentParser('HOD pre-training', add_help=False)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--config_file', default='configs/no_decoder/debug_clip_base.yml', type=str,
                        help='config file')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):

    dist_utils.init_distributed_mode(args)
    config = get_config(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    dist_utils.random_seed(args.seed, dist_utils.get_rank())

    transform_train = build_transform(config.model.name,mode='train')
    transform_val = build_transform(config.model.name,mode='val')

    crop_size = 336 if "_336PX" in config.model.name  else 224
    tokenizer = None

    train_dataset = EgoExoDataset(
        config.data, transform=transform_train, is_training=True, tokenizer=tokenizer, crop_size=crop_size
    )
    val_dataset = EK100Dataset(config.test.ek100_mir, transform=transform_val, is_training=False, tokenizer=None, crop_size=crop_size)

    if dist_utils.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=(train_sampler is None),
        # collate_fn=collect if config.data.dataset == 'htego_feat' else None,
        collate_fn = None,
        num_workers=config.train.workers, pin_memory=False, sampler=train_sampler, drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.test.batch_size, shuffle=False,
        num_workers=config.train.workers, pin_memory=False, sampler=val_sampler, drop_last=False
    ) 

    model_name = config.model.name
    if model_name == 'CLIP_VITB16':
        model = CLIP_VITB16(
            config=config.model,
            freeze_temperature=config.model.freeze_temperature,
            use_grad_checkpointing=config.model.grad_checkpointing,
            context_length=config.data.context_length,
            vocab_size=config.data.vocab_size,
            patch_dropout=config.model.patch_dropout,
            num_frames=config.data.clip_length,
            drop_path_rate=config.model.drop_path_rate,
            use_fast_conv1=config.model.use_fast_conv1,
            use_flash_attn=config.model.use_flash_attn,
            use_quick_gelu=True,
            project_embed_dim=config.model.project_embed_dim,
            pretrain_zoo=config.model.pretrain_zoo,
            pretrain_path=config.model.pretrain_path,
        )
    elif model_name == 'CLIP_VITL14_336PX':
        model = CLIP_VITL14_336PX(
            config=config.model,
            freeze_temperature=config.model.freeze_temperature,
            use_grad_checkpointing=config.model.grad_checkpointing,
            context_length=config.data.context_length,
            vocab_size=config.data.vocab_size,
            patch_dropout=config.model.patch_dropout,
            num_frames=config.data.clip_length,
            drop_path_rate=config.model.drop_path_rate,
            use_fast_conv1=config.model.use_fast_conv1,
            use_flash_attn=config.model.use_flash_attn,
            use_quick_gelu=True,
            project_embed_dim=config.model.project_embed_dim,
            pretrain_zoo=config.model.pretrain_zoo,
            pretrain_path=config.model.pretrain_path,
        )
    elif model_name == 'CLIP_VITL14_336PX_Slowfast':
        model = CLIP_VITL14_336PX_Slowfast(
            config=config.model,
            freeze_temperature=config.model.freeze_temperature,
            use_grad_checkpointing=config.model.grad_checkpointing,
            context_length=config.data.context_length,
            vocab_size=config.data.vocab_size,
            patch_dropout=config.model.patch_dropout,
            num_frames=config.data.clip_length,
            drop_path_rate=config.model.drop_path_rate,
            use_fast_conv1=config.model.use_fast_conv1,
            use_flash_attn=config.model.use_flash_attn,
            use_quick_gelu=True,
            project_embed_dim=config.model.project_embed_dim,
            pretrain_zoo=config.model.pretrain_zoo,
            pretrain_path=config.model.pretrain_path,
        )     
    elif model_name == 'CLIP_VITB16_Slowfast':
        model = CLIP_VITB16_Slowfast(
            config=config.model,
            freeze_temperature=config.model.freeze_temperature,
            use_grad_checkpointing=config.model.grad_checkpointing,
            context_length=config.data.context_length,
            vocab_size=config.data.vocab_size,
            patch_dropout=config.model.patch_dropout,
            num_frames=config.data.clip_length,
            drop_path_rate=config.model.drop_path_rate,
            use_fast_conv1=config.model.use_fast_conv1,
            use_flash_attn=config.model.use_flash_attn,
            use_quick_gelu=True,
            project_embed_dim=config.model.project_embed_dim,
            pretrain_zoo=config.model.pretrain_zoo,
            pretrain_path=config.model.pretrain_path,
        )   

    model.to(device)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, config.train.optimizer.wd)
    optimizer = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=(0.9, 0.999))
    print(optimizer)
    scaler = amp.GradScaler(enabled=not config.train.disable_amp)

    print(f"Start training for {config.train.epochs} epochs")
    start_time = time.time()
    start_epoch = 0

    criterion = loss.ClipLoss(
        local_loss=config.train.local_loss,
        gather_with_grad=config.train.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
    ).cuda(args.gpu)

    if config.resume:
        print("=> loading resume checkpoint '{}'".format(config.resume))
        curr_checkpoint = torch.load(config.resume, map_location='cpu')

        result = model.load_state_dict(curr_checkpoint['state_dict'])
        


    for epoch in range(start_epoch, config.train.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, scaler,
            log_writer=log_writer,
            args=args,criterion=criterion
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        val_stats = validate_ek100_mir_zeroshot(val_loader,  model=model, criterion=criterion, args=args, config=config,split=epoch) 

        dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'val_state':val_stats,
            }, config.output_dir)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
