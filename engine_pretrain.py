# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from einops import rearrange
import numpy as np
import pandas as pd
from util.meter import *
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def build_transform(model_name, mode):

    mean, std = (0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)
    input_size = 336 if model_name.endswith("_336PX") else 224
    # simple augmentation
    if mode == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=3),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean, std=std)])
    return transform
    
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, scaler,
                    log_writer=None,
                    args=None,criterion=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, inputs in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        optimizer.zero_grad()
        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        with torch.cuda.amp.autocast():
            inputs[0] = inputs[0].permute(0, 2, 1, 3, 4) # [b t 3 t w -> b c t h w]
            inputs[1] = inputs[1].permute(0, 2, 1, 3, 4) # [b t 3 t w -> b c t h w]
            image_features, text_features, logit_scale = model(*inputs)
            loss_dict = criterion(image_features, text_features, logit_scale)
            loss = loss_dict['loss']

        scaler.scale(loss).backward()
        metric_logger.update(loss=loss.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None and (data_iter_step + 1) % 10 == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss.item(), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def validate_ek100_mir_zeroshot(val_loader, model, criterion, args, config,split):
    from torch import nn
    import torch.nn.functional as F
    import torch
    # switch to eval mode
    model.eval()

    all_video_embed = [[] for _ in range(args.world_size)]
    all_text_embed = [[] for _ in range(args.world_size)]
    total_num = 0
    with torch.cuda.amp.autocast(enabled=not config.train.disable_amp):
        with torch.no_grad():
            for i, inputs in enumerate(val_loader):

                inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]
                _ = inputs.pop()  # loader will a "relevancy" variable which is not needed except ek100_mir

                inputs[0] = inputs[0].permute(0, 2, 1, 3, 4) # [b t 3 t w -> b c t h w]
                inputs[1] = inputs[1].permute(0, 2, 1, 3, 4) # [b t 3 t w -> b c t h w]      
                
                image_features, text_features, logit_scale = model(*inputs)
                gathered_image_features = [torch.zeros_like(image_features) for _ in range(args.world_size)]
                gathered_text_features = [torch.zeros_like(text_features) for _ in range(args.world_size)]
                torch.distributed.all_gather(gathered_image_features, image_features)
                torch.distributed.all_gather(gathered_text_features, text_features)
                for j in range(args.world_size):
                    all_video_embed[j].append(gathered_image_features[j].detach().cpu())
                    all_text_embed[j].append(gathered_text_features[j].detach().cpu())
                
                total_num += image_features.shape[0] * args.world_size
                if i % 10 == 0:
                    print(f'step {i}/{len(val_loader)}')
    for j in range(args.world_size):
        all_video_embed[j] = torch.cat(all_video_embed[j], dim=0).numpy()
        all_text_embed[j] = torch.cat(all_text_embed[j], dim=0).numpy()
    all_text_embed_reorg, all_video_embed_reorg = [], []
    for i in range(total_num):
        all_video_embed_reorg.append(all_video_embed[i % args.world_size][i // args.world_size])
        all_text_embed_reorg.append(all_text_embed[i % args.world_size][i // args.world_size])
    all_text_embed = np.vstack(all_text_embed_reorg)
    all_video_embed = np.vstack(all_video_embed_reorg)
    all_text_embed = all_text_embed[:9668, :]
    all_video_embed = all_video_embed[:9668, :]
    similarity_matrix = np.matmul(all_video_embed, all_text_embed.T)
    similarity_matrix = (similarity_matrix + 1) / 2
    
    video_id = pd.read_csv(config.test.ek100_mir.metadata).values[:, 0]
    text_id = pd.read_csv(config.test.ek100_mir.metadata.replace('test', 'test_sentence')).values[:, 0]
    indexes = [video_id.tolist().index(elem) for elem in text_id]
    similarity_matrix = similarity_matrix[:, indexes]
    # similarity_matrix = torch.from_numpy(similarity_matrix)
    # similarity_matrix = similarity_matrix * F.softmax(similarity_matrix, dim=0)*len(similarity_matrix)
    # similarity_matrix = similarity_matrix.numpy()
    print(similarity_matrix.shape,text_id.shape,video_id.shape)
    rel_matrix = pd.read_pickle(config.test.ek100_mir.relevancy_path)
    vis_map, txt_map, avg_map = get_mAP(similarity_matrix, rel_matrix)
    print('mAP: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_map, txt_map, avg_map))
    vis_nDCG, txt_nDCG, avg_nDCG = get_nDCG(similarity_matrix, rel_matrix)
    print('nDCG: V->T: {:.3f} T->V: {:.3f} AVG: {:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))
    return {'vis_map': vis_map, 'txt_map': txt_map, 'avg_map': avg_map,
            'vis_ndcg': vis_nDCG, 'txt_ndcg': txt_nDCG, 'avg_ndcg': avg_nDCG}