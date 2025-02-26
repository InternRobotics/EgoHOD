import json
import json

import time
import func_timeout
from func_timeout import func_set_timeout
import os
import torch
import numpy as np
import csv
import glob
import os.path as osp
import pickle
import random
import numpy as np
import pandas as pd
import torch
import os
import decord
from model.clip import *
from util.config import get_config
from dataset.data_utils import video_loader
import argparse
from tqdm import tqdm
import decord
from decord import cpu
import io
from ipdb import set_trace
from petrel_client.client import Client
import ast
client = Client()
import torchvision.transforms as transforms
import torchvision.datasets as datasets


chunk_sec = 600  # Each segment is up to 600s
noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

def get_args_parser():
    parser = argparse.ArgumentParser('egomcq eval', add_help=False)

    parser.add_argument('--config_file', default='configs/no_decoder/clip_base_eval.yml', type=str,help='config file')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--root', default='ego4d/videos_short320_chunked_15s/', type=str,help='root of egtea video clips')
    parser.add_argument('--metadata', default='egomcq.json', type=str,help='root of egtea annotations')    
    parser.add_argument('--crop_size', default=224, type=int,help='root of egtea annotations')    
    return parser

def _get_caption(sample):
    noun_vec = torch.zeros(noun_dim)
    verb_vec = torch.zeros(verb_dim)
    noun_idx = eval(sample['tag_noun'])
    verb_idx = eval(sample['tag_verb'])
    for i in noun_idx:
        noun_vec[i] = 1
    for i in verb_idx:
        verb_vec[i] = 1

    return sample['clip_text'], noun_vec, verb_vec

def build_transform(model_name):

    mean, std = (0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)
    input_size = 336 if model_name.endswith("_336PX") else 224
    # simple augmentation

    transform = transforms.Compose([
        transforms.Resize(input_size, interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.Normalize(mean=mean, std=std)])
    return transform

def main(args):
    config = get_config(args)
    ego4d_root = args.root
    crop_size = args.crop_size
    mean = (0.48145466 * 255,0.4578275 * 255,0.40821073 * 255)
    std = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
    with open(args.metadata,'r') as f:
        json_data = json.load(f)
        print(len(json_data))
    # mean, std = [0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255] 
    import kornia as K
    gpu_val_transform_ls = [K.enhance.Normalize(mean=mean, std=std)]
    transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)                                                                                                                                                                                                                                           
    from tqdm import tqdm
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
    if config.resume:
        print("=> loading resume checkpoint '{}'".format(config.resume))
        curr_checkpoint = torch.load(config.resume, map_location='cpu')
        new_ckpt = {}

        for key,value in curr_checkpoint['state_dict'].items():
            new_key = key.replace('module.','')
            new_ckpt[new_key] = value
        result = model.load_state_dict(new_ckpt)
        print(result)
    model = model.to('cuda')
    
    model = model.eval().cuda().half()
    
    transform = build_transform(model_name)
    total0 = 0
    total1 = 0
    ac0 = 0
    ac1 = 0

    def sim_matrix(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    for key,item in tqdm(json_data.items()):

        itemMCQ = json_data[key]

        answerIndex = itemMCQ['answer']
        sampleQuery = itemMCQ['query']

        textQuery, _, _ = _get_caption(sampleQuery)

        sampleOptions = itemMCQ['choices']
        num_options = len(sampleOptions)
        textOptions = []
        videoOptions = torch.zeros([num_options, 16,crop_size,crop_size,3])

        for id, option in enumerate(sampleOptions):
            si = sampleOptions[option]
            video_id = si['video_uid']
            start = float(si['clip_start'])
            end = float(si['clip_end'])

            caption, _, _ = _get_caption(si)
            textOptions.append(caption)

            frames = video_loader(ego4d_root, video_id, 'mp4', start, end,
                    chunk_len=15, clip_length=16, threads=1, fps=30,
                    fast_rcc=True, rcc_params=(crop_size,), jitter=False)
            frames = frames.float() / 255.0
            frames = transform(frames.permute(0, 3, 1, 2))
            frames = frames.permute(0, 2, 3, 1)
            videoOptions[id] = frames

        type =  itemMCQ['types']

        videoOptions = rearrange(videoOptions,'b t h w c->b c t h w')

        videoOptions = videoOptions.to(torch.float16).to('cuda')

        data = {'video': videoOptions, 'text': textQuery, 'text_ops':textOptions, 'correct': answerIndex, 'type': type}

        data['text'] = data['text']

        text = clip.tokenize(data['text'],truncate=True).to('cuda')

        text_embed = model.encode_text(text)
        text_embed = F.normalize(text_embed, dim=-1)

        vid_embed = model.encode_image(videoOptions)[0]
        vid_embed = F.normalize(vid_embed,dim=-1)
        # text_embed = model.encode_text(text,mask)
        # text_embed = F.normalize(text_embed, dim=-1)

        data_gt = data['correct']
        data_pred = sim_matrix(text_embed, vid_embed)
        
        index = torch.argmax(data_pred)
        data_type = data['type']
        if data_type == 1:
            total0 += 1
            if index == data_gt:
                ac0 += 1
        else:
            total1 += 1
            if index == data_gt:
                ac1 += 1
        
        acc0 = round(ac0 / max(1,total0),2)
        acc1 = round(ac1 / max(1,total1),2)
        print(f'number of sample {total0 + total1}')
        print(f'inter acc: {acc0} acc0:{ac0} total:{total0}')
        print(f'intra acc: {acc1} acc1:{ac1} total:{total1}')
        print('---------------------------------------')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
