import numpy as np
import os.path as osp

import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import pandas as pd
import torch
from ipdb import set_trace
import cv2
import io,os

import torch
import os
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

import time
import func_timeout
from func_timeout import func_set_timeout

import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import pandas as pd
import torch
from decord import cpu
import cv2
import io,os
import argparse

import decord
from model.clip import *
from util.config import get_config
from dataset.data_utils import video_loader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
try:
    from petrel_client.client import Client
    client = Client()

    # Disable boto logger
    import logging
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)
except:
    client = None

def get_args_parser():
    parser = argparse.ArgumentParser('EK-CLS eval', add_help=False)

    parser.add_argument('--config_file', default='configs/no_decoder/clip_base_eval.yml', type=str,help='config file')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--root', default='epic/epic_video_320p/', type=str)
    parser.add_argument('--metadata', default='epic_kitchen/', type=str,help='root of egtea annotations')    
    parser.add_argument('--crop_size', default=224, type=int,help='root of egtea annotations')    
    return parser

def generate_label_map(dataset, metapath):
    if dataset == 'ek100_cls':
        print("Preprocess ek100 action label space")
        vn_list = []
        mapping_vn2narration = {}
        for f in [
            f'{metapath}epic-kitchens-100-annotations/EPIC_100_train.csv',
            f'{metapath}epic-kitchens-100-annotations/EPIC_100_validation.csv',
        ]:
            csv_reader = csv.reader(open(f))
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                vn = '{}:{}'.format(int(row[10]), int(row[12]))
                narration = row[8]
                if vn not in vn_list:
                    vn_list.append(vn)
                if vn not in mapping_vn2narration:
                    mapping_vn2narration[vn] = [narration]
                else:
                    mapping_vn2narration[vn].append(narration)
                # mapping_vn2narration[vn] = [narration]
        vn_list = sorted(vn_list)
        print('# of action= {}'.format(len(vn_list)))
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        labels = [list(set(mapping_vn2narration[vn_list[i]])) for i in range(len(mapping_vn2act))]
        print(labels[:5])
    elif dataset == 'charades_ego':
        print("=> preprocessing charades_ego action label space")
        vn_list = []
        labels = []
        with open(f'{metapath}Charades_v1_classes.txt') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                vn = row[0][:4]
                vn_list.append(vn)
                narration = row[0][5:]
                labels.append(narration)
        mapping_vn2act = {vn: i for i, vn in enumerate(vn_list)}
        print(labels[:5])
    elif dataset == 'egtea':
        print("=> preprocessing egtea action label space")
        labels = []
        with open(f'{metapath}action_idx.txt') as f:
            for row in f:
                row = row.strip()
                narration = ' '.join(row.split(' ')[:-1])
                labels.append(narration.replace('_', ' ').lower())
                # labels.append(narration)
        mapping_vn2act = {label: i for i, label in enumerate(labels)}
        print(len(labels), labels[:5])
    else:
        raise NotImplementedError
    return labels, mapping_vn2act

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

class EK100Dataset_CLS(torch.utils.data.Dataset):
    def __init__(self,root,metadata,crop_size=224):
        ### common setups ###
        self.root = root
        self.metadata = f'{metadata}/EPIC_100_retrieval_test.csv'
        self.clip_length = 16
        self.clip_stride = 2
        
        ### maybe customized ###
        self.transform = None
        self.is_training = False
        
        self.chunk_len = -1
        self.fps = -1
        self.threads = True


        self.fast_rcc = True
        self.rcc_params = (crop_size,)
        mean, std = (0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)
        self.transform = transforms.Compose([
            transforms.Resize(crop_size, interpolation=3),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=mean, std=std)])
        self.samples = []
        with open(self.metadata) as f:
            csv_reader = csv.reader(f)
            _ = next(csv_reader)  # skip the header
            for row in csv_reader:
                pid, vid = row[1:3]
                # start_frame, end_frame = int(row[6]), int(row[7])
                # Deprecated: some videos might have fps mismatch issue
                start_timestamp, end_timestamp = datetime2sec(row[4]), datetime2sec(row[5])
                narration = row[8]
                verb, noun = int(row[10]), int(row[12])
                
                vid_path = '{}.mp4'.format(vid)
                self.samples.append((vid_path, start_timestamp, end_timestamp, narration, verb, noun))

        self.labels, self.label_mapping = generate_label_map('ek100_cls', metadata)

    def __len__(self):
        return len(self.samples)
    
    def get_raw_item(self, i):
        # vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]
        # frames = video_loader_by_timestamp(self.root, vid_path, 
        #     start_timestamp=start_frame, end_timestamp=end_frame, 
        #     clip_length=self.clip_length, is_training=self.is_training, 
        #     threads=self.threads, fast_rcc=self.fast_rcc, rcc_params=self.rcc_params
        # )
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]

        if self.is_training:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
            # frames_slow = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
            #     chunk_len=self.chunk_len, clip_length=4, threads=self.threads, fps=self.fps,
            #     fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
            frames_slow = frames
        else:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rcc=self.fast_rcc, rcc_params=self.rcc_params, jitter=self.is_training)
            frames_slow = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=4, threads=self.threads, fps=self.fps,
                fast_rcc=self.fast_rcc, rcc_params=self.rcc_params, jitter=self.is_training)   
            frames = frames.float() / 255.0
            frames = self.transform(frames.permute(0, 3, 1, 2))

        return frames,frames_slow, f'{verb}:{noun}', narration
    
    def __getitem__(self, i):
        ### for record info only ###
        vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]

        frames, frames_slow, label, narration = self.get_raw_item(i)
        raw_caption = narration


        if isinstance(label, list):
            # multi-label case
            res_array = np.zeros(len(self.label_mapping))
            for lbl in label:
                res_array[self.label_mapping[lbl]] = 1.
            label = res_array
        else:
            raw_label = label
            label = self.label_mapping[label]

        return frames, frames_slow, label

def main(args):
    config = get_config(args)
    root = args.root
    crop_size = args.crop_size
    metadata = args.metadata
    dataset = EK100Dataset_CLS(root,metadata)


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
    ans = []


    text_features = []
    labels = dataset.labels
    num_clips = 16
    templates = ['{}']
    with torch.no_grad():
        for label in labels:
            if isinstance(label, list):
                texts = [tmpl.format(lbl) for tmpl in templates for lbl in label]
            else:
                texts = [tmpl.format(label) for tmpl in templates]
            texts = clip.tokenize(texts).to('cuda')
            
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

            text_features.append(class_embeddings)
    text_features = torch.stack(text_features, dim=0)

    mean, std = [0.485* 255, 0.456* 255, 0.406* 255], [0.229* 255, 0.224* 255, 0.225* 255]  
    mean = (0.48145466 * 255,0.4578275 * 255,0.40821073 * 255)
    std = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
    import kornia as K
    gpu_val_transform_ls = [K.enhance.Normalize(mean=mean, std=std)]
    transform_gpu = torch.nn.Sequential(*gpu_val_transform_ls)

    top1ac = 0
    top1total = 0

    top5ac = 0
    top5total = 0

    for i in range(len(dataset)):
        with torch.no_grad():
            frames,frames_slow,label = dataset[i]
            frames = frames.to('cuda').unsqueeze(0).to(torch.float16)
            frames = frames.permute(0, 2, 1, 3, 4)
            image_embed = model.encode_image(frames)[0]
            image_embed = F.normalize(image_embed, dim=-1)
            similarities = F.cosine_similarity(image_embed, text_features, dim=1)
            top1_values, top1_indices = torch.topk(similarities, k=1, dim=-1)
            top5_values, top5_indices = torch.topk(similarities, k=5, dim=-1)
            #label2word = dataset.mapping_act2narration[label]

            top1total += 1
            top5total += 1

            if label in top1_indices:
                top1ac += 1
            if label in top5_indices:
                top5ac += 1

            print(f'top1acc: {top1ac / top1total}')
            print(f'large top5acc: {top5ac / top5total}')
            print('---------------------------------')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
