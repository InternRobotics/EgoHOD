
import csv
import glob
import json
import numpy as np
import os.path as osp
import pickle
import random

import decord
import pandas as pd
import torch
from ipdb import set_trace
import cv2
import io,os

from nltk.stem import WordNetLemmatizer
from .data_utils import datetime2sec, get_frame_ids
from .data_utils import video_loader_by_frames, video_loader_by_timestamp, video_loader
from .data_utils import generate_label_map
from petrel_client.client import Client
import clip 

class EK100Dataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None, is_training=False, tokenizer=None, crop_size=224):
        ### common setups ###
        self.config = config
        self.root = config.root
        self.metadata = config.metadata
        self.clip_length = config.clip_length
        self.clip_stride = config.clip_stride
        ### maybe customized ###
        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        
        self.chunk_len = config.video_chunk_len
        self.fps = config.fps
        self.threads = config.decode_threads
        
        if is_training:
            self.fast_rrc = config.fused_decode_crop
            self.rrc_params = (crop_size, (0.5, 1.0))
        else:
            self.fast_rcc = config.fused_decode_crop
            self.rcc_params = (crop_size,)

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
        
        # if self.dataset == 'ek100_mir':
        self.metadata_sentence = pd.read_csv(self.metadata[:self.metadata.index('.csv')] + '_sentence.csv')
        if 'train' in self.metadata:
            self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_train.pkl'), 'rb'))
        elif 'test' in self.metadata:
            self.relevancy_mat = pickle.load(open(osp.join(osp.dirname(self.metadata), 'relevancy', 'caption_relevancy_EPIC_100_retrieval_test.pkl'), 'rb'))
        else:
            raise ValueError('{} should contain either "train" or "test"!'.format(self.metadata))
        self.relevancy = .1

        print(self.threads)
    def __len__(self):
        return len(self.samples)
    
    def get_raw_item(self, i):
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]
        # frames = video_loader_by_timestamp(self.root, vid_path, 
        #     start_timestamp=start_frame, end_timestamp=end_frame, 
        #     clip_length=self.clip_length, is_training=self.is_training, 
        #     threads=self.threads, fast_rcc=self.fast_rcc, rcc_params=self.rcc_params
        # )

        if self.is_training:
            frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
            frames_slow = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                chunk_len=self.chunk_len, clip_length=4, threads=self.threads, fps=self.fps,
                fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
        else:
            while True:
                try:
                    frames = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                        chunk_len=self.chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.fps,
                        fast_rcc=self.fast_rcc, rcc_params=self.rcc_params, jitter=self.is_training)
                    frames_slow = video_loader(self.root, vid_path.replace('.mp4',''), 'mp4', start_timestamp, end_timestamp,
                        chunk_len=self.chunk_len, clip_length=4, threads=self.threads, fps=self.fps,
                        fast_rcc=self.fast_rcc, rcc_params=self.rcc_params, jitter=self.is_training)   
                    break
                except:
                    continue     

        if self.transform is not None:
            frames = frames.float() / 255.0
            frames = self.transform(frames.permute(0, 3, 1, 2))
            frames_slow = self.transform(frames_slow.permute(0, 3, 1, 2))
                    
        if self.is_training:
            positive_list = np.where(self.relevancy_mat[i] > self.relevancy)[0].tolist()
            if positive_list != []:
                pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                    return frames, frames_slow, self.metadata_sentence.iloc[pos][1], self.relevancy_mat[i][pos]
        else:
            return frames, frames_slow,narration, 1
        
    
    def __getitem__(self, i):
        ### for record info only ###
        vid_path, start_timestamp, end_timestamp, narration, verb, noun = self.samples[i]
        uid = vid_path
        raw_caption = narration

        frames, frames_slow,narration, relevancy = self.get_raw_item(i)
        
        #### this is for ek100_cls ###
        # if self.config.dataset == 'ek100_cls':
        #     return frames, '{}:{}'.format(verb, noun)

        #### this is for ek100_mir ###
        caption = clip.tokenize(narration,context_length=77, truncate=True)
                
        return frames,frames_slow, caption,relevancy     



class EK100Dataset_CLS(torch.utils.data.Dataset):
    def __init__(self, config, transform=None, is_training=False, tokenizer=None, crop_size=224,use_bert=False):
        ### common setups ###
        self.root = config.root
        self.metadata = config.metadata
        self.clip_length = config.clip_length
        self.clip_stride = config.clip_stride
        
        ### maybe customized ###
        self.transform = transform
        self.is_training = is_training
        self.use_bert = use_bert
        self.tokenizer = tokenizer
        
        self.chunk_len = config.video_chunk_len
        self.fps = config.fps
        self.threads = config.decode_threads
        
        if is_training:
            self.fast_rrc = config.fused_decode_crop
            self.rrc_params = (crop_size, (0.5, 1.0))
        else:
            self.fast_rcc = config.fused_decode_crop
            self.rcc_params = (crop_size,)

        
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

        self.labels, self.label_mapping = generate_label_map('ek100_cls', config.metapath)

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
        return frames,frames_slow, f'{verb}:{noun}', narration
    
    def __getitem__(self, i):
        ### for record info only ###

        vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]

        frames, frames_slow, label, narration = self.get_raw_item(i)
        raw_caption = narration

        frames = self.transform(frames) if self.transform is not None else None

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
        

if __name__ == "__main__":
    import os
    import time
    from pathlib import Path

    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.tensorboard import SummaryWriter
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    mean, std = (0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)
    transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=mean, std=std)])
    val_dataset = EK100Dataset('epic_kitchen/', transform=transform_val, is_training=False, tokenizer=None, crop_size=224)

