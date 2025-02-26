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
from decord import cpu
import io
from ipdb import set_trace
from .data_utils import video_loader
from petrel_client.client import Client
import ast
import clip
client = Client()

class EgoExoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, transform=None, is_training=True, tokenizer=None, crop_size=224,
            subsample_stride=None):
        self.cfg = cfg
        self.dataset = cfg.dataset
        self.ego4d_root = cfg.ego4d_root
        self.ego4d_metadata = cfg.ego4d_metadata
        self.ego4d_chunk_len = cfg.ego4d_video_chunk_len
        self.ego4d_fps = cfg.ego4d_fps

        self.howto_root = cfg.howto_root
        self.howto_metadata = cfg.howto_metadata
        self.howto_chunk_len = cfg.howto_video_chunk_len        
        self.howto_fps = cfg.howto_fps
        
        self.is_trimmed = cfg.is_trimmed
        ### hardcode this for now ###
        self.narration_selection = 'random'

        if self.dataset == 'ego4d':
            self.samples = pd.read_csv(self.ego4d_metadata)
            if cfg.ego4d_metadata_aux is not None:
                self.aux_samples = pd.read_csv(cfg.ego4d_metadata_aux)
                self.samples = pd.concat([self.samples, self.aux_samples])

        elif self.dataset == 'htego':
            self.samples = pd.read_csv(self.howto_metadata)
        elif self.dataset == 'ego4d_htego':
            self.ego4d_samples = pd.read_csv(self.ego4d_metadata)
            if cfg.ego4d_metadata_aux is not None:
                self.aux_samples = pd.read_csv(cfg.ego4d_metadata_aux)
                self.ego4d_samples = pd.concat([self.ego4d_samples, self.aux_samples])
            
            self.htego_samples = pd.read_csv(self.howto_metadata)
            self.samples = pd.concat([self.ego4d_samples, self.htego_samples])
        else:
            raise NotImplementedError
        print(len(self.samples))
        self.full_samples = self.samples.copy()
        if isinstance(subsample_stride, int):
            self.samples = self.samples[::subsample_stride]
        

        self.transform = transform
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.clip_length = cfg.clip_length
        self.clip_stride = cfg.clip_stride
        self.threads = cfg.decode_threads
        self.context_length = cfg.context_length
        print(f'sentence length {self.context_length}')
        self.multiview = cfg.multiview

        self.fast_rrc = cfg.fused_decode_crop
        self.rrc_params = (crop_size, (0.5, 1.0))


    def __len__(self):
        return len(self.samples)
    
    def process_text(self, narration):
        ### this is a list of narrations ###
        if narration[0] == '[' and narration[-1] == ']':
            narration = ast.literal_eval(narration)
            if self.narration_selection == 'random':
                narration = random.choice(narration)
            elif self.narration_selection == 'concat':
                narration = '. '.join(narration)
            else:
                raise NotImplementedError
        
        return narration

    def __getitem__(self, i):
        try:
            ### get indicator ###
            curr = self.samples.iloc[i]
            curr_dataset = curr['dataset'] if 'dataset' in curr else 'howto_ego'
            exo_vid_path = ''
            #print(curr['video_id'],curr_dataset)
            ### get data ###

            if curr_dataset == 'ego4d':
                
                vid, start_second, end_second, narration = curr['video_id'], curr['start_second'], curr['end_second'], curr['text']
                # print(f'Getting ego video {vid} from {start_second} to {end_second}')

                frames = video_loader(self.ego4d_root, vid, 'mp4', start_second, end_second,
                        chunk_len=self.ego4d_chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.ego4d_fps,
                        fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)


                narration = self.process_text(narration)
                frames_slow = frames
                exo_frames = torch.zeros_like(frames)

            else:
                vid = vid_path = curr['video_id'] 
                start_second, end_second, narration = curr['start_second'], curr['end_second'], curr['text']
                uid = curr['uid'] if 'uid' in curr else '{}_{}'.format(vid, start_second)

                frames = video_loader(self.howto_root, vid_path, 'mp4', start_second, end_second,
                        chunk_len=self.howto_chunk_len, clip_length=self.clip_length, threads=self.threads, fps=self.howto_fps,
                        fast_rrc=self.fast_rrc, rrc_params=self.rrc_params, jitter=self.is_training)
                frames_slow = frames
                
            raw_caption = narration

            if self.transform is not None:
                frames = frames.float() / 255.0
                frames = self.transform(frames.permute(0, 3, 1, 2))
                frames_slow = self.transform(frames_slow.permute(0, 3, 1, 2))

            if self.tokenizer is not None:
                narration = narration.replace('\n','')
                caption = self.tokenizer(narration)   
            else:
                narration = narration.replace('\n','')
                caption = clip.tokenize(narration,context_length=77, truncate=True)
            return frames, frames_slow,caption
                    
        except Exception as e:
            print(f'Error with sample {i}: {exo_vid_path} dataset:{curr_dataset} error {e}')
            ids = np.random.randint(0, len(self.samples))
            return self.__getitem__(ids)
