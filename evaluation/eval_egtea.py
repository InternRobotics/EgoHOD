import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import argparse
from model.clip import *
from util.config import get_config
import numpy as np
import os.path as osp
import decord
import clip

def get_args_parser():
    parser = argparse.ArgumentParser('EGTEA eval', add_help=False)

    parser.add_argument('--config_file', default='clip_base_eval.yml', type=str,help='config file')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--root', default='egtea_gaze/cropped_clips', type=str,help='root of egtea video clips')
    parser.add_argument('--metadata', default='./egtea', type=str,help='root of egtea annotations')    
    parser.add_argument('--crop_size', default=224, type=int,help='root of egtea annotations')    
    return parser

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    return frame_ids.astype(int).tolist()


class VideoClassyDataset(torch.utils.data.Dataset):
    def __init__(
        self, root, metadata,crop_size=224, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        clip_length=32, clip_stride=2,
        sparse_sample=False,
        is_trimmed=True,
        anno_dir=''
    ):
        super().__init__()

        metadata = metadata

        # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]    
        mean = (0.48145466,0.4578275,0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        transform = transforms.Compose(
            [
                T.Resize((crop_size), interpolation=InterpolationMode.BICUBIC),
                T.CenterCrop(crop_size),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                T.Normalize(mean=mean, std=std)
            ]
        )  
        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.sparse_sample = sparse_sample
        self.anno_dir = anno_dir
        self.root = root
        vn_list = []
        labels = []
        for row in open(f'{metadata}/action_idx.txt'):
            row = row.strip()
            vn = int(row.split(' ')[-1])
            vn_list.append(vn)
            narration = ' '.join(row.split(' ')[:-1])
            labels.append(narration.replace('_', ' ').lower())
        self.mapping_act2narration = {vn: narration for vn, narration in zip(vn_list, labels)}
        self.samples = []
        with open(f'{metadata}/test_split1.txt') as f:
            for row in f:
                clip_id, action_idx = row.strip().split(' ')[:2]
                video_id = '-'.join(clip_id.split('-')[:3])
                vid_relpath = osp.join(video_id, '{}.mp4'.format(clip_id))
                vid_fullpath = osp.join(self.root, video_id, '{}.mp4'.format(clip_id))
                self.samples.append((vid_relpath, 0, self.mapping_act2narration[int(action_idx)]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
      
        vid,_,label = self.samples[i]

        vr = decord.VideoReader(os.path.join(self.root,vid),num_threads=1)
        fps = vr.get_avg_fps()
        if len(vr) > 12000000:
            frame_ids = get_frame_ids(0,int(len(vr) * 0.75),16)
            frame_ids_slow = get_frame_ids(0,int(len(vr) * 0.75),4)
        else:
            frame_ids = get_frame_ids(0,int(len(vr)),16)
            frame_ids_slow = get_frame_ids(0,int(len(vr)),4)
        frames = torch.from_numpy(vr.get_batch(frame_ids).asnumpy())
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        frames = self.transform(frames)


        frames_slow = torch.from_numpy(vr.get_batch(frame_ids_slow).asnumpy())
        frames_slow = frames_slow.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
        frames_slow = self.transform(frames_slow)

        return frames,frames_slow,label



def main(args):
    config = get_config(args)

    root = args.root
    metadata = args.metadata
    crop_size = args.crop_size

    dataset = VideoClassyDataset(root,metadata,crop_size)

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

    model = model.to('cuda')

    if config.resume:
        print("=> loading resume checkpoint '{}'".format(config.resume))
        curr_checkpoint = torch.load(config.resume, map_location='cpu')
        new_ckpt = {}

        for key,value in curr_checkpoint['state_dict'].items():
            new_key = key.replace('module.','')
            new_ckpt[new_key] = value
        result = model.load_state_dict(new_ckpt)
        print(result)
    
    model = model.eval().cuda().half()
    
    words = []
    words_origin = []
    narration2act = {}
    for i in range(1,107):
        word = dataset.mapping_act2narration[i]
        narration2act[word] = i
        text = clip.tokenize(word).to('cuda')
        text_embed = model.encode_text(text)
        words.append(F.normalize(text_embed, dim=-1))
        
    words = torch.stack(words)
    words = words.squeeze()

    ans = []
    total = 0
    acc = 0



    acc_total = [0 for i in range(106)]
    acc_acc = [0 for i in range(106)]

    for i in range(len(dataset)):
        with torch.no_grad():
            frames,frames_slow,label = dataset[i]
            frames = frames.to('cuda').unsqueeze(0).to(torch.float16)
            frames = frames.permute(0, 2, 1, 3, 4)

            frames_slow = frames_slow.to('cuda').unsqueeze(0).to(torch.float16)
            frames_slow = frames_slow.permute(0, 2, 1, 3, 4)

            image_embed = model.encode_image(frames)[0]
            image_embed = F.normalize(image_embed, dim=-1)

            similarities = F.cosine_similarity(image_embed, words, dim=1)
            
            most_similar_index = torch.argmax(similarities)
            index2word = dataset.mapping_act2narration[most_similar_index.item() + 1]
            #label2word = dataset.mapping_act2narration[label]
            print(f"ans: {label} our: {index2word}")

            ans.append(most_similar_index.item())
            total += 1
            if most_similar_index.item() + 1 == narration2act[label]:
                acc += 1
                acc_acc[narration2act[label] - 1] += 1

            acc_total[narration2act[label] - 1] += 1
            
            print(f'acc: {acc / total}')
            print('---------------------------------')

    mean_acc = 0
    for k in range(106):
        mean_acc += acc_acc[k] / acc_total[k]
    mean_acc = mean_acc / 106.0
    print(f'mean acc is {mean_acc}')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
