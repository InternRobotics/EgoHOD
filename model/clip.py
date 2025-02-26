import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .transformer import TextTransformer, VisionTransformer, VisionTransformer_Slowfast
from .timesformer import SpaceTimeTransformer
from ipdb import set_trace
from einops import rearrange
import torch.cuda.amp as amp

from easydict import EasyDict
import sys
import torch
from PIL import Image
import pickle
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace
import functools

# util functions to convert OpenCLIP-style model keys to ViT-style
def remap_keys_from_open_clip_to_vit(
    clip_state_dict,
    visual_transformer_layers=12,
    textual_transformer_layers=12,
    context_length=77,
    vocab_size=49408,
    use_fast_conv1=False,
    use_flash_attn=False,
):
    if 'state_dict' in clip_state_dict:
        clip_state_dict = clip_state_dict['state_dict']
    if list(clip_state_dict.keys())[0].startswith('module.'):
        clip_state_dict = OrderedDict({
            k.replace('module.', ''): v for k, v in clip_state_dict.items()
        })
    remapped_state_dict = OrderedDict()
    key_mapping = {
        "logit_scale": "logit_scale",
        "visual.proj": "visual.image_projection",
        "positional_embedding": "textual.positional_embedding",
        "text_projection": "textual.text_projection",
        "token_embedding.weight": "textual.token_embedding.weight",
        "ln_final.weight": "textual.ln_final.weight",
        "ln_final.bias": "textual.ln_final.bias"
    }

    for layer in range(visual_transformer_layers):
        if use_flash_attn:
            for src_name, tgt_name in {
                'attn.in_proj_weight': 'attn.Wqkv.weight', 'attn.in_proj_bias': 'attn.Wqkv.bias',
                'attn.out_proj.weight': 'attn.out_proj.weight', 'attn.out_proj.bias': 'attn.out_proj.bias',
                'mlp.c_fc.weight': 'mlp.fc1.weight', 'mlp.c_fc.bias': 'mlp.fc1.bias',
                'mlp.c_proj.weight': 'mlp.fc2.weight', 'mlp.c_proj.bias': 'mlp.fc2.bias',
            }.items():
                key_mapping[f"visual.transformer.resblocks.{layer}.{src_name}"] = f"visual.transformer.resblocks.{layer}.{tgt_name}"


    for layer in range(textual_transformer_layers):
        for name in [
            'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight', 'attn.out_proj.bias',
            'ln_1.weight', 'ln_1.bias', 'ln_2.weight', 'ln_2.bias',
             'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias',
        ]:
            key_mapping[f"transformer.resblocks.{layer}.{name}"] = f"textual.transformer.resblocks.{layer}.{name}"

    for key in clip_state_dict:
        if key in ["visual.proj", "text_projection", "logit_scale"]:
            continue
        if use_fast_conv1 and key == 'visual.conv1.weight':
            remapped_state_dict['visual.conv1.weight'] = clip_state_dict[key].flatten(1)
        elif key not in key_mapping:
            remapped_state_dict[key] = clip_state_dict[key]
        else:
            if key == 'positional_embedding':
                old_context_length, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                if context_length <= old_context_length:
                    remapped_state_dict[key_mapping[key]] = clip_state_dict[key][:context_length, :]
                else:
                    remapped_state_dict[key_mapping[key]] = torch.cat(
                        (clip_state_dict[key], torch.zeros((context_length - old_context_length, dim), dtype=old_dtype)), dim=0
                    )
            elif key == 'token_embedding.weight':
                old_vocab_size, dim = clip_state_dict[key].shape
                old_dtype = clip_state_dict[key].dtype
                assert vocab_size >= old_vocab_size
                remapped_state_dict[key_mapping[key]] = torch.cat(
                    (clip_state_dict[key], torch.zeros((vocab_size - old_vocab_size, dim), dtype=old_dtype)), dim=0
                )
            else:
                remapped_state_dict[key_mapping[key]] = clip_state_dict[key]

    return remapped_state_dict

def CLIP_VITB16(
    config,
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224, 16, 768, 12, 12, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )

    text_model = TextTransformer(context_length=77, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature,ckpt_path=config.lavila_path)
    
    print("=> loading openai model")
    clip_model, preprocess = clip.load(config.ckpt_path, device='cpu')
    remapped_state_dict = remap_keys_from_open_clip_to_vit(
        clip_model.state_dict(),
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )

    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)

    return model

def CLIP_VITL14_336PX(
    config,
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    vision_model = VisionTransformer(
        336, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    model = CLIP(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature,ckpt_path=config.lavila_path)

    print("=> loading openai model")
    clip_model, preprocess = clip.load(config.ckpt_path, device='cpu')
    remapped_state_dict = remap_keys_from_open_clip_to_vit(
        clip_model.state_dict(),
        visual_transformer_layers=24,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)

    return model

def CLIP_VITL14_336PX_Slowfast(
    config,
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer_Slowfast(
        336, 14, 1024, 24, 16, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=vocab_size, width=768, heads=12, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    model = CLIP_Slowfast(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature,ckpt_path=config.lavila_path)

    print("=> loading openai model")
    clip_model, preprocess = clip.load(config.ckpt_path, device='cpu')
    remapped_state_dict = remap_keys_from_open_clip_to_vit(
        clip_model.state_dict(),
        visual_transformer_layers=24,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)
    return model

def CLIP_VITB16_Slowfast(
    config,
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.,
    drop_path_rate=0.,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo='openai',
    pretrain_path=None,
    **kwargs
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer_Slowfast(
        224, 16, 768, 12, 12, 4,
        output_dim=project_embed_dim, patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(context_length=context_length, vocab_size=49408, width=512, heads=8, layers=12, output_dim=project_embed_dim, causal_mask=not use_bidirectional_lm)
    model = CLIP_Slowfast(embed_dim=project_embed_dim, vision_model=vision_model, text_model=text_model, freeze_temperature=freeze_temperature,ckpt_path=config.lavila_path)
    
    print("=> loading openai model")
    clip_model, preprocess = clip.load(config.ckpt_path, device='cpu')
    remapped_state_dict = remap_keys_from_open_clip_to_vit(
        clip_model.state_dict(),
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)
    print("missing_keys: ", missing_keys)
    print("unexpected_keys: ", unexpected_keys)

    return model

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 freeze_temperature=False,
                 ckpt_path=None,
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)

        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None
        self.init_parameters()

    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)

    def encode_visual(self, image):
        return self.encode_image(image)

    def encode_image(self, image):

        x_pooling,x = self.visual(image)
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x_pooling,x

    def encode_text(self, text, cast_dtype=None):
        if len(text.shape) > 2:
            text = text.squeeze()
        x = self.textual(text)
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x

    def forward(self,image,slow, text,eval_mode=False):

        image_embed,_ = self.encode_image(image)
        #print(image_embed.dtype)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)
        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1), self.logit_scale.exp()


class CLIP_Slowfast(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 vision_model: nn.Module,
                 text_model: nn.Module,
                 vision_width: int = None,
                 text_width: int = None,
                 freeze_temperature=False,
                 ckpt_path=None,
                 **kwargs
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)

        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None
        
        self.slowfast_projection = nn.Parameter(torch.empty(embed_dim*2, embed_dim))
        self.init_parameters()

        for n, p in self.named_parameters():
            if ('adapter' in n) or ('projection' in n) or  ('visual.class_embedding' in n) or ('visual.temporal_embedding') in n:
                p.requires_grad_(True)
                print(n,p.requires_grad)
            else:
                p.requires_grad_(False)

        
        n_trainable_params = 0
        for n, p in self.named_parameters():
            if p.requires_grad:
                n_trainable_params += p.numel()
        print('Total trainable params:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))

        
    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width ** -0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width ** -0.5)
        
        trunc_normal_(self.slowfast_projection, std=(1024) ** -0.5)

    def encode_visual(self, image):
        return self.encode_image(image)

    def encode_image(self, image):

        x_pooling,x = self.visual(image)
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x_pooling,x

    def encode_text(self, text, cast_dtype=None):
        text = text.squeeze()
        if len(text.shape) == 1:
            text = text.unsqueeze(0)
        x = self.textual(text)
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x

    def encode_slowfast(self,image, image_slow):
        image_embed_slow,all_embed_slow = self.encode_image(image_slow)
        #image_temp = rearrange(image,'b c (t1 t2) h w->(b t1) c t2 h w',t1=4,t2=4)
        image_embed,all_embed= self.encode_image(image)
        # image_embed = rearrange(image_embed,'(b t1) c->b t1 c',t1=4)
        # image_embed = image_embed.mean(dim=1)

        image_embed = torch.cat((image_embed_slow,image_embed),dim=-1)
        image_embed = image_embed @ self.slowfast_projection.to(image_embed.dtype)

        return F.normalize(image_embed, dim=-1)

    def forward(self, image, image_slow,text,eval_mode=False):

        image_embed_slow,all_embed_slow = self.encode_image(image_slow)
        #image_temp = rearrange(image,'b c (t1 t2) h w->(b t1) c t2 h w',t1=4,t2=4)
        image_embed,all_embed= self.encode_image(image)
        # image_embed = rearrange(image_embed,'(b t1) c->b t1 c',t1=4)
        # image_embed = image_embed.mean(dim=1)

        image_embed = torch.cat((image_embed_slow,image_embed),dim=-1)
        image_embed = image_embed @ self.slowfast_projection.to(image_embed.dtype)

    
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)

        return F.normalize(image_embed, dim=-1), F.normalize(text_embed, dim=-1), self.logit_scale.exp()
