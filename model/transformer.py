# adapted from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py

from collections import OrderedDict
from functools import partial
from typing import Callable, List, Optional
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.layers import trunc_normal_
from timm.models.layers import to_2tuple
from timm.models.layers import DropPath

try:
    from flash_attn.modules.mha import MHA as FlashMHA
    from flash_attn.modules.mlp import Mlp as FlashMlp
except:
    print('First pip install flash-attn')
    
from ipdb import set_trace
from typing import Tuple
from collections import OrderedDict
import math
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x

from mmengine.model.weight_init import constant_init, trunc_normal_init

class Adapter(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # temporal depth-wise convolution
        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=hidden_dims,
        )

        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)

        # self.dwconv2 = nn.Conv1d(
        #     hidden_dims,
        #     hidden_dims,
        #     kernel_size=kernel_size,
        #     stride=1,
        #     padding=(kernel_size // 2) * dilation,
        #     dilation=dilation,
        #     groups=hidden_dims,
        # )

        #self.conv2 = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.conv2d = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3),padding=1)
        # self.conv_spatial = nn.Conv1d(hidden_dims, hidden_dims, 1)

        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()
        self.conv2d.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.conv2d.bias.data.zero_()
        # self.conv_spatial.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        # self.conv_spatial.bias.data.zero_()


        # self.batch2d = nn.BatchNorm2d(384)
        # self.relu = nn.ReLU(inplace=True)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0


    def forward(self, x):

        if x.shape[-1] == 768:
            h = 14
            w = 14
        else:
            h = 24
            w = 24         
        x_origin = x
        inputs = x[:,1:,:]
        x = x[:,1:,:]
        # [B,16*196,768]
        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape  # 48, 16*14*14, 384  

        # attn_spatial = rearrange(x,'b (t h w) c->(b t) c h w',t=self.temporal_size,h=h,w=w)
        # attn_spatial = self.conv2d(attn_spatial)

        # attn_spatial = self.relu(self.batch2d)

        # attn_spatial = rearrange(attn_spatial,'B C H W->B C (H W)')
        # attn_spatial = self.conv_spatial(attn_spatial)
        # attn_spatial = rearrange(attn_spatial,'B C (H W)->B C H W',H=h,W=w)
        # attn_spatial = rearrange(attn_spatial,'(B T) C H W->B T H W C',T=self.temporal_size)
        # attn_spatial = attn_spatial.reshape(B, N, C)

        # attn = rearrange(attn_spatial,'(B T) C H W->B T H W C',T=self.temporal_size)
        attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]

        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
        attn = attn.reshape(B, N, C)
        #x = x + attn + attn_spatial
        
        x = x + attn

        x = self.up_proj(x)
        x_new = x * self.gamma + inputs
        x_origin[:,1:,:] = x_new
        return x_origin

# class Adapter(nn.Module):
#     def __init__(
#         self,
#         embed_dims: int,
#         mlp_ratio: float = 0.25,
#         kernel_size: int = 3,
#         dilation: int = 1,
#         temporal_size: int = 384,
#     ) -> None:
#         super().__init__()

#         hidden_dims = int(embed_dims * mlp_ratio)

#         # temporal depth-wise convolution
#         self.temporal_size = temporal_size
#         self.dwconv = nn.Conv1d(
#             hidden_dims,
#             hidden_dims,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=(kernel_size // 2) * dilation,
#             dilation=dilation,
#             groups=hidden_dims,
#         )

#         self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)

#         # self.dwconv2 = nn.Conv1d(
#         #     hidden_dims,
#         #     hidden_dims,
#         #     kernel_size=kernel_size,
#         #     stride=1,
#         #     padding=(kernel_size // 2) * dilation,
#         #     dilation=dilation,
#         #     groups=hidden_dims,
#         # )

#         #self.conv2 = nn.Conv1d(hidden_dims, hidden_dims, 1)
#         self.conv2d = nn.Conv2d(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=(3, 3),padding=1)
#         # self.conv_spatial = nn.Conv1d(hidden_dims, hidden_dims, 1)

#         self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
#         self.dwconv.bias.data.zero_()
#         self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
#         self.conv.bias.data.zero_()
#         self.conv2d.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
#         self.conv2d.bias.data.zero_()
#         # self.conv_spatial.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
#         # self.conv_spatial.bias.data.zero_()

#         #self.layer_norm = nn.LayerNorm([384, 14, 14])  # Normalize over C, H, W dimensions
#         self.batch2d = nn.BatchNorm2d(hidden_dims)
#         self.relu = nn.ReLU(inplace=True)

#         # adapter projection
#         self.down_proj = nn.Linear(embed_dims, hidden_dims)
#         self.act = nn.GELU()
#         self.up_proj = nn.Linear(hidden_dims, embed_dims)
#         self.gamma = nn.Parameter(torch.ones(1))
#         trunc_normal_init(self.down_proj, std=0.02, bias=0)
#         constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

#     def forward(self, x):
#         if x.shape[-1] == 768:
#             h = 14
#             w = 14
#         else:
#             h = 24
#             w = 24         
#         x_origin = x
#         inputs = x[:,1:,:]
#         x = x[:,1:,:]
#         # [B,16*196,768]
#         # down and up projection
#         x = self.down_proj(x)
#         x = self.act(x)

#         # temporal depth-wise convolution
#         B, N, C = x.shape  # 48, 16*14*14, 384  

#         attn_spatial = rearrange(x,'b (t h w) c->(b t) c h w',t=self.temporal_size,h=h,w=w)
#         attn_spatial = self.conv2d(attn_spatial)

#         #attn_spatial = self.layer_norm(attn_spatial)
#         attn_spatial - self.batch2d(attn_spatial)
#         attn_spatial = rearrange(attn_spatial,'(B T) C H W->B (T H W) C',T=self.temporal_size)
#         attn_spatial = self.relu(attn_spatial)
#         attn_spatial = rearrange(attn_spatial,'B (T H W) C->(B T) C H W',T=self.temporal_size,H=h,W=w)

#         attn = rearrange(attn_spatial,'(B T) C H W->B T H W C',T=self.temporal_size)

#         #attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]
        
#         attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
#         attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
#         attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
#         attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
#         attn = attn.reshape(B, N, C)
#         #x = x + attn + attn_spatial
        
#         x = x + attn

#         x = self.up_proj(x)
#         x_new = x * self.gamma + inputs
#         x_origin[:,1:,:] = x_new
#         return x_origin


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        else:
            # self.attn = FlashMHA(d_model, n_head, cross_attn=False, bias=True, dropout=attn_drop, use_flash_attn=True)
            self.attn = FlashMHA(d_model, n_head, cross_attn=False, qkv_proj_bias=True,
                                 out_proj_bias=True, dropout=attn_drop, use_flash_attn=True)
            
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if not use_flash_attn:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("drop1", nn.Dropout(drop)),
                ("c_proj", nn.Linear(mlp_width, d_model)),
                ("drop2", nn.Dropout(drop)),
            ]))
        else:
            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.use_flash_attn:
            x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask)))
        else:
            x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x))))
        x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))
        return x
    
class ResidualAttentionBlock_Slowfast(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()

        self.embed_dim = d_model
        self.ln_1 = norm_layer(d_model)
        self.use_flash_attn = use_flash_attn
        if not use_flash_attn:
            self.attn = nn.MultiheadAttention(d_model, n_head, dropout=attn_drop)
        else:
            # self.attn = FlashMHA(d_model, n_head, cross_attn=False, bias=True, dropout=attn_drop, use_flash_attn=True)
            self.attn = FlashMHA(d_model, n_head, cross_attn=False, qkv_proj_bias=True,
                                 out_proj_bias=True, dropout=attn_drop, use_flash_attn=True)
            
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        if not use_flash_attn:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("drop1", nn.Dropout(drop)),
                ("c_proj", nn.Linear(mlp_width, d_model)),
                ("drop2", nn.Dropout(drop)),
            ]))
        else:
            self.mlp = FlashMlp(d_model, hidden_features=mlp_width, activation=act_layer())
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.embed_dim == 768:
            self.adapter = Adapter(
                embed_dims=768,
                kernel_size=3,
                dilation=1,
                temporal_size=16,
                mlp_ratio=0.5,
            )
        else:
            self.adapter = Adapter(
                embed_dims=1024,
                kernel_size=3,
                dilation=1,
                temporal_size=16,
                mlp_ratio=0.5,
            )            

        if self.embed_dim == 768:
            self.adapter_2 = Adapter(
                embed_dims=768,
                kernel_size=3,
                dilation=1,
                temporal_size=16,
                mlp_ratio=0.5,
            )
        else:
            self.adapter_2 = Adapter(
                embed_dims=1024,
                kernel_size=3,
                dilation=1,
                temporal_size=16,
                mlp_ratio=0.5,
            )  
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        attn_mask = attn_mask.to(x.dtype) if attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):     
        # if self.embed_dim == 768:
        #     if x.shape[1] != 785:
        #         x = self.adapter_2(x)  
        # else:
        #     if x.shape[1] != 2305:
        #         x = self.adapter_2(x)   
        #x = self.adapter_pre_attn(x, 16)
        if x.shape[1] != 785:
            with torch.no_grad():
                if not self.use_flash_attn:
                    x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask)))
                else:
                    x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x))))

                x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))  
        else:
            if not self.use_flash_attn:
                x = x + self.drop_path(self.ls_1(self.attention(self.ln_1(x), attn_mask=attn_mask)))
            else:
                x = x + self.drop_path(self.ls_1(self.attn(self.ln_1(x))))

            x = x + self.drop_path(self.ls_2(self.mlp(self.ln_2(x))))              

        if self.embed_dim == 768:
            if x.shape[1] != 785:
                x = self.adapter(x)  
        else:
            if x.shape[1] != 2305:
                x = self.adapter(x)              
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_flash_attn=use_flash_attn)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
    
class Transformer_Slowfast(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock_Slowfast(
                width, heads, mlp_ratio, ls_init_value=ls_init_value,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                use_flash_attn=use_flash_attn)
            for nnnn in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            num_frames: int = 1,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            output_dim: int = None,
            patch_dropout: float = 0.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            ln_pre: bool = True,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
            use_fast_conv1: bool = False,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_fast_conv1 = use_fast_conv1
        self.use_flash_attn = use_flash_attn
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.width = width
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.output_dim = output_dim
        if use_fast_conv1:
            self.conv1 = nn.Linear(in_features=3 * patch_size ** 2, out_features=width, bias=not ln_pre)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        assert num_frames >= 1
        self.num_frames = num_frames
        if num_frames > 1:
            self.temporal_embedding = nn.Parameter(torch.zeros(num_frames, width))

        assert not (patch_dropout > 0. and drop_rate > 0.)
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        if ln_pre:
            self.ln_pre = norm_layer(width)
        else:
            self.ln_pre = nn.Identity()

        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_flash_attn=use_flash_attn,
        )

        self.global_average_pool = global_average_pool
        self.ln_post = norm_layer(width)
        if output_dim is None:
            self.image_projection = None
        else:
            self.image_projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        # TODO: compare the two styles
        # Mimicking timm's initialization
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'weight' in n:
                    trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError('Unknown parameters named {}'.format(n)) 
        if self.image_projection is not None:
            nn.init.normal_(self.image_projection, std=self.width ** -0.5)

        # Same init as TextTransformer
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.image_projection is not None:
        #     nn.init.normal_(self.image_projection, std=self.output_dim ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def get_pos_embedding(self, x: torch.Tensor, curr_frames: int):
        # x: [b,c,f,h,w]
        cls_embed = self.positional_embedding[0, :].unsqueeze(0)
        if self.num_frames == curr_frames:
            tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
            tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
        else:
            tile_pos_embed = self.positional_embedding[1:, :].repeat(curr_frames, 1)
            new_temporal_embed = F.interpolate(self.temporal_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.temporal_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            tile_temporal_embed = torch.nn.Parameter(new_temporal_embed).to(self.temporal_embedding.device)
            tile_temporal_embed = tile_temporal_embed.repeat_interleave(self.patches_per_frame, 0)
            
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
        return total_pos_embed

    def forward(self, x: torch.Tensor, return_dense=False):
        x = x.to(torch.float16)
        curr_frames = x.size(2)
        if self.use_fast_conv1:
            if self.num_frames == 1:
                x = rearrange(x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:

                x = rearrange(x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])

                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                
                # cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                # tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                # tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                # total_pos_embed = tile_pos_embed + tile_temporal_embed
                # total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                
                total_pos_embed = self.get_pos_embedding(x, curr_frames)

                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        else:
            if self.num_frames == 1:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W =>  B, T, C, H, W
                B, F, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                x = self.conv1(x)
                x = x.flatten(2).transpose(2, 1)    # BT, C', H, W => BT, HW, C'
                x = x.reshape(B, -1, self.width)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                # cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                # tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                # tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                # total_pos_embed = tile_pos_embed + tile_temporal_embed
                # total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                
                total_pos_embed = self.get_pos_embedding(x, curr_frames)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.pos_drop(x)

        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x)

        if return_dense:
            if self.global_average_pool:
                return self.ln_post(x)
            else:
                return self.ln_post(x[:,1:])
        
        if self.global_average_pool:
            x_pooling = x.mean(dim=1)
        else:
            x_pooling = x[:, 0]

        x_pooling = self.ln_post(x_pooling)

        if self.image_projection is not None:
            x = x @ self.image_projection
            x_pooling = x_pooling @ self.image_projection
        return x_pooling,x


class VisionTransformer_Slowfast(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            num_frames: int = 1,
            ls_init_value: float = None,
            global_average_pool: bool = False,
            output_dim: int = None,
            patch_dropout: float = 0.,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            ln_pre: bool = True,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
            use_fast_conv1: bool = False,
            use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_fast_conv1 = use_fast_conv1
        self.use_flash_attn = use_flash_attn
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.width = width
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.patches_per_frame = self.grid_size[0] * self.grid_size[1]
        self.output_dim = output_dim
        if use_fast_conv1:
            self.conv1 = nn.Linear(in_features=3 * patch_size ** 2, out_features=width, bias=not ln_pre)
        else:
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=not ln_pre)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        assert num_frames >= 1
        self.num_frames = num_frames
        if num_frames > 1:
            self.temporal_embedding = nn.Parameter(torch.zeros(num_frames, width))

        assert not (patch_dropout > 0. and drop_rate > 0.)
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        if ln_pre:
            self.ln_pre = norm_layer(width)
        else:
            self.ln_pre = nn.Identity()

        self.transformer = Transformer_Slowfast(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_flash_attn=use_flash_attn,
        )

        self.global_average_pool = global_average_pool
        self.ln_post = norm_layer(width)
        if output_dim is None:
            self.image_projection = None
        else:
            self.image_projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        # TODO: compare the two styles
        # Mimicking timm's initialization
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.02)

        for block in self.transformer.resblocks:
            for n, p in block.named_parameters():
                if 'adapter' in n:
                    continue
                if 'weight' in n:
                    trunc_normal_(p, std=0.02)
                elif 'bias' in n:
                    nn.init.zeros_(p)
                else:
                    raise NotImplementedError('Unknown parameters named {}'.format(n)) 
        if self.image_projection is not None:
            nn.init.normal_(self.image_projection, std=self.width ** -0.5)

        # Same init as TextTransformer
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.image_projection is not None:
        #     nn.init.normal_(self.image_projection, std=self.output_dim ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def get_pos_embedding(self, x: torch.Tensor, curr_frames: int):
        # x: [b,c,f,h,w]
        cls_embed = self.positional_embedding[0, :].unsqueeze(0)
        if self.num_frames == curr_frames:
            tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
            tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
        else:
            tile_pos_embed = self.positional_embedding[1:, :].repeat(curr_frames, 1)
            new_temporal_embed = F.interpolate(self.temporal_embedding.unsqueeze(0).unsqueeze(0), (curr_frames, self.temporal_embedding.shape[-1]), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
            tile_temporal_embed = torch.nn.Parameter(new_temporal_embed).to(self.temporal_embedding.device)
            tile_temporal_embed = tile_temporal_embed.repeat_interleave(self.patches_per_frame, 0)
            
        total_pos_embed = tile_pos_embed + tile_temporal_embed
        total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
        return total_pos_embed

    def forward(self, x: torch.Tensor, return_dense=False):
        curr_frames = x.size(2)
        if self.use_fast_conv1:
            if self.num_frames == 1:
                x = rearrange(x, "b c (hh sh) (ww sw) -> b (hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:

                x = rearrange(x, "b c t (hh sh) (ww sw) -> b (t hh ww) (c sh sw)", sh=self.patch_size[0], sw=self.patch_size[1])
                x = self.conv1(x)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                
                # cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                # tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                # tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                # total_pos_embed = tile_pos_embed + tile_temporal_embed
                # total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                
                total_pos_embed = self.get_pos_embedding(x, curr_frames)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)
        else:
            if self.num_frames == 1:
                x = self.conv1(x)  # shape = [*, width, grid, grid]
                x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
                x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                x = x + self.positional_embedding.to(x.dtype)
            else:
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # B, C, T, H, W =>  B, T, C, H, W
                B, F, C, H, W = x.shape
                x = x.view(-1, C, H, W)
                x = self.conv1(x)
                x = x.flatten(2).transpose(2, 1)    # BT, C', H, W => BT, HW, C'
                x = x.reshape(B, -1, self.width)
                x = torch.cat(
                    [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x], dim=1)  # shape = [*, grid ** 2 + 1, width]
                # cls_embed = self.positional_embedding[0, :].unsqueeze(0)
                # tile_pos_embed = self.positional_embedding[1:, :].repeat(self.num_frames, 1)
                # tile_temporal_embed = self.temporal_embedding.repeat_interleave(self.patches_per_frame, 0)
                # total_pos_embed = tile_pos_embed + tile_temporal_embed
                # total_pos_embed = torch.cat([cls_embed, total_pos_embed], dim=0)
                
                total_pos_embed = self.get_pos_embedding(x, curr_frames)
                x = x + total_pos_embed.to(x.dtype).unsqueeze(0)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = self.pos_drop(x)

        if not self.use_flash_attn:
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        else:
            x = self.transformer(x)

        if return_dense:
            if self.global_average_pool:
                return self.ln_post(x)
            else:
                return self.ln_post(x[:,1:])
        
        if self.global_average_pool:
            x_pooling = x.mean(dim=1)
        else:
            x_pooling = x[:, 0]

        x_pooling = self.ln_post(x_pooling)

        if self.image_projection is not None:
            x = x @ self.image_projection
            x_pooling = x_pooling @ self.image_projection
        return x_pooling,x
class TextTransformer(nn.Module):

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            ls_init_value: float = None,
            output_dim: int = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            causal_mask: float = True,
            flash_attn: bool = False,
            flash_mlp: bool = False,
            fused_bias_fc: bool = False,
    ):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim

        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)
        if output_dim is None:
            self.text_projection = None
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.causal_mask = causal_mask

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.width ** -0.5)
            # trunc_normal_(self.text_projection, std=0.001)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, cast_dtype=None):
        #with torch.no_grad():
        if cast_dtype is None:
            cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask if self.causal_mask else None)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.to(cast_dtype)

        return x
