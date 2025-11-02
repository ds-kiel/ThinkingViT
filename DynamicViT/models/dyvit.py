""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import logging
import numbers
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.distributions import Categorical

from utils import batch_index_select

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


class Identity(nn.Module):
    def forward(self, input: Tensor, keep_shape: int) -> Tensor:
        return input


class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, keep_shape_in: int, keep_shape_out: int = -1) -> Tensor:
        if keep_shape_out == -1:
            keep_shape_out = keep_shape_in
        if keep_shape_in:
            if input.dim() == 2:
                return F.linear(
                    input[:, 0:keep_shape_in],
                    self.weight[0:keep_shape_out, 0:keep_shape_in],
                    None if self.bias is None else self.bias[0:keep_shape_out],
                )
            if input.dim() == 3:
                return F.linear(
                    input[:, :, 0:keep_shape_in],
                    self.weight[0:keep_shape_out, 0:keep_shape_in],
                    None if self.bias is None else self.bias[0:keep_shape_out],
                )
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class QKVLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, keep_shape: int) -> Tensor:
        if keep_shape:
            third = self.out_features // 3
            idx = list(range(keep_shape))
            gather = idx + [i + third for i in idx] + [i + 2 * third for i in idx]
            if input.dim() == 2:
                sliced = input[:, 0:keep_shape]
            else:
                sliced = input[:, :, 0:keep_shape]
            return F.linear(
                sliced,
                self.weight[gather, 0:keep_shape],
                None if self.bias is None else self.bias[gather],
            )
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor, keep_shape: int) -> Tensor:
        if keep_shape:
            if input.dim() == 2:
                target = input[:, 0:keep_shape]
            else:
                target = input[:, :, 0:keep_shape]
            return F.layer_norm(
                target,
                (keep_shape,),
                None if self.weight is None else self.weight[0:keep_shape],
                None if self.bias is None else self.bias[0:keep_shape],
                self.eps,
            )
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x: Tensor, p: float) -> Tensor:
        keep_1 = max(int(p * self.in_features), 1)
        keep_hidden = max(int(p * self.hidden_features), 1)
        keep_out = max(int(p * self.out_features), 1)
        x = self.fc1(x, keep_1, keep_hidden)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, keep_hidden, keep_out)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm: bool = False,
        attn_drop=0.,
        proj_drop=0.,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = QKVLinear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dim = dim

    def softmax_with_policy(self, attn: Tensor, policy: Tensor, eps: float = 1e-6) -> Tensor:
        b, n, _ = policy.size()
        _, h, _, _ = attn.size()
        attn_policy = policy.reshape(b, 1, 1, n)
        eye = torch.eye(n, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, n, n)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / n) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x: Tensor, p: float, policy: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        keep_shape = max(int(p * self.dim), 1)
        keep_heads = max(int(p * self.num_heads), 1)
        qkv = self.qkv(x, keep_shape).reshape(B, N, 3, keep_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if isinstance(self.q_norm, LayerNorm):
            q = self.q_norm(q, self.head_dim)
        else:
            q = self.q_norm(q)
        if isinstance(self.k_norm, LayerNorm):
            k = self.k_norm(k, self.head_dim)
        else:
            k = self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, keep_shape)
        x = self.proj(x, keep_shape)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor, keep_shape: int) -> Tensor:
        scale = self.gamma[0:keep_shape]
        return x.mul_(scale) if self.inplace else x * scale


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, qk_norm: bool = False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm,
                 mlp_layer=Mlp, init_values=None):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor, p: float, policy: Tensor | None = None) -> Tensor:
        keep_shape = max(int(p * self.dim), 1)
        attn_out = self.attn(self.norm1(x, keep_shape), p, policy)
        attn_out = self.ls1(attn_out, keep_shape)
        x = x[:, :, 0:keep_shape] + self.drop_path1(attn_out)
        mlp_out = self.mlp(self.norm2(x, keep_shape), p)
        mlp_out = self.ls2(mlp_out, keep_shape)
        x = x[:, :, 0:keep_shape] + self.drop_path2(mlp_out)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


class VisionTransformerDiffPruning(nn.Module):
    """ThinkingViT backbone with DynamicViT token pruning."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        hybrid_backbone=None,
        norm_layer=None,
        pruning_loc=None,
        token_ratio=None,
        distill=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.max_heads = num_heads
        norm_layer = norm_layer or partial(LayerNorm, eps=1e-6)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = Linear(embed_dim, num_classes) if num_classes > 0 else Identity()
        self.pre_logits = nn.Identity()
        self.distill = distill
        self.pruning_loc = pruning_loc or []
        self.token_ratio = token_ratio or []
        self.score_predictor = nn.ModuleList()

        self.thinking_enabled = False
        self.teacher_mode = False
        self.stage_pruning_enabled = True
        self.eval_pruning_enabled = False
        self.thinking_stages: list[int] = []
        self.stage_ratios: list[float] = []
        self.stage_dims: list[int] = []
        self.stage_threshold = None
        self.alpha = nn.ParameterList()
        self.proj_layer: nn.Module | None = None

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def configure_thinking(self, thinking_stages=(3, 6), threshold=None, teacher_mode=False):
        stages = list(thinking_stages)
        if len(stages) < 1:
            raise ValueError('thinking_stages must contain at least one stage.')
        self.thinking_enabled = True
        self.teacher_mode = teacher_mode
        self.stage_pruning_enabled = not teacher_mode
        self.stage_threshold = threshold
        self.thinking_stages = stages
        self.stage_ratios = [stage / self.max_heads for stage in stages]
        self.stage_dims = [max(int(self.embed_dim * ratio), 1) for ratio in self.stage_ratios]

        # Projection & gating parameters must exist prior to loading checkpoints
        if len(self.alpha) != self.depth:
            self.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.depth)])
        if self.stage_dims:
            self.proj_layer = nn.Linear(self.stage_dims[0], self.embed_dim)

        final_stage_dim = self.stage_dims[-1]
        if self.stage_pruning_enabled and self.pruning_loc:
            self.score_predictor = nn.ModuleList([PredictorLG(final_stage_dim) for _ in self.pruning_loc])
        else:
            self.score_predictor = nn.ModuleList()

    def set_eval_pruning(self, enabled: bool) -> None:
        self.eval_pruning_enabled = bool(enabled)

    def _prepare_tokens(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def _apply_stage_projection(self, tokens: Tensor, prev_tokens: Tensor | None, stage_idx: int) -> Tensor:
        if prev_tokens is None or stage_idx == 0 or self.proj_layer is None:
            return tokens
        alpha_idx = min(stage_idx - 1, len(self.alpha) - 1)
        projected = self.proj_layer(prev_tokens)
        return tokens + self.alpha[alpha_idx] * projected

    def _forward_blocks(
        self,
        tokens: Tensor,
        p: float,
        apply_pruning: bool,
        training_pruning: bool,
    ):
        B, N, _ = tokens.shape
        policy = None
        prev_decision = None
        out_pred_prob = []
        init_n = N - 1
        predictor_index = 0

        if apply_pruning and (self.training or self.eval_pruning_enabled):
            prev_decision = torch.ones(B, init_n, 1, dtype=tokens.dtype, device=tokens.device)
            policy = torch.ones(B, init_n + 1, 1, dtype=tokens.dtype, device=tokens.device)

        for idx, blk in enumerate(self.blocks):
            if apply_pruning and idx in self.pruning_loc and predictor_index < len(self.score_predictor):
                spatial_tokens = tokens[:, 1:]
                if training_pruning:
                    pred_score = self.score_predictor[predictor_index](spatial_tokens, prev_decision).reshape(B, -1, 2)
                    hard_keep = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(hard_keep.reshape(B, init_n))
                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep.dtype, device=hard_keep.device)
                    policy = torch.cat([cls_policy, hard_keep], dim=1)
                    tokens = blk(tokens, p, policy=policy)
                    prev_decision = hard_keep
                else:
                    pred_score = self.score_predictor[predictor_index](spatial_tokens, prev_decision).reshape(B, -1, 2)
                    score = pred_score[:, :, 0]
                    keep_count = max(int(init_n * self.token_ratio[predictor_index]), 1)
                    keep_idx = torch.argsort(score, dim=1, descending=True)[:, :keep_count]
                    cls_policy = torch.zeros(B, 1, dtype=keep_idx.dtype, device=keep_idx.device)
                    gather_idx = torch.cat([cls_policy, keep_idx + 1], dim=1)
                    tokens = batch_index_select(tokens, gather_idx)
                    prev_decision = batch_index_select(prev_decision, keep_idx)
                    policy = None
                    tokens = blk(tokens, p)
                predictor_index += 1
            else:
                tokens = blk(tokens, p, policy=policy if training_pruning else None)
            # print(tokens.shape)

        keep_shape = max(int(p * self.embed_dim), 1)
        tokens = self.norm(tokens, keep_shape)
        return tokens, prev_decision, out_pred_prob

    def forward_head(self, tokens: Tensor, p: float, pre_logits: bool = False) -> Tensor:
        cls = tokens[:, 0]
        cls = self.pre_logits(cls)
        cls = self.head_drop(cls)
        keep_shape = max(int(p * self.embed_dim), 1)
        if pre_logits:
            return cls
        return self.head(cls, keep_shape, self.num_classes)

    def entropy(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs=probs).entropy().unsqueeze(1)

    def _forward_stage(
        self,
        x: Tensor,
        stage_idx: int,
        prev_tokens: Tensor | None = None,
        apply_pruning: bool = False,
        training_pruning: bool = False,
    ):
        if not self.thinking_enabled:
            raise RuntimeError('configure_thinking must be called before using staged forward.')
        p = self.stage_ratios[stage_idx]
        tokens = self._prepare_tokens(x)
        tokens = self._apply_stage_projection(tokens, prev_tokens, stage_idx)
        tokens, prev_decision, out_pred_prob = self._forward_blocks(tokens, p, apply_pruning, training_pruning)
        logits = self.forward_head(tokens, p)
        info = {
            'mask': prev_decision,
            'out_pred_prob': out_pred_prob,
            'p': p,
        }
        return tokens, logits, info

    def _forward_train(self, x: Tensor):
        stage_logits = []
        prev_tokens = None
        final_tokens = None
        mask = None
        out_pred_prob = []

        for idx in range(len(self.thinking_stages)):
            apply_pruning = (
                self.stage_pruning_enabled
                and idx == len(self.thinking_stages) - 1
                and bool(self.pruning_loc)
            )
            tokens, logits, info = self._forward_stage(
                x,
                idx,
                prev_tokens=prev_tokens,
                apply_pruning=apply_pruning,
                training_pruning=True,
            )
            stage_logits.append(logits)
            prev_tokens = tokens
            final_tokens = tokens
            if apply_pruning:
                mask = info['mask']
                out_pred_prob = info['out_pred_prob']

        if final_tokens is None:
            raise RuntimeError('No stages executed; ensure thinking_stages is configured.')

        if mask is None:
            B = x.shape[0]
            init_n = self.patch_embed.num_patches
            mask = final_tokens.new_ones((B, init_n, 1))

        if self.distill:
            token_pred = final_tokens[:, 1:, :]
            return tuple(stage_logits), token_pred, mask.detach(), out_pred_prob
        return tuple(stage_logits), out_pred_prob

    def _forward_eval(self, x: Tensor, threshold=None):
        threshold = self.stage_threshold if threshold is None else threshold
        tokens0, logits0, _ = self._forward_stage(
            x,
            0,
            prev_tokens=None,
            apply_pruning=False,
            training_pruning=False,
        )

        if len(self.thinking_stages) == 1:
            return logits0

        if threshold is None:
            tokens_last, logits_last, _ = self._forward_stage(
                x,
                len(self.thinking_stages) - 1,
                prev_tokens=tokens0,
                apply_pruning=self.eval_pruning_enabled,
                training_pruning=False,
            )
            return logits_last

        ent = self.entropy(logits0).squeeze(1)
        need_second = ent > threshold
        if not need_second.any():
            return logits0

        tokens_last, logits_last, _ = self._forward_stage(
            x,
            len(self.thinking_stages) - 1,
            prev_tokens=tokens0,
            apply_pruning=self.eval_pruning_enabled,
            training_pruning=False,
        )
        logits_final = logits0.clone()
        logits_final[need_second] = logits_last[need_second]
        return logits_final

    def forward(self, x: Tensor, threshold=None):
        if self.training:
            return self._forward_train(x)
        return self._forward_eval(x, threshold=threshold)

    @torch.no_grad()
    def forward_thinking(self, x: Tensor, enable_pruning: bool | None = None):
        results = []
        prev_tokens = None
        pruning_flag = enable_pruning if enable_pruning is not None else (self.eval_pruning_enabled and not self.training)
        for idx in range(len(self.thinking_stages)):
            tokens, logits, info = self._forward_stage(
                x,
                idx,
                prev_tokens=prev_tokens,
                apply_pruning=pruning_flag and idx == len(self.thinking_stages) - 1,
                training_pruning=self.training and idx == len(self.thinking_stages) - 1,
            )
            results.append({'tokens': tokens, 'logits': logits, 'info': info})
            prev_tokens = tokens
        return results

class VisionTransformerTeacher(VisionTransformerDiffPruning):
    """Teacher wrapper for ThinkingViT backbone without pruning."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('pruning_loc', [])
        kwargs.setdefault('token_ratio', [])
        kwargs.setdefault('distill', False)
        super().__init__(*args, **kwargs)
        self.stage_pruning_enabled = False

    def configure_thinking(self, thinking_stages=(3, 6), threshold=None, teacher_mode=True):
        super().configure_thinking(thinking_stages=thinking_stages, threshold=threshold, teacher_mode=True)

    def forward(self, x: Tensor, threshold=None):
        was_training = self.training
        try:
            self.eval()
            stages = self.forward_thinking(x, enable_pruning=False)
            logits = stages[-1]['logits']
            tokens = stages[-1]['tokens'][:, 1:, :]
            return logits, tokens
        finally:
            self.train(was_training)

def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict