
"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""
# Modified from the Segmenter paper implementation

# Copyright 2024 Kiel University
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
# Modifications:
# - Adjusted training loop to support the thinking approach

import math
import numbers
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torch.nn.parameter import Parameter

from segm.model.utils import init_weights, resize_pos_embed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import _load_weights


class Identity(nn.Module):
    def forward(self, input: torch.Tensor, keep_shape: Optional[int] = None) -> torch.Tensor:
        return input


class Linear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        input: torch.Tensor,
        keep_shape_in: Optional[int] = None,
        keep_shape_out: Optional[int] = None,
    ) -> torch.Tensor:
        if keep_shape_in is None:
            keep_shape_in = self.in_features
        if keep_shape_out is None:
            keep_shape_out = self.out_features

        keep_shape_in = max(1, min(keep_shape_in, self.in_features))
        keep_shape_out = max(1, min(keep_shape_out, self.out_features))

        if input.dim() == 3:
            input = input[:, :, :keep_shape_in]
        else:
            input = input[:, :keep_shape_in]

        weight = self.weight[:keep_shape_out, :keep_shape_in]
        bias = self.bias[:keep_shape_out] if self.bias is not None else None
        return F.linear(input, weight, bias)


class QKVLinear(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert out_features % 3 == 0, "QKV Linear out_features must be divisible by 3"
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, keep_shape: Optional[int]) -> torch.Tensor:
        if keep_shape is None:
            return F.linear(input, self.weight, self.bias)

        keep_shape = max(1, min(keep_shape, min(self.in_features, self.out_features // 3)))
        if input.dim() == 3:
            input = input[:, :, :keep_shape]
        else:
            input = input[:, :keep_shape]

        third = self.out_features // 3
        idx = (
            list(range(0, keep_shape))
            + list(range(third, third + keep_shape))
            + list(range(2 * third, 2 * third + keep_shape))
        )

        weight = self.weight[idx, :keep_shape]
        bias = self.bias[idx] if self.bias is not None else None
        return F.linear(input, weight, bias)


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: tuple
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor, keep_shape: Optional[int]) -> torch.Tensor:
        if keep_shape is None or keep_shape >= self.normalized_shape[0]:
            return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

        keep_shape = max(1, min(keep_shape, self.normalized_shape[0]))
        if input.dim() == 3:
            input_slice = input[:, :, :keep_shape]
        else:
            input_slice = input[:, :keep_shape]
        weight = self.weight[:keep_shape] if self.elementwise_affine else None
        bias = self.bias[:keep_shape] if self.elementwise_affine else None
        return F.layer_norm(input_slice, (keep_shape,), weight, bias, self.eps)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, keep_shape: int) -> torch.Tensor:
        keep_shape = max(1, min(keep_shape, self.in_features))
        ratio = keep_shape / self.in_features
        hidden_keep = max(1, min(self.hidden_features, int(round(ratio * self.hidden_features))))
        out_keep = keep_shape

        x = self.fc1(x, keep_shape, hidden_keep)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x, hidden_keep, out_keep)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = QKVLinear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        keep_dim: int,
        mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = x.shape
        keep_heads = max(1, min(self.num_heads, keep_dim // self.head_dim))
        keep_dim = keep_heads * self.head_dim

        qkv = (
            self.qkv(x, keep_dim)
            .reshape(B, N, 3, keep_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, keep_dim)
        x = self.proj(x, keep_dim, keep_dim)
        x = self.proj_drop(x)
        return x, attn


class ThinkingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=dropout)

    def forward(
        self,
        x: torch.Tensor,
        keep_dim: int,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        y, attn = self.attn(self.norm1(x, keep_dim), keep_dim, mask)
        if return_attention:
            return attn

        x_residual = x[:, :, :keep_dim]
        x = x_residual + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x, keep_dim), keep_dim))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = image_size[0] // patch_size, image_size[1] // patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls
        self.distilled = distilled

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        if self.distilled:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 2, d_model)
            )
            self.head_dist = Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.patch_embed.num_patches + 1, d_model)
            )

        mlp_ratio = d_ff / d_model
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [
                ThinkingBlock(d_model, n_heads, mlp_ratio, dropout, dpr[i])
                for i in range(n_layers)
            ]
        )

        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

        self._thinking_config: Optional[Sequence[int]] = None
        self._thinking_proj_layers: Optional[nn.ModuleList] = None
        self._thinking_alphas: Optional[nn.ParameterList] = None

    @torch.jit.ignore
    def no_weight_decay(self):
        tokens = {"pos_embed", "cls_token"}
        if self.distilled:
            tokens.add("dist_token")
        return tokens

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def _ensure_thinking_modules(self, thinking_stages: Sequence[int]) -> None:
        if not thinking_stages or len(thinking_stages) <= 1:
            self._thinking_config = tuple(thinking_stages) if thinking_stages else None
            self._thinking_proj_layers = None
            self._thinking_alphas = None
            return

        config = tuple(int(h) for h in thinking_stages)
        if getattr(self, "_thinking_config", None) == config:
            return

        device = self.cls_token.device
        dtype = self.cls_token.dtype
        proj_layers = nn.ModuleList()
        alphas = nn.ParameterList()

        prev_dim = max(1, min(self.n_heads, config[0])) * self.head_dim
        for stage_heads in config[1:]:
            cur_dim = max(1, min(self.n_heads, stage_heads)) * self.head_dim
            layer = nn.Linear(prev_dim, cur_dim, device=device, dtype=dtype)
            nn.init.zeros_(layer.bias)
            proj_layers.append(layer)
            alphas.append(nn.Parameter(torch.zeros(1, device=device, dtype=dtype)))
            prev_dim = cur_dim

        self._thinking_proj_layers = proj_layers
        self._thinking_alphas = alphas
        self._thinking_config = config

    def forward_stage(
        self,
        im: torch.Tensor,
        stage_heads: int,
        prev_tokens: Optional[torch.Tensor] = None,
        stage_idx: int = 0,
        thinking_stages: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + int(self.distilled)
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)

        keep_heads = max(1, min(int(stage_heads), self.n_heads))
        keep_dim = keep_heads * self.head_dim
        keep_dim = min(keep_dim, self.d_model)

        if thinking_stages is not None:
            self._ensure_thinking_modules(thinking_stages)
            if stage_idx > 0 and self._thinking_proj_layers is not None:
                if self._thinking_proj_layers is None or self._thinking_alphas is None:
                    raise RuntimeError("Thinking modules are not initialized correctly.")
                if stage_idx - 1 >= len(self._thinking_proj_layers):
                    raise ValueError(
                        f"stage_idx {stage_idx} exceeds available thinking stages"
                    )
                if prev_tokens is None:
                    raise ValueError("prev_tokens must be provided for stage_idx > 0")
                proj = self._thinking_proj_layers[stage_idx - 1]
                alpha = self._thinking_alphas[stage_idx - 1]
                addition = proj(prev_tokens)
                x = x.clone()
                x[:, :, :keep_dim] = x[:, :, :keep_dim] + alpha * addition[:, :, :keep_dim]

        for blk in self.blocks:
            # print(x.shape)
            x = blk(x, keep_dim)
        # print('----')
        x = self.norm(x, keep_dim)
        return x

    def forward(
        self,
        im: torch.Tensor,
        return_features: bool = False,
        stage_heads: Optional[int] = None,
        prev_tokens: Optional[torch.Tensor] = None,
        stage_idx: int = 0,
        thinking_stages: Optional[Sequence[int]] = None,
    ):
        heads = stage_heads or self.n_heads
        tokens = self.forward_stage(
            im,
            heads,
            prev_tokens=prev_tokens,
            stage_idx=stage_idx,
            thinking_stages=thinking_stages,
        )

        if return_features:
            return tokens

        keep_dim = tokens.shape[-1]
        if self.distilled:
            x, x_dist = tokens[:, 0], tokens[:, 1]
            x = self.head(x, keep_dim, self.n_cls)
            x_dist = self.head_dist(x_dist, keep_dim, self.n_cls)
            x = (x + x_dist) / 2
        else:
            x = tokens[:, 0]
            x = self.head(x, keep_dim, self.n_cls)
        return x

    def get_attention_map(self, im: torch.Tensor, layer_id: int):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + int(self.distilled)
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (H // PS, W // PS),
                num_extra_tokens,
            )
        x = x + pos_embed

        keep_dim = self.d_model
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x, keep_dim)
            else:
                return blk(x, keep_dim, return_attention=True)
