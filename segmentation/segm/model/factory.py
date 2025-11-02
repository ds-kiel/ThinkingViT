from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn
import pickle

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from segm.model.vit import VisionTransformer
from segm.model.utils import checkpoint_filter_fn
from segm.model.decoder import DecoderLinear
from segm.model.decoder import MaskTransformer
from segm.model.segmenter import Segmenter
import segm.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    pretrained_path = model_cfg.pop("pretrained_path", None)
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = model_cfg.pop("mlp_ratio", 4)
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)

    def _ensure_thinking_modules_from_state(model, state_dict):
        """Prepare legacy thinking parameters found in a pretrained checkpoint."""
        info = dict(
            initialized=False,
            stage_heads=None,
            legacy_alpha_found=False,
            legacy_alpha_mapped=False,
            legacy_alpha_count=0,
            legacy_proj_found=False,
            legacy_proj_mapped=False,
        )
        if not isinstance(state_dict, dict) or not hasattr(model, "_ensure_thinking_modules"):
            return info

        def _collect_entries(target):
            entries = []
            for key in state_dict.keys():
                parts = key.split(".")
                for idx, part in enumerate(parts):
                    if part == target:
                        next_idx = idx + 1
                        pos = 0
                        if next_idx < len(parts) and parts[next_idx].isdigit():
                            pos = int(parts[next_idx])
                        entries.append((pos, key))
                        break
            return entries

        legacy_alpha_entries = _collect_entries("alpha")
        info["legacy_alpha_found"] = len(legacy_alpha_entries) > 0
        info["legacy_alpha_count"] = len(legacy_alpha_entries)

        head_dim = getattr(model, "head_dim", None)
        n_heads = getattr(model, "n_heads", None)
        if head_dim is None or n_heads is None or head_dim <= 0 or n_heads <= 0:
            return info

        proj_entries = []
        for key, tensor in state_dict.items():
            if not key.endswith("weight") or not torch.is_tensor(tensor) or tensor.ndim != 2:
                continue
            parts = key.split(".")
            entry_type = None
            idx_value = None
            if "_thinking_proj_layers" in key:
                entry_type = "modern"
                for part in parts:
                    if part.isdigit():
                        idx_value = int(part)
                        break
            else:
                for pos, part in enumerate(parts):
                    if part == "proj_layer":
                        entry_type = "legacy_single"
                        idx_value = 0
                        break
                    if part == "proj_layers":
                        entry_type = "legacy_list"
                        if pos + 1 < len(parts) and parts[pos + 1].isdigit():
                            idx_value = int(parts[pos + 1])
                        else:
                            idx_value = 0
                        break
            if entry_type is None or idx_value is None:
                continue
            info["legacy_proj_found"] = info["legacy_proj_found"] or entry_type.startswith("legacy")
            bias_key = key[:-6] + "bias" if key.endswith("weight") else None
            proj_entries.append(
                dict(
                    idx=idx_value,
                    key=key,
                    tensor=tensor,
                    bias_key=bias_key if bias_key in state_dict else None,
                    kind=entry_type,
                )
            )

        if not proj_entries:
            return info

        proj_entries.sort(key=lambda x: x["idx"])
        stage_heads = []
        for order, entry in enumerate(proj_entries):
            prev_dim = int(entry["tensor"].shape[1])
            cur_dim = int(entry["tensor"].shape[0])
            if not stage_heads:
                if prev_dim % head_dim != 0:
                    return info
                stage0 = max(1, min(n_heads, prev_dim // head_dim))
                stage_heads.append(stage0)
            else:
                expected_prev = stage_heads[-1] * head_dim
                if expected_prev != prev_dim:
                    return info
            if cur_dim % head_dim != 0:
                return info
            next_stage = max(1, min(n_heads, cur_dim // head_dim))
            stage_heads.append(next_stage)

        if len(stage_heads) <= 1:
            return info

        model._ensure_thinking_modules(stage_heads)
        info["initialized"] = True
        info["stage_heads"] = stage_heads

        # Rename legacy projection weights/biases to new names
        mapped_proj = False
        for order, entry in enumerate(proj_entries):
            if entry["kind"] == "modern":
                continue
            new_weight_key = f"_thinking_proj_layers.{order}.weight"
            new_bias_key = f"_thinking_proj_layers.{order}.bias"
            if entry["key"] in state_dict:
                state_dict[new_weight_key] = state_dict[entry["key"]]
                del state_dict[entry["key"]]
                mapped_proj = True
            if entry["bias_key"]:
                state_dict[new_bias_key] = state_dict[entry["bias_key"]]
                del state_dict[entry["bias_key"]]
        info["legacy_proj_mapped"] = mapped_proj

        # Legacy alpha remapping
        expected_alphas = len(stage_heads) - 1
        modern_alpha_keys = [k for k in state_dict.keys() if "_thinking_alphas" in k]
        if modern_alpha_keys:
            info["legacy_alpha_mapped"] = True
            return info

        if legacy_alpha_entries and expected_alphas > 0:
            legacy_alpha_entries.sort(key=lambda x: x[0])
            mapped = 0
            for idx, (_, key) in enumerate(legacy_alpha_entries[:expected_alphas]):
                new_key = f"_thinking_alphas.{idx}"
                if key in state_dict:
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
                    mapped += 1
            info["legacy_alpha_mapped"] = mapped == expected_alphas
        return info

    if pretrained_path is not None:
        load_kwargs = dict(map_location="cpu")
        try:
            state_dict = torch.load(pretrained_path, weights_only=True, **load_kwargs)
        except TypeError:
            # Older torch without weights_only support
            state_dict = torch.load(pretrained_path, **load_kwargs)
        except (pickle.UnpicklingError, RuntimeError):
            state_dict = torch.load(pretrained_path, weights_only=False, **load_kwargs)

        if isinstance(state_dict, dict):
            # common checkpoint layouts
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        thinking_info = _ensure_thinking_modules_from_state(model, filtered_dict)
        if thinking_info["initialized"]:
            stage_heads = thinking_info["stage_heads"]
            print(
                f"Detected thinking stages {stage_heads} in pretrained checkpoint; initialized matching thinking modules."
            )
            if thinking_info["legacy_proj_found"]:
                if thinking_info["legacy_proj_mapped"]:
                    print("Loaded thinking-stage projector weights from legacy checkpoint entries.")
                else:
                    print("Found legacy thinking-stage projector weights but could not map them cleanly; they will be ignored.")
            if thinking_info["legacy_alpha_found"]:
                if thinking_info["legacy_alpha_mapped"]:
                    print(
                        f"Loaded {len(stage_heads) - 1} thinking alpha parameter(s) from legacy checkpoint entries."
                    )
                else:
                    print(
                        f"Found {thinking_info['legacy_alpha_count']} legacy alpha parameter(s) but could not map them cleanly "
                        "to the current thinking-stage layout; they will be ignored."
                    )
        elif thinking_info["legacy_alpha_found"]:
            print(
                f"Found {thinking_info['legacy_alpha_count']} legacy alpha parameter(s) in pretrained checkpoint, "
                "but detected no compatible thinking-stage projection weights; skipping them."
            )
        else:
            print("No thinking-stage alpha parameters found in pretrained checkpoint.")
        model.load_state_dict(filtered_dict, strict=False)
    elif backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)

    thinking_cfg = variant.get("thinking_kwargs") or {}
    thinking_stages = thinking_cfg.get("stages")
    if thinking_stages:
        model.encoder._ensure_thinking_modules(thinking_stages)

    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
