#!/usr/bin/env python3
'''Export ThinkingViT checkpoints to Hugging Face Hub-ready folders.

The script intentionally exports EMA state dicts as safetensors instead of the
full training checkpoints, which also contain optimizer state and require pickle
loading. Run once without --push to inspect the generated folders, then run with
--push after `hf auth login` or after setting HF_TOKEN.
'''

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timm.models import create_model
from timm.models.hub import save_for_hf

DEFAULT_OUTPUT_DIR = REPO_ROOT / "hf_exports"
PAPER_URL = "https://arxiv.org/abs/2507.10800"
HF_PAPER_URL = "https://huggingface.co/papers/2507.10800"
PROJECT_URL = "https://ds-kiel.github.io/ThinkingViT-project-page/"
GITHUB_URL = "https://github.com/ds-kiel/ThinkingViT"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    slug: str
    title: str
    checkpoint: Path
    architecture: str
    model_args: Mapping[str, Any]
    threshold_note: str
    result_table: str
    usage: str


def _deit_usage(repo_id: str) -> str:
    return f'''```python
import torch
from timm.models import create_model

# Run from the ThinkingViT repository root, or put this repository on PYTHONPATH.
model = create_model("hf-hub:{repo_id}", pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits, stage = model(x, threshold=1.0)
print(logits.shape, stage)
```
'''


def _swin_usage(repo_id: str) -> str:
    return f'''```python
import torch
from timm.models import create_model

# Run from the ThinkingViT repository root, or put this repository on PYTHONPATH.
model = create_model("hf-hub:{repo_id}", pretrained=True)
model.eval()

x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits, stage = model(x, threshold=1.0)
print(logits.shape, stage)
```
'''


SPECS: Dict[str, ModelSpec] = {
    "thinkingvit_deit": ModelSpec(
        key="thinkingvit_deit",
        slug="thinkingvit_deit-3h-6h-imagenet1k",
        title="ThinkingViT DeiT 3H -> 6H ImageNet-1K",
        checkpoint=REPO_ROOT / "ThinkingViT_re_3_6.pth.tar",
        architecture="thinkingvit",
        model_args={"thinking_stages": [3, 6]},
        threshold_note=(
            "The entropy threshold controls early exit. Lower thresholds send more samples "
            "to the 6-head stage; higher thresholds exit earlier at the 3-head stage."
        ),
        result_table='''| Threshold | Acc@1 (%) | GMACs |
|---:|---:|---:|
| 0.0 | 81.440 | 5.850 |
| 0.5 | 81.368 | 3.977 |
| 1.0 | 80.310 | 2.907 |
| 1.6 | 77.292 | 1.944 |
| 10.0 | 73.536 | 1.250 |''',
        usage="thinkingvit_deit",
    ),
    "thinkingvit_800epochs": ModelSpec(
        key="thinkingvit_800epochs",
        slug="thinkingvit_800epochs",
        title="ThinkingViT 800 Epochs DeiT 3H -> 6H ImageNet-1K",
        checkpoint=REPO_ROOT / "ThinkingViT_3_6_800epochs.pth.tar",
        architecture="thinkingvit",
        model_args={"thinking_stages": [3, 6]},
        threshold_note=(
            "The entropy threshold controls early exit. Lower thresholds send more samples "
            "to the 6-head stage; higher thresholds exit earlier at the 3-head stage."
        ),
        result_table='''| Threshold | Acc@1 (%) | GMACs |
|---:|---:|---:|
| 0.0 | 81.850 | 5.850 |
| 0.1 | 81.848 | 5.385 |
| 0.2 | 81.846 | 4.751 |
| 0.3 | 81.832 | 4.363 |
| 0.5 | 81.758 | 3.841 |
| 0.8 | 81.386 | 3.189 |
| 1.0 | 80.636 | 2.781 |
| 1.2 | 79.764 | 2.433 |
| 1.4 | 78.846 | 2.136 |
| 1.6 | 77.688 | 1.865 |
| 2.0 | 75.500 | 1.417 |
| 5.0 | 74.514 | 1.250 |
| 10.0 | 74.514 | 1.250 |''',
        usage="thinkingvit_deit",
    ),
    "swin": ModelSpec(
        key="swin",
        slug="thinkingvit-swin-s-imagenet1k",
        title="ThinkingViT-Swin / Swin-S ImageNet-1K",
        checkpoint=REPO_ROOT / "ThinkingViTSwin.pth.tar",
        architecture="swin_small_patch4_window7_224",
        model_args={
            "head_rounds": [[3, 3, 6, 12], [3, 6, 12, 24]],
            "build_token_projector": True,
        },
        threshold_note=(
            "The entropy threshold controls early exit. Lower thresholds send more samples "
            "to the full Swin-S round; higher thresholds exit after the reduced-head round."
        ),
        result_table='''| Threshold | Acc@1 (%) | GMACs |
|---:|---:|---:|
| 0.0 | 83.516 | 11.68 |
| 0.5 | 83.386 | 6.76 |
| 1.0 | 82.124 | 4.88 |
| 1.6 | 79.746 | 3.53 |
| 5.0 | 77.990 | 2.82 |''',
        usage="swin",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "deit", *SPECS.keys()],
        default="all",
        help="Which model export to build.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated Hub-ready model folders.",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("HF_NAMESPACE", ""),
        help="Optional Hugging Face namespace, for example ds-kiel or your username.",
    )
    parser.add_argument(
        "--state-key",
        default="state_dict_ema",
        help="Checkpoint key to export. Falls back to state_dict if unavailable.",
    )
    parser.add_argument("--push", action="store_true", help="Upload generated folders to the Hugging Face Hub.")
    parser.add_argument("--private", action="store_true", help="Create private Hub repos when pushing.")
    parser.add_argument(
        "--skip-create-repo",
        action="store_true",
        help="Skip Hub repo creation and only upload to an existing repo.",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Defaults to HF_TOKEN/login cache.")
    parser.add_argument("--revision", default=None, help="Optional Hub revision/branch for upload.")
    return parser.parse_args()


def selected_specs(model: str) -> Iterable[ModelSpec]:
    aliases = {"deit": "thinkingvit_deit"}
    model = aliases.get(model, model)
    if model == "all":
        return SPECS.values()
    return [SPECS[model]]


def repo_id_for(spec: ModelSpec, namespace: str) -> str:
    namespace = namespace.strip().strip("/")
    return f"{namespace}/{spec.slug}" if namespace else spec.slug


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())


def extract_state_dict(checkpoint: Any, preferred_key: str) -> tuple[OrderedDict[str, torch.Tensor], str]:
    if isinstance(checkpoint, Mapping):
        keys = [preferred_key, "state_dict_ema", "model_ema", "state_dict", "model"]
        for key in dict.fromkeys(keys):
            state_dict = checkpoint.get(key)
            if state_dict is not None:
                return clean_state_dict(state_dict), key
    return clean_state_dict(checkpoint), "raw"


def build_model(spec: ModelSpec) -> torch.nn.Module:
    model = create_model(
        spec.architecture,
        pretrained=False,
        num_classes=1000,
        **dict(spec.model_args),
    )
    model.eval()
    return model


def model_card(spec: ModelSpec, repo_id: str, checkpoint_key: str) -> str:
    usage = _deit_usage(repo_id) if spec.usage == "thinkingvit_deit" else _swin_usage(repo_id)
    return f'''---
license: mit
library_name: timm
pipeline_tag: image-classification
tags:
- image-classification
- vision-transformer
- adaptive-inference
- elastic-inference
- imagenet-1k
- arxiv:2507.10800
datasets:
- imagenet-1k
metrics:
- accuracy
---

# {spec.title}

This repository contains the ImageNet-1K EMA weights for **{spec.title}** from
[ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference]({PAPER_URL}).

- Paper: {PAPER_URL}
- Hugging Face paper: {HF_PAPER_URL}
- Code: {GITHUB_URL}
- Project page: {PROJECT_URL}
- Exported checkpoint key: `{checkpoint_key}`
- Weight format: `model.safetensors`

## Usage

{usage}

This is a custom timm-based architecture. Use the code from the ThinkingViT repository when loading this model.

## Threshold Behavior

{spec.threshold_note}

## ImageNet-1K Results

{spec.result_table}

## Citation

Please cite the ThinkingViT paper if you use this model: {PAPER_URL}
'''


def export_one(spec: ModelSpec, output_dir: Path, namespace: str, state_key: str) -> tuple[Path, str]:
    if not spec.checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {spec.checkpoint}")

    checkpoint = torch_load(spec.checkpoint)
    state_dict, actual_key = extract_state_dict(checkpoint, state_key)
    model = build_model(spec)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict load failed for {spec.slug}: missing={missing}, unexpected={unexpected}")

    repo_id = repo_id_for(spec, namespace)
    save_dir = output_dir / spec.slug
    save_dir.mkdir(parents=True, exist_ok=True)
    save_for_hf(
        model,
        save_dir,
        model_config={"num_classes": 1000},
        model_args=dict(spec.model_args),
        safe_serialization=True,
    )
    (save_dir / "README.md").write_text(model_card(spec, repo_id, actual_key))
    print(f"Exported {spec.slug} -> {save_dir} using {actual_key} ({len(state_dict)} tensors)")
    return save_dir, repo_id


def push_folder(folder: Path, repo_id: str, args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi, upload_folder

    if not args.skip_create_repo:
        api = HfApi(token=args.token)
        api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
    upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        token=args.token,
        revision=args.revision,
        commit_message="Add ThinkingViT weights",
    )
    print(f"Pushed {repo_id}")


def main() -> None:
    args = parse_args()
    exported: List[tuple[Path, str]] = []
    for spec in selected_specs(args.model):
        exported.append(export_one(spec, args.output_dir, args.namespace, args.state_key))

    if args.push:
        for folder, repo_id in exported:
            push_folder(folder, repo_id, args)
    else:
        print("Dry run complete. Re-run with --push after confirming the generated folders.")


if __name__ == "__main__":
    main()
