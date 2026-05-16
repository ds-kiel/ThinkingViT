#!/usr/bin/env python3
"""Smoke-test the uploaded ThinkingViT-DeiT Hugging Face model.

Examples:
  python scripts/test_hf_thinkingvit_deit.py
  python scripts/test_hf_thinkingvit_deit.py --image /path/to/image.jpg --threshold 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from timm.models import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="alihjt/thinkingvit_deit-3h-6h-imagenet1k",
        help="Hugging Face model repository id.",
    )
    parser.add_argument("--image", type=Path, default=None, help="Optional image path. Uses random input if omitted.")
    parser.add_argument("--threshold", type=float, default=1.0, help="Entropy threshold for early exit.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def imagenet_label(class_id: int) -> str:
    synsets_path = REPO_ROOT / "timm" / "data" / "_info" / "imagenet_synsets.txt"
    lemmas_path = REPO_ROOT / "timm" / "data" / "_info" / "imagenet_synset_to_lemma.txt"
    synset = synsets_path.read_text().splitlines()[class_id]
    for line in lemmas_path.read_text().splitlines():
        key, label = line.split("\t", 1)
        if key == synset:
            return label
    return synset


def load_input(image_path: Path | None, device: str) -> torch.Tensor:
    if image_path is None:
        return torch.randn(1, 3, 224, 224, device=device)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading hf-hub:{args.repo_id} on {device}...")
    model = create_model(f"hf-hub:{args.repo_id}", pretrained=True)
    model.to(device)
    model.eval()

    x = load_input(args.image, str(device))
    with torch.no_grad():
        logits, stage = model(x, threshold=args.threshold)
        probs = logits.softmax(dim=1)
        confidence, class_id = probs.max(dim=1)

    predicted_id = class_id.item()
    print(f"threshold: {args.threshold}")
    print(f"predicted class id: {predicted_id}")
    print(f"predicted label: {imagenet_label(predicted_id)}")
    print(f"confidence: {confidence.item():.4f}")
    print(f"thinking stage: {stage.flatten()[0].item()}  (0 = 3 heads, 1 = 6 heads)")
    print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
