# ThinkingViT + DynamicViT

## Overview

- To show that token pruning approaches can make the results of ThinkingViT better, this repo applies DynamicViT during the second round of thinking in the ThinkingViT 3->6 schedule using a base pruning rate of `0.8`. This keeps the early thinking pass intact while aggressively reducing compute in the later pass.
- The provided checkpoint `ThinkingViT_DynamicViT_0_8.pth` is already trained with this schedule and expects ImageNet-1K style preprocessing.

## Setup

- Environment: Follow the setup instructions from the official DynamicViT repository: https://github.com/raoyongming/DynamicViT

- Checkpoints: Download the pretrained weights from https://zenodo.org/records/17504412

## Evaluation

```bash
python3 infer.py \
  --model deit-b \
  --data_path /path/to/imagenet \
  --model_path checkpoints/ThinkingViT_DynamicViT_0_8.pth \
  --thinking-stages 3,6 \
  --thinking-threshold [THRESHOLD] \
  --eval-pruning true \
  --base_rate 0.8
```

Set `--thinking-threshold` to the gating value you wish to sweep; lower values retain more tokens, higher values prune more aggressively during the 6-head stage.

| GMACs | Top-1 Acc (%) |
|-------|---------------|
| 4.650 | 81.204 |
| 4.611 | 81.206 |
| 4.196 | 81.214 |
| 3.678 | 81.172 |
| 3.219 | 80.982 |
| 2.720 | 80.372 |
| 2.254 | 79.228 |
| 1.913 | 77.752 |
| 1.655 | 76.416 |
| 1.470 | 75.168 |
| 1.343 | 74.218 |
| 1.275 | 73.750 |
| 1.250 | 73.580 |
