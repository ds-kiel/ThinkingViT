# 🚀 ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference - ES-FoMo@ICML2025 


**[📄 PAPER: [arXiv:2507.10800](https://arxiv.org/abs/2507.10800)]**

Vision Transformers deliver state-of-the-art performance, yet their fixed computational budget prevents scalable deployment across heterogeneous hardware. Recent Matryoshka-based Transformer architectures mitigate this by embedding nested subnetworks within a single model to enable scalable inference. However, these models allocate the same amount of compute to all inputs, regardless of their complexity, which leads to inefficiencies. To address this, we introduce ThinkingViT, a nested ViT architecture that employs progressive thinking stages to dynamically adjust inference computation based on input difficulty. ThinkingViT initiates inference by activating a small subset of the most important attention heads and terminates early if predictions reach sufficient certainty. Otherwise, it activates additional attention heads and re-evaluates the input. At the core of ThinkingViT is our Token Recycling mechanism, which conditions each subsequent inference stage on the embeddings from the previous stage, enabling progressive improvement. Due to its backbone-preserving design, ThinkingViT also serves as a plugin upgrade for vanilla ViT. Experiments show that ThinkingViT surpasses nested baselines by up to 2.0 percentage points (p.p.) in accuracy at the same throughput and by up to 2.9 p.p. at equal GMACs on ImageNet-1K.

<div align="center">
  <img src="figures/demo.png" width="100%">
</div>



## 🗓️ Updates

- **Update 24.10.2025**: Added Swin Transformer variant with its pretrained checkpoint.
- **Update 2.11.2025**: Added semantic segmentation support (Segmenter + ThinkingViT).
- **Update 2.11.2025**: Added implementation combining ThinkingViT and DynamicViT (Segmenter + DynamicViT).


## 📦 Quick Start

### Installation

Create and activate the Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 📦 Pretrained Checkpoints

We provide a set of pretrained **ThinkingViT** checkpoints for all configurations reported in the paper. You can download all models from [Zenodo](https://zenodo.org/records/17507118).

#### ✅ Available Configurations:
- `2H → 3H`
- `2H → 3H → 6H`
- `2H → 6H`
- `3H → 6H`
- `3H → 6H → 12H`
- `3H → 9H`
- `3H → 12H`


## 🛠️ Inference

Run inference using `validate.py` with the following parameters:

```bash
python3 validate.py \
  -m [MODEL_ARCHITECTURE] \
  --checkpoint [CHECKPOINT_PATH] \
  --data-dir [IMAGENET_DIR] \
  -b [BATCH_SIZE] \
  --use-ema \
  --threshold [CONFIDENCE_THRESHOLD] \
  --thinking_stages [H1 H2 ... Hn]
```

**Example:** Two-stage thinking

```bash
python3 validate.py \
  -m deit_base_patch16_224 \
  --checkpoint /path/to/ThinkingViT_3_6.pth.tar \
  --data-dir IMAGENET_DIR \
  -b 2048 \
  --use-ema \
  --threshold 1.5 \
  --thinking_stages 3 6
```

**Example:** Three-stage thinking

```bash
python3 validate.py \
  -m deit_base_patch16_224 \
  --checkpoint /path/to/ThinkingViT_3_6_12.pth.tar \
  --data-dir IMAGENET_DIR \
  -b 2048 \
  --use-ema \
  --threshold CONFIDENCE_THRESHOLD \
  --thinking_stages 3 6 12
```

## 📌 Parameter Explanation

* `-m`: Specify ViT architecture (e.g., `deit_base_patch16_224`).
* `--checkpoint`: Path to pretrained ThinkingViT model.
* `--data-dir`: Path to the ImageNet.
* `-b`: Batch size for inference.
* `--threshold`: Entropy threshold.
* `--use-ema`: Use Exponential Moving Average weights.
* `--thinking_stages`: Attention heads used in each inference stage.

---

## 🎯 Training

To train ThinkingViT, use the following code. The `args.yaml` file contains all the parameters needed to train ThinkingViT. We train ThinkingViT on 2 A100 GPUs with a global batch size of `1024`. You can download the DeiT-Tiny checkpoint from [here](https://zenodo.org/records/17429320).

```bash
./distributed_train.sh 2 \
  --config args.yaml \
  --output [CHECKPOINT_SAVING_PATH]\
  --initial-checkpoint [PATH_TO_PRETRAINED_CHECKPOINT] \
  --thinking_stages H1 H2
```


### Performance Metrics of `ThinkingViT 3H → 6H` Across Different Entropy Thresholds

| **Entropy Threshold** | **Accuracy** | **Throughput [#/s]** | **Params [M]** | **GMACs** | **Second Round Call Ratio [%]** |
|-----------------------|--------------|----------------------|----------------|-----------|----------------------------------|
| 0.0                   | 81.444       | 3157.09              | 22.01          | 5.85      | 100.0                            |
| 0.1                   | 81.440       | 3347.69              | 22.01          | 5.47      | 71.7                             |
| 0.3                   | 81.438       | 3955.05              | 22.01          | 4.50      | 70.58                            |
| 0.5                   | 81.386       | 4380.71              | 22.01          | 3.98      | 59.29                            |
| 0.7                   | 81.230       | 4807.04              | 22.01          | 3.55      | 49.95                            |
| 0.9                   | 80.714       | 5342.47              | 22.01          | 3.11      | 40.36                            |
| 1.1                   | 79.990       | 5918.90              | 22.01          | 2.72      | 31.97                            |
| 1.3                   | 79.114       | 6535.13              | 22.01          | 2.38      | 24.63                            |
| 1.5                   | 77.936       | 7201.46              | 22.01          | 2.08      | 18.11                            |
| 1.7                   | 76.766       | 7944.38              | 22.01          | 1.81      | 12.13                            |
| 2.0                   | 74.736       | 9203.90              | 22.01          | 1.44      | 4.20                             |
| 2.5                   | 73.580       | 10047.60             | 22.01          | 1.25      | 0.0                              |


## Evaluation

Comparison of ThinkingViT and baselines on ImageNet-1K: We compare ThinkingViT with MatFormer, HydraViT, SortedNet, and DynaBERT, all built on DeiT-Tiny using the same ViT training setup. ThinkingViT uses two progressive stages (3 and 6 heads) and achieves up to 2.0 p.p. higher accuracy at equal throughput and 2.9 p.p. at equal GMACs on an A100.

<div align="center">
  <img src="figures/gmacs_imagenet_val.png" width="45%">
  <img src="figures/throughput_imagenet_val.png" width="45%">
</div>


## 🌲 Swin Transformer Variant

We also provide a Swin Transformer adaptation that reuses the progressive head scheduling defined in `swin_transformer.py`. 

<div align="center">
  <img src="figures/swin_accuracy_vs_gmacs.png" width="45%">
</div>

### Training

```bash
torchrun --nproc_per_node=4 train_swin.py --config args_swin.yaml
```

### Inference

```bash
torchrun validate_swin.py \
  -m swin_small_patch4_window7_224 \
  --checkpoint [CHECKPOINT_PATH] \
  --data-dir [IMAGENET_DIR] \
  -b 128 \
  --use-ema \
  --threshold [CONFIDENCE_THRESHOLD]
```
You can download pretrained checkpoints models [Zenodo](https://zenodo.org/records/17429320).


## Contributing

- [pytorch-image-modes (timm)](https://github.com/huggingface/pytorch-image-models)
