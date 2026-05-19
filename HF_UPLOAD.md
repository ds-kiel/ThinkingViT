# Hugging Face Publishing Guide

This repository can export the ThinkingViT checkpoints into Hugging Face Hub-ready model folders.
The exporter writes EMA weights only, using `model.safetensors`, and creates a model card for each checkpoint.

## 1. Export locally

Install the export dependencies if your environment does not already have them:

```bash
pip install -r requirements.txt
```

From the repository root:

```bash
python scripts/export_hf_models.py --namespace YOUR_HF_NAMESPACE
```

For the Kiel organization, use:

```bash
python scripts/export_hf_models.py --namespace NCPS
```

This creates:

```text
hf_exports/thinkingvit_deit-3h-6h-imagenet1k/
hf_exports/thinkingvit-swin-s-imagenet1k/
```

Each folder contains:

- `model.safetensors`: EMA weights only.
- `config.json`: timm Hub config with the custom model arguments.
- `README.md`: model card with paper, project, GitHub, usage, and ImageNet results.

## 2. Push to Hugging Face

Log in once:

```bash
huggingface-cli login
```

Then push both model repos:

```bash
python scripts/export_hf_models.py --namespace NCPS --push
```

To test privately first:

```bash
python scripts/export_hf_models.py --namespace NCPS --private --push
```

You can also push one model at a time:

```bash
python scripts/export_hf_models.py --namespace NCPS --model thinkingvit_deit --push
python scripts/export_hf_models.py --namespace NCPS --model swin --push
```

## 3. Submit and claim the paper

Submit the paper at:

```text
https://huggingface.co/papers/submit
```

Use arXiv ID:

```text
2507.10800
```

After the paper page exists, claim it as an author and add:

- GitHub: https://github.com/ds-kiel/ThinkingViT
- Project page: https://ds-kiel.github.io/ThinkingViT-project-page/
- Models:
  - `NCPS/thinkingvit_deit-3h-6h-imagenet1k`
  - `NCPS/thinkingvit-swin-s-imagenet1k`

## Suggested Reply to Niels

```text
Hi Niels,

Thanks for reaching out. Yes, we would be happy to submit the paper to hf.co/papers and host the ThinkingViT checkpoints on the Hub.

We have prepared Hub-ready exports for the ImageNet-1K EMA checkpoints as safetensors, with model cards linking the arXiv paper, GitHub repository, project page, and evaluation results. The implementation is a custom timm-based PyTorch model, so the cards include usage through the ThinkingViT repository code.

A Space with a threshold slider for elastic inference would also be useful, so a ZeroGPU grant would be appreciated once the model repositories are public.

Best,
<your name>
```
