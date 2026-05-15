#!/usr/bin/env python3
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn

from timm.models import create_model
from train_swin import SwinTokenProjector, apply_swin_head_rounds, swin_projector_dims


class SwinMacCounter:
    def __init__(self, model: nn.Module):
        self.model = model
        self.total_macs = 0
        self.by_type: Dict[str, int] = {}
        self.handles = []

    def reset(self) -> None:
        self.total_macs = 0
        self.by_type = {}

    def add(self, kind: str, macs: int) -> None:
        macs = int(macs)
        self.total_macs += macs
        self.by_type[kind] = self.by_type.get(kind, 0) + macs

    def _linear_hook(self, module, args, kwargs, output) -> None:
        if not torch.is_tensor(output) or output.shape[-1] == 0:
            return
        x = args[0]
        keep_in = kwargs.get('keep_shape_in')
        if keep_in is None and len(args) > 1:
            keep_in = args[1]
        keep_in = int(keep_in) if keep_in is not None else int(x.shape[-1])

        out_dim = int(output.shape[-1])
        positions = output.numel() // out_dim
        self.add('linear', positions * keep_in * out_dim)

    def _conv2d_hook(self, kind: str, module: nn.Conv2d, args, kwargs, output) -> None:
        if not torch.is_tensor(output):
            return
        b, out_channels, out_h, out_w = output.shape
        kernel_h, kernel_w = module.kernel_size
        in_per_group = module.in_channels // module.groups
        self.add(kind, b * out_channels * out_h * out_w * in_per_group * kernel_h * kernel_w)

    def _conv_transpose2d_hook(self, kind: str, module: nn.ConvTranspose2d, args, kwargs, output) -> None:
        x = args[0]
        b, in_channels, in_h, in_w = x.shape
        kernel_h, kernel_w = module.kernel_size
        out_per_group = module.out_channels // module.groups
        self.add(kind, b * in_channels * in_h * in_w * out_per_group * kernel_h * kernel_w)

    def _attention_hook(self, module, args, kwargs, output) -> None:
        q_shape = getattr(module, 'last_q_shape', None)
        if not q_shape:
            return
        batch_windows, heads, tokens, head_dim = [int(v) for v in q_shape]
        self.add('attention_matmul', 2 * batch_windows * heads * tokens * tokens * head_dim)

    def __enter__(self):
        for name, module in self.model.named_modules():
            class_name = module.__class__.__name__
            if class_name == 'Linear':
                self.handles.append(module.register_forward_hook(self._linear_hook, with_kwargs=True))
            elif isinstance(module, nn.Conv2d):
                kind = 'projection' if name.startswith('proj_layer.') else 'conv2d'
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, kwargs, out, kind=kind: self._conv2d_hook(kind, mod, args, kwargs, out),
                    with_kwargs=True,
                ))
            elif isinstance(module, nn.ConvTranspose2d):
                kind = 'projection' if name.startswith('proj_layer.') else 'conv_transpose2d'
                self.handles.append(module.register_forward_hook(
                    lambda mod, args, kwargs, out, kind=kind: self._conv_transpose2d_hook(kind, mod, args, kwargs, out),
                    with_kwargs=True,
                ))
            elif class_name == 'WindowAttention':
                self.handles.append(module.register_forward_hook(self._attention_hook, with_kwargs=True))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _source_hw(model: nn.Module) -> Tuple[int, int]:
    return tuple(dim // (2 ** (len(model.layers) - 1)) for dim in model.patch_grid)


def _build_model(args, device: torch.device) -> nn.Module:
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
    )
    apply_swin_head_rounds(model, args.head_rounds, args.head_round_1, args.head_round_2)
    projector_in, projector_out = swin_projector_dims(model)
    model.alpha = nn.ParameterList([nn.Parameter(torch.zeros(1, device=device))])
    model.proj_layer = SwinTokenProjector(
        in_channels=projector_in,
        out_channels=projector_out,
        target_hw=model.patch_grid,
        source_hw=_source_hw(model),
        device=device,
    )
    model.to(device)
    model.eval()
    return model


def _run_round(model: nn.Module, x: torch.Tensor, round_idx: int, prev_tokens=None):
    tokens, keep_dim = model._forward_round(x, model.head_rounds[round_idx], prev_tokens=prev_tokens)
    logits = model.forward_head(tokens, keep_dim=keep_dim)
    return tokens, logits


def calculate_round_gmacs(model: nn.Module, image_size: int, batch_size: int, device: torch.device):
    x = torch.zeros(batch_size, 3, image_size, image_size, device=device)
    results = []

    with torch.no_grad(), SwinMacCounter(model) as counter:
        counter.reset()
        tokens0, _ = _run_round(model, x, 0)
        round0_macs = counter.total_macs / batch_size
        results.append((0, model.head_rounds[0], round0_macs, dict(counter.by_type)))

        counter.reset()
        tokens0, _ = _run_round(model, x, 0)
        counter.reset()
        _run_round(model, x, 1, prev_tokens=tokens0)
        round1_macs = counter.total_macs / batch_size
        results.append((1, model.head_rounds[1], round1_macs, dict(counter.by_type)))

    return results


def _format_gmacs(macs: float) -> str:
    return f'{macs / 1e9:.4f}'


def main():
    parser = argparse.ArgumentParser(description='Calculate ThinkingViT-Swin GMACs per round.')
    parser.add_argument('--model', default='swin_small_patch4_window7_224')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-classes', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--head-rounds', '--head_rounds', nargs='+', default=None)
    parser.add_argument('--head-round-1', '--head_round_1', nargs='+', default=None)
    parser.add_argument('--head-round-2', '--head_round_2', nargs='+', default=None)
    args = parser.parse_args()
    if args.head_rounds is None and args.head_round_1 is None and args.head_round_2 is None:
        args.head_round_1 = [3, 3, 6, 12]
        args.head_round_2 = [3, 6, 12, 24]

    device = torch.device(args.device)
    model = _build_model(args, device)
    results = calculate_round_gmacs(model, args.img_size, args.batch_size, device)

    print(f'Model: {args.model}')
    print(f'Input: {args.img_size}x{args.img_size}')
    print('MACs counted: Conv2d, ConvTranspose2d, sliced Linear layers, attention QK/AV matmuls.')
    print('Projection GMACs are included in the incremental and cumulative GMACs.')
    print('')
    print('| Round | Head round | Incremental GMACs | Projection GMACs | Cumulative GMACs |')
    print('|---|---|---:|---:|---:|')

    cumulative = 0.0
    for round_idx, head_round, macs, by_type in results:
        cumulative += macs
        projection_macs = by_type.get('projection', 0) / args.batch_size
        print(
            f'| {round_idx + 1} | {tuple(head_round)} | '
            f'{_format_gmacs(macs)} | {_format_gmacs(projection_macs)} | {_format_gmacs(cumulative)} |'
        )

    print('')
    for round_idx, _, macs, by_type in results:
        print(f'Round {round_idx + 1} breakdown:')
        for kind, value in sorted(by_type.items()):
            print(f'  {kind}: {_format_gmacs(value / args.batch_size)} GMACs')


if __name__ == '__main__':
    main()
