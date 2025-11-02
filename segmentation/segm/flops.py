"""
DeepSpeed-based FLOP counter for Segmenter checkpoints.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import click
import torch
import torch.nn as nn

from segm.model.factory import load_model
import segm.utils.torch as ptu


def _ensure_deepspeed():
    try:
        from deepspeed.profiling.flops_profiler import FlopsProfiler  # type: ignore
        from deepspeed.profiling.flops_profiler.profiler import MODULE_HOOK_MAPPING  # type: ignore
    except ImportError as err:  # pragma: no cover - import guard
        raise click.ClickException(
            "DeepSpeed is required for this script. Install it with `pip install deepspeed`."
        ) from err
    return FlopsProfiler


def _parse_stages(stage_str: Optional[str], fallback: Optional[Sequence[int]]) -> List[int]:
    if stage_str is None:
        return list(fallback or [])
    stages: List[int] = []
    for value in stage_str.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            stages.append(int(value))
        except ValueError as exc:
            raise click.BadParameter(f"Invalid stage value '{value}'. Expected integers.") from exc
    return stages


def _resolve_input_size(
    height: Optional[int],
    width: Optional[int],
    dataset_cfg,
) -> Tuple[int, int]:
    size_default = dataset_cfg.get("image_size", 512)
    if isinstance(size_default, (list, tuple)):
        default_h, default_w = size_default
    else:
        default_h = default_w = int(size_default)
    return height or default_h, width or default_w


def _resolve_device(device: str) -> torch.device:
    device = device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        click.echo("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _sum_attr(module: nn.Module, attr: str) -> float:
    total = float(getattr(module, attr, 0.0))
    for child in module.children():
        total += _sum_attr(child, attr)
    return total


def _profile_stage(
    model,
    profiler_cls,
    input_tensor: torch.Tensor,
    stages: Sequence[int],
) -> Tuple[float, float, float]:
    profiler = profiler_cls(model)
    profiler.start_profile()
    with torch.no_grad():
        _ = model(input_tensor, thinking_stages=stages, threshold=None)
    profiler.stop_profile()
    flops = _sum_attr(model, "__flops__")
    macs = _sum_attr(model, "__macs__")
    params = _sum_attr(model, "__params__")
    profiler.end_profile()
    return float(flops), float(macs), float(params)


@click.command()
@click.argument("model_path", type=click.Path(dir_okay=False, exists=True))
@click.option(
    "--thinking-stages",
    type=str,
    default=None,
    help="Comma-separated head counts per stage. Defaults to variant.yml if omitted.",
)
@click.option("--height", type=int, default=None, help="Input height (defaults to dataset config).")
@click.option("--width", type=int, default=None, help="Input width (defaults to dataset config).")
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cpu")
@click.option(
    "--dtype",
    type=click.Choice(["float32", "float16", "bfloat16"]),
    default="float32",
    help="Input tensor dtype used for profiling.",
)
def main(model_path, thinking_stages, height, width, device, dtype):
    profiler_cls = _ensure_deepspeed()

    model_path = Path(model_path).resolve()
    torch_device = _resolve_device(device)
    torch_dtype = _dtype_from_name(dtype)

    ptu.set_gpu_mode(torch_device.type == "cuda")
    model, variant = load_model(str(model_path))
    model.eval()
    model.to(torch_device)

    stage_list = _parse_stages(thinking_stages, (variant.get("thinking_kwargs") or {}).get("stages"))
    if not stage_list:
        raise click.UsageError("No thinking stages provided or found in the checkpoint variant.")

    dataset_cfg = variant.get("dataset_kwargs", {})
    h, w = _resolve_input_size(height, width, dataset_cfg)

    input_tensor = torch.randn(1, 3, h, w, device=torch_device, dtype=torch_dtype)

    click.echo(f"Computing FLOPs for model: {model_path}")
    click.echo(f"Stages: {stage_list}")
    click.echo(f"Input size: {h}x{w}, device={torch_device}, dtype={torch_dtype}")
    click.echo("")
    click.echo(
        f"{'Stop @ Stage':>12} {'Heads':>10} "
        f"{'Total FLOPs (G)':>18} {'Extra Stage FLOPs (G)':>24}"
    )
    click.echo("-" * 70)

    cumulative = 0.0
    for idx, heads in enumerate(stage_list):
        active_stages = stage_list[: idx + 1]
        flops, _, _ = _profile_stage(model, profiler_cls, input_tensor, active_stages)
        gflops = flops / 1e9
        incremental = gflops - cumulative
        cumulative = gflops
        click.echo(
            f"{idx + 1:>12} {heads:>10} "
            f"{gflops:>18.3f} {incremental:>24.3f}"
        )

    click.echo("-" * 70)
    click.echo("Done.")


if __name__ == "__main__":
    try:
        main()
    except click.ClickException as err:
        click.echo(f"Error: {err}", err=True)
        raise SystemExit(1)
