import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

import segm.utils.torch as ptu

from segm.data.utils import STATS
from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb

from segm.model.factory import load_model
from segm.model.utils import inference


def _parse_stages(stage_str):
    if stage_str is None or stage_str == "":
        return None
    return [int(v.strip()) for v in stage_str.split(",") if v.strip()]


def _parse_thresholds(threshold_vals, default_threshold):
    if not threshold_vals:
        return [default_threshold]
    parsed = []
    for val in threshold_vals:
        if val is None:
            parsed.append(None)
            continue
        text = val.strip()
        if text.lower() in ("none", "null"):
            parsed.append(None)
        else:
            parsed.append(float(text))
    return parsed


@click.command()
@click.argument("model_path_arg", type=str, required=False)
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
@click.option(
    "--thinking-stages",
    type=str,
    default=None,
    help="Comma-separated head counts per stage (overrides model variant).",
)
@click.option(
    "--thinking-threshold",
    type=str,
    multiple=True,
    help="Entropy threshold(s) for gating. "
         "Use multiple times for different values, or 'none' for no gating.",
)
def main(model_path_arg, model_path, input_dir, output_dir, gpu, thinking_stages, thinking_threshold):
    model_path = model_path or model_path_arg
    if not model_path:
        raise click.BadParameter("Please provide a model path either positionally or via --model-path.")
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    model.to(ptu.device)

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    _, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)

    variant_thinking = variant.get("thinking_kwargs", {}) or {}
    stage_list = _parse_stages(thinking_stages)
    if stage_list is None:
        stage_list = variant_thinking.get("stages")
    thresholds = _parse_thresholds(thinking_threshold, variant_thinking.get("threshold"))

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    list_dir = list(input_dir.iterdir())
    for threshold in thresholds:
        threshold_dir = output_dir
        if len(thresholds) > 1:
            name = "no_threshold" if threshold is None else f"threshold_{threshold:.3f}"
            threshold_dir = output_dir / name
            threshold_dir.mkdir(exist_ok=True, parents=True)

        for filename in tqdm(list_dir, ncols=80, desc=f"thr={threshold}"):
            pil_im = Image.open(filename).copy()
            im = F.pil_to_tensor(pil_im).float() / 255
            im = F.normalize(im, normalization["mean"], normalization["std"])
            im = im.to(ptu.device).unsqueeze(0)

            im_meta = dict(flip=False)
            logits = inference(
                model,
                [im],
                [im_meta],
                ori_shape=im.shape[2:4],
                window_size=variant["inference_kwargs"]["window_size"],
                window_stride=variant["inference_kwargs"]["window_stride"],
                batch_size=2,
                thinking_stages=stage_list,
                thinking_threshold=threshold,
            )
            seg_map = logits.argmax(0, keepdim=True)
            seg_rgb = seg_to_rgb(seg_map, cat_colors)
            seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
            pil_seg = Image.fromarray(seg_rgb[0])

            pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
            pil_blend.save(threshold_dir / filename.name)


if __name__ == "__main__":
    main()
