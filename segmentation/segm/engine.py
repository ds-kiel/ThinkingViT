import torch
import math

from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    thinking_stages=None,
    thinking_loss_weights=None,
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)
    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        with amp_autocast():
            seg_pred = model.forward(im, thinking_stages=thinking_stages)
            if isinstance(seg_pred, (list, tuple)):
                stage_outputs = list(seg_pred)
            else:
                stage_outputs = [seg_pred]

            stage_losses = [criterion(out, seg_gt) for out in stage_outputs]
            if len(stage_losses) > 1:
                if thinking_loss_weights is not None and len(thinking_loss_weights):
                    if len(thinking_loss_weights) != len(stage_losses):
                        raise ValueError(
                            f"Thinking loss weights length {len(thinking_loss_weights)} "
                            f"does not match number of stage outputs {len(stage_losses)}."
                        )
                    weight_tensor = torch.tensor(
                        thinking_loss_weights,
                        device=stage_losses[0].device,
                        dtype=stage_losses[0].dtype,
                    )
                    weight_tensor = weight_tensor / weight_tensor.sum()
                else:
                    weight_tensor = torch.full(
                        (len(stage_losses),),
                        1.0 / len(stage_losses),
                        device=stage_losses[0].device,
                        dtype=stage_losses[0].dtype,
                    )
                loss = torch.stack(stage_losses).mul(weight_tensor).sum()
            else:
                loss = stage_losses[0]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        num_updates += 1
        lr_scheduler.step_update(num_updates=num_updates)

        torch.cuda.synchronize()

        logger.update(
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]["lr"],
        )
        if len(stage_losses) > 1:
            for idx, stage_loss in enumerate(stage_losses):
                logger.update(**{f"loss_stage{idx}": stage_loss.item()})

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
    thinking_stages=None,
    thinking_threshold=None,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    val_seg_pred = {}
    model.eval()
    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]
        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        with amp_autocast():
            seg_pred = utils.inference(
                model_without_ddp,
                ims,
                ims_metas,
                ori_shape,
                window_size,
                window_stride,
                batch_size=1,
                thinking_stages=thinking_stages,
                thinking_threshold=thinking_threshold,
            )
            seg_pred = seg_pred.argmax(0)

        seg_pred = seg_pred.cpu().numpy()
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        data_loader.unwrapped.n_cls,
        ignore_index=IGNORE_LABEL,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
