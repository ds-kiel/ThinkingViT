import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

from segm.model.utils import padding, unpadding


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def _strip_extra_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        num_extra_tokens = 1 + int(self.encoder.distilled)
        return tokens[:, num_extra_tokens:]

    def _prepare_decoder_input(self, tokens: torch.Tensor):
        features = self._strip_extra_tokens(tokens)
        keep_dim = min(features.shape[-1], self.encoder.d_model)
        if features.shape[-1] != keep_dim:
            features = features[..., :keep_dim]
        return features, keep_dim

    def _decode_stage(
        self,
        tokens: torch.Tensor,
        padded_size,
    ):
        features, keep_dim = self._prepare_decoder_input(tokens)
        masks = self.decoder(features, padded_size, keep_dim=keep_dim)
        masks = F.interpolate(masks, size=padded_size, mode="bilinear")
        entropy = self._mean_pixel_entropy(masks)
        return masks, entropy

    def _finalize_masks(self, masks: torch.Tensor, original_size):
        return unpadding(masks, original_size)

    def _mean_pixel_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
        return entropy.flatten(1).mean(dim=1, keepdim=True)

    def forward(
        self,
        im,
        threshold: Optional[float] = None,
        thinking_stages: Optional[Sequence[int]] = None,
        train_val: bool = False,
    ):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        padded_size = (H, W)
        original_size = (H_ori, W_ori)

        if not thinking_stages:
            tokens = self.encoder(im, return_features=True)
            masks, _ = self._decode_stage(tokens, padded_size)
            return self._finalize_masks(masks, original_size)

        stages = [int(s) for s in thinking_stages]
        if len(stages) == 0:
            tokens = self.encoder(im, return_features=True)
            masks, _ = self._decode_stage(tokens, padded_size)
            return self._finalize_masks(masks, original_size)
        if len(stages) > 3:
            raise NotImplementedError("Thinking segmentation currently supports up to three stages.")

        if self.training:
            outputs = []
            prev_tokens = None
            for idx, heads in enumerate(stages):
                # print('stages:', idx)
                tokens = self.encoder(
                    im,
                    return_features=True,
                    stage_heads=heads,
                    prev_tokens=prev_tokens,
                    stage_idx=idx,
                    thinking_stages=stages,
                )
                masks, _ = self._decode_stage(tokens, padded_size)
                outputs.append(self._finalize_masks(masks, original_size))
                prev_tokens = tokens
            return tuple(outputs)

        if train_val:
            prev_tokens = None
            masks = None
            for idx, heads in enumerate(stages):
                tokens = self.encoder(
                    im,
                    return_features=True,
                    stage_heads=heads,
                    prev_tokens=prev_tokens,
                    stage_idx=idx,
                    thinking_stages=stages,
                )
                masks, _ = self._decode_stage(tokens, padded_size)
                prev_tokens = tokens
            return self._finalize_masks(masks, original_size)

        tokens_stage_0 = self.encoder(
            im,
            return_features=True,
            stage_heads=stages[0],
            stage_idx=0,
            thinking_stages=stages,
        )
        masks_stage_0, entropy_stage_0 = self._decode_stage(tokens_stage_0, padded_size)

        if threshold is None:
            masks_final = masks_stage_0
            prev_tokens = tokens_stage_0
            for stage_idx, heads in enumerate(stages[1:], start=1):
                tokens_stage = self.encoder(
                    im,
                    return_features=True,
                    stage_heads=heads,
                    prev_tokens=prev_tokens,
                    stage_idx=stage_idx,
                    thinking_stages=stages,
                )
                masks_final, _ = self._decode_stage(tokens_stage, padded_size)
                prev_tokens = tokens_stage
            return self._finalize_masks(masks_final, original_size)

        prev_tokens = tokens_stage_0

        if len(stages) == 1:
            stage_mask = torch.zeros(entropy_stage_0.shape[0], device=im.device, dtype=torch.long)
            return self._finalize_masks(masks_stage_0, original_size), stage_mask

        tokens_stage_1 = self.encoder(
            im,
            return_features=True,
            stage_heads=stages[1],
            prev_tokens=prev_tokens,
            stage_idx=1,
            thinking_stages=stages,
        )
        masks_stage_1, entropy_stage_1 = self._decode_stage(tokens_stage_1, padded_size)
        prev_tokens = tokens_stage_1

        mask_stage_0_bool = entropy_stage_0 < threshold
        mask_stage_0 = mask_stage_0_bool.view(-1, 1, 1, 1)

        if len(stages) == 2:
            output = torch.where(mask_stage_0, masks_stage_0, masks_stage_1)
            stage_mask = torch.where(
                mask_stage_0_bool,
                torch.zeros_like(entropy_stage_0, dtype=torch.long),
                torch.ones_like(entropy_stage_0, dtype=torch.long),
            ).squeeze(1)
            return self._finalize_masks(output, original_size), stage_mask

        tokens_stage_2 = self.encoder(
            im,
            return_features=True,
            stage_heads=stages[2],
            prev_tokens=prev_tokens,
            stage_idx=2,
            thinking_stages=stages,
        )
        masks_stage_2, _ = self._decode_stage(tokens_stage_2, padded_size)

        mask_stage_1_bool = (~mask_stage_0_bool) & (entropy_stage_1 < threshold)
        mask_stage_1 = mask_stage_1_bool.view(-1, 1, 1, 1)
        output = masks_stage_2.clone()
        output = torch.where(mask_stage_0, masks_stage_0, output)
        output = torch.where(mask_stage_1, masks_stage_1, output)

        stage_mask = torch.full_like(entropy_stage_0, 2, dtype=torch.long)
        stage_mask = torch.where(mask_stage_1_bool, torch.ones_like(stage_mask), stage_mask)
        stage_mask = torch.where(mask_stage_0_bool, torch.zeros_like(stage_mask), stage_mask).squeeze(1)

        return self._finalize_masks(output, original_size), stage_mask

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
