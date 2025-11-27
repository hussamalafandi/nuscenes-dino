"""Utilities for dense feature extraction and cross-view correspondences using DINOv3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image

from nuscenes_dino.vlm import DinoImageEncoder
import matplotlib.pyplot as plt



@dataclass
class DenseFeatureMap:
    """Container holding patch-level DINO features for a single image."""

    embeddings: torch.Tensor
    grid_size: tuple[int, int]
    patch_size: int
    original_size: tuple[int, int]
    resized_size: tuple[int, int]

    def patch_center(self, index: int) -> tuple[float, float]:
        """Return pixel-space center for a flattened patch index."""

        h, w = self.grid_size
        row, col = divmod(index, w)
        scale_x = self.original_size[0] / self.resized_size[1]
        scale_y = self.original_size[1] / self.resized_size[0]
        x = (col + 0.5) * self.patch_size * scale_x
        y = (row + 0.5) * self.patch_size * scale_y
        return x, y


@dataclass
class Correspondence:
    """Represents a match between two patch centers."""

    score: float
    left: tuple[float, float]
    right: tuple[float, float]


class DinoMatcher:
    """Compute dense correspondences across images with a DINO encoder."""

    def __init__(self, encoder: DinoImageEncoder) -> None:
        self.encoder = encoder

    @torch.inference_mode()
    def extract_dense_maps(self, images: Iterable[Image.Image]) -> list[DenseFeatureMap]:
        batch = list(images)
        inputs = self.encoder.processor(images=batch, return_tensors="pt")
        if self.encoder.device_map is None:
            inputs = inputs.to(self.encoder.device)
        elif hasattr(self.encoder.model, "device"):
            inputs = inputs.to(self.encoder.model.device)

        outputs = self.encoder.model(**inputs)
        tokens = outputs.last_hidden_state

        resized_h, resized_w = inputs["pixel_values"].shape[-2:]
        patch_size = getattr(self.encoder.model.config, "patch_size", None)
        if patch_size is None:
            grid_guess = int(tokens.shape[1] ** 0.5)
            patch_size = resized_h // max(grid_guess, 1)

        if resized_h % patch_size != 0 or resized_w % patch_size != 0:
            msg = (
                "Resized image dimensions are not divisible by the patch size; "
                "cannot compute grid shape."
            )
            raise ValueError(msg)

        grid_h = resized_h // patch_size
        grid_w = resized_w // patch_size
        patch_tokens = grid_h * grid_w
        if tokens.shape[1] < patch_tokens:
            msg = (
                "Model output does not contain enough patch tokens to fill the grid. "
                f"Got {tokens.shape[1]}, expected {patch_tokens}."
            )
            raise ValueError(msg)

        # Drop any leading special tokens (e.g., CLS or register tokens) and keep the
        # most recent patch tokens so the reshape corresponds to the image grid.
        tokens = tokens[:, -patch_tokens:, :]

        batch_embeddings = tokens.reshape(tokens.shape[0], grid_h, grid_w, -1)

        dense_maps: list[DenseFeatureMap] = []
        for emb, image in zip(batch_embeddings, batch, strict=False):
            dense_maps.append(
                DenseFeatureMap(
                    embeddings=emb.detach().cpu(),
                    grid_size=(grid_h, grid_w),
                    patch_size=patch_size,
                    original_size=image.size,
                    resized_size=(resized_h, resized_w),
                )
            )
        return dense_maps

    def match(self, left: DenseFeatureMap, right: DenseFeatureMap, top_k: int = 40) -> list[Correspondence]:
        """Return the top-k cosine-similar patch correspondences between two views."""

        left_feats = left.embeddings.reshape(-1, left.embeddings.shape[-1])
        right_feats = right.embeddings.reshape(-1, right.embeddings.shape[-1])
        left_feats = F.normalize(left_feats, dim=-1)
        right_feats = F.normalize(right_feats, dim=-1)

        sim = left_feats @ right_feats.T
        values, indices = torch.topk(sim.flatten(), k=min(top_k, sim.numel()))

        matches: list[Correspondence] = []
        right_total = right.embeddings.shape[0] * right.embeddings.shape[1]
        for score, flat_idx in zip(values.tolist(), indices.tolist(), strict=False):
            left_idx = flat_idx // right_total
            right_idx = flat_idx % right_total
            matches.append(
                Correspondence(
                    score=score,
                    left=left.patch_center(int(left_idx)),
                    right=right.patch_center(int(right_idx)),
                )
            )
        return matches


def plot_correspondences(
    left_image: Image.Image,
    right_image: Image.Image,
    correspondences: list[Correspondence],
    max_lines: int = 20,
) -> "plt.figure.Figure":
    """Visualize matches by drawing lines between patch centers."""


    corr = correspondences[:max_lines]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(left_image)
    axes[1].imshow(right_image)
    axes[0].axis("off")
    axes[1].axis("off")

    for match in corr:
        x0, y0 = match.left
        x1, y1 = match.right
        axes[0].scatter(x0, y0, c="lime", s=24)
        axes[1].scatter(x1, y1, c="lime", s=24)

    # draw connecting lines on a stitched canvas
    stitched = Image.new("RGB", (left_image.width + right_image.width, max(left_image.height, right_image.height)))
    stitched.paste(left_image, (0, 0))
    stitched.paste(right_image, (left_image.width, 0))
    axes_full = fig.add_axes((0, 0, 1, 1), frameon=False)
    axes_full.imshow(stitched)
    axes_full.axis("off")

    for match in corr:
        x0, y0 = match.left
        x1, y1 = match.right
        axes_full.plot([x0, x1 + left_image.width], [y0, y1], c="yellow", linewidth=1.5)

    return fig