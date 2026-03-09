"""Visualization helpers for segmentation results."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Distinct colors for GT class overlays
CLASS_COLORS: List[Tuple[float, float, float]] = [
    (0.1, 0.75, 0.2),   # green
    (0.2, 0.5, 1.0),    # blue
    (1.0, 0.6, 0.1),    # orange
    (0.7, 0.2, 0.8),    # purple
]

# Red for result masks (NCut / PANC)
RESULT_COLOR: Tuple[float, float, float] = (1.0, 0.2, 0.1)


def load_image(image_path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 numpy array."""
    with Image.open(image_path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.1),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an image with a given color and transparency."""
    base = image_rgb.astype(np.float32) / 255.0
    overlay = base.copy()
    m = mask.astype(bool)
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * np.array(color, dtype=np.float32)
    return np.clip(overlay, 0.0, 1.0)


def upscale_map(
    patch_map: np.ndarray,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> np.ndarray:
    """Upscale a (grid_h, grid_w) map to (height, width)."""
    t = torch.from_numpy(patch_map).float().unsqueeze(0).unsqueeze(0)
    align = False if mode == "nearest" else True
    kwargs: dict = {"mode": mode, "size": (height, width)}
    if mode == "bilinear":
        kwargs["align_corners"] = align
    up = F.interpolate(t, **kwargs).squeeze().numpy()
    return up


def plot_target_with_gt(
    image_rgb: np.ndarray,
    samples: Sequence[Any],
) -> None:
    """Plot the target image with per-class GT masks in distinct colours."""
    n = len(samples)
    fig, axes = plt.subplots(1, 1 + n, figsize=(6 * (1 + n), 5))
    if n == 0:
        axes = np.atleast_1d(axes)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Target image")
    for i, s in enumerate(samples):
        c = CLASS_COLORS[i % len(CLASS_COLORS)]
        axes[i + 1].imshow(overlay_mask(image_rgb, s.mask, color=c))
        axes[i + 1].set_title(f"GT: {s.category_name}")
    for ax in np.atleast_1d(axes):
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_prior_gallery(
    priors_by_class: Dict[str, List[Any]],
) -> None:
    """Plot prior images and masks for each class."""
    for class_name, samples in priors_by_class.items():
        n = len(samples)
        fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))
        if n == 1:
            axes = np.array([axes])
        fig.suptitle(f"Priors for '{class_name}'")
        for row, s in enumerate(samples):
            axes[row, 0].imshow(load_image(s.image_path))
            axes[row, 0].set_title(f"Image {row + 1}")
            axes[row, 1].imshow(s.mask, cmap="gray")
            axes[row, 1].set_title(f"Mask {row + 1}")
            for ax in axes[row]:
                ax.axis("off")
        fig.tight_layout()
        plt.show()
        plt.close(fig)


def plot_unsupervised(
    image_rgb: np.ndarray,
    ncut_data: Dict[str, Any],
    target_samples: Sequence[Any] = (),
) -> None:
    """Plot unsupervised NCut results (single row).

    Panels: GT overlay (all classes), Fiedler (eig 2), NCut result overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Unsupervised NCut", fontsize=16, fontweight="bold")

    # GT overlay with all target classes in distinct colours
    gt_vis = image_rgb.astype(np.float32) / 255.0
    for i, s in enumerate(target_samples):
        c = np.array(CLASS_COLORS[i % len(CLASS_COLORS)], dtype=np.float32)
        m = s.mask.astype(bool)
        gt_vis[m] = 0.55 * gt_vis[m] + 0.45 * c
    axes[0].imshow(np.clip(gt_vis, 0, 1))
    axes[0].set_title("GT overlay")

    axes[1].imshow(ncut_data["fiedler"], cmap="viridis")
    axes[1].set_title("Eigen-attention (eig 2)")

    axes[2].imshow(overlay_mask(image_rgb, ncut_data["mask"], color=RESULT_COLOR))
    axes[2].set_title(f"NCut overlay (thr={ncut_data['threshold']:.3f})")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_panc_class(
    image_rgb: np.ndarray,
    target: Any,
    panc_data: Dict[str, Any],
    class_index: int = 0,
) -> None:
    """Plot PANC results for a single class (single row).

    Panels: GT overlay, Eigen-attention (eig 2), PANC result overlay.
    """
    name = target.category_name
    gt_color = CLASS_COLORS[class_index % len(CLASS_COLORS)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"PANC: '{name}'", fontsize=16, fontweight="bold")

    axes[0].imshow(overlay_mask(image_rgb, target.mask, color=gt_color))
    axes[0].set_title(f"GT: {name}")

    axes[1].imshow(panc_data["scores"], cmap="viridis")
    axes[1].set_title("Eigen-attention (eig 2)")

    axes[2].imshow(overlay_mask(image_rgb, panc_data["mask"], color=RESULT_COLOR))
    axes[2].set_title(f"PANC overlay (thr={panc_data['threshold']:.3f})")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    plt.close(fig)