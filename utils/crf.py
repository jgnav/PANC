from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _bilateral_message(
    image_rgb: np.ndarray,
    q_fg: np.ndarray,
    spatial_std: float,
    color_std: float,
) -> np.ndarray:
    """Approximate bilateral message passing.

    Weights the current q by color similarity, then spatially blurs.
    """
    img = image_rgb.astype(np.float32) / 255.0

    # Compute a smoothed version of the image for local color reference
    img_smooth = gaussian_filter(img, sigma=[spatial_std, spatial_std, 0])

    # Color affinity: how similar each pixel is to its local neighborhood average
    color_diff_sq = ((img - img_smooth) ** 2).sum(axis=2)
    color_weight = np.exp(-color_diff_sq / (2.0 * color_std ** 2 + 1e-8))

    # Spatially blur the color-weighted q
    weighted = q_fg * color_weight
    blurred_weighted = gaussian_filter(weighted, sigma=spatial_std)
    blurred_weight = gaussian_filter(color_weight, sigma=spatial_std)

    return np.clip(blurred_weighted / (blurred_weight + 1e-8), 0.0, 1.0)


# Default CRF hyper-parameters (hard-coded so the demo stays simple)
_DEFAULT_CRF_ITERS = 5
_DEFAULT_POS_XY_STD = 3.0
_DEFAULT_POS_W = 3.0
_DEFAULT_BIL_XY_STD = 80.0
_DEFAULT_BIL_RGB_STD = 13.0
_DEFAULT_BIL_W = 10.0


def refine_with_crf(
    score_map: np.ndarray,
    image_rgb: np.ndarray,
    num_iterations: int = _DEFAULT_CRF_ITERS,
    pos_xy_std: float = _DEFAULT_POS_XY_STD,
    pos_w: float = _DEFAULT_POS_W,
    bilateral_xy_std: float = _DEFAULT_BIL_XY_STD,
    bilateral_rgb_std: float = _DEFAULT_BIL_RGB_STD,
    bilateral_w: float = _DEFAULT_BIL_W,
) -> np.ndarray:
    """Refine a soft score map into a binary mask using mean-field CRF inference.

    Args:
        score_map: [H, W] array in [0, 1] — foreground probability.
        image_rgb: [H, W, 3] uint8 image.
        num_iterations: Number of mean-field iterations.
        pos_xy_std: Spatial std for the smoothness (Gaussian) kernel.
        pos_w: Weight for the smoothness kernel.
        bilateral_xy_std: Spatial std for the bilateral kernel.
        bilateral_rgb_std: Color std for the bilateral kernel.
        bilateral_w: Weight for the bilateral kernel.

    Returns:
        Binary mask [H, W] as uint8 (0 or 1).
    """
    score_map = np.clip(score_map.astype(np.float32), 1e-6, 1.0 - 1e-6)

    # Unary potentials (negative log-likelihood)
    unary_fg = -np.log(score_map)
    unary_bg = -np.log(1.0 - score_map)

    q_fg = score_map.copy()

    for _ in range(num_iterations):
        # Pairwise message 1: Gaussian smoothness
        msg_gauss = gaussian_filter(q_fg, sigma=pos_xy_std)

        # Pairwise message 2: Bilateral (appearance)
        msg_bilateral = _bilateral_message(
            image_rgb, q_fg,
            spatial_std=bilateral_xy_std,
            color_std=bilateral_rgb_std,
        )

        # Combine unary + pairwise
        energy_fg = unary_fg - pos_w * msg_gauss - bilateral_w * msg_bilateral
        energy_bg = unary_bg - pos_w * (1.0 - msg_gauss) - bilateral_w * (1.0 - msg_bilateral)

        # Softmax update
        max_e = np.maximum(energy_fg, energy_bg)
        exp_fg = np.exp(-(energy_fg - max_e))
        exp_bg = np.exp(-(energy_bg - max_e))
        q_fg = exp_fg / (exp_fg + exp_bg + 1e-8)
        q_fg = np.clip(q_fg, 1e-6, 1.0 - 1e-6)

    return (q_fg >= 0.5).astype(np.uint8)
