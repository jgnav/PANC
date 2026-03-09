"""Unsupervised Normalized Cut on DINOv3 patch tokens.

Applies standard NCut (no priors / anchors) and thresholds the Fiedler vector
with the mean of its values.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def ncut_unsupervised(
    query_features: torch.Tensor,
    tau: float = 1.0,
    eig_eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unsupervised Normalized Cut (no priors).

    Args:
        query_features: [N, D] patch-token features.
        tau: Temperature for the cosine-similarity affinity.
        eig_eps: Small epsilon for numerical stability.

    Returns:
        query_mask:  [N] binary mask (1 = foreground).
        fiedler_scores: [N] min-max normalised 2nd eigenvector (Fiedler).
        eig3_scores: [N] min-max normalised 3rd eigenvector.
        threshold: scalar tensor — mean of the Fiedler scores.
    """
    if query_features.ndim != 2:
        raise ValueError("query_features must be 2-D [N, D].")

    device = query_features.device
    N = query_features.size(0)

    # 1. Build cosine-similarity affinity matrix
    normed = F.normalize(query_features, p=2, dim=1)
    similarity = normed @ normed.T
    affinity = torch.exp(similarity / tau)
    affinity.fill_diagonal_(0.0)

    # 2. Symmetric normalised graph Laplacian  L_sym = I - D^{-1/2} W D^{-1/2}
    degree = affinity.sum(dim=1).clamp_min(eig_eps)
    inv_sqrt_deg = torch.rsqrt(degree)
    norm_aff = inv_sqrt_deg[:, None] * affinity * inv_sqrt_deg[None, :]
    laplacian = torch.eye(N, device=device, dtype=query_features.dtype) - norm_aff

    # 3. Eigendecomposition — smallest eigenvalues first
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

    # Skip the constant eigenvector (eigenvalue ≈ 0)
    nonzero = torch.nonzero(eigenvalues > eig_eps, as_tuple=False).view(-1)
    fiedler_idx = int(nonzero[0].item()) if nonzero.numel() > 0 else min(1, N - 1)

    # Fiedler vector (2nd smallest eigenvector)
    fiedler_raw = inv_sqrt_deg * eigenvectors[:, fiedler_idx]

    # 3rd eigenvector
    eig3_idx = fiedler_idx + 1
    if eig3_idx < N:
        eig3_raw = inv_sqrt_deg * eigenvectors[:, eig3_idx]
    else:
        eig3_raw = torch.zeros_like(fiedler_raw)

    # 4. Min-max normalisation → [0, 1]
    def _minmax(v: torch.Tensor) -> torch.Tensor:
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + eig_eps)

    fiedler_scores = _minmax(fiedler_raw)
    eig3_scores = _minmax(eig3_raw)

    # 5. Threshold = mean of the Fiedler scores
    threshold = fiedler_scores.mean()
    query_mask = (fiedler_scores > threshold).to(torch.int64)

    return query_mask, fiedler_scores, eig3_scores, threshold
