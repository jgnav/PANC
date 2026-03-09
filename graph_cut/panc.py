"""PANC — Prior-Anchored Normalized Cut.

Spectral segmentation that augments the token affinity graph with two anchor
nodes (positive / negative) connected to labeled prior patches.
Supports multiple threshold strategies: roc, median_midpoint, gmm, platt.
"""
from __future__ import annotations

from typing import Any, Literal, Tuple

import torch
import torch.nn.functional as F


ThresholdStrategy = Literal["roc", "median_midpoint", "gmm", "platt"]


# ── label parsing ────────────────────────────────────────────────

def _to_mask(indices_or_mask: Any, length: int, device: torch.device) -> torch.Tensor:
    if isinstance(indices_or_mask, torch.Tensor):
        tensor = indices_or_mask.to(device)
    else:
        tensor = torch.as_tensor(indices_or_mask, device=device)

    if tensor.dtype == torch.bool:
        if tensor.numel() != length:
            raise ValueError(f"Boolean mask must have length={length}, got {tensor.numel()}.")
        return tensor.view(-1)

    if tensor.ndim == 1 and tensor.numel() == length and torch.is_floating_point(tensor):
        return (tensor > 0).view(-1)

    if tensor.ndim == 1 and tensor.numel() == length and tensor.dtype in {
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
    }:
        if tensor.min() >= 0 and tensor.max() <= 1:
            return tensor.bool().view(-1)

    flat_idx = tensor.view(-1).long()
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    if flat_idx.numel() > 0:
        if flat_idx.min() < 0 or flat_idx.max() >= length:
            raise ValueError("Index labels contain values outside valid prior range.")
        mask[flat_idx] = True
    return mask


def _parse_prior_labels(
    prior_labels: Any,
    num_prior: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(prior_labels, dict):
        pos_key = next((k for k in ["pos", "positive", "positive_idx", "positive_mask"] if k in prior_labels), None)
        neg_key = next((k for k in ["neg", "negative", "negative_idx", "negative_mask"] if k in prior_labels), None)
        if pos_key is None or neg_key is None:
            raise ValueError("Dictionary prior_labels must include positive and negative entries.")
        pos_mask = _to_mask(prior_labels[pos_key], num_prior, device)
        neg_mask = _to_mask(prior_labels[neg_key], num_prior, device)
    elif isinstance(prior_labels, (tuple, list)) and len(prior_labels) == 2:
        pos_mask = _to_mask(prior_labels[0], num_prior, device)
        neg_mask = _to_mask(prior_labels[1], num_prior, device)
    else:
        labels = prior_labels.to(device) if isinstance(prior_labels, torch.Tensor) else torch.as_tensor(prior_labels, device=device)
        if labels.ndim != 1 or labels.numel() != num_prior:
            raise ValueError("1D prior label tensor must have one value per prior token.")
        pos_mask = labels > 0
        neg_mask = labels <= 0

    if not bool(pos_mask.any()):
        raise ValueError("No positive prior nodes were provided.")
    if not bool(neg_mask.any()):
        raise ValueError("No negative prior nodes were provided.")
    if bool((pos_mask & neg_mask).any()):
        raise ValueError("Positive and negative prior sets must be disjoint.")
    return pos_mask, neg_mask


# ── affinity graph ───────────────────────────────────────────────

def _build_affinity(features: torch.Tensor, tau: float) -> torch.Tensor:
    normed = F.normalize(features, p=2, dim=1)
    sim = normed @ normed.T
    aff = torch.exp(sim / tau)
    aff.fill_diagonal_(0.0)
    return aff


def _augment_with_anchors(
    affinity: torch.Tensor,
    num_query: int,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    kappa: float,
    eps: float,
) -> torch.Tensor:
    N = affinity.size(0)
    num_prior = pos_mask.numel()
    connection = torch.zeros(N, 2, device=affinity.device, dtype=affinity.dtype)

    prior_offset = num_query
    seed_idx = torch.arange(num_prior, device=affinity.device) + prior_offset

    if num_query > 0:
        local_mean = affinity[seed_idx, :num_query].mean(dim=1).clamp_min(eps)
    else:
        local_mean = affinity[seed_idx].mean(dim=1).clamp_min(eps)
    anchor_w = (kappa * local_mean).clamp(min=1e-4, max=1e3)

    for p in range(num_prior):
        gi = prior_offset + p
        col = 0 if pos_mask[p] else 1
        connection[gi, col] = anchor_w[p]

    anchor_block = torch.zeros(2, 2, device=affinity.device, dtype=affinity.dtype)
    anchor_block[0, 0] = eps
    anchor_block[1, 1] = eps

    top = torch.cat([affinity, connection], dim=1)
    bot = torch.cat([connection.T, anchor_block], dim=1)
    aug = torch.cat([top, bot], dim=0)
    return 0.5 * (aug + aug.T)


# ── eigen-decomposition ─────────────────────────────────────────

def _fiedler_scores(aug_aff: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    deg = aug_aff.sum(dim=1).clamp_min(eps)
    inv_sqrt_d = torch.rsqrt(deg)
    norm_aff = inv_sqrt_d[:, None] * aug_aff * inv_sqrt_d[None, :]
    L = torch.eye(aug_aff.size(0), device=aug_aff.device, dtype=aug_aff.dtype) - norm_aff

    evals, evecs = torch.linalg.eigh(L)
    nz = torch.nonzero(evals > eps, as_tuple=False).view(-1)
    idx2 = int(nz[0].item()) if nz.numel() > 0 else min(1, evals.numel() - 1)

    fiedler = inv_sqrt_d * evecs[:, idx2]
    idx3 = idx2 + 1
    eig3 = inv_sqrt_d * evecs[:, idx3] if idx3 < evals.numel() else torch.zeros_like(fiedler)
    return fiedler, eig3


# ── threshold strategies ─────────────────────────────────────────

def _threshold_roc(
    prior_scores: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    n_thresh: int = 200,
) -> torch.Tensor:
    cands = torch.linspace(0, 1, n_thresh, device=prior_scores.device, dtype=prior_scores.dtype)
    tpr = (prior_scores[pos_mask, None] > cands[None, :]).float().mean(0)
    fpr = (prior_scores[neg_mask, None] > cands[None, :]).float().mean(0)
    return cands[torch.argmax(tpr - fpr)]


def _threshold_median_midpoint(
    prior_scores: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (prior_scores[pos_mask].median() + prior_scores[neg_mask].median())


def _threshold_gmm(
    all_scores: torch.Tensor,
    prior_scores: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    max_iter: int,
    eps: float,
) -> torch.Tensor:
    try:
        from sklearn.mixture import GaussianMixture
        import numpy as np

        s = all_scores.cpu().numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
        if prior_scores.numel() > 0:
            fg_med = prior_scores[pos_mask].median().item() if pos_mask.any() else float(s.mean())
            bg_med = prior_scores[neg_mask].median().item() if neg_mask.any() else float(s.mean())
            gmm.means_init = np.array([[bg_med], [fg_med]])
        gmm.fit(s)
        means = gmm.means_.ravel()
        fg_comp = int(np.argmax(means))
        grid = np.linspace(0, 1, 1000).reshape(-1, 1)
        post = gmm.predict_proba(grid)[:, fg_comp]
        t_idx = int(np.argmin(np.abs(post - 0.5)))
        return torch.tensor(float(grid[t_idx]), device=all_scores.device, dtype=all_scores.dtype)
    except ImportError:
        # Fallback to torch EM
        init_means = torch.stack([prior_scores[neg_mask].mean(), prior_scores[pos_mask].mean()])
        data = all_scores.view(-1)
        means_t = init_means.clone()
        stds = torch.full((2,), data.std().clamp_min(0.1), device=data.device, dtype=data.dtype)
        weights = torch.full((2,), 0.5, device=data.device, dtype=data.dtype)
        x = data.view(-1, 1)
        for _ in range(max_iter):
            var = (stds ** 2).clamp_min(eps)
            log_prob = (torch.log(weights.clamp_min(eps))[None, :]
                        - 0.5 * torch.log(2 * torch.pi * var)[None, :]
                        - 0.5 * ((x - means_t[None, :]) ** 2) / var[None, :])
            log_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)
            resp = torch.exp(log_prob - log_norm)
            nk = resp.sum(dim=0).clamp_min(eps)
            weights = nk / nk.sum()
            means_t = (resp * x).sum(dim=0) / nk
            stds = torch.sqrt((resp * (x - means_t[None, :]) ** 2).sum(dim=0) / nk).clamp_min(eps)
        # Intersection of the two fitted Gaussians (equal-posterior boundary).
        # For unequal variances solve the quadratic from equating log-pdfs;
        # fall back to the midpoint only when variances are nearly equal.
        m1, m2 = means_t[0], means_t[1]
        s1, s2 = stds[0], stds[1]
        v1, v2 = (s1 ** 2).clamp_min(eps), (s2 ** 2).clamp_min(eps)
        if torch.abs(v1 - v2) < eps:
            return (0.5 * (m1 + m2)).clamp(0.0, 1.0)
        # Quadratic coefficients: a*x^2 + b*x + c = 0
        a = v1 - v2
        b = 2 * (m1 * v2 - m2 * v1)
        c = m2 ** 2 * v1 - m1 ** 2 * v2 + 2 * v1 * v2 * torch.log((s2 / s1).clamp_min(eps))
        disc = (b ** 2 - 4 * a * c).clamp_min(0.0)
        r1 = (-b + torch.sqrt(disc)) / (2 * a + eps)
        r2 = (-b - torch.sqrt(disc)) / (2 * a + eps)
        # Pick the root that lies between the two means
        lo, hi = torch.min(m1, m2), torch.max(m1, m2)
        between1 = (r1 >= lo) & (r1 <= hi)
        between2 = (r2 >= lo) & (r2 <= hi)
        if between1:
            return r1.clamp(0.0, 1.0)
        if between2:
            return r2.clamp(0.0, 1.0)
        # Neither root is between the means — pick the closest to midpoint
        mid = 0.5 * (m1 + m2)
        return (r1 if torch.abs(r1 - mid) < torch.abs(r2 - mid) else r2).clamp(0.0, 1.0)


def _threshold_platt(
    prior_scores: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    max_iter: int = 200,
    lr: float = 1e-2,
    eps: float = 1e-9,
) -> torch.Tensor:
    x = prior_scores.view(-1, 1)
    y = torch.zeros(prior_scores.numel(), device=prior_scores.device, dtype=prior_scores.dtype)
    y[pos_mask] = 1.0

    w = torch.zeros(1, device=prior_scores.device, dtype=prior_scores.dtype, requires_grad=True)
    b = torch.zeros(1, device=prior_scores.device, dtype=prior_scores.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(max_iter):
        optimizer.zero_grad()
        logits = x[:, 0] * w + b
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    a = float(w.detach()[0].item())
    b_val = float(b.detach()[0].item())

    if abs(a) < eps:
        return _threshold_roc(prior_scores, pos_mask, neg_mask)

    return torch.tensor(-b_val / (a + eps), device=prior_scores.device, dtype=prior_scores.dtype).clamp(0.0, 1.0)


def _select_threshold(
    strategy: ThresholdStrategy,
    all_scores: torch.Tensor,
    prior_scores: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    em_max_iter: int,
    platt_max_iter: int,
    eps: float,
) -> torch.Tensor:
    if strategy == "roc":
        return _threshold_roc(prior_scores, pos_mask, neg_mask)
    if strategy == "median_midpoint":
        return _threshold_median_midpoint(prior_scores, pos_mask, neg_mask)
    if strategy == "gmm":
        return _threshold_gmm(all_scores, prior_scores, pos_mask, neg_mask, max_iter=em_max_iter, eps=eps)
    if strategy == "platt":
        return _threshold_platt(prior_scores, pos_mask, neg_mask, max_iter=platt_max_iter, eps=eps)
    raise ValueError(f"Unknown threshold_strategy '{strategy}'. Use: roc, median_midpoint, gmm, platt.")


# ── main entry point ─────────────────────────────────────────────

def panc_segment(
    query_features: torch.Tensor,
    prior_features: torch.Tensor,
    prior_labels: Any,
    tau: float = 1.0,
    kappa: float = 1.0,
    threshold_strategy: ThresholdStrategy = "roc",
    eig_eps: float = 1e-8,
    em_max_iter: int = 50,
    platt_max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """PANC spectral segmentation.

    Args:
        query_features:  [Nq, D] query patch tokens.
        prior_features:  [Np, D] labeled prior tokens.
        prior_labels:    Labels as 1-D tensor, (pos, neg) tuple, or dict.
        tau:             Affinity temperature.
        kappa:           Anchor coupling strength.
        threshold_strategy: 'roc', 'median_midpoint', 'gmm', or 'platt'.
        eig_eps:         Numerical epsilon.
        em_max_iter:     EM iterations for GMM thresholding.
        platt_max_iter:  Optimizer iterations for Platt scaling.

    Returns:
        query_mask, query_scores, query_scores_eig3,
        prior_scores, anchor_scores, threshold
    """
    device = query_features.device
    dtype = query_features.dtype
    prior_features = prior_features.to(device=device, dtype=dtype)
    Nq = query_features.size(0)
    Np = prior_features.size(0)

    pos_mask, neg_mask = _parse_prior_labels(prior_labels, Np, device)

    features = torch.cat([query_features, prior_features], dim=0)
    aff = _build_affinity(features, tau)
    aug = _augment_with_anchors(aff, Nq, pos_mask, neg_mask, kappa, eig_eps)

    fiedler_vec, eig3_vec = _fiedler_scores(aug, eig_eps)

    # Sign stabilisation: positive priors should score higher
    fg_med = fiedler_vec[Nq:Nq + Np][pos_mask].median()
    bg_med = fiedler_vec[Nq:Nq + Np][neg_mask].median()
    if fg_med < bg_med:
        fiedler_vec = -fiedler_vec
        eig3_vec = -eig3_vec

    # Min-max normalise — exclude anchor nodes so they don't
    # compress the dynamic range of query / prior scores.
    def _norm(v: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        lo, hi = ref.min(), ref.max()
        return (v - lo) / (hi - lo + eig_eps)

    token_scores = fiedler_vec[:Nq + Np]
    token_eig3 = eig3_vec[:Nq + Np]

    all_scores = _norm(fiedler_vec, token_scores)
    all_eig3 = _norm(eig3_vec, token_eig3)

    anchor_scores = all_scores[-2:].clamp(0, 1)
    q_scores = all_scores[:Nq].clamp(0, 1)
    q_eig3 = all_eig3[:Nq].clamp(0, 1)
    p_scores = all_scores[Nq:Nq + Np].clamp(0, 1)

    threshold = _select_threshold(
        strategy=threshold_strategy,
        all_scores=all_scores[:Nq + Np],
        prior_scores=p_scores,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        em_max_iter=em_max_iter,
        platt_max_iter=platt_max_iter,
        eps=eig_eps,
    ).clamp(0, 1)

    q_mask = (q_scores > threshold).to(torch.int64)
    return q_mask, q_scores, q_eig3, p_scores, anchor_scores, threshold


__all__ = ["panc_segment", "ThresholdStrategy"]
