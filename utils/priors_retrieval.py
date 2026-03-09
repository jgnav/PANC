"""Improved prior-bank construction with relevance scoring and MMR selection.

Pipeline:
1. Extract patch tokens + binary labels from each prior image.
2. Score every prior patch by k-NN similarity to the query target patches.
3. Pre-filter the top-M most relevant patches (separately per label).
4. Apply Maximal Marginal Relevance (MMR) to select diverse, high-quality patches.
5. Return a balanced (equal pos / neg) prior bank.
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from .coco_dataset import CocoSample


def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


# ──────────────────────────────────────────────────────────────────
# 1. Patch extraction from prior images
# ──────────────────────────────────────────────────────────────────

def extract_prior_patches(
    prior_samples: Sequence[CocoSample],
    extract_fn: Callable[[Image.Image], torch.Tensor],
    grid_side: int,
    mask_pos_threshold: float = 0.6,
    mask_neg_threshold: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract all patch tokens and per-patch labels from prior images.

    Returns:
        all_tokens: [P, D] L2-normalised patch features (CPU).
        all_labels: [P] binary labels (1 = positive, 0 = negative).
    """
    token_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    for sample in prior_samples:
        image_pil = Image.open(sample.image_path).convert("RGB")
        tokens = extract_fn(image_pil).detach().cpu()  # [N, D]

        # Resize mask to patch grid via area interpolation
        mask_t = torch.from_numpy(sample.mask).float().unsqueeze(0).unsqueeze(0)
        patch_mask = F.interpolate(mask_t, size=(grid_side, grid_side), mode="area")
        patch_mask = patch_mask.squeeze().clamp(0, 1).reshape(-1)

        pos_idx = torch.where(patch_mask >= mask_pos_threshold)[0]
        neg_idx = torch.where(patch_mask <= mask_neg_threshold)[0]

        if pos_idx.numel() == 0 and neg_idx.numel() == 0:
            continue

        sel = torch.cat([pos_idx, neg_idx])
        labels = torch.cat([
            torch.ones(pos_idx.numel(), dtype=torch.long),
            torch.zeros(neg_idx.numel(), dtype=torch.long),
        ])
        token_chunks.append(tokens[sel])
        label_chunks.append(labels)

    if len(token_chunks) == 0:
        raise RuntimeError("No valid prior patches could be extracted.")

    all_tokens = _l2_normalize(torch.cat(token_chunks, dim=0))
    all_labels = torch.cat(label_chunks, dim=0)
    return all_tokens, all_labels


# ──────────────────────────────────────────────────────────────────
# 2. Relevance scoring
# ──────────────────────────────────────────────────────────────────

def compute_relevance_scores(
    bank_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    k_sim: int = 8,
) -> torch.Tensor:
    """Score each bank patch by its average cosine similarity to the k most
    similar query patches (higher = more relevant to the target image).

    Args:
        bank_tokens:  [P, D] L2-normalised prior patches.
        query_tokens: [Nq, D] L2-normalised query patches.
        k_sim: Number of nearest query patches to average over.

    Returns:
        relevance: [P] per-patch relevance scores.
    """
    bank = _l2_normalize(bank_tokens)
    query = _l2_normalize(query_tokens)
    sim = bank @ query.T  # [P, Nq]
    k = min(k_sim, sim.shape[1])
    topk_vals = torch.topk(sim, k=k, dim=1, largest=True, sorted=False).values
    return topk_vals.mean(dim=1)


# ──────────────────────────────────────────────────────────────────
# 3. Maximal Marginal Relevance (MMR) selection
# ──────────────────────────────────────────────────────────────────

def select_with_mmr(
    candidate_tokens: torch.Tensor,
    candidate_scores: torch.Tensor,
    n_select: int,
    mmr_lambda: float = 0.35,
) -> torch.Tensor:
    """Greedy MMR selection: balances relevance with diversity.

    MMR(i) = score(i)  −  λ · max_{j ∈ selected} sim(i, j)

    Args:
        candidate_tokens: [C, D] L2-normalised features of candidates.
        candidate_scores: [C] relevance scores.
        n_select: How many to select.
        mmr_lambda: Trade-off (0 = pure relevance, 1 = pure diversity).

    Returns:
        Indices into the candidate set (length ≤ n_select).
    """
    n = candidate_tokens.shape[0]
    if n == 0 or n_select <= 0:
        return torch.empty((0,), dtype=torch.long)

    n_select = min(n_select, n)
    device = candidate_tokens.device

    selected: List[int] = []
    remaining = torch.arange(n, device=device)

    # Pick the most relevant patch first
    first = torch.argmax(candidate_scores)
    selected.append(int(first.item()))
    remaining = remaining[remaining != first]

    if n_select == 1:
        return torch.tensor(selected, dtype=torch.long, device=device)

    # Pre-compute pairwise similarity for fast greedy updates
    sims = candidate_tokens @ candidate_tokens.T

    while len(selected) < n_select and remaining.numel() > 0:
        sel_idx = torch.tensor(selected, device=device, dtype=torch.long)
        max_sim_to_sel = sims[remaining][:, sel_idx].max(dim=1).values
        mmr_scores = candidate_scores[remaining] - mmr_lambda * max_sim_to_sel
        best_pos = torch.argmax(mmr_scores)
        best_idx = remaining[best_pos]
        selected.append(int(best_idx.item()))
        remaining = remaining[remaining != best_idx]

    return torch.tensor(selected, dtype=torch.long, device=device)


# ──────────────────────────────────────────────────────────────────
# 4. Main entry point
# ──────────────────────────────────────────────────────────────────

def build_prior_bank(
    prior_samples: Sequence[CocoSample],
    extract_fn: Callable[[Image.Image], torch.Tensor],
    query_tokens: torch.Tensor,
    grid_side: int = 28,
    *,
    mask_pos_threshold: float = 0.6,
    mask_neg_threshold: float = 0.05,
    k_sim: int = 8,
    prefilter_top_m: int = 4096,
    mmr_lambda: float = 0.35,
    final_total: int = 1500,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a balanced, relevance-scored, diversity-selected prior bank.

    Args:
        prior_samples: COCO prior images for one target class.
        extract_fn: Maps a PIL image to [N_patches, D] patch tokens.
        query_tokens: [Nq, D] query target patch tokens.
        grid_side: Patch-grid side length (e.g. 28 for 448 / 16).
        mask_pos_threshold: Area-interpolated mask value to count as positive.
        mask_neg_threshold: Area-interpolated mask value to count as negative.
        k_sim: k for k-NN relevance scoring.
        prefilter_top_m: Keep top-M relevant patches before MMR.
        mmr_lambda: MMR diversity trade-off.
        final_total: Total patches in the final bank (split 50/50).

    Returns:
        prior_feats:  [N, D] selected patch tokens.
        prior_labels: [N] binary labels.
    """
    # 1. Extract all patches and labels
    all_tokens, all_labels = extract_prior_patches(
        prior_samples, extract_fn, grid_side,
        mask_pos_threshold=mask_pos_threshold,
        mask_neg_threshold=mask_neg_threshold,
    )
    device = query_tokens.device
    all_tokens = all_tokens.to(device)
    all_labels = all_labels.to(device)
    query_norm = _l2_normalize(query_tokens)

    # 2. Compute relevance to the query image
    relevance = compute_relevance_scores(all_tokens, query_norm, k_sim=k_sim)

    # 3. Pre-filter top-M per label
    pos_mask = all_labels == 1
    neg_mask = all_labels == 0
    pos_idx = torch.where(pos_mask)[0]
    neg_idx = torch.where(neg_mask)[0]

    def _prefilter(indices: torch.Tensor, m: int) -> torch.Tensor:
        if indices.numel() <= m:
            return indices
        topk = torch.topk(relevance[indices], k=m, largest=True, sorted=False).indices
        return indices[topk]

    half_m = max(prefilter_top_m // 2, 1)
    pos_idx = _prefilter(pos_idx, half_m)
    neg_idx = _prefilter(neg_idx, half_m)

    # 4. MMR selection — balanced
    final_each = min(final_total // 2, pos_idx.numel(), neg_idx.numel())

    pos_selected = select_with_mmr(
        candidate_tokens=all_tokens[pos_idx],
        candidate_scores=relevance[pos_idx],
        n_select=final_each,
        mmr_lambda=mmr_lambda,
    )
    neg_selected = select_with_mmr(
        candidate_tokens=all_tokens[neg_idx],
        candidate_scores=relevance[neg_idx],
        n_select=final_each,
        mmr_lambda=mmr_lambda,
    )

    selected = torch.cat([pos_idx[pos_selected], neg_idx[neg_selected]])
    return all_tokens[selected].cpu(), all_labels[selected].cpu()
