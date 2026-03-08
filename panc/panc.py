from __future__ import annotations

from typing import Any, Dict, Iterable, Literal, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn


ThresholdStrategy = Literal["roc", "median_midpoint", "gmm", "platt"]


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
		torch.int8,
		torch.int16,
		torch.int32,
		torch.int64,
		torch.uint8,
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
	pos_mask: torch.Tensor
	neg_mask: torch.Tensor

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


def _build_affinity(features: torch.Tensor, tau: float) -> torch.Tensor:
	normalized = F.normalize(features, p=2, dim=1)
	similarity = normalized @ normalized.T
	affinity = torch.exp(similarity / tau)
	affinity.fill_diagonal_(0.0)
	return affinity


def _augment_with_anchors(
	affinity: torch.Tensor,
	num_query: int,
	pos_mask_prior: torch.Tensor,
	neg_mask_prior: torch.Tensor,
	kappa: float,
	eps: float,
) -> torch.Tensor:
	num_total = affinity.size(0)
	num_prior = pos_mask_prior.numel()
	
	# Per-seed adaptive anchor weighting (legacy approach)
	connection = torch.zeros((num_total, 2), device=affinity.device, dtype=affinity.dtype)
	
	if num_prior > 0:
		prior_offset = num_query
		seed_indices = torch.arange(num_prior, device=affinity.device) + prior_offset
		
		# Compute per-seed mean of affinities to query tokens
		if num_query > 0:
			seed_local_mean = affinity[seed_indices, :num_query].mean(dim=1).clamp_min(eps)
		else:
			seed_local_mean = affinity[seed_indices].mean(dim=1).clamp_min(eps)
		
		# Per-seed anchor weight (scale with kappa, clamp extremes)
		seed_anchors = (kappa * seed_local_mean).clamp(min=1e-4, max=1e3)
		
		# Assign each prior to its corresponding anchor
		for p in range(num_prior):
			global_idx = prior_offset + p
			if pos_mask_prior[p]:
				connection[global_idx, 0] = seed_anchors[p]
			else:
				connection[global_idx, 1] = seed_anchors[p]
	
	# Tiny self-loop to avoid zero-degree anchors
	anchor_block = torch.zeros((2, 2), device=affinity.device, dtype=affinity.dtype)
	anchor_block[0, 0] = eps
	anchor_block[1, 1] = eps
	
	top = torch.cat([affinity, connection], dim=1)
	bottom = torch.cat([connection.T, anchor_block], dim=1)
	augmented = torch.cat([top, bottom], dim=0)
	
	# Symmetrize
	return 0.5 * (augmented + augmented.T)


def _fiedler_scores(aug_affinity: torch.Tensor, eig_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
	"""Return (fiedler_vector, third_eigvec) as generalized eigenvectors."""
	degree = aug_affinity.sum(dim=1).clamp_min(eig_eps)
	inv_sqrt_degree = torch.rsqrt(degree)
	normalized_affinity = inv_sqrt_degree[:, None] * aug_affinity * inv_sqrt_degree[None, :]
	laplacian_sym = torch.eye(
		aug_affinity.size(0), device=aug_affinity.device, dtype=aug_affinity.dtype
	) - normalized_affinity

	eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_sym)
	nonzero_idx = torch.nonzero(eigenvalues > eig_eps, as_tuple=False).view(-1)
	if nonzero_idx.numel() == 0:
		vector_idx = min(1, eigenvalues.numel() - 1)
	else:
		vector_idx = int(nonzero_idx[0].item())

	# Second smallest eigenvector (Fiedler)
	fiedler = inv_sqrt_degree * eigenvectors[:, vector_idx]

	# Third smallest eigenvector
	third_idx = vector_idx + 1
	if third_idx < eigenvalues.numel():
		third_vec = inv_sqrt_degree * eigenvectors[:, third_idx]
	else:
		third_vec = torch.zeros_like(fiedler)

	return fiedler, third_vec


def _ncut_query_only(
	query_features: torch.Tensor,
	tau: float,
	eig_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""Pure Normalized Cut on query tokens only (no priors/anchors).

	Returns (query_mask, query_scores, query_scores_eig3, threshold).
	The threshold is the mean of the 2nd eigenvector scores (lambda-2 mean).
	"""
	query_affinity = _build_affinity(query_features, tau=tau)
	degree = query_affinity.sum(dim=1).clamp_min(eig_eps)
	inv_sqrt_degree = torch.rsqrt(degree)
	normalized_affinity = inv_sqrt_degree[:, None] * query_affinity * inv_sqrt_degree[None, :]
	laplacian_sym = torch.eye(
		query_affinity.size(0), device=query_affinity.device, dtype=query_affinity.dtype
	) - normalized_affinity

	eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_sym)
	# Smallest non-trivial eigenvalue index (skip constant eigenvector)
	nonzero_idx = torch.nonzero(eigenvalues > eig_eps, as_tuple=False).view(-1)
	if nonzero_idx.numel() == 0:
		vector_idx = min(1, eigenvalues.numel() - 1)
	else:
		vector_idx = int(nonzero_idx[0].item())

	fiedler_vec = inv_sqrt_degree * eigenvectors[:, vector_idx]

	# Third eigenvector
	third_idx = vector_idx + 1
	if third_idx < eigenvalues.numel():
		third_vec = inv_sqrt_degree * eigenvectors[:, third_idx]
	else:
		third_vec = torch.zeros_like(fiedler_vec)

	# Normalize 2nd eigenvector
	min_val = fiedler_vec.min()
	max_val = fiedler_vec.max()
	query_scores = (fiedler_vec - min_val) / (max_val - min_val + eig_eps)

	# Normalize 3rd eigenvector
	v3_min = third_vec.min()
	v3_max = third_vec.max()
	query_scores_eig3 = (third_vec - v3_min) / (v3_max - v3_min + eig_eps)

	# Threshold = mean of Fiedler scores (lambda-2 mean)
	threshold = query_scores.mean()
	query_mask = (query_scores > threshold).to(torch.int64)
	return query_mask, query_scores, query_scores_eig3, threshold


def _stabilize_and_normalize(
	raw_scores_wo_anchors: torch.Tensor,
	num_query: int,
	pos_mask_prior: torch.Tensor,
	neg_mask_prior: torch.Tensor,
	eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	prior_scores = raw_scores_wo_anchors[num_query:]
	pos_mean = prior_scores[pos_mask_prior].mean()
	neg_mean = prior_scores[neg_mask_prior].mean()

	sign = torch.tensor(1.0, device=raw_scores_wo_anchors.device, dtype=raw_scores_wo_anchors.dtype)
	stabilized = raw_scores_wo_anchors
	if pos_mean < neg_mean:
		sign = -sign
		stabilized = sign * stabilized

	min_val = stabilized.min()
	max_val = stabilized.max()
	normalized = (stabilized - min_val) / (max_val - min_val + eps)
	return normalized, sign, min_val, max_val


def _threshold_roc(prior_scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor, n_thresh: int = 200) -> torch.Tensor:
	"""ROC-based thresholding using Youden's J statistic.
	
	Uses fixed number of uniformly spaced thresholds (legacy behavior).
	"""
	candidates = torch.linspace(0.0, 1.0, n_thresh, device=prior_scores.device, dtype=prior_scores.dtype)
	if candidates.numel() == 0:
		return torch.tensor(0.5, device=prior_scores.device, dtype=prior_scores.dtype)
	pos_scores = prior_scores[pos_mask]
	neg_scores = prior_scores[neg_mask]
	tpr = (pos_scores[:, None] > candidates[None, :]).float().mean(dim=0)
	fpr = (neg_scores[:, None] > candidates[None, :]).float().mean(dim=0)
	j_stat = tpr - fpr
	return candidates[torch.argmax(j_stat)]


def _threshold_median_midpoint(prior_scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
	pos_median = prior_scores[pos_mask].median()
	neg_median = prior_scores[neg_mask].median()
	return 0.5 * (pos_median + neg_median)


def _fit_gmm_1d_two_components(
	data: torch.Tensor,
	init_means: torch.Tensor,
	max_iter: int,
	eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	means = init_means.clone()
	stds = torch.full((2,), data.std().clamp_min(0.1), device=data.device, dtype=data.dtype)
	weights = torch.full((2,), 0.5, device=data.device, dtype=data.dtype)

	x = data.view(-1, 1)
	for _ in range(max_iter):
		var = (stds**2).clamp_min(eps)
		log_prob = (
			torch.log(weights.clamp_min(eps))[None, :]
			- 0.5 * torch.log(2 * torch.pi * var)[None, :]
			- 0.5 * ((x - means[None, :]) ** 2) / var[None, :]
		)
		log_norm = torch.logsumexp(log_prob, dim=1, keepdim=True)
		resp = torch.exp(log_prob - log_norm)

		nk = resp.sum(dim=0).clamp_min(eps)
		weights = nk / nk.sum()
		means = (resp * x).sum(dim=0) / nk
		var_new = (resp * (x - means[None, :]) ** 2).sum(dim=0) / nk
		stds = torch.sqrt(var_new.clamp_min(eps))

	return weights, means, stds


def _gmm_intersection_threshold(
	weights: torch.Tensor,
	means: torch.Tensor,
	stds: torch.Tensor,
	eps: float,
) -> torch.Tensor:
	w1, w2 = weights[0], weights[1]
	m1, m2 = means[0], means[1]
	s1, s2 = stds[0].clamp_min(eps), stds[1].clamp_min(eps)

	a = (1.0 / (2.0 * s2 * s2)) - (1.0 / (2.0 * s1 * s1))
	b = (m1 / (s1 * s1)) - (m2 / (s2 * s2))
	c = (
		(m2 * m2) / (2.0 * s2 * s2)
		- (m1 * m1) / (2.0 * s1 * s1)
		+ torch.log((w2 / s2).clamp_min(eps))
		- torch.log((w1 / s1).clamp_min(eps))
	)

	if torch.abs(a) < 1e-10:
		threshold = -c / (b + eps)
		return threshold.clamp(0.0, 1.0)

	disc = b * b - 4.0 * a * c
	if disc < 0:
		return (0.5 * (m1 + m2)).clamp(0.0, 1.0)

	sqrt_disc = torch.sqrt(disc)
	root1 = (-b + sqrt_disc) / (2.0 * a)
	root2 = (-b - sqrt_disc) / (2.0 * a)

	lo = torch.minimum(m1, m2)
	hi = torch.maximum(m1, m2)
	in_range_1 = (root1 >= lo) & (root1 <= hi)
	in_range_2 = (root2 >= lo) & (root2 <= hi)

	if in_range_1 and in_range_2:
		threshold = 0.5 * (root1 + root2)
	elif in_range_1:
		threshold = root1
	elif in_range_2:
		threshold = root2
	else:
		midpoint = 0.5 * (m1 + m2)
		threshold = root1 if torch.abs(root1 - midpoint) < torch.abs(root2 - midpoint) else root2

	return threshold.clamp(0.0, 1.0)


def _threshold_gmm(
	all_scores: torch.Tensor,
	prior_scores: torch.Tensor,
	pos_mask: torch.Tensor,
	neg_mask: torch.Tensor,
	max_iter: int,
	eps: float,
	use_priors_init: bool = True,
) -> torch.Tensor:
	"""GMM-based thresholding (legacy behavior).
	
	Uses sklearn's GaussianMixture for consistency with legacy implementation.
	"""
	try:
		from sklearn.mixture import GaussianMixture
		import numpy as np
		
		# Convert to numpy
		s = all_scores.cpu().numpy().reshape(-1, 1)
		
		# Initialize GMM
		gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
		
		# Initialize means from priors if requested
		if use_priors_init and prior_scores.numel() > 0:
			fg_med = prior_scores[pos_mask].median().item() if pos_mask.any() else s.mean()
			bg_med = prior_scores[neg_mask].median().item() if neg_mask.any() else s.mean()
			gmm.means_init = np.array([[bg_med], [fg_med]])
		
		gmm.fit(s)
		means = gmm.means_.ravel()
		
		# Assign foreground to component with higher mean
		fg_comp = int(np.argmax(means))
		
		# Find threshold where posterior_fg = 0.5
		grid = np.linspace(0, 1, 1000).reshape(-1, 1)
		post = gmm.predict_proba(grid)[:, fg_comp]
		t_idx = np.argmin(np.abs(post - 0.5))
		thresh = float(grid[t_idx])
		
		return torch.tensor(thresh, device=all_scores.device, dtype=all_scores.dtype)
	except ImportError:
		# Fallback to torch implementation if sklearn not available
		init_means = torch.stack([prior_scores[pos_mask].mean(), prior_scores[neg_mask].mean()])
		weights, means, stds = _fit_gmm_1d_two_components(all_scores, init_means, max_iter=max_iter, eps=eps)
		return _gmm_intersection_threshold(weights, means, stds, eps=eps)


def _threshold_platt(
	prior_scores: torch.Tensor,
	pos_mask: torch.Tensor,
	neg_mask: torch.Tensor,
	max_iter: int = 200,
	lr: float = 1e-2,
	weight_decay: float = 1e-4,
	delta: float = 0.05,
	eps: float = 1e-9,
) -> torch.Tensor:
	"""Platt scaling for threshold selection (legacy behavior).
	
	Fits logistic regression on prior scores to find decision boundary.
	Fallback to Youden's J if slope is near-zero.
	"""
	x = prior_scores.view(-1, 1)
	y = torch.zeros(prior_scores.numel(), device=prior_scores.device, dtype=prior_scores.dtype)
	y[pos_mask] = 1.0
	y[neg_mask] = 0.0

	w = torch.zeros(1, device=prior_scores.device, dtype=prior_scores.dtype, requires_grad=True)
	b = torch.zeros(1, device=prior_scores.device, dtype=prior_scores.dtype, requires_grad=True)
	optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
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
		# Fallback: Youden's J statistic
		ths = torch.linspace(0.0, 1.0, steps=201, device=prior_scores.device, dtype=prior_scores.dtype)
		preds = (prior_scores.unsqueeze(0) > ths.unsqueeze(1))  # (T, P)
		labs_pos = (pos_mask.unsqueeze(0)).expand_as(preds)
		labs_neg = (neg_mask.unsqueeze(0)).expand_as(preds)
		
		tp = (preds & labs_pos).sum(dim=1).float()
		fn = (~preds & labs_pos).sum(dim=1).float()
		fp = (preds & labs_neg).sum(dim=1).float()
		tn = (~preds & labs_neg).sum(dim=1).float()
		
		tpr = tp / (tp + fn + eps)
		fpr = fp / (fp + tn + eps)
		J = tpr - fpr
		t = float(ths[J.argmax()].item())
		return torch.tensor(t, device=prior_scores.device, dtype=prior_scores.dtype).clamp(0.0, 1.0)
	
	threshold = -b_val / (a + eps)
	return torch.tensor(threshold, device=prior_scores.device, dtype=prior_scores.dtype).clamp(0.0, 1.0)


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
	"""Select threshold based on strategy with edge case handling (legacy behavior)."""
	strategy = strategy.lower()
	
	# Edge case: both positive and negative priors exist
	if pos_mask.any() and neg_mask.any():
		if strategy == "roc":
			return _threshold_roc(prior_scores, pos_mask, neg_mask)
		if strategy == "median_midpoint":
			return _threshold_median_midpoint(prior_scores, pos_mask, neg_mask)
		if strategy == "gmm":
			return _threshold_gmm(all_scores, prior_scores, pos_mask, neg_mask, max_iter=em_max_iter, eps=eps)
		if strategy == "platt":
			return _threshold_platt(prior_scores, pos_mask, neg_mask, max_iter=platt_max_iter, eps=eps)
		
	# Edge case: only positive priors exist
	elif pos_mask.any():
		delta = 0.05
		return (prior_scores[pos_mask].median() - delta).clamp(0.0, 1.0)
	
	# Edge case: only negative priors exist
	elif neg_mask.any():
		delta = 0.05
		return (prior_scores[neg_mask].median() + delta).clamp(0.0, 1.0)
	
	# Edge case: no priors at all
	else:
		return all_scores.median().clamp(0.0, 1.0)
	
	raise ValueError(
		"Unsupported threshold_strategy. Use one of: 'roc', 'median_midpoint', 'gmm', 'platt'."
	)


def panc_segment(
	query_features: torch.Tensor,
	prior_features: torch.Tensor,
	prior_labels: Any,
	tau: float = 0.07,
	kappa: float = 1.0,
	threshold_strategy: ThresholdStrategy = "roc",
	eig_eps: float = 1e-8,
	em_max_iter: int = 50,
	platt_max_iter: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	PANC segmentation over token features with anchor-based spectral partitioning.

	Args:
		query_features: Tensor [Nq, D] for unannotated image tokens.
		prior_features: Tensor [Np, D] for labeled prior tokens.
		prior_labels: Positive/negative labels as either:
			- tuple/list: (positive_indices_or_mask, negative_indices_or_mask)
			- dict with positive/negative keys
			- 1D tensor/list of length Np where values > 0 are positive and <= 0 are negative
		tau: Affinity temperature.
		kappa: Anchor coupling multiplier.
		threshold_strategy: One of 'roc', 'median_midpoint', 'gmm', 'platt'.
		eig_eps: Small epsilon for numerical stability.
		em_max_iter: EM iterations for GMM thresholding.
		platt_max_iter: Optimizer iterations for Platt scaling.

	Returns:
		query_mask: Tensor [Nq] with values in {0, 1}.
		query_scores: Tensor [Nq] normalized scores in [0, 1] (2nd eigenvector).
		query_scores_eig3: Tensor [Nq] normalized scores in [0, 1] (3rd eigenvector).
		prior_scores: Tensor [Np] normalized scores in [0, 1] (empty if Np=0).
		anchor_scores: Tensor [2] normalized scores in [0, 1], order [positive_anchor, negative_anchor]
			or empty if Np=0.
		threshold: Scalar tensor threshold in [0, 1].
	"""
	if query_features.ndim != 2 or prior_features.ndim != 2:
		raise ValueError("query_features and prior_features must be 2D tensors [N, D].")
	if query_features.size(1) != prior_features.size(1):
		raise ValueError("Feature dimensions for query and prior must match.")
	if tau <= 0:
		raise ValueError("tau must be > 0.")

	device = query_features.device
	dtype = query_features.dtype
	prior_features = prior_features.to(device=device, dtype=dtype)

	num_query = query_features.size(0)
	num_prior = prior_features.size(0)
	if num_query == 0:
		raise ValueError("query_features must be non-empty.")

	if num_prior == 0:
		query_mask, query_scores, query_scores_eig3, threshold = _ncut_query_only(
			query_features=query_features,
			tau=tau,
			eig_eps=eig_eps,
		)
		empty_prior_scores = torch.empty(0, device=device, dtype=dtype)
		empty_anchor_scores = torch.empty(0, device=device, dtype=dtype)
		return query_mask, query_scores, query_scores_eig3, empty_prior_scores, empty_anchor_scores, threshold

	pos_mask_prior, neg_mask_prior = _parse_prior_labels(prior_labels, num_prior, device)

	features = torch.cat([query_features, prior_features], dim=0)
	affinity = _build_affinity(features, tau=tau)
	aug_affinity = _augment_with_anchors(
		affinity,
		num_query=num_query,
		pos_mask_prior=pos_mask_prior,
		neg_mask_prior=neg_mask_prior,
		kappa=kappa,
		eps=eig_eps,
	)

	generalized_eigvec, third_eigvec = _fiedler_scores(aug_affinity, eig_eps=eig_eps)
	
	# Sign stabilization based on prior medians (legacy approach)
	if num_prior > 0:
		prior_indices = torch.arange(num_prior, device=device) + num_query
		fg_idx = prior_indices[pos_mask_prior]
		bg_idx = prior_indices[neg_mask_prior]
		
		fg_med = generalized_eigvec[fg_idx].median() if fg_idx.numel() > 0 else None
		bg_med = generalized_eigvec[bg_idx].median() if bg_idx.numel() > 0 else None
		
		# Flip sign if foreground median < background median
		if fg_med is not None and bg_med is not None:
			if (fg_med - bg_med).abs() > 1e-6 and (fg_med - bg_med) < 0:
				generalized_eigvec = -generalized_eigvec
				third_eigvec = -third_eigvec
		elif fg_med is not None and fg_med < 0:
			generalized_eigvec = -generalized_eigvec
			third_eigvec = -third_eigvec
	
	# Normalize ALL nodes together (query + priors + anchors) using global min-max
	v_min = generalized_eigvec.min()
	v_max = generalized_eigvec.max()
	all_scores = (generalized_eigvec - v_min) / (v_max - v_min + eig_eps)
	
	# Normalize 3rd eigenvector the same way
	v3_min = third_eigvec.min()
	v3_max = third_eigvec.max()
	all_scores3 = (third_eigvec - v3_min) / (v3_max - v3_min + eig_eps)
	
	# Extract anchor scores (last 2 elements)
	anchor_scores = all_scores[-2:].clamp(0.0, 1.0)

	# Extract scores for query, priors (excluding anchors)
	query_scores = all_scores[:num_query].clamp(0.0, 1.0)
	query_scores_eig3 = all_scores3[:num_query].clamp(0.0, 1.0)
	prior_scores = all_scores[num_query:num_query + num_prior].clamp(0.0, 1.0)
	
	threshold = _select_threshold(
		strategy=threshold_strategy,
		all_scores=all_scores[:num_query + num_prior],  # Exclude anchors from threshold computation
		prior_scores=prior_scores,
		pos_mask=pos_mask_prior,
		neg_mask=neg_mask_prior,
		em_max_iter=em_max_iter,
		platt_max_iter=platt_max_iter,
		eps=eig_eps,
	)
	
	# Clamp threshold to [0, 1]
	threshold = threshold.clamp(0.0, 1.0)
	
	query_mask = (query_scores > threshold).to(torch.int64)
	return query_mask, query_scores, query_scores_eig3, prior_scores, anchor_scores, threshold


__all__ = ["panc_segment"]
