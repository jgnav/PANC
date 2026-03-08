from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from torchvision.transforms import v2


def download_file(url: str, destination: Path) -> None:
	destination.parent.mkdir(parents=True, exist_ok=True)
	if destination.exists():
		return
	response = requests.get(url, stream=True, timeout=120)
	response.raise_for_status()
	with destination.open("wb") as handle:
		for chunk in response.iter_content(chunk_size=1024 * 1024):
			if chunk:
				handle.write(chunk)


def ensure_dinov3_repo(repo_dir: Path) -> Path:
	if repo_dir.exists():
		return repo_dir
	repo_dir.parent.mkdir(parents=True, exist_ok=True)
	subprocess.run(
		["git", "clone", "--depth", "1", "https://github.com/facebookresearch/dinov3.git", str(repo_dir)],
		check=True,
	)
	return repo_dir


def ensure_annotations_json(cache_dir: Path, annotations_zip: Path, annotations_json: Path) -> Path:
	if annotations_json.exists():
		return annotations_json
	download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", annotations_zip)
	with zipfile.ZipFile(annotations_zip, "r") as archive:
		archive.extract("annotations/instances_val2017.json", cache_dir)
	return annotations_json


def coco_url(file_name: str) -> str:
	return f"http://images.cocodataset.org/val2017/{file_name}"


def ensure_sample_asset(coco: COCO, sample: dict, images_dir: Path, masks_dir: Path) -> dict:
	image_path = images_dir / f"{sample['name']}.jpg"
	mask_path = masks_dir / f"{sample['name']}.png"
	download_file(coco_url(sample["file_name"]), image_path)

	if not mask_path.exists():
		category_names = {cat["name"]: cat["id"] for cat in coco.loadCats(coco.getCatIds())}
		image_info = coco.loadImgs([sample["image_id"]])[0]
		mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
		for class_name in sample["classes"]:
			ann_ids = coco.getAnnIds(imgIds=[sample["image_id"]], catIds=[category_names[class_name]], iscrowd=False)
			for ann in coco.loadAnns(ann_ids):
				mask = np.maximum(mask, coco.annToMask(ann).astype(np.uint8))
		Image.fromarray(mask * 255).save(mask_path)

	return {**sample, "image_path": image_path, "mask_path": mask_path}


def overlay_mask(image: Image.Image, mask: Image.Image, alpha: float = 0.5) -> np.ndarray:
	image_arr = np.asarray(image.convert("RGB"), dtype=np.float32)
	mask_arr = (np.asarray(mask.convert("L"), dtype=np.float32) > 0).astype(np.float32)[..., None]
	red = np.array([255, 0, 0], dtype=np.float32)
	blended = image_arr * (1.0 - alpha * mask_arr) + red * (alpha * mask_arr)
	return blended.astype(np.uint8)


def make_transform(resize_size: int):
	return v2.Compose(
		[
			v2.ToImage(),
			v2.Resize((resize_size, resize_size), antialias=True),
			v2.ToDtype(torch.float32, scale=True),
			v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		]
	)


def resize_image_and_mask(image_path: Path, mask_path: Path, resize_size: int) -> tuple[Image.Image, Image.Image]:
	image = Image.open(image_path).convert("RGB").resize((resize_size, resize_size), Image.BICUBIC)
	mask = Image.open(mask_path).convert("L").resize((resize_size, resize_size), Image.NEAREST)
	return image, mask


def load_dinov3_backbone(weights_path: Path, repo_dir: Path, device: str) -> torch.nn.Module:
	if not weights_path.exists():
		raise FileNotFoundError(
			f"Missing local DINOv3 checkpoint: {weights_path}. Place the small pretrained checkpoint there before running this notebook."
		)
	if weights_path.stat().st_size < 1024 * 1024:
		raise RuntimeError(f"Checkpoint at {weights_path} is too small to be valid.")

	ensure_dinov3_repo(repo_dir)
	if str(repo_dir) not in sys.path:
		sys.path.insert(0, str(repo_dir))

	from dinov3.hub.backbones import dinov3_vits16

	model = dinov3_vits16(pretrained=False)
	state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
	model.load_state_dict(state_dict, strict=True)
	model.eval().to(device)
	return model


def extract_patch_tokens(
	model: torch.nn.Module,
	image: Image.Image,
	resize_size: int,
	device: str,
) -> tuple[torch.Tensor, int, int, Image.Image]:
	transform = make_transform(resize_size)
	resized = image.convert("RGB").resize((resize_size, resize_size), Image.BICUBIC)
	batch = transform(resized).unsqueeze(0).to(device)
	with torch.inference_mode():
		patch_map, _ = model.get_intermediate_layers(batch, n=1, reshape=True, return_class_token=True)[0]
	grid_h, grid_w = patch_map.shape[-2:]
	tokens = patch_map.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1).float().cpu()
	return tokens, grid_h, grid_w, resized


def mask_to_patch_labels(mask: Image.Image, grid_h: int, grid_w: int, resize_size: int, threshold: float = 0.25) -> torch.Tensor:
	resized_mask = mask.convert("L").resize((resize_size, resize_size), Image.NEAREST)
	mask_tensor = torch.from_numpy((np.asarray(resized_mask, dtype=np.float32) / 255.0)).unsqueeze(0).unsqueeze(0)
	pooled = F.interpolate(mask_tensor, size=(grid_h, grid_w), mode="area").squeeze(0).squeeze(0)
	return (pooled >= threshold).long().reshape(-1)


def sample_class_indices(labels: torch.Tensor, value: int, max_count: int, generator: torch.Generator) -> torch.Tensor:
	indices = torch.where(labels == value)[0]
	if indices.numel() <= max_count:
		return indices
	order = torch.randperm(indices.numel(), generator=generator)[:max_count]
	return indices[order]


def build_prior_bank(
	model: torch.nn.Module,
	samples: list[dict],
	resize_size: int,
	device: str,
	max_patches_per_class_per_image: int = 128,
	seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
	feature_chunks = []
	label_chunks = []
	generator = torch.Generator().manual_seed(seed)

	for sample in samples:
		image, mask = resize_image_and_mask(sample["image_path"], sample["mask_path"], resize_size)
		tokens, grid_h, grid_w, _ = extract_patch_tokens(model, image, resize_size, device)
		patch_labels = mask_to_patch_labels(mask, grid_h, grid_w, resize_size)

		pos_idx = sample_class_indices(patch_labels, 1, max_patches_per_class_per_image, generator)
		neg_idx = sample_class_indices(patch_labels, 0, max_patches_per_class_per_image, generator)

		sampled_idx = torch.cat([pos_idx, neg_idx])
		sampled_labels = torch.cat(
			[
				torch.ones(pos_idx.numel(), dtype=torch.long),
				torch.zeros(neg_idx.numel(), dtype=torch.long),
			]
		)

		feature_chunks.append(tokens[sampled_idx])
		label_chunks.append(sampled_labels)

	return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def upscale_score_map(scores_1d: torch.Tensor, grid_h: int, grid_w: int, target_h: int, target_w: int) -> np.ndarray:
	grid = scores_1d.float().reshape(1, 1, grid_h, grid_w)
	upscaled = F.interpolate(grid, size=(target_h, target_w), mode="bilinear", align_corners=False)
	return upscaled.squeeze(0).squeeze(0).cpu().numpy()


def edge_aware_refine_mask(scores_1d: torch.Tensor, grid_h: int, grid_w: int, image: Image.Image, threshold: float = 0.5) -> np.ndarray:
	image_rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
	height, width = image_rgb.shape[:2]

	score_map = upscale_score_map(scores_1d, grid_h, grid_w, height, width)
	score_tensor = torch.from_numpy(np.clip(score_map, 0.0, 1.0)).float().unsqueeze(0).unsqueeze(0)

	gray = image_rgb.mean(axis=2)
	grad_y, grad_x = np.gradient(gray)
	edge_map = np.sqrt(grad_x * grad_x + grad_y * grad_y)
	edge_map = edge_map / (edge_map.max() + 1e-6)
	edge_tensor = torch.from_numpy(edge_map).float().unsqueeze(0).unsqueeze(0)

	for _ in range(3):
		blurred = F.avg_pool2d(F.pad(score_tensor, (1, 1, 1, 1), mode="replicate"), kernel_size=3, stride=1)
		score_tensor = blurred * (1.0 - 0.65 * edge_tensor) + score_tensor * (0.65 * edge_tensor)

	refined = score_tensor.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
	return (refined >= threshold).astype(np.uint8)