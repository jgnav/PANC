from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
from pycocotools.coco import COCO


ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass(frozen=True)
class CocoSample:
	image_id: int
	category_id: int
	category_name: str
	image_path: Path
	mask: np.ndarray
	image_info: Dict


def _download(url: str, destination: Path) -> Path:
	destination.parent.mkdir(parents=True, exist_ok=True)
	if not destination.exists():
		urlretrieve(url, destination)
	return destination


def load_coco(cache_dir: Path) -> COCO:
	annotations_dir = f"{cache_dir}/annotations"
	json_path = Path(annotations_dir) / "instances_val2017.json"
	if not json_path.exists():
		zip_path = _download(ANNOTATIONS_URL, Path(cache_dir) / "annotations_trainval2017.zip")
		with ZipFile(zip_path) as archive:
			archive.extract("annotations/instances_val2017.json", path=cache_dir)
	return COCO(str(json_path))


def download_coco_image(image_info: Dict, cache_dir: Path) -> Path:
	images_dir = Path(cache_dir) / "val2017"
	image_path = images_dir / image_info["file_name"]
	if not image_path.exists():
		_download(image_info["coco_url"], image_path)
	return image_path


def build_category_mask(coco: COCO, image_id: int, category_id: int) -> np.ndarray:
	image_info = coco.loadImgs([image_id])[0]
	height = int(image_info["height"])
	width = int(image_info["width"])
	mask = np.zeros((height, width), dtype=np.uint8)
	ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=[category_id], iscrowd=False)
	for annotation in coco.loadAnns(ann_ids):
		mask = np.maximum(mask, coco.annToMask(annotation).astype(np.uint8))
	return mask


def _category_area_ratios(coco: COCO, image_info: Dict, min_area_ratio: float) -> Dict[int, float]:
	image_area = float(image_info["width"] * image_info["height"])
	area_by_category: Dict[int, float] = {}
	ann_ids = coco.getAnnIds(imgIds=[image_info["id"]], iscrowd=False)
	for annotation in coco.loadAnns(ann_ids):
		area_ratio = float(annotation.get("area", 0.0)) / image_area
		if area_ratio < min_area_ratio:
			continue
		category_id = int(annotation["category_id"])
		area_by_category[category_id] = area_by_category.get(category_id, 0.0) + area_ratio
	return area_by_category


def select_demo_target(coco: COCO, cache_dir: Path, min_area_ratio: float = 0.03, seed: int = 0) -> Tuple[Dict, List[CocoSample]]:
	category_lookup = {category["id"]: category["name"] for category in coco.loadCats(coco.getCatIds())}
	all_image_ids = sorted(coco.getImgIds())
	# Shuffle deterministically with seed so users can pick different images
	rng = np.random.RandomState(seed)
	rng.shuffle(all_image_ids)
	for image_id in all_image_ids:
		image_info = coco.loadImgs([image_id])[0]
		area_by_category = _category_area_ratios(coco, image_info, min_area_ratio=min_area_ratio)
		if len(area_by_category) < 2:
			continue

		selected_categories = sorted(area_by_category.items(), key=lambda item: item[1], reverse=True)[:2]
		image_path = download_coco_image(image_info, cache_dir)
		samples = []
		for category_id, _ in selected_categories:
			mask = build_category_mask(coco, image_id=image_id, category_id=category_id)
			samples.append(
				CocoSample(
					image_id=image_id,
					category_id=category_id,
					category_name=category_lookup[category_id],
					image_path=image_path,
					mask=mask,
					image_info=image_info,
				)
			)
		return image_info, samples

	raise RuntimeError("Could not find a COCO validation image with at least two large object classes.")


def collect_prior_samples(coco: COCO, cache_dir: Path, category_id: int, category_name: str, exclude_image_id: int, limit: int, min_area_ratio: float) -> List[CocoSample]:
	prior_samples: List[CocoSample] = []
	for image_id in sorted(coco.getImgIds(catIds=[category_id])):
		if image_id == exclude_image_id:
			continue
		image_info = coco.loadImgs([image_id])[0]
		mask = build_category_mask(coco, image_id=image_id, category_id=category_id)
		area_ratio = float(mask.sum()) / float(mask.shape[0] * mask.shape[1])
		if area_ratio < min_area_ratio:
			continue
		image_path = download_coco_image(image_info, cache_dir)
		prior_samples.append(
			CocoSample(
				image_id=image_id,
				category_id=category_id,
				category_name=category_name,
				image_path=image_path,
				mask=mask,
				image_info=image_info,
			)
		)
		if len(prior_samples) >= limit:
			break

	if len(prior_samples) < limit:
		raise RuntimeError(f"Only found {len(prior_samples)} prior images for class '{category_name}' with min_area_ratio={min_area_ratio}.")

	return prior_samples