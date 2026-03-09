# PANC: Prior-Aware Normalized Cut

## [[arXiv]](https://arxiv.org/abs/2602.06912)

PANC is a spectral segmentation method that improves upon standard Normalized Cut (NCut) by injecting labeled prior information. It augments the token affinity graph with positive and negative anchor nodes connected to labeled prior patches, guiding eigenvector-based segmentation toward a specific target class.

## Overview

Given a query image and a set of prior images with known class labels, PANC:

1. Extracts patch-level features using a DINOv3 (ViT) backbone
2. Builds a cosine-similarity affinity graph over patch tokens
3. Augments the graph with anchor nodes linked to prior patches (controlled by `kappa`)
4. Solves for the Fiedler eigenvector and thresholds to produce a class-specific segmentation mask
5. Optionally refines the mask with CRF post-processing

## Repository Structure

```
graph_cut/
  panc.py                 # Core PANC algorithm
  ncut_unsupervised.py    # Baseline unsupervised NCut
utils/
  priors_retrieval.py     # Prior bank construction (k-NN + MMR selection)
  crf.py                  # Mean-field CRF post-processing
  coco_dataset.py         # COCO val2017 data loading
  visualization.py        # Plotting helpers
dinov3/                   # Vendored DINOv3 backbone
weights/                  # Pretrained model weights
demo.ipynb                # End-to-end demo notebook
```

## Getting Started

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r dinov3/requirements.txt
```

Download the DINOv3 pretrained weights from the [DINOv3 repository](https://github.com/facebookresearch/dinov3) and place the checkpoint in `weights/` (the demo defaults to `weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`).

## Demo

Run [demo.ipynb](demo.ipynb) for an end-to-end example on COCO val2017 — from feature extraction through unsupervised NCut, PANC segmentation, CRF refinement, and visualization.

## Citation

```bibtex
@misc{gutiérrez2026pancpriorawarenormalizedcut,
      title={PANC: Prior-Aware Normalized Cut for Object Segmentation}, 
      author={Juan Gutiérrez and Victor Gutiérrez-Garcia and José Luis Blanco-Murillo},
      year={2026},
      eprint={2602.06912},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.06912}, 
}
```
