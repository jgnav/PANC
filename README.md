# PANC

Minimal repository for the PANC paper demo.

Run [demo.ipynb](demo.ipynb) to reproduce a short didactic example on COCO. The notebook:

1. clones DINOv3 into a cache directory,
2. downloads one query image and two prior sets from COCO,
3. shows the query and priors with mask overlays,
4. extracts DINOv3 patch tokens,
5. shows the unsupervised NCut baseline using the mean Fiedler threshold, and
6. compares it against PANC with priors for both query classes.

Place the small DINOv3 checkpoint at `weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` before running the notebook.
