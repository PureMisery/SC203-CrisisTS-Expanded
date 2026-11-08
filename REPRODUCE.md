# CM4IR reproduction

Upstream: [tirer-lab/CM4IR](https://github.com/tirer-lab/CM4IR), commit `e5fe0a02ea63184a26ac34a1c504818d26b3c592`. Paper: CVPR 2025, [arXiv:2412.20596](https://arxiv.org/abs/2412.20596).

## Verified scope

The [Colab notebook](colab_baseline.ipynb) ran LSUN Cat 256 ×4 bicubic super-resolution on a Tesla T4. It used the authors' `cd_cat256_lpips.pt` checkpoint and 57 images from their validation folder. The observed averages were PSNR **26.24 dB** and LPIPS **0.3492**. See [RUN_REPORT.md](RUN_REPORT.md) for the command and limits.

This is a partial baseline. The authors' Drive folder has more than 50 images and `gdown --folder --remaining-ok` downloads only the first batch. The paper's other datasets, degradations, and checkpoints were not run. Do not compare these subset metrics with the paper's full-split metrics.

## Run on Colab

Open [colab_baseline.ipynb](colab_baseline.ipynb) in Google Colab and select a GPU runtime. Run cells from top to bottom. The notebook clones the upstream source, installs dependencies, downloads the author checkpoint and an available subset of LSUN Cat images, replaces obsolete `flash-attn` with PyTorch scaled dot product attention, and executes the baseline on the downloaded images. Outputs appear under `/content/CM4IR/exp/image_samples/` and are lost when the Colab runtime expires unless copied elsewhere.

Colab notebook used for the verified run: https://colab.research.google.com/drive/1AnxTVDhjrXW-oOr36B8UfAw-_KBKZvY4

For the complete paper, obtain all author validation data, checkpoints, and masks linked in the upstream README and run the commands in `evaluate_CM4IR.sh`. The original environment specifies Python 3.8, PyTorch 1.9.0 and `flash-attn==0.2.8`. The Colab compatibility change may yield numerically different results and requires a fixed full split before paper-level comparison.

Checkpoints, validation images, and generated images are excluded from Git history.
