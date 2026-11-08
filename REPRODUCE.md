# CM4IR reproduction record

Upstream: https://github.com/tirer-lab/CM4IR (commit `e5fe0a02ea63184a26ac34a1c504818d26b3c592`). Paper: CVPR 2025, https://arxiv.org/abs/2412.20596.

## Scope

This repository preserves the authors' code and experiment commands. `evaluate_CM4IR.sh` contains the paper settings for LSUN Bedroom, LSUN Cat, and ImageNet64, across super-resolution, deblurring, and inpainting. Results have **not** been reproduced on this machine: no NVIDIA GPU or Docker was detected. Do not report paper metrics as independently verified.

## Setup on a CUDA machine

The authors tested Python 3.8 and PyTorch 1.9.0. Use a compatible Linux CUDA environment; the upstream Dockerfile uses `nvcr.io/nvidia/pytorch:21.04-py3`. Install dependencies with `pip install -r requirements.txt`. Note that `flash-attn==0.2.8` is an old CUDA extension and may require the original environment. Record `python --version`, `nvidia-smi`, and `pip freeze` with every run.

Download the official pretrained checkpoints linked in README.md into `exp/logs/lsun_bedroom/`, `exp/logs/lsun_cat/`, and `exp/logs/imagenet/`. Download the authors' validation datasets using the README links and place them in `exp/datasets/<dataset>/`. ImageNet also needs `exp/imagenet_val.txt`. For inpainting, use masks in `exp/inpainting_masks/`.

Run `bash evaluate_CM4IR.sh` for all published configurations, or run one command from that file to smoke-test first. Results are saved under `exp/image_samples/`. Keep checkpoints, datasets, and generated images outside Git history. Preserve the random seed (default 1234), exact checkpoint, dataset split, command line, GPU model, and package versions in each report.

## Verification checklist

1. Confirm checkpoint files and dataset counts against the authors' README.
2. Run one ImageNet64 SRx2 case, then all listed configurations.
3. Compare generated metrics with the paper using the same split and preprocessing. Record deviations rather than tuning on the test split.
4. Only after baseline matches, modify one component at a time (for example guidance schedule or number of model evaluations) and compare quality and runtime at fixed seed.

The upstream README does not provide a single command that downloads all datasets and checkpoints. These are external prerequisites, not bundled artifacts.
