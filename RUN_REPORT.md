# Verified Colab run — LSUN Cat SR×4

Run date: 2026-11-08 UTC (2026-11-09 Asia/Bangkok). Notebook: [Colab](https://colab.research.google.com/drive/1AnxTVDhjrXW-oOr36B8UfAw-_KBKZvY4).

| Item | Value |
| --- | --- |
| Upstream commit | `e5fe0a02ea63184a26ac34a1c504818d26b3c592` |
| GPU | Tesla T4, 15 GB |
| Colab PyTorch | `2.14.1+cu130` |
| Checkpoint | `cd_cat256_lpips.pt`, 2,105,395,349 bytes |
| Dataset | 57 PNGs downloaded from the authors' LSUN Cat validation folder |
| Task | Bicubic super-resolution ×4, noise σ=0.05 |
| Seed | 1234 |
| Samples completed | 57/57 |
| Mean PSNR | 26.24 dB |
| Mean LPIPS | 0.3492 |

Command used in Colab after applying the SDPA compatibility fallback:

```bash
python main.py --config lsun_cat_256.yml --path_y lsun_cat --deg sr_bicubic --deg_scale 4 --sigma_y 0.05 -i CM4IR_lsun_cat_sr4_subset57 --iN 250 --gamma 0.2 --model_ckpt lsun_cat/cd_cat256_lpips.pt --subset_start 0 --subset_end 57
```

The first one-image smoke test also completed, with PSNR 30.93 dB and LPIPS 0.2859. The full subset run produced 57 PNG files and `0_logs.log` in `exp/image_samples/CM4IR_lsun_cat_sr4_subset57/` on the Colab runtime. A 5,409,466-byte ZIP was created in `/content/cm4ir_subset57.zip`. Copying it into Google Drive stalled at the Drive mount authorization stage, so the archive is not claimed as persisted outside the current runtime.

On the same T4 runtime, the improved public-repository runner was verified separately. Its one-image smoke test matched PSNR 30.93 / LPIPS 0.2859. Its 57-image run completed with 57 generated PNGs and matched PSNR **26.24** / LPIPS **0.3492**. It wrote `exp/repro_runs/verification_57images.json` and `.log` inside the Colab runtime. The JSON includes the exact input and checkpoint hashes, but the file has not been committed; the notebook now offers an artifact ZIP download after each run.

Limitations: `gdown` caps the retrieved folder listing at 50 files; this run included 57 images accumulated across two download attempts. The exact list was not saved. The entire official validation split and the paper's other configurations have not been evaluated. The SDPA fallback is mathematically equivalent attention but may differ in floating point details from the old FlashAttention implementation. These metrics are evidence that the code executes and produces outputs, not verification of the paper table.
