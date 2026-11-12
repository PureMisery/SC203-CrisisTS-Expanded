# SC203 Group 9 — CM4IR extension plan

## Shared starting point

`main` contains Quỳnh Anh's CM4IR reproduction. The verified baseline is LSUN Cat bicubic super-resolution ×4 on 57 author-provided images (PSNR 26.24 dB, LPIPS 0.3492). This is a subset run, not the full paper benchmark; see `REPRODUCE.md` and `RUN_REPORT.md`.

The group question is whether additional semantic context helps **image restoration**. The proposed example, “this region is missing,” belongs to CM4IR's inpainting setting. A natural-language description alone cannot condition the current CM4IR model: the implementation accepts an image, degradation operator, and mask, but has no text encoder. Phúc's branch should implement and test an explicit conditioning path, or report a controlled image-derived context alternative if a text-conditioned model is infeasible. Do not claim an improvement until a paired experiment establishes it.

## Workstreams

| Branch | Owner | Deliverable |
| --- | --- | --- |
| `main` | Quỳnh Anh | Stable reproduced CM4IR baseline and integration decisions. |
| `codex/phuc-context` | Phúc | Context input, conditioning mechanism, examples, ablation with context on/off, and run instructions. Preserve the baseline path. |
| `codex/long-evaluation` | Long | Fixed evaluation manifest, paired baseline/variant runner, PSNR and LPIPS plus region-focused metrics for inpainting, per-image results, summary with uncertainty, and integrity checks. |

Nguyên's CrisisTS repository studies disaster messages and meteorological data. Its labels, inputs, and metric targets do not align with CM4IR image restoration. Keep that work as a separate research track unless the group defines a concrete image-restoration contribution.

## Integration contract

1. Freeze the image list, input hashes, checkpoint hash, degradation, mask, seed, and software environment before comparing methods. The current 57-image super-resolution result is a baseline reference; for the proposed missing-region hypothesis, both baseline and context method must run on the **same inpainting images and masks**.
2. Phúc writes context features/conditioning in a separate module and exposes an optional argument. With the argument off, outputs must follow the original CM4IR path. Record the source or generation rule for every description; never derive context from hidden ground-truth pixels at test time.
3. Long evaluates the baseline and context method on exactly the same held-out images. Save per-image metrics and failures, not only averages. Include a no-context ablation and a shuffled/irrelevant-context control to test whether the text itself helps.
4. Each branch opens a pull request to `main` with commands, input manifest, results, and limitations. Merge the evaluation interface first, then the context implementation; rerun the paired comparison after merging. Keep model checkpoints, datasets, and generated images out of Git.

## Branch workflow

```bash
git clone https://github.com/winanhle/CS203-GROUP-9.git
cd CS203-GROUP-9
git switch codex/phuc-context  # or codex/long-evaluation
# work, commit, then push your branch
git push -u origin HEAD
```

No branch currently contains a demonstrated context improvement. The reported 57-image scores apply only to the existing super-resolution baseline.
