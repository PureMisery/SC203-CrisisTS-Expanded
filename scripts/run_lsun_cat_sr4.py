"""Run and record the CM4IR LSUN Cat SRx4 baseline on a local CUDA runtime."""

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "exp/datasets/lsun_cat/cat"
CHECKPOINT = ROOT / "exp/logs/lsun_cat/cd_cat256_lpips.pt"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Run the first N sorted images; default is all available images")
    parser.add_argument("--output-name", help="New output folder name under exp/image_samples")
    args = parser.parse_args()

    images = sorted(DATASET.glob("*.png"))
    if not images:
        parser.error(f"No PNG images found in {DATASET}")
    if not CHECKPOINT.is_file():
        parser.error(f"Checkpoint missing: {CHECKPOINT}")
    if args.limit is not None and not 1 <= args.limit <= len(images):
        parser.error(f"--limit must be between 1 and {len(images)}")
    count = args.limit or len(images)
    output_name = args.output_name or f"CM4IR_lsun_cat_sr4_{count}images"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", output_name):
        parser.error("--output-name must contain only letters, digits, underscores, or hyphens")
    output_dir = ROOT / "exp/image_samples" / output_name
    if output_dir.exists():
        parser.error(f"Output already exists; choose another --output-name: {output_dir}")

    record_dir = ROOT / "exp/repro_runs"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"{output_name}.json"
    stdout_path = record_dir / f"{output_name}.log"
    command = [
        sys.executable, "main.py", "--config", "lsun_cat_256.yml",
        "--path_y", "lsun_cat", "--deg", "sr_bicubic", "--deg_scale", "4",
        "--sigma_y", "0.05", "-i", output_name, "--iN", "250",
        "--gamma", "0.2", "--model_ckpt", "lsun_cat/cd_cat256_lpips.pt",
        "--subset_start", "0", "--subset_end", str(count),
    ]
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "dataset_total_available": len(images),
        "dataset_used": count,
        "dataset_files": [
            {"name": image.name, "sha256": sha256(image)} for image in images[:count]
        ],
        "checkpoint": {"name": CHECKPOINT.name, "bytes": CHECKPOINT.stat().st_size,
                       "sha256": sha256(CHECKPOINT)},
        "command": command,
        "output_dir": str(output_dir.relative_to(ROOT)),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    with stdout_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncode = process.wait()

    output = stdout_path.read_text(encoding="utf-8")
    psnr = re.search(r"Total Average PSNR:\s*([0-9.]+)", output)
    lpips = re.search(r"Total Average LPIPS:\s*([0-9.]+)", output)
    sample_count = re.search(r"Number of samples:\s*(\d+)", output)
    saved_images = len(list(output_dir.glob("*_0.png"))) if output_dir.exists() else 0
    record.update({
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "saved_images": saved_images,
        "psnr_db": float(psnr.group(1)) if psnr else None,
        "lpips": float(lpips.group(1)) if lpips else None,
        "reported_samples": int(sample_count.group(1)) if sample_count else None,
        "stdout_log": str(stdout_path.relative_to(ROOT)),
    })
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if returncode or not psnr or not lpips or not sample_count or saved_images != count or int(sample_count.group(1)) != count:
        raise RuntimeError(f"Incomplete baseline run; inspect {stdout_path} and {record_path}")
    print(f"Verified {count} images; PSNR {psnr.group(1)} dB; LPIPS {lpips.group(1)}")
    print(f"Record: {record_path}")


if __name__ == "__main__":
    main()
