# CrisisTS-Expanded Research Import Manifest

This directory contains the complete tracked research workspace for the CrisisTS-Expanded multimodal benchmark audit and extension study.

## Source Snapshot

- **Author / Student**: Lê Quang Nguyên Phúc (Student ID: `25125070`)
- **Course**: SC203 — Scientific Method (*Phương pháp Nghiên cứu Khoa học*)
- **Affiliation**: Faculty of Information Technology, University of Science, VNU-HCM
- **Source Repository**: `https://github.com/PureMisery/SC203-CrisisTS-Expanded.git`
- **Source Branch**: `main`
- **Source Commit**: `0da26bd137c8257bfbe9b159bda451b23a9ea054`
- **Import Date**: 2026-11-12
- **Import Method**: `git archive` of exact source commit `0da26bd`, preserving all 204 tracked files.

## Repository Structure & Quick Start

1. **Academic Paper**:
   - `SC203_Research_Paper.pdf` (5-page formal conference paper, ReportLab compiled).
   - Run `python crisists/compile_paper_pdf.py` to recompile.
2. **Benchmark Reports & Evidence**:
   - `comparative_benchmark_report.md` and `SC203_Expanded_Benchmark_Report.pdf` (4-page benchmark audit report).
   - `comparative_benchmark_summary.json` & `crisists/expanded_benchmark_results.json`.
3. **Datasets**:
   - `data/CrisisTS_Expanded_MultiDataset.csv.gz` (202,312 aligned samples, 65 events).
   - `data/intermediate_extended_tweets.csv.gz`.
4. **Reproducing the Pipeline**:
   - Harmonize text corpora: `python crisists/ingest_extended_datasets.py`
   - Align ERA5 weather: `python crisists/align_era5_weather.py`
   - Run multimodal benchmark: `python crisists/run_expanded_benchmark.py`

Large raw uncompressed CSVs and temporary download caches are intentionally Git-ignored and can be regenerated or unpacked from the committed `.csv.gz` archives.
