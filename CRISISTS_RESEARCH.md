# CrisisTS-Expanded Multimodal Research Contribution

The group member's complete CrisisTS-Expanded research contribution is located at:

`research/crisists-expanded/`

## Overview

This research contribution audits, repairs, and expands the ACL 2025 *CrisisTS* benchmark (Meunier et al., 2025), coupling social media microblog streams with 11-channel hourly ERA5 meteorological time series.

### Key Research Assets in `research/crisists-expanded/`:
- **Research Paper**: `SC203_Research_Paper.pdf` (5-page academic paper adhering to VNU-HCM SC203 scientific standards).
- **Expanded Multimodal Corpus**: `data/CrisisTS_Expanded_MultiDataset.csv.gz` (N = 202,312 samples across 65 international disaster events covering 2012–2019, incorporating HumAID, CrisisLex T6/T26, CrisisMMD, and CrisisNLP paired with ERA5 weather reanalysis).
- **Empirical Benchmark Report**: `comparative_benchmark_report.md` and `SC203_Expanded_Benchmark_Report.pdf` (comprehensive 5-fold CV, LOEO-CV cross-disaster generalization, and split conformal risk control at $\alpha=0.05$).
- **Pipeline Source Code**:
  - `crisists/ingest_extended_datasets.py`: Multi-dataset ingestion and unified schema alignment.
  - `crisists/align_era5_weather.py`: Open-Meteo ERA5 historical weather acquisition and sliding window constructor.
  - `crisists/run_expanded_benchmark.py`: High-performance 5-fold CV and LOEO-CV evaluation with Dynamic MoE Gating and Conformal Risk Control.
  - `crisists/compile_paper_pdf.py`: Publication-grade ReportLab PDF compiler.
- **Oral Defense Materials**: `SC203_Oral_Defense_Guide.md` and `SC203_Oral_Defense_Guide.pdf`.

See `research/crisists-expanded/IMPORTED_FROM.md` for exact provenance, source commit, and execution guidelines.

This contribution is isolated from the CM4IR code at repository root so it can be reviewed or merged without overwriting the existing group baseline.
