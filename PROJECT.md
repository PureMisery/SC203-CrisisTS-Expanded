# Project: Extended Disaster Tweet Dataset Ingestion, ERA5 Alignment, and Crisis Modeling Benchmark

## Architecture
- **Data Ingestion Module** (`crisists/ingest_extended_datasets.py`): Downloads and harmonizes extended crisis tweet corpora (HumAID full, CrisisLex T26 & T6, CrisisMMD text/metadata, CrisisNLP) via official HuggingFace QCRI datasets, decodes Twitter Snowflake UTC timestamps, and normalizes categorical labels to a unified 10-class / 3-class / 2-class taxonomy.
- **Meteorological Alignment Module** (`crisists/align_era5_weather.py`): Queries the Open-Meteo ERA5 Historical Reanalysis Archive for 27+ international disaster coordinate centers (2012–2015), caches 11 physical channels, standardizes them over local climatology, and constructs $(5, 11)$ daily sliding window tensors matched to tweet UTC dates.
- **Multimodal Dataset Assembler** (`data/CrisisTS_Expanded_MultiDataset.csv`): Combines cleaned microblog text, event metadata, and $(5, 11)$ ERA5 weather window tensors into a single unified multimodal dataset with zero test leakage.
- **Evaluation & Benchmarking Engine** (`crisists/run_expanded_benchmark.py`): Executes Stratified 5-Fold Cross-Validation and Leave-One-Crisis-Type-Out (LOEO) evaluation on the expanded multi-dataset corpus across 4 tasks (`label_urgent`, `label_humanitarian`, `label_useful`, `label_sudden`) utilizing Text TF-IDF FeatureUnion, Physical Weather Random Forest, Q1 Dynamic MoE Gating, and Split Conformal Risk Control.
- **Comparative Reporting Engine**: Emits `comparative_benchmark_report.md` (Markdown table) and `comparative_benchmark_summary.json` (machine-readable metrics) benchmarked against the 76,481 sample baseline.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | HumAID Ingestion Verification | Verify full 19-event HumAID benchmark ($N=76,481$) integrity in `data/CrisisTS_Full_76k.csv` | M1 | Survey / R1 |
| 2 | CrisisLex Ingestion | Ingest CrisisLex T26 (26 events, ~20k labeled) and T6 (6 events, ~50k labeled) | M1 | Survey / R1 |
| 3 | CrisisMMD Ingestion | Ingest CrisisMMD (~16k unique tweets across 7 events), cleanly omitting image processing | M1 | Survey / R1 |
| 4 | CrisisNLP Ingestion | Ingest CrisisNLP (~40k curated across 19 events, 100k+ extended tweet IDs) | M1 | Survey / R1 |
| 5 | Snowflake UTC Timestamp Decoding | Deterministic millisecond UTC date extraction via `(tid >> 22) + 1288834974657` | M1 | Survey / R1 |
| 6 | Common Taxonomy Schema Alignment | Map all native categorical labels into 10-class `label_humanitarian`, 3-class `label_urgent`, 2-class `label_useful`, and 3-class `label_sudden` | M1 | Survey / R1 |
| 7 | Global Deduplication & Filtering | Deduplicate across datasets by `tweet_id` and `(date, clean_text)` to prevent train/test leakage | M1 | Survey / R1 |
| 8 | ERA5 Geographic Coordinates Registry | Map all 27+ newly added disaster events to spatial coordinates $(\text{lat}, \text{lon})$ | M2 | Survey / R2 |
| 9 | ERA5 11-Channel Weather Acquisition | Query Open-Meteo ERA5 archive for 2012–2015 historical weather across 11 physical channels | M2 | Survey / R2 |
| 10 | Climatological Standardization | Fit location-specific `StandardScaler` to normalize multi-year baseline weather distributions | M2 | Survey / R2 |
| 11 | $(5, 11)$ Spatio-Temporal Window Construction | Build 5-day sliding window matrices ending on tweet UTC dates with 1:1 backward compatibility | M2 | Survey / R2 |
| 12 | Expanded Multimodal Dataset Export | Serialize unified dataset to `data/CrisisTS_Expanded_MultiDataset.csv` and `.csv.gz` | M2 | Survey / R2 |
| 13 | Model Evaluation Rerun Pipeline | Run Stratified 5-Fold Cross-Validation on expanded multi-dataset corpus across 4 crisis tasks | M3 | Survey / R3 |
| 14 | LOEO-CV Evaluation Rerun | Execute Leave-One-Crisis-Type-Out cross-validation to assess out-of-domain generalization | M3 | Survey / R3 |
| 15 | Conformal Risk Control Calibration | Calibrate split conformal prediction on expanded corpus for $\ge 95\%$ rescue recall | M3 | Survey / R3 |
| 16 | Comparative Markdown Report Generation | Produce `comparative_benchmark_report.md` comparing baseline (76,481) vs expanded corpus | M3 | Survey / R3 |
| 17 | Comparative JSON Summary Generation | Produce `comparative_benchmark_summary.json` capturing structured before/after metrics | M3 | Survey / R3 |
| 18 | Empirical Verification & Review | 2 Reviewers and 2 Challengers verify data validity, zero leakage, and script execution | M4 | Quality / Gate |
| 19 | Forensic Integrity Audit | `teamwork_auditor` executes anti-cheating and authenticity verification checks | M4 | Integrity / Gate |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Dataset Ingestion & Schema Alignment | Ingest HumAID, CrisisLex (T6 & T26), CrisisMMD, CrisisNLP; decode timestamps; map to unified schema; deduplicate | none | DONE |
| 2 | M2: Hourly 11-Channel ERA5 Weather Alignment | Download 2012-2015 ERA5 reanalysis via Open-Meteo for new events; standardize; construct $(5, 11)$ sliding window tensors; build expanded dataset | M1 | DONE |
| 3 | M3: Model Evaluation Rerun & Comparative Benchmarking | Run 5-Fold CV and LOEO-CV on expanded corpus across 4 tasks; produce Markdown table and JSON summary comparing vs 76,481 baseline | M2 | IN_PROGRESS |
| 4 | M4: Adversarial Review & Forensic Audit | Reviewers, Challengers, and Forensic Auditor verify pipeline authenticity and metric consistency | M3 | PLANNED |

## Interface Contracts

### M1 (Dataset Ingestion) ↔ M2 (ERA5 Alignment)
- **Intermediate Artifact**: `data/intermediate_extended_tweets.parquet` (or `.csv.gz`)
- **Schema**:
  - `tweet_id`: `str` (64-bit Twitter Snowflake identifier)
  - `Text`: `str` (cleaned, normalized tweet text)
  - `Date`: `str` (`YYYY-MM-DD` UTC date)
  - `Event_Name`: `str` (canonical snake_case event identifier)
  - `Crisis_Type`: `str` (`Earthquake`, `Flood`, `Hurricane`, `Wildfire`)
  - `Source_Dataset`: `str` (`HumAID`, `CrisisLexT6`, `CrisisLexT26`, `CrisisMMD`, `CrisisNLP`)
  - `label_humanitarian`: `str` (10-class taxonomy)
  - `label_urgent`: `str` (`urgent`, `not_urgent`, `not_humanitarian`)
  - `label_useful`: `str` (`useful`, `not_humanitarian`)
  - `label_sudden`: `str` (`Sudden_Crisis`, `Non_Sudden_Crisis`, `Not_Crisis_period`)

### M2 (ERA5 Alignment) ↔ M3 (Model Evaluation)
- **Output Artifact**: `data/CrisisTS_Expanded_MultiDataset.csv` (and `.csv.gz`)
- **Schema**:
  - All columns from M1 above plus:
  - `Window`: `str` (stringified $(5, 11)$ array parsable via `crisists/run_full_76k_benchmark.py:parse_window_3d`)
- **Shape Requirement**: Exactly 5 rows $\times$ 11 physical channels per sample.
- **Physical Channels**: `['mean_speed_wind', 'precipitation', 'snow_fall', 'snow_depth', 'temp_max', 'temp_min', 'temp_mean', 'wind_dir_angle_2min', 'wind_dir_angle_5sec', 'wind_max_speed_2min', 'wind_max_speed_5sec']`.

### M3 (Model Evaluation) ↔ Final Deliverables
- **Comparative Markdown Deliverable**: `comparative_benchmark_report.md`
- **Comparative JSON Deliverable**: `comparative_benchmark_summary.json`
- **Required Baseline Comparison**: Baseline metrics from `crisists/full_76k_benchmark_results.json` ($N = 76,481$).

## Code Layout
- `crisists/ingest_extended_datasets.py`: Extended multi-dataset ingestion and schema normalization script.
- `crisists/align_era5_weather.py`: Open-Meteo ERA5 downloader, local climatology scaler, and $(5, 11)$ sliding window builder.
- `crisists/run_expanded_benchmark.py`: Benchmark rerun script executing 5-Fold Stratified CV, LOEO-CV, and conformal calibration on expanded corpus.
- `data/CrisisTS_Expanded_MultiDataset.csv`: Complete expanded multimodal dataset.
- `comparative_benchmark_report.md`: Markdown comparison table report.
- `comparative_benchmark_summary.json`: JSON structured metrics summary.
