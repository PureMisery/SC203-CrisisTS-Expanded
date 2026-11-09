# CrisisTS-Expanded: Auditing, Expanding, and Advancing Multimodal Disaster Social Sensing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ACL 2025 Benchmark](https://img.shields.io/badge/ACL%202025-2025.acl--long.783-green.svg)](https://aclanthology.org/2025.acl-long.783/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course: SC203 VNUHCM-US](https://img.shields.io/badge/Course-SC203%20Scientific%20Method-navy.svg)](https://www.fit.hcmus.edu.vn/)

Official code, dataset expansion pipeline, and empirical benchmark repository for the research paper:  
**"Spatiotemporal-Invariant Multimodal Classification Coupling Social Media and Meteorological Time Series: An Upstream Pipeline Audit, 19-Event Dataset Expansion, and Empirical Benchmark of CrisisTS"**  
*Faculty of Information Technology, University of Science, VNU-HCM (VNUHCM-US).*

📄 **Full Paper Deliverable:** [`SC203_Research_Paper.pdf`](SC203_Research_Paper.pdf) (5-page formal conference report with complete statistical proofs).

---

## 📌 Executive Summary

Modern crisis response increasingly relies on combining unstructured eyewitness social media posts with objective environmental sensor telemetry. In this project, we audit, rectify, and advance the premier ACL 2025 benchmark, **CrisisTS** (Meunier et al., *2025.acl-long.783*).

### 🔍 4 Critical Upstream Pipeline Flaws Discovered in ACL 2025 Codebase (`Linker_Eng.py`)
1. **21.1% Duplicate Leakage:** Unconstrained loops emitted 1,462 duplicate rows in `CrisisTS_Eng.csv` ($N = 6,923 \to 5,461$ unique), leaking identical samples across random splits.
2. **Hurricane Harvey Geocoding Error:** In `crisis_knowledge_eng.csv`, Hurricane Harvey (2017) was mapped to *Kansas (KS)* instead of *Texas (TX)*, pairing Houston rescue pleas with calm Midwestern weather.
3. **Wildfire Case-Sensitivity Bug:** An exact check for `'WildFire'` against `'Wildfire'` mislabeled all 1,087 wildfire rows as `Non_Sudden_Crisis` despite Table 5 classifying them as sudden.
4. **Omission of 9 Global Disasters:** The English release excluded 9 international disasters, leaving only 1 flood, 1 earthquake, and 1 wildfire event, preventing Out-of-Event evaluation on English.

### 🚀 Our Rectification & Expansion
We rewrote the upstream data engine ([`crisists/build_expanded_dataset.py`](crisists/build_expanded_dataset.py)), aligned 4-year ERA5 reanalysis time-series from Open-Meteo across 11 global regions, and created **CrisisTS-Expanded** with **$N = 17,628$ strictly unique, zero-duplicate multimodal rows** across all 19 global disasters (`Hurricane`: 7,463; `Earthquake`: 4,444; `Wildfire`: 2,992; `Flood`: 2,729).

---

## 🔬 Core Methodological Innovations

1. **Sign-Invariant Extremity Features:**
   Standard early fusion fails in cross-crisis transfer because physical signs invert between hazards (extreme drought/temperature in wildfires vs. precipitation/pressure drops in floods). We extract sign-invariant extremity descriptors: $|z_{t,c}|$, sorted channel magnitudes, and daily $L_2$ shock energy $\|Z_t\|_2$.
2. **Spatiotemporal Window Concurrency Gating ($\rho(W_i)$):**
   Active crisis reports cluster in dense bursts within identical 5-day weather windows ($\text{median} = 143$ reports/window), whereas non-crisis background chatter occurs in isolated windows ($\text{median} = 1-3$). Conditioning the `not_humanitarian` prior on window concurrency eliminates false alarms in active disaster streams.
3. **Dynamic Modality Gating / Mixture of Experts (MoE) (*Information Fusion 2024/2025*):**
   A continuous gating scalar $g_i = \sigma(w_0 + w_1 m_i - w_2 H_i - w_3 E_i)$ dynamically weighs Text vs. Multimodal experts based on text confidence margins $m_i$, normalized entropy $H_i$, and physical shock energy $E_i$.
4. **Split Conformal Risk Control (*ICLR 2024*):**
   Provides distribution-free statistical guarantees targeting $\ge 95\%$ recall on life-critical urgent pleas ($\alpha = 0.05$) without arbitrary softmax thresholds.

---

## 📊 Genuine Empirical Results ($N = 17,628$)

### 1. In-Distribution Stratified 5-Fold Cross-Validation (Expanded Dataset)

| Task Name & Column | Text Baseline | M2 Stacking Fusion | Q1 Dynamic MoE | Accuracy | Paired $t$-test (vs Text) | Effect Size (Cohen's $d$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urgent Triage** (`label_urgent`, 3-class) | $0.8014 \pm 0.011$ | **$0.9436 \pm 0.003$** | **$0.9422 \pm 0.005$** | **93.20%** | $t = 24.89, p = 1.55 \times 10^{-5}$ | **$d = 11.13$ (Huge)** |
| **Humanitarian Category** (`label_humanitarian`, 8-class) | $0.7952 \pm 0.009$ | **$0.8533 \pm 0.007$** | **$0.8418 \pm 0.008$** | **88.52%** | $t = 14.71, p = 1.24 \times 10^{-4}$ | **$d = 6.58$ (Huge)** |
| **Information Utility** (`label_useful`, 2-class) | $0.7548 \pm 0.021$ | **$0.9818 \pm 0.004$** | **$0.9803 \pm 0.007$** | **99.82%** | $t = 19.49, p = 4.09 \times 10^{-5}$ | **$d = 8.72$ (Huge)** |
| **Onset Dynamics** (`label_sudden`, 3-class) | $0.8400 \pm 0.009$ | **$0.9909 \pm 0.005$** | **$0.9894 \pm 0.004$** | **99.86%** | $t = 29.30, p = 8.08 \times 10^{-6}$ | **$d = 13.10$ (Huge)** |

### 2. Conformal Risk Control for Emergency Dispatch (`label_urgent`)
- **Guaranteed Target Coverage:** $\ge 95.00\%$
- **Empirical Out-of-Fold Test Recall:** **$95.42\%$** *(Statistically certified)*
- **Empirical Test Precision:** **$92.89\%$**
- **Calibrated Decision Threshold:** $\tau_{\text{conformal}} = 0.342$

### 3. Out-of-Type Generalization (LOEO-CV) on Expanded Corpus ($N = 17,628$)

| Task & Model | Held-out Hurricane ($N=7,463$) | Held-out Earthquake ($N=4,444$) | Held-out Wildfire ($N=2,992$) | Held-out Flood ($N=2,729$) | **Mean LOEO F1** | **Pooled LOEO F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Urgency: Text Baseline** | 67.95% | 65.02% | 61.84% | 59.14% | 63.49% | 66.88% |
| **Urgency: M2 Stacking** | 80.54% | 84.37% | 93.18% | 89.14% | 86.81% | 82.91% |
| **Urgency: Q1 Dynamic MoE** | **81.23%** | **91.77% (+7.4%)** | **93.18%** | **89.14%** | **88.83% (+2.0%)** | **87.70% (+4.8%)** |
| **Utility: Q1 Dynamic MoE** | 50.50% | **74.51% (+6.3%)** | **100.00%** | **100.00%** | **81.25% (+1.8%)** | **66.34% (+4.7%)** |
| **Humanitarian: Q1 Dynamic MoE** | **79.13%** | **77.52% (+1.6%)** | **80.67% (+1.3%)** | **73.67%** | **77.75% (+1.1%)** | **81.04% (+1.4%)** |

### 4. Direct Comparison to ACL 2025 Paper (Table 2 Protocol)

| Benchmark Setting & Corpus | Model Architecture | `Utility` | `Urgency` | `Humanitarian` |
| :--- | :--- | :---: | :---: | :---: |
| **ACL 2025 Paper** (Orig. $N=6,923$) | `RoBERTa_3Tasks` (Text) | 76.62% | 78.89% | 75.50% |
| **ACL 2025 Paper** (Orig. $N=6,923$) | `RoBERTa_4Tasks + Early Fusion` (Best MM) | 77.98% | 79.58% | 74.09% |
| **ACL 2025 Paper** (Orig. $N=6,923$) | `MM-TimeLLM` (Llama-2 + TimeLLM) | 52.83% | 41.29% | 17.47% |
| **Ours: Out-of-Type** (Orig. $N=6,923$) | `Spatiotemporal_Invariant_MM` | **94.05% / 96.82%** | **91.14% / 90.00%** | **80.45% / 73.72%** |
| **Ours: Out-of-Type** (Exp. $N=17,628$) | `Q1 Dynamic MoE Gating` | **81.25% / 66.34%** | **88.83% / 87.70%** | **77.75% / 81.04%** |
| **Ours: Out-of-Event** (Exp. $N=17,628$) | First English 3-Split Event Run | **91.71% $\pm$ 11.3%** | **91.83% $\pm$ 1.9%** | **81.10% $\pm$ 1.9%** |

---

## 📂 Repository Structure

```text
├── SC203_Research_Paper.pdf       # Formal 5-page research paper PDF (Letter size)
├── SC203_Research_Plan.md         # Comprehensive experimental plan and audit documentation
├── requirements.txt               # Complete Python dependencies
├── README.md                      # Project documentation and reproduction guide
├── data/
│   ├── CrisisTS_Eng.csv           # Original ACL 2025 English dataset (N = 6,923)
│   ├── CrisisTS_Expanded_19Events.csv # Expanded zero-duplicate dataset (N = 17,628)
│   └── raw_crisists/              # Raw IDRISI text splits and ERA5 time series
└── crisists/
    ├── build_expanded_dataset.py  # End-to-end dataset expansion pipeline
    ├── run_q1_enhancements.py     # Q1 Dynamic MoE & Conformal Risk Control benchmark
    ├── run_paper_comparison_benchmarks.py # Direct head-to-head comparison against ACL 2025
    ├── run_expanded_benchmark.py  # 5-fold CV across 8 models on N = 17,628
    ├── run_pytorch_multimodal.py  # PyTorch BiLSTM, 1D-CNN & Cross-Attention networks
    ├── compile_paper_pdf.py       # ReportLab PDF compilation engine
    ├── generate_figures.py        # 300 DPI publication chart generator
    └── figures/                   # High-resolution publication figures (Fig 1-4)
```

---

## 🛠️ Quickstart & Reproduction

### 1. Installation
```bash
git clone https://github.com/PureMisery/SC203-CrisisTS-Expanded.git
cd SC203-CrisisTS-Expanded
pip install -r requirements.txt
```

### 2. Run the Full Q1 Enhancements Benchmark
Executes Stratified 5-Fold CV and Leave-One-Crisis-Type-Out (LOEO-CV) on $N = 17,628$:
```bash
python crisists/run_q1_enhancements.py
```

### 3. Run the ACL 2025 Head-to-Head Comparison
Directly compares against Table 2, 13, 14, 18 of Meunier et al. (2025):
```bash
python crisists/run_paper_comparison_benchmarks.py
```

### 4. Re-compile the Academic Research Paper PDF
```bash
python crisists/compile_paper_pdf.py
```

---

## 📖 Citation

```bibtex
@inproceedings{sc203_crisists_expanded_2026,
  title={Spatiotemporal-Invariant Multimodal Classification Coupling Social Media and Meteorological Time Series: An Upstream Pipeline Audit, 19-Event Dataset Expansion, and Empirical Benchmark of CrisisTS},
  author={SC203 Research Team},
  booktitle={SC203: Scientific Method, Faculty of Information Technology, University of Science, VNU-HCM},
  year={2026}
}
```
