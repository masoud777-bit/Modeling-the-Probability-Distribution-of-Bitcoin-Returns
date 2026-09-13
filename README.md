# An Explainable Deep Learning Framework for Bitcoin Risk Quantification: From Parsimonious Feature Selection to Modeling the Return Distribution

[![License: Proprietary](https://img.shields.io/badge/License-All_Rights_Reserved-red.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow/Keras](https://img.shields.io/badge/TensorFlow%2FKeras-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Under_Peer_Review-orange.svg)]()

## Intellectual Property & Copyright Notice

> **Copyright (c) 2026 Maghsoud Fadakar. All Rights Reserved.**
>
> This repository contains the **original implementation** developed for the research program underlying the manuscript **"An Explainable Deep Learning Framework for Bitcoin Risk Quantification: From Parsimonious Feature Selection to Modeling the Return Distribution."** In response to peer‑review feedback, the manuscript has since undergone substantive revision — including an updated dataset and refinements to the feature‑selection and modeling methodology — and is presently **under further review at a different venue**. This repository reflects the earlier implementation stage of the project and will be updated to match the revised methodology once the review process concludes.
>
> **Terms of Use**
> - This codebase is published strictly for **academic evaluation**, **peer‑review transparency**, and **portfolio verification**.
> - Commercial use, redistribution, or modification of any part of this code without explicit written permission from the author is prohibited.
> - See [`LICENSE`](LICENSE) for the full terms.
> - Feature counts, model comparisons, and other specifics described below correspond to this earlier implementation and may differ from the results reported in the manuscript's current, revised version.

---

## Project Status & Evolution

This repository reflects an **earlier stage** of an ongoing research program on modeling the probability distribution of Bitcoin returns with explainable deep learning.

- **Original implementation (this repository):** monthly Bitcoin log‑returns over an earlier sample period; an initial broad feature set narrowed via Spearman correlation and XGBoost filtering; four Spline Quantile Function (SQF) architectures (Perceptron, GRU, Transformer, LSTM) evaluated against a wide set of candidate statistical distributions.
- **Current manuscript:** in response to reviewer feedback, the study was substantially revised — the dataset was extended through 2025, and the feature‑selection and modeling methodology were refined, yielding a more parsimonious feature set and an updated comparative evaluation. This revised study, under the title above, is currently under peer review.

The codebase in this repository will be updated to reflect the revised methodology once the review process concludes.

---

## Overview

Quantifying tail risk in cryptocurrency markets requires methods that can capture non‑Gaussian dynamics, non‑stationarity, and high‑dimensional dependencies across technical and market‑derived features. This repository implements an end‑to‑end, explainable deep learning pipeline that models the **full probability distribution** of Bitcoin returns — rather than a single point forecast — using **Spline Quantile Function (SQF)** neural architectures, combined with parsimonious feature selection and SHAP‑based interpretability.

```
┌─────────────────────┬───────────────────────┬───────────────────────┬─────────────────────────┐
│ 1. Feature           │ 2. Feature            │ 3. SQF Deep Learning   │ 4. Distribution &        │
│    Engineering       │    Selection           │    Models              │    Risk Evaluation        │
├─────────────────────┼───────────────────────┼───────────────────────┼─────────────────────────┤
│ Technical indicators │ Spearman correlation  │ SQF‑MP  (Perceptron)   │ Candidate distribution   │
│ from OHLCV data      │ filtering + XGBoost    │ SQF‑MGRU (GRU)         │ fitting (AHP + TOPSIS)   │
│ (50+ features);      │ importance ranking;    │ SQF‑MT  (Transformer)  │ VaR / CVaR / GlueVaR at  │
│ log‑return           │ SHAP‑based             │ SQF‑MLSTM (LSTM)       │ 90% / 95% / 99%          │
│ construction &       │ interpretability and   │ + classical ML         │ Calibration via PIT +    │
│ robust preprocessing │ leakage checks         │ baseline comparison    │ isotonic regression      │
└─────────────────────┴───────────────────────┴───────────────────────┴─────────────────────────┘
```

Each stage of this pipeline is documented in detail under [`docs/`](docs/), with one file per script.

---

## Repository Structure

```text
├── data/
│   └── sample/                        # Anonymized sample data for pipeline verification
├── docs/                              # Module-level documentation (one file per script)
├── notebooks/                         # Exploratory notebooks and run logs
│   └── logs/                          # Training / feature-selection run logs
├── outputs/                           # Example outputs (plots, metrics) from each stage
│   ├── baseline/                      # Classical ML baseline results
│   ├── distribution/                  # Distribution-fitting results
│   ├── feature selection/             # SHAP plots, leakage analysis, CV results
│   ├── model/                         # SQF model prediction plots
│   └── preprocessing/                 # Preprocessing diagnostic plots
├── src/                               # Source code, organized by pipeline stage
│   ├── feature engineering (indicators)/
│   ├── preprocessing/
│   ├── feature selection/
│   ├── models/                        # SQF-MP, SQF-MGRU, SQF-MT, SQF-MLSTM
│   ├── distribution/                  # Probabilistic forecasting with SQF-MT
│   └── comparison/                    # ML baseline comparison framework
├── requirements.txt
├── LICENSE
└── README.md
```

> `data/` contains only an anonymized **sample** subset, sufficient to run and verify the pipeline end‑to‑end. The full research dataset is not distributed in this repository.

---

## Pipeline Stages

### 1. Feature Engineering (`src/feature engineering (indicators)/`)
Transforms raw OHLCV (Open, High, Low, Close, Volume) data into a rich technical feature set — over 50 indicators across multiple timeframes, including moving averages (SMA, EMA, WMA, DEMA), momentum measures (RSI, ROC, Momentum), trend indicators (MACD), volatility measures (ATR, rolling std, Beta), volume-weighted measures, and stochastic oscillators. Built on `pandas`, `NumPy`, and `TA‑Lib`.

### 2. Return Construction & Preprocessing (`src/preprocessing/`)
- **No.1 – Convert to log 30**: computes 30‑day logarithmic returns from Bitcoin price history and generates descriptive statistics (mean, std, skewness, kurtosis) and diagnostic plots.
- **No.2 – Preprocessing of main feature**: a robust preprocessing pipeline for the target series — cleaning, missing/zero handling, quantile‑regression‑based outlier treatment, `RobustScaler` scaling (with save/load support), and chronological train/test splitting.
- **No.3 – Preprocessing of features**: applies a signed log transform (`signed_log1p`) to the full feature set, detects and replaces outliers via quantile regression, and normalizes with `RobustScaler`; handles constant / all‑NaN columns separately.

### 3. Feature Selection & Interpretability (`src/feature selection/`)
- **No.4(a) – Feature selection**: a full feature‑selection and modeling framework combining Spearman correlation filtering, XGBoost importance ranking, correlation heatmaps, Random Forest / XGBoost regressors tuned via Bayesian optimization (`BayesSearchCV`), k‑fold cross‑validation, and leakage analysis.
- **No.4(b) – SHAP Visualization Framework**: generates SHAP summary and importance plots (dot plots and ranked bar charts) for transparent, reproducible feature‑attribution analysis, exported as both PNG and CSV.

### 4. Spline Quantile Function (SQF) Models (`src/models/`)
Four deep learning architectures parameterize the full conditional return distribution via spline‑based quantile regression, trained with a pinball‑loss objective across a dense grid of quantiles (with extra density in the tails for accurate risk estimation):
- **No.5 – SQF‑MP**: Multi‑Layer Perceptron backbone.
- **No.6 – SQF‑MGRU**: GRU‑based recurrent backbone.
- **No.7 – SQF‑MT**: Transformer encoder backbone with multi‑head attention.
- **No.8 – SQF‑MLSTM**: LSTM‑based recurrent backbone.

All four share a common design: hyperparameter optimization via **Ray Tune** (HyperOpt / ASHA), isotonic‑regression‑based calibration using the Probability Integral Transform (PIT), and evaluation via CRPS, MSIS, R², MASE, KL divergence, and Wasserstein distance.

### 5. Probabilistic Forecasting & Risk Quantification (`src/distribution/`)
- **No.9 – Probabilistic Forecasting by SQF‑MT**: runs the full inference pipeline for the Transformer‑based SQF model — training, PIT‑based calibration, and generation of calibrated predictive distributions. Computes **Value‑at‑Risk (VaR)**, **Conditional VaR (CVaR)**, and **GlueVaR** at the 90%, 95%, and 99% confidence levels, and benchmarks the fitted distribution against 20 standard candidate distributions (e.g. Johnson SU, Normal, Log‑Normal) using an AHP‑weighted **TOPSIS** ranking over KL divergence, AIC, BIC, and the KS‑statistic.

### 6. Classical ML Baseline Comparison (`src/comparison/`)
- **No.10 – Machine Learning Baseline Comparison Framework**: benchmarks seven classical / tree‑based models (Quantile Regression, Random Forest Quantile, LightGBM Quantile, CatBoost Quantile, scikit‑learn Gradient Boosting, AdaBoost, and SVR) against the SQF models, using the same evaluation suite — point‑forecast accuracy (RMSE, MAE, R²), probabilistic accuracy (CRPS, MSIS), calibration (PIT), and distributional similarity (KL divergence, Wasserstein distance).

Full technical detail for every stage — architectures, loss functions, configuration spaces, generated artifacts, and dependencies — is available in the corresponding file under [`docs/`](docs/).

---

## Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/masoud777-bit/Modeling-the-Probability-Distribution-of-Bitcoin-Returns.git
cd Modeling-the-Probability-Distribution-of-Bitcoin-Returns

python -m venv .venv
source .venv/bin/activate      # On Windows: .\venv\Scripts\activate.ps1
pip install -r requirements.txt
```

### 2. Running the Pipeline on Sample Data
The scripts under `src/` are organized by stage and are intended to be run in sequence, using the anonymized data in `data/sample/`:

```bash
python "src/feature engineering (indicators)/Feature engineering on technical data.py"
python "src/preprocessing/No.1 - Convert to log 30.py"
python "src/preprocessing/No.2 - preprocessing of main feature.py"
python "src/preprocessing/No.3 - preprocessing of features.py"
python "src/feature selection/No.4(a)  - Feature selection.py"
python "src/feature selection/No.4(b)  - Feature selection (SHAP Visualization Framework).py"
python "src/models/No.5  - Spline Quantile Function Multi Perceptron (SQF_MP) Model.py"
python "src/models/No.6 - Spline Quantile Function Multi GRU (SQF_MGRU) Model.py"
python "src/models/No.7 - Spline Quantile Function Multi Transformer (SQF-MT) Model.py"
python "src/models/No.8 - Spline Quantile Function Multi LSTM(SQF_MLSTM) Model.py"
python "src/distribution/No.9 - Probabilistic Forecasting by SQF_MT.py"
python "src/comparison/No.10 - Machine Learning Baseline Comparison Framework for Probabilistic Forecasting.py"
```

> Each script reads its inputs from and writes its outputs to the paths documented in the corresponding file under `docs/`. See individual scripts for exact expected file names and CLI options.

---

## Citation

This code accompanies the following manuscript, currently **under peer review**:

> Fadakar, M., Ebadian, A., & Ashtab, A. *An Explainable Deep Learning Framework for Bitcoin Risk Quantification: From Parsimonious Feature Selection to Modeling the Return Distribution.* Manuscript under review.

A complete citation (journal name, volume, DOI) will be added here once the paper is published.

---

## License

This project is released under an **all‑rights‑reserved** license — see [`LICENSE`](LICENSE) for full terms. The code is made available for academic evaluation and peer‑review transparency only; no commercial use, redistribution, or modification is permitted without the author's explicit written permission.

---

## Authors

**Maghsoud Fadakar** — *Methodology, Software, Data Curation, Formal Analysis, Investigation, Original Draft*
Master of Finance Graduate, Department of Accounting and Finance, Faculty of Economics and Management, Urmia University, Urmia, Iran
Email: fadakar.masoud777@gmail.com

**Ali Ebadian, Ph.D.** — *Corresponding Author; Conceptualization, Supervision, Writing – Review & Editing*
Professor, Department of Sciences, Faculty of Mathematics, Urmia University, Urmia, Iran
Email: ebadian.ali@gmail.com · [Google Scholar](https://scholar.google.com/citations?user=0jZj6YsAAAAJ&hl=en)

**Ali Ashtab, Ph.D.** — *Supervision, Writing – Review & Editing*
Associate Professor, Department of Accounting and Finance, Faculty of Economics and Management, Urmia University, Urmia, Iran
Email: a.ashtab@urmia.ac.ir · [Google Scholar](https://scholar.google.com/citations?user=zv_sq6cAAAAJ&hl=en)

Correspondence regarding the manuscript should be directed to the corresponding author, Dr. Ali Ebadian. Correspondence regarding this codebase specifically may be directed to Maghsoud Fadakar.
