# Prediction Streaming 5G - Tree-Based Models Suite (2, 4, and 6 Classes)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-fast%20packaging-purple.svg)](https://github.com/astral-sh/uv)
[![Models](https://img.shields.io/badge/ML-Tree--Based-success.svg)]()
[![Optimization](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-orange.svg)](https://optuna.org/)

This repository contains the complete Machine Learning pipeline for predicting video streaming quality over real-world 5G networks based on client-side physical-layer radio frequency (RF) measurements. The evaluation considers both **Static** and **Vehicular/Driving** mobility scenarios across two major commercial streaming providers: **Amazon Prime Video** and **Netflix**.

The methodology investigates the impact of target resolution granularities (**6, 4, and 2 classes**), features automated hyperparameter optimization via **Optuna**, rigorous class balancing using **SMOTE** (applied strictly to the training partition to avoid data leakage), multi-seed evaluations (Mean ± Std), and automatic generation of publication-ready figures (normalized confusion matrices and comparison charts).

---

## Project Overview & Research Context

The primary goal is to proactively estimate achievable video delivery quality tiers from 5G physical radio link KPIs:
- `RSRP` (Reference Signal Received Power)
- `RSRQ` (Reference Signal Received Quality)
- `SNR` (Signal-to-Noise Ratio)
- `CQI` (Channel Quality Indicator)
- `RSSI` (Received Signal Strength Indicator)

### Evaluated Resolution Granularities:

| Granularity | Target Classes | Aggregated Resolutions | Bitrate Range (YouTube Proxy) |
| :--- | :--- | :--- | :--- |
| **6 Classes** | `144p`, `360p`, `480p`, `720p`, `1080p`, `4K` | Individual quality tiers | `<0.7`, `0.7–1.1`, `1.1–2.5`, `2.5–5`, `5–20`, `≥20 Mbps` |
| **4 Classes** | `SD`, `HD`, `FHD`, `4K` | • `SD` (`144p` + `360p`)<br>• `HD` (`480p` + `720p`)<br>• `FHD` (`1080p`)<br>• `4K` (`4K`) | `<1.1 Mbps`<br>`1.1 – 5.0 Mbps`<br>`5.0 – 20.0 Mbps`<br>`≥ 20.0 Mbps` |
| **2 Classes** | `SD`, `HD` | • `SD` (`144p`, `360p`, `480p`)<br>• `HD` (`720p`, `1080p`, `4K`) | `<2.5 Mbps` (Standard Definition)<br>`≥2.5 Mbps` (High Definition) |

---

## Class Imbalance & Leakage-Free SMOTE Pipeline

To counter extreme class imbalances (e.g., `144p` representing ~80% and `4K` less than 1% of raw measurements):
- **Stratified Splitting First**: Datasets are partitioned into 80% train and 20% test strictly **before** applying any synthetic oversampling.
- **100% Authentic Test Set**: The test split (`X_test, y_test`) is never subjected to SMOTE, ensuring evaluation reflects authentic real-world 5G traffic distributions without data leakage.

---

## Environment Setup & Installation

This project is managed with **[`uv`](https://github.com/astral-sh/uv)**, an extremely fast Python package manager and resolver.

### 1. Install `uv`

If you do not have `uv` installed, install it using the official standalone installer:

* **Windows (PowerShell):**
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

* **Via pip (alternative):**
  ```bash
  pip install uv
  ```

### 2. Sync Dependencies
Once `uv` is installed, navigate to the project directory and run:
```powershell
uv sync
```
This automatically sets up the virtual environment with all required dependencies (Optuna, Scikit-learn, XGBoost, LightGBM, Imbalanced-learn, Seaborn, etc.).

---

## Repository Structure

```text
Prediction_streaming_5G-v2/
├── datasets/                                 # Centralized datasets directory
│   ├── Amazon/                               # Raw Amazon recordings (Static and Driving)
│   ├── Netflix/                              # Raw Netflix recordings (Static and Driving)
│   ├── Amazon-{2,4,6}/                       # Generated datasets per class granularity (gitignored)
│   └── Netflix-{2,4,6}/                      # Generated datasets per class granularity (gitignored)
├── Scripts/
│   ├── treatment_data.py                     # Bitrate-to-class dataset generator (2, 4, and 6 classes)
│   ├── smote_treatment.py                    # Reference minority oversampling (SMOTE)
│   ├── dataset_statistics.py                 # Summary stats of mobility, RF features, and class counts
│   ├── data_loader.py                        # Dataset loading, train-only SMOTE, and stratified 80/20 split
│   ├── models_optuna.py                      # Optuna Bayesian hyperparameter optimization
│   ├── evaluation.py                         # Evaluation metrics, Cohen's Kappa, CMs, and comparison charts
│   └── training_pipeline.py                  # Main runner and interactive CLI menu
├── Results/                                  # Exported CSV reports and figures
│   ├── dataset_statistics/                   # Characterization tables
│   ├── classes_2/                            # Results for 2 classes (With_SMOTE / Without_SMOTE)
│   ├── classes_4/                            # Results for 4 classes (With_SMOTE / Without_SMOTE)
│   └── classes_6/                            # Results for 6 classes (With_SMOTE / Without_SMOTE)
├── pyproject.toml / uv.lock                  # Project dependencies lockfile
└── README.md
```

---

## Running the Experiments

### 1. Interactive Terminal Menu (Recommended)
Launch the interactive terminal interface:
```powershell
uv run python -m Scripts.training_pipeline
```

#### Menu Overview:
```text
===========================================================================
    5G STREAMING PREDICTION - TREE-BASED MODELS PIPELINE
===========================================================================
 Current Granularity : [6 Classes] ['144p', '360p', '480p', '720p', '1080p', '4K']
 Current Mode        : [WITHOUT SMOTE]
 Optuna Trials       : [15]
 Seeds Evaluation    : [Single (42)]
---------------------------------------------------------------------------
 [1] Run Pipeline for Current Configuration
 [2] Run All Scenarios and Classes (6, 4, and 2) in Current Mode
 [3] Run Complete Benchmark (6, 4, 2 Classes x WITHOUT and WITH SMOTE)
 [4] Select Class Granularity (2, 4, or 6)
 [5] Select a Specific Scenario
 [6] Select a Specific Tree Model
 [7] Toggle Dataset Mode (WITHOUT SMOTE / WITH SMOTE)
 [8] Set Number of Optuna Trials (Current: 15)
 [9] Toggle Multi-Seed Evaluation (Mean +- Std)
---------------------------------------------------------------------------
 [10] Generate Resolution Datasets (2, 4, and 6 Classes from raw data)
 [11] Generate SMOTE Datasets (Balanced for 2, 4, and 6 Classes)
 [12] Generate Dataset Characterization and Descriptive Statistics Tables
 [13] Run Quick Test Run (2 Trials / Confusion Matrix)
 [0] Exit
===========================================================================
```

---

### 2. Command-Line Interface (CLI / Automation)

You can also run tasks directly using CLI flags:

* **Generate all resolution datasets (2, 4, and 6 classes):**
  ```powershell
  uv run python -m Scripts.training_pipeline --generate-datasets
  ```

* **Generate dataset summary tables for the paper:**
  ```powershell
  uv run python -m Scripts.training_pipeline --dataset-stats
  ```

* **Run the Full Benchmark (All 6, 4, and 2 classes x Without and With SMOTE):**
  ```powershell
  uv run python -m Scripts.training_pipeline --full-benchmark --n-trials 15
  ```

* **Run Full Benchmark with Multi-Seed Evaluation (Seeds 42, 100, 2024 for Mean ± Std):**
  ```powershell
  uv run python -m Scripts.training_pipeline --full-benchmark --multi-seeds --n-trials 15
  ```

* **Fast validation run (2 trials + figure generation):**
  ```powershell
  uv run python -m Scripts.training_pipeline --test-run --classes 4
  ```

---

## Output Metrics and Reports

All experimental findings are automatically written to `Results/`:

1. **`model_results_global.csv`**:
   - Comprehensive performance: `accuracy`, `balanced_accuracy`, `f1_score_macro`, `f1_score_weighted`, `precision_weighted`, `recall_weighted`, `mcc`, `cohen_kappa`, `roc_auc`, `log_loss`.
   - Computational cost: `train_time_seconds`, `inference_time_seconds`, `inference_time_per_sample_seconds`.
   - Variance metrics (when using `--multi-seeds`): `balanced_accuracy_std`, `f1_macro_std`.

2. **`model_results_per_class.csv`**:
   - Individual class breakdown: `precision`, `recall`, and `f1_score`.

3. **`optuna_best_params.csv`**:
   - Best hyperparameter configurations discovered during cross-validation.

4. **`confusion_matrices/`**:
   - High-resolution (300 DPI) normalized confusion matrices exported in both **PNG** and publication-ready **PDF**.

5. **`figures/`**:
   - Macro F1-score comparison charts across models and scenarios.
