# Prediction Streaming 5G - 6 Resolution Classes

This repository contains the complete Machine Learning pipeline for dataset generation, hyperparameter optimization with **Optuna**, model training, and performance evaluation for 5G video streaming quality prediction across **6 resolution classes**.

---

## 📌 Project Overview

The goal of this project is to predict the video streaming quality class (`144p`, `360p`, `480p`, `720p`, `1080p`, `4K`) based on 5G network signal metrics (`RSRP`, `RSRQ`, `SNR`, `CQI`, `RSSI`), evaluating **4 test scenarios**:

---

## 🛠️ 1. 6-Class Dataset Generation

The `Scripts/treatment_data.py` script converts the raw downlink bitrate column (`DL_bitrate`) into 6 video quality resolution classes and selects only the relevant signal features.

### How to Run Dataset Generation:
```powershell
uv run -m Scripts.treatment_data
```

### 6-Class Resolution Mapping:
- `(0, 0.7 Mbps)` -> **144p**
- `[0.7, 1.1 Mbps)` -> **360p**
- `[1.1, 2.5 Mbps)` -> **480p**
- `[2.5, 5.0 Mbps)` -> **720p**
- `[5.0, 20.0 Mbps)` -> **1080p**
- `[>= 20.0 Mbps)` -> **4K**
- Inactive / invalid streaming -> **Class 0** *(filtered out during model training)*

Processed datasets are automatically stored under `Amazon-6/` and `Netflix-6/`.

---

## 🧠 2. Training Pipeline & Hyperparameter Optimization (Optuna)

The training pipeline in `Scripts/training_pipeline.py` performs automated hyperparameter optimization using **Optuna** (with 3-fold cross-validation) and calculates comprehensive classification metrics.

### Supported Machine Learning Models:
1. **KNN** (`KNeighborsClassifier`)
2. **Gaussian Naive Bayes** (`GaussianNB`)
3. **MLP - Neural Network** (`MLPClassifier`)
4. **Random Forest** (`RandomForestClassifier`)
5. **SVM** (`SVC`)
6. **LightGBM** (`LGBMClassifier`)

---

## 🚀 3. How to Run the Pipeline

### Interactive Terminal Menu
Launch the interactive CLI menu to select specific scenarios, models, or Optuna trials:

```powershell
uv run python -m Scripts.training_pipeline
```

---

## 📊 4. Output CSV Reports

Upon completion, all experimental results are exported into 3 structured CSV files ready for paper tables and figure plotting:

1. **`model_results_global.csv`**:
   - Overall metrics: `balanced_accuracy`, `f1_score_weighted`, `precision_weighted`, `recall_weighted`, `mcc`, `log_loss`.
   - Computational performance: `train_time_seconds`, `inference_time_seconds`, `inference_time_per_sample_seconds`.

2. **`model_results_per_class.csv`**:
   - Per-class metrics: `precision`, `recall`, and `f1_score` calculated individually for each resolution target (`144p`, `360p`, `480p`, `720p`, `1080p`, `4K`).

3. **`optuna_best_params.csv`**:
   - Optimal hyperparameter configurations selected by Optuna for each model and scenario.

---

## 📁 Repository Structure

```text
Prediction_streaming_5G-v2/
├── Amazon-6/                # Processed Amazon datasets (Static and Driving)
├── Netflix-6/               # Processed Netflix datasets (Static and Driving)
├── Scripts/
│   ├── treatment_data.py    # Generates 6-class datasets from raw DL_bitrate
│   ├── data_loader.py       # Dataset loading, class 0 filtering, and train/test split
│   ├── models_optuna.py     # Optuna optimization functions and model pipelines
│   ├── evaluation.py        # Global & per-class evaluation metrics and CSV exporters
│   └── training_pipeline.py # Main runner script with interactive CLI menu
├── model_results_global.csv # Global experimental results
├── model_results_per_class.csv # Per-class experimental results
├── optuna_best_params.csv   # Best Optuna hyperparameter parameters
└── README.md
```
