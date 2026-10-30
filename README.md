# Prediction Streaming 5G - Tree-Based Models (2, 4 and 6 Resolution Classes)

This repository contains the Machine Learning pipeline developed for evaluating **tree-based classification models** in predicting 5G video streaming quality across multiple resolution granularities (**2, 4, and 6 classes**), with hyperparameter optimization via **Optuna** and class balancing using **SMOTE**.

---

## 📌 Project Overview

The objective of this research is to estimate achievable video streaming quality from 5G physical-layer radio frequency measurements (`RSRP`, `RSRQ`, `SNR`, `CQI`, `RSSI`) across two mobility patterns (**Static** and **Vehicular/Driving**) and two major streaming providers (**Amazon Prime Video** and **Netflix**).

### Evaluated Granularities:
1. **6 Classes**: `144p`, `360p`, `480p`, `720p`, `1080p`, `4K`
2. **4 Classes**: `Low_SD` (<1.1 Mbps), `SD_HD` (1.1-5.0 Mbps), `FHD` (5.0-20.0 Mbps), `4K` (>=20.0 Mbps)
3. **2 Classes**: `SD` (<2.5 Mbps) vs `HD_Plus` (>=2.5 Mbps)

---

## 🌲 Tree-Based Machine Learning Models

The pipeline focuses exclusively on tree-based architectures:
1. **Decision Tree** (`DecisionTreeClassifier`) - Interpretable baseline
2. **Random Forest** (`RandomForestClassifier`) - Bagging ensemble
3. **Extra Trees** (`ExtraTreesClassifier`) - Extremely randomized trees
4. **Gradient Boosting** (`GradientBoostingClassifier`) - Sequential boosting
5. **LightGBM** (`LGBMClassifier`) - Fast histogram-based gradient boosting
6. **XGBoost** (`XGBClassifier`) - Scalable regularized gradient boosting

All models are tuned automatically with **Optuna** (5-fold stratified cross-validation, optimizing Balanced Accuracy / Macro F1).

---

## 🛠️ Data Preparation

### 1. Generate 2, 4, and 6-Class Datasets
Converts raw downlink bitrate (`DL_bitrate`) into categorized classes and isolates signal features:
```powershell
uv run python -m Scripts.treatment_data
```

### 2. Generate SMOTE Datasets
Generates class-balanced training datasets for all configurations:
```powershell
uv run python -m Scripts.smote_treatment
```

---

## 🚀 Running the Experiments

### Interactive Terminal Menu
Launch the interactive CLI menu:
```powershell
uv run python -m Scripts.training_pipeline
```

### Command-Line Arguments (CLI)
- **Run all classes (6, 4, 2) automatically:**
  ```powershell
  uv run python -m Scripts.training_pipeline --all-classes --n-trials 10
  ```
- **Run with SMOTE:**
  ```powershell
  uv run python -m Scripts.training_pipeline --all-classes --smote --n-trials 10
  ```
- **Run a specific class configuration (e.g. 4 classes):**
  ```powershell
  uv run python -m Scripts.training_pipeline --all --classes 4 --n-trials 10
  ```
- **Fast validation run:**
  ```powershell
  uv run python -m Scripts.training_pipeline --test-run --classes 2
  ```

---

## 📊 Output Structure

Results are organized systematically by class granularity and balancing technique:

```text
Results/
├── classes_2/
│   ├── Without_SMOTE/
│   │   ├── model_results_global.csv
│   │   ├── model_results_per_class.csv
│   │   └── optuna_best_params.csv
│   └── With_SMOTE/
├── classes_4/
│   ├── Without_SMOTE/
│   └── With_SMOTE/
└── classes_6/
    ├── Without_SMOTE/
    └── With_SMOTE/
```
