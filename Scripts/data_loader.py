import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
TARGET_COLUMN = "resolution_class"
RESOLUTION_CLASSES = ["144p", "360p", "480p", "720p", "1080p", "4K"]

DATASETS_STANDARD = {
    "Amazon Static": Path("Amazon/Static/amazon_static.csv"),
    "Amazon Driving": Path("Amazon/Driving/amazon_driving.csv"),
    "Netflix Static": Path("Netflix/Static/netflix_static.csv"),
    "Netflix Driving": Path("Netflix/Driving/netflix_driving.csv"),
}

DATASETS_SMOTE = {
    "Amazon Static": Path("Amazon-6-SMOTE/Static/amazon_6_static_smote.csv"),
    "Amazon Driving": Path("Amazon-6-SMOTE/Driving/amazon_6_driving_smote.csv"),
    "Netflix Static": Path("Netflix-6-SMOTE/Static/netflix_6_static_smote.csv"),
    "Netflix Driving": Path("Netflix-6-SMOTE/Driving/netflix_6_driving_smote.csv"),
}

# Alias for backwards compatibility
DATASETS = DATASETS_STANDARD


def load_and_preprocess_dataset(dataset_key: str, use_smote: bool = False, test_size: float = 0.20, random_state: int = 42):
    """
    Loads standard or SMOTE dataset, filters out invalid rows (class '0'),
    converts features to numeric, and splits into train/test with stratification.
    """
    dataset_map = DATASETS_SMOTE if use_smote else DATASETS_STANDARD

    if dataset_key not in dataset_map:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(dataset_map.keys())}")

    file_path = dataset_map[dataset_key]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    # Filter out class 0 / '0' / missing target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df_filtered = df[df[TARGET_COLUMN].isin(RESOLUTION_CLASSES)].copy()

    for col in FEATURE_COLUMNS:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    X = df_filtered[FEATURE_COLUMNS]
    y = df_filtered[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
