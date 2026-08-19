import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
TARGET_COLUMN = "resolution_class"
RESOLUTION_CLASSES = ["144p", "360p", "480p", "720p", "1080p", "4K"]

DATASETS = {
    "Amazon Static": Path("Amazon-6/Static/amazon_6_static.csv"),
    "Amazon Driving": Path("Amazon-6/Driving/amazon_6_driving.csv"),
    "Netflix Static": Path("Netflix-6/Static/netflix_6_static.csv"),
    "Netflix Driving": Path("Netflix-6/Driving/netflix_6_driving.csv"),
}


def load_and_preprocess_dataset(dataset_key: str, test_size: float = 0.20, random_state: int = 42, use_smote: bool = False):
    """
    Loads dataset, filters out non-streaming/invalid rows (class '0'),
    prepares features (RSRP, RSRQ, SNR, CQI, RSSI) and target (resolution_class),
    splits into train/test with stratification, and optionally applies SMOTE to train set.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(DATASETS.keys())}")

    file_path = DATASETS[dataset_key]
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

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train)
            smote = SMOTE(random_state=random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train)
            X_train = pd.DataFrame(X_train_res, columns=FEATURE_COLUMNS)
            y_train = pd.Series(y_train_res, name=TARGET_COLUMN)
        except ImportError:
            print("Warning: imbalanced-learn is not installed. Skipping SMOTE.")

    return X_train, X_test, y_train, y_test
