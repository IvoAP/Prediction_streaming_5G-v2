import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
TARGET_COLUMN = "resolution_class"
DATASETS_DIR = Path("datasets")

CLASSES_BY_CONFIG = {
    6: ["144p", "360p", "480p", "720p", "1080p", "4K"],
    4: ["SD", "HD", "FHD", "4K"],
    2: ["SD", "HD"],
}

# Alias for default 6 classes
RESOLUTION_CLASSES = CLASSES_BY_CONFIG[6]


def get_dataset_map(num_classes: int = 6, use_smote: bool = False):
    """
    Returns paths to standard or SMOTE datasets for a given number of classes.
    """
    suffix = f"-{num_classes}-SMOTE" if use_smote else f"-{num_classes}"
    smote_tag = "_smote" if use_smote else ""

    return {
        "Amazon Static": DATASETS_DIR / f"Amazon{suffix}" / "Static" / f"amazon_{num_classes}_static{smote_tag}.csv",
        "Amazon Driving": DATASETS_DIR / f"Amazon{suffix}" / "Driving" / f"amazon_{num_classes}_driving{smote_tag}.csv",
        "Netflix Static": DATASETS_DIR / f"Netflix{suffix}" / "Static" / f"netflix_{num_classes}_static{smote_tag}.csv",
        "Netflix Driving": DATASETS_DIR / f"Netflix{suffix}" / "Driving" / f"netflix_{num_classes}_driving{smote_tag}.csv",
    }


# Backwards compatibility dictionaries
DATASETS_STANDARD = get_dataset_map(num_classes=6, use_smote=False)
DATASETS_SMOTE = get_dataset_map(num_classes=6, use_smote=True)
DATASETS = DATASETS_STANDARD


def load_and_preprocess_dataset(
    dataset_key: str,
    num_classes: int = 6,
    use_smote: bool = False,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Loads dataset for specified class configuration (2, 4, or 6),
    filters out invalid rows (class '0'), converts features to numeric,
    and splits into train/test with stratification.
    """
    dataset_map = get_dataset_map(num_classes=num_classes, use_smote=use_smote)

    if dataset_key not in dataset_map:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(dataset_map.keys())}")

    file_path = dataset_map[dataset_key]
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")

    df = pd.read_csv(file_path)

    valid_classes = CLASSES_BY_CONFIG.get(num_classes)
    if not valid_classes:
        raise ValueError(f"Unsupported num_classes: {num_classes}. Choose from [2, 4, 6].")

    # Filter out class 0 / '0' / invalid target values
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
    df_filtered = df[df[TARGET_COLUMN].isin(valid_classes)].copy()

    for col in FEATURE_COLUMNS:
        df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce")

    X = df_filtered[FEATURE_COLUMNS]
    y = df_filtered[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
