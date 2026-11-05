from pathlib import Path
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
TARGET_COLUMN = "resolution_class"
RANDOM_SEED = 42

DATASETS_DIR = Path("datasets")


def generate_smote_for_all(target_classes=(2, 4, 6)):
    for num_classes in target_classes:
        print(f"\n=== Generating SMOTE datasets for {num_classes} classes ===")
        datasets = [
            ("Amazon", "Static", DATASETS_DIR / f"Amazon-{num_classes}" / "Static" / f"amazon_{num_classes}_static.csv"),
            ("Amazon", "Driving", DATASETS_DIR / f"Amazon-{num_classes}" / "Driving" / f"amazon_{num_classes}_driving.csv"),
            ("Netflix", "Static", DATASETS_DIR / f"Netflix-{num_classes}" / "Static" / f"netflix_{num_classes}_static.csv"),
            ("Netflix", "Driving", DATASETS_DIR / f"Netflix-{num_classes}" / "Driving" / f"netflix_{num_classes}_driving.csv"),
        ]

        for provider, movement_type, input_path in datasets:
            output_dir = DATASETS_DIR / f"{provider}-{num_classes}-SMOTE" / movement_type
            output_name = f"{provider.lower()}_{num_classes}_{movement_type.lower()}_smote.csv"
            output_path = output_dir / output_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if not input_path.exists():
                print(f"[AVISO] Arquivo {input_path} nao encontrado.")
                continue

            df = pd.read_csv(input_path)
            # Filter class 0
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip()
            df = df[~df[TARGET_COLUMN].isin(["0", "nan", ""])]

            for col in FEATURE_COLUMNS:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Impute median before SMOTE
            df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

            X = df[FEATURE_COLUMNS]
            y = df[TARGET_COLUMN]

            counts_before = Counter(y)
            min_class_count = min(counts_before.values())
            k_neighbors = min(5, min_class_count - 1) if min_class_count > 1 else 1

            if min_class_count > 1:
                smote = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_SEED)
                X_res, y_res = smote.fit_resample(X, y)
            else:
                X_res, y_res = X, y

            df_res = pd.concat([pd.DataFrame(X_res, columns=FEATURE_COLUMNS), pd.Series(y_res, name=TARGET_COLUMN)], axis=1)
            df_res.to_csv(output_path, index=False)

            counts_after = Counter(y_res)
            print(f"{input_path} -> {output_path}")
            print(f"  before: {dict(counts_before)}")
            print(f"  after : {dict(counts_after)}")


if __name__ == "__main__":
    generate_smote_for_all()
