from pathlib import Path
import pandas as pd
import numpy as np

DATASETS_DIR = Path("datasets")
RAW_DATASETS = [
    ("Amazon", "Static", DATASETS_DIR / "Amazon" / "Static" / "amazon_static.csv"),
    ("Amazon", "Driving", DATASETS_DIR / "Amazon" / "Driving" / "amazon_driving.csv"),
    ("Netflix", "Static", DATASETS_DIR / "Netflix" / "Static" / "netflix_static.csv"),
    ("Netflix", "Driving", DATASETS_DIR / "Netflix" / "Driving" / "netflix_driving.csv"),
]

FEATURE_COLUMNS = ["RSRP", "RSRQ", "SNR", "CQI", "RSSI"]
TARGET_COLUMN = "resolution_class"
OUTPUT_DIR = Path("Results") / "dataset_statistics"


def generate_dataset_characterization():
    """
    Computes summary tables addressing reviewers' concerns:
    1. Sample counts per scenario (raw, valid active, and filtered).
    2. Distribution per class before and after SMOTE (for 2, 4, and 6 classes).
    3. Descriptive statistics of the RF physical layer features (Mean, Std, Min, Max).
    4. Mobility characteristics (Speed statistics for Static vs Driving).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 70)
    print(" GERANDO ESTATÍSTICAS E CARACTERIZAÇÃO DOS DATASETS")
    print("=" * 70)

    # 1. Raw Dataset and Mobility Stats
    mobility_summary = []
    rf_feature_stats = []

    for provider, movement_type, csv_path in RAW_DATASETS:
        if not csv_path.exists():
            continue
        df_raw = pd.read_csv(csv_path)
        total_raw_samples = len(df_raw)

        # Speed stats if present
        speed_mean, speed_std, speed_max = np.nan, np.nan, np.nan
        if "Speed" in df_raw.columns:
            s_clean = pd.to_numeric(df_raw["Speed"], errors="coerce")
            speed_mean = s_clean.mean()
            speed_std = s_clean.std()
            speed_max = s_clean.max()

        # Clean RF features
        df_rf = df_raw[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")

        mobility_summary.append({
            "Provider": provider,
            "Movement": movement_type,
            "Total_Raw_Samples": total_raw_samples,
            "Speed_Mean_kmh": round(speed_mean, 2) if not np.isnan(speed_mean) else "N/A",
            "Speed_Std_kmh": round(speed_std, 2) if not np.isnan(speed_std) else "N/A",
            "Speed_Max_kmh": round(speed_max, 2) if not np.isnan(speed_max) else "N/A",
        })

        for feat in FEATURE_COLUMNS:
            feat_series = df_rf[feat].dropna()
            rf_feature_stats.append({
                "Provider": provider,
                "Movement": movement_type,
                "Feature": feat,
                "Mean": round(feat_series.mean(), 2),
                "Std": round(feat_series.std(), 2),
                "Min": round(feat_series.min(), 2),
                "25%": round(feat_series.quantile(0.25), 2),
                "Median": round(feat_series.median(), 2),
                "75%": round(feat_series.quantile(0.75), 2),
                "Max": round(feat_series.max(), 2),
            })

    df_mobility = pd.DataFrame(mobility_summary)
    df_mobility.to_csv(OUTPUT_DIR / "scenario_mobility_summary.csv", index=False)
    print(f"-> Resumo de mobilidade salvo em: {OUTPUT_DIR / 'scenario_mobility_summary.csv'}")

    df_rf_stats = pd.DataFrame(rf_feature_stats)
    df_rf_stats.to_csv(OUTPUT_DIR / "rf_features_descriptive_stats.csv", index=False)
    print(f"-> Estatísticas descritivas das RF features salvas em: {OUTPUT_DIR / 'rf_features_descriptive_stats.csv'}")

    # 2. Class Distributions Before and After SMOTE (for 2, 4, 6 classes)
    class_distribution_rows = []

    for num_classes in [2, 4, 6]:
        for provider, movement_type, _ in RAW_DATASETS:
            std_path = DATASETS_DIR / f"{provider}-{num_classes}" / movement_type / f"{provider.lower()}_{num_classes}_{movement_type.lower()}.csv"
            smote_path = DATASETS_DIR / f"{provider}-{num_classes}-SMOTE" / movement_type / f"{provider.lower()}_{num_classes}_{movement_type.lower()}_smote.csv"

            if not std_path.exists():
                continue

            df_std = pd.read_csv(std_path)
            df_std[TARGET_COLUMN] = df_std[TARGET_COLUMN].astype(str).str.strip()
            df_std_filtered = df_std[~df_std[TARGET_COLUMN].isin(["0", "nan", ""])].copy()

            counts_before = df_std_filtered[TARGET_COLUMN].value_counts().to_dict()
            total_before = len(df_std_filtered)

            counts_after = {}
            total_after = 0
            if smote_path.exists():
                df_smote = pd.read_csv(smote_path)
                df_smote[TARGET_COLUMN] = df_smote[TARGET_COLUMN].astype(str).str.strip()
                counts_after = df_smote[TARGET_COLUMN].value_counts().to_dict()
                total_after = len(df_smote)

            for cls_name, count_b in counts_before.items():
                pct_b = (count_b / total_before) * 100 if total_before > 0 else 0
                count_a = counts_after.get(cls_name, 0)
                pct_a = (count_a / total_after) * 100 if total_after > 0 else 0

                class_distribution_rows.append({
                    "Config_Classes": num_classes,
                    "Provider": provider,
                    "Movement": movement_type,
                    "Class": cls_name,
                    "Count_Before_SMOTE": count_b,
                    "Percentage_Before_SMOTE": round(pct_b, 2),
                    "Count_After_SMOTE": count_a,
                    "Percentage_After_SMOTE": round(pct_a, 2),
                })

    df_class_dist = pd.DataFrame(class_distribution_rows)
    df_class_dist.to_csv(OUTPUT_DIR / "class_distribution_pre_post_smote.csv", index=False)
    print(f"-> Distribuição de classes (Pré e Pós SMOTE) salva em: {OUTPUT_DIR / 'class_distribution_pre_post_smote.csv'}")

    print("\n" + "=" * 70)
    print(" CARACTERIZAÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    generate_dataset_characterization()

