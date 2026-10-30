import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)

from Scripts.data_loader import CLASSES_BY_CONFIG


def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    target_classes: list,
    model_name: str,
    dataset_name: str,
    output_dir: Path,
):
    """
    Computes and plots both normalized (percentages) and raw confusion matrices,
    saving them as publication-ready PNG and PDF figures.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_ds = dataset_name.replace(" ", "_").lower()
    clean_m = model_name.replace(" ", "_").lower()

    # Normalized Confusion Matrix (by true rows)
    cm_norm = confusion_matrix(y_true, y_pred, labels=target_classes, normalize="true")

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_classes,
        yticklabels=target_classes,
        cbar=True,
        annot_kws={"size": 11, "weight": "bold"},
    )
    plt.title(f"Normalized Confusion Matrix\n{model_name} - {dataset_name}", fontsize=12, pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.tight_layout()

    fig_png = output_dir / f"cm_{clean_ds}_{clean_m}.png"
    fig_pdf = output_dir / f"cm_{clean_ds}_{clean_m}.pdf"
    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf, dpi=300)
    plt.close()


def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    model_name: str,
    dataset_name: str,
    num_classes: int = 6,
    use_smote: bool = False,
    generate_plots: bool = False,
    output_dir: Path = None,
):
    """
    Fits model, calculates performance metrics, latencies, per-class metrics,
    and optionally renders publication-ready confusion matrices.
    """
    target_classes = CLASSES_BY_CONFIG.get(num_classes)
    if not target_classes:
        raise ValueError(f"Unsupported num_classes: {num_classes}")

    # Training
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    # Inference
    start_inference = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = time.perf_counter() - start_inference
    inference_time_per_sample = inference_time / len(X_test) if len(X_test) > 0 else 0.0

    # Probabilities for log_loss / roc_auc
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        pass

    # Global Metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    loss = np.nan
    roc_auc = np.nan

    if y_proba is not None:
        try:
            loss = log_loss(y_test, y_proba, labels=model.classes_)
        except Exception:
            pass

        try:
            if num_classes == 2:
                pos_idx = list(model.classes_).index(target_classes[1])
                roc_auc = roc_auc_score((y_test == target_classes[1]).astype(int), y_proba[:, pos_idx])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        except Exception:
            pass

    global_result = {
        "dataset": dataset_name,
        "model": model_name,
        "num_classes": num_classes,
        "use_smote": use_smote,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "f1_score_macro": f1_macro,
        "f1_score_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "log_loss": loss,
        "train_time_seconds": train_time,
        "inference_time_seconds": inference_time,
        "inference_time_per_sample_seconds": inference_time_per_sample,
    }

    # Per-Class Metrics
    per_class_precision = precision_score(y_test, y_pred, labels=target_classes, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, labels=target_classes, average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, labels=target_classes, average=None, zero_division=0)

    per_class_rows = []
    for idx, cls in enumerate(target_classes):
        p = per_class_precision[idx]
        r = per_class_recall[idx]
        f = per_class_f1[idx]

        global_result[f"precision_{cls}"] = p
        global_result[f"recall_{cls}"] = r
        global_result[f"f1_{cls}"] = f

        per_class_rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "num_classes": num_classes,
            "use_smote": use_smote,
            "resolution_class": cls,
            "precision": p,
            "recall": r,
            "f1_score": f,
        })

    # Plot Confusion Matrix if requested
    if generate_plots and output_dir is not None:
        plot_and_save_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            target_classes=target_classes,
            model_name=model_name,
            dataset_name=dataset_name,
            output_dir=output_dir / "confusion_matrices",
        )

    return global_result, per_class_rows


def plot_comparison_bar_charts(results_csv: Path, output_dir: Path):
    """
    Generates grouped bar charts comparing Macro F1 and Balanced Accuracy across models and scenarios.
    """
    if not results_csv.exists():
        return

    df = pd.read_csv(results_csv)
    if df.empty or "f1_score_macro" not in df.columns:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Macro F1 Chart
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df,
        x="dataset",
        y="f1_score_macro",
        hue="model",
        palette="viridis",
    )
    plt.title("Macro F1-Score Comparison Across Scenarios", fontsize=13, weight="bold")
    plt.ylabel("Macro F1-Score", fontsize=11)
    plt.xlabel("Scenario", fontsize=11)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.tight_layout()

    plt.savefig(output_dir / "comparison_macro_f1.png", dpi=300)
    plt.savefig(output_dir / "comparison_macro_f1.pdf", dpi=300)
    plt.close()


def save_results(
    global_results: list,
    per_class_results: list,
    best_params_list: list,
    num_classes: int = 6,
    use_smote: bool = False,
    generate_summary_plots: bool = True,
):
    """
    Saves results to CSV files and optionally renders comparative charts.
    """
    smote_dir = "With_SMOTE" if use_smote else "Without_SMOTE"
    output_dir = Path("Results") / f"classes_{num_classes}" / smote_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    global_csv = output_dir / "model_results_global.csv"
    per_class_csv = output_dir / "model_results_per_class.csv"
    best_params_csv = output_dir / "optuna_best_params.csv"

    if global_results:
        df_global = pd.DataFrame(global_results)
        df_global.sort_values(by=["dataset", "balanced_accuracy"], ascending=[True, False], inplace=True)
        df_global.to_csv(global_csv, index=False)
        print(f"-> Global metrics saved to: {global_csv}")

        if generate_summary_plots:
            plot_comparison_bar_charts(results_csv=global_csv, output_dir=output_dir / "figures")
            print(f"-> Summary comparison plots saved to: {output_dir / 'figures'}")

    if per_class_results:
        df_per_class = pd.DataFrame(per_class_results)
        df_per_class.to_csv(per_class_csv, index=False)
        print(f"-> Per-class metrics saved to: {per_class_csv}")

    if best_params_list:
        df_params = pd.DataFrame(best_params_list)
        df_params.to_csv(best_params_csv, index=False)
        print(f"-> Optuna best parameters saved to: {best_params_csv}")
