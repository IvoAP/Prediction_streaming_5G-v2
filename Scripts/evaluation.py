import time
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    log_loss,
    roc_auc_score,
)

from Scripts.data_loader import CLASSES_BY_CONFIG


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
):
    """
    Fits the model, evaluates overall performance metrics, measures training/inference speed,
    and calculates per-class Precision, Recall, and F1-score for all classes of the target configuration.
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
                # Binary ROC-AUC
                pos_idx = list(model.classes_).index(target_classes[1])
                roc_auc = roc_auc_score((y_test == target_classes[1]).astype(int), y_proba[:, pos_idx])
            else:
                # Multiclass OvR ROC-AUC
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

    return global_result, per_class_rows


def save_results(
    global_results: list,
    per_class_results: list,
    best_params_list: list,
    num_classes: int = 6,
    use_smote: bool = False,
):
    """
    Saves results to CSV files partitioned by class count and SMOTE status.
    E.g.: Results/classes_6/With_SMOTE/ or Results/classes_4/Without_SMOTE/
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

    if per_class_results:
        df_per_class = pd.DataFrame(per_class_results)
        df_per_class.to_csv(per_class_csv, index=False)
        print(f"-> Per-class metrics saved to: {per_class_csv}")

    if best_params_list:
        df_params = pd.DataFrame(best_params_list)
        df_params.to_csv(best_params_csv, index=False)
        print(f"-> Optuna best parameters saved to: {best_params_csv}")
