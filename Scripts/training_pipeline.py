import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from Scripts.data_loader import (
    CLASSES_BY_CONFIG,
    get_dataset_map,
    load_and_preprocess_dataset,
)
from Scripts.models_optuna import MODEL_OPTIMIZERS, build_final_model
from Scripts.evaluation import evaluate_model, save_results
from Scripts.dataset_statistics import generate_dataset_characterization
from Scripts.treatment_data import generate_all_resolution_datasets
from Scripts.smote_treatment import generate_smote_for_all

# Exclusively Tree-Based Models (AdaBoost removed due to poor multi-class performance)
TREE_MODELS = [
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "LightGBM",
    "XGBoost",
]


def run_experiment(
    num_classes: int = 6,
    datasets_to_run=None,
    models_to_run=None,
    n_trials: int = 15,
    use_smote: bool = False,
    seeds: list = None,
    generate_plots: bool = True,
):
    """
    Orchestrates dataset loading, Optuna tuning, training across seeds,
    and performance evaluation with optional confusion matrices and charts.
    """
    if seeds is None:
        seeds = [42]

    dataset_map = get_dataset_map(num_classes=num_classes, use_smote=use_smote)
    if datasets_to_run is None:
        datasets_to_run = list(dataset_map.keys())
    if models_to_run is None:
        models_to_run = TREE_MODELS

    global_results = []
    per_class_results = []
    best_params_list = []

    mode_str = "WITH SMOTE" if use_smote else "WITHOUT SMOTE"
    output_base_dir = Path("Results") / f"classes_{num_classes}" / ("With_SMOTE" if use_smote else "Without_SMOTE")

    print("\n" + "=" * 75)
    print(f" PIPELINE TREE-BASED MODELS - {num_classes} CLASSES ({mode_str})")
    print("=" * 75)
    print(f"Target Classes Config : {num_classes} {CLASSES_BY_CONFIG.get(num_classes)}")
    print(f"Selected Scenarios    : {datasets_to_run}")
    print(f"Tree-Based Models     : {models_to_run}")
    print(f"Optuna Trials         : {n_trials}")
    print(f"Random Seeds          : {seeds}")
    print(f"Dataset Mode          : {mode_str}")
    print("=" * 75)

    for dataset_name in datasets_to_run:
        print(f"\n============================================================")
        print(f" LOADING DATASET ({num_classes} classes | {mode_str}): {dataset_name}")
        print(f"============================================================")

        for model_name in models_to_run:
            if model_name not in MODEL_OPTIMIZERS:
                print(f"[WARNING] Model '{model_name}' not supported. Skipping...")
                continue

            # Run Optuna on base seed (seeds[0])
            try:
                X_train_opt, _, y_train_opt, _ = load_and_preprocess_dataset(
                    dataset_key=dataset_name,
                    num_classes=num_classes,
                    use_smote=use_smote,
                    random_state=seeds[0],
                )
            except Exception as e:
                print(f"[ERROR] Failed to load dataset '{dataset_name}': {e}")
                break

            print(f"\n------------------------------------------------------------")
            print(f"[OPTUNA] Optimizing ({model_name}) for {num_classes} classes ({n_trials} trials)")
            print(f"------------------------------------------------------------")

            optimizer_fn = MODEL_OPTIMIZERS[model_name]
            best_params = optimizer_fn(X_train_opt, y_train_opt, n_trials=n_trials)
            print(f"[OK] Best parameters [{model_name}]: {best_params}")

            best_params_list.append({
                "dataset": dataset_name,
                "model": model_name,
                "num_classes": num_classes,
                "use_smote": use_smote,
                "best_params": str(best_params),
            })

            # Evaluate across all seeds
            seed_metrics = []
            for seed in seeds:
                X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
                    dataset_key=dataset_name,
                    num_classes=num_classes,
                    use_smote=use_smote,
                    random_state=seed,
                )

                final_model = build_final_model(model_name, best_params)
                gen_plot = generate_plots and (seed == seeds[0])

                global_res, per_class_res = evaluate_model(
                    model=final_model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    num_classes=num_classes,
                    use_smote=use_smote,
                    generate_plots=gen_plot,
                    output_dir=output_base_dir,
                )

                global_res["seed"] = seed
                for r in per_class_res:
                    r["seed"] = seed

                seed_metrics.append(global_res)
                per_class_results.extend(per_class_res)

            # Average performance across seeds
            if len(seeds) > 1:
                df_seeds = pd.DataFrame(seed_metrics)
                mean_res = seed_metrics[0].copy()
                mean_res["seed"] = "mean"
                mean_res["accuracy"] = df_seeds["accuracy"].mean()
                mean_res["balanced_accuracy"] = df_seeds["balanced_accuracy"].mean()
                mean_res["f1_score_macro"] = df_seeds["f1_score_macro"].mean()
                mean_res["f1_score_weighted"] = df_seeds["f1_score_weighted"].mean()
                mean_res["mcc"] = df_seeds["mcc"].mean()
                mean_res["cohen_kappa"] = df_seeds["cohen_kappa"].mean()
                mean_res["train_time_seconds"] = df_seeds["train_time_seconds"].mean()
                mean_res["inference_time_seconds"] = df_seeds["inference_time_seconds"].mean()
                mean_res["balanced_accuracy_std"] = df_seeds["balanced_accuracy"].std()
                mean_res["f1_macro_std"] = df_seeds["f1_score_macro"].std()

                global_results.extend(seed_metrics)
                global_results.append(mean_res)

                print(f"   [MEAN +- STD] {model_name}:")
                print(f"     Bal Acc : {mean_res['balanced_accuracy']:.4f} +- {mean_res['balanced_accuracy_std']:.4f}")
                print(f"     Macro F1: {mean_res['f1_score_macro']:.4f} +- {mean_res['f1_macro_std']:.4f}")
            else:
                global_results.append(seed_metrics[0])
                res = seed_metrics[0]
                print(f"   Accuracy             : {res['accuracy']:.4f}")
                print(f"   Balanced Accuracy    : {res['balanced_accuracy']:.4f}")
                print(f"   F1-Score (Macro)     : {res['f1_score_macro']:.4f}")
                print(f"   F1-Score (Weighted)  : {res['f1_score_weighted']:.4f}")
                print(f"   MCC                  : {res['mcc']:.4f}")
                print(f"   Cohen's Kappa        : {res['cohen_kappa']:.4f}")
                if not pd.isna(res["roc_auc"]):
                    print(f"   ROC-AUC              : {res['roc_auc']:.4f}")
                print(f"   Training Time (s)    : {res['train_time_seconds']:.2f}s")
                print(f"   Inference Time (s)   : {res['inference_time_seconds']:.4f}s")

    # Save all results to CSV & charts
    save_results(
        global_results=global_results,
        per_class_results=per_class_results,
        best_params_list=best_params_list,
        num_classes=num_classes,
        use_smote=use_smote,
        generate_summary_plots=generate_plots,
    )

    print("\n============================================================")
    print(f" EXPERIMENTS ({num_classes} CLASSES | {mode_str}) FINISHED SUCCESSFULLY!")
    print("============================================================\n")


def run_full_benchmark(n_trials: int = 15, seeds: list = None):
    """
    Runs the complete benchmark: all granularities (6, 4, 2)
    both WITHOUT SMOTE and WITH SMOTE across all tree models and scenarios.
    """
    if seeds is None:
        seeds = [42]

    print("\n" + "=" * 75)
    print(" STARTING COMPLETE BENCHMARK")
    print(" Granularities : 6, 4, and 2 Classes")
    print(" Modes         : WITHOUT SMOTE and WITH SMOTE")
    print(" Models        : DecisionTree, RandomForest, ExtraTrees,")
    print("                 GradientBoosting, LightGBM, XGBoost")
    print("=" * 75)

    for use_smote in [False, True]:
        for n_cls in [6, 4, 2]:
            run_experiment(
                num_classes=n_cls,
                n_trials=n_trials,
                use_smote=use_smote,
                seeds=seeds,
                generate_plots=True,
            )

    print("\n" + "=" * 75)
    print(" COMPLETE BENCHMARK COMPLETED SUCCESSFULLY!")
    print(" All reports, plots, and confusion matrices saved to 'Results/'")
    print("=" * 75 + "\n")


def display_menu():
    """
    Displays the CLI interactive terminal menu in English without emojis.
    """
    n_trials = 15
    use_smote = False
    current_num_classes = 6
    multi_seeds = False

    while True:
        mode_label = "WITH SMOTE" if use_smote else "WITHOUT SMOTE"
        seeds = [42, 100, 2024] if multi_seeds else [42]
        seed_label = f"Multiple ({seeds})" if multi_seeds else "Single (42)"

        print("\n" + "=" * 75)
        print("    5G STREAMING PREDICTION - TREE-BASED MODELS PIPELINE")
        print("=" * 75)
        print(f" Current Granularity : [{current_num_classes} Classes] {CLASSES_BY_CONFIG.get(current_num_classes)}")
        print(f" Current Mode        : [{mode_label}]")
        print(f" Optuna Trials       : [{n_trials}]")
        print(f" Seeds Evaluation    : [{seed_label}]")
        print("-" * 75)
        print(" [1] Run Pipeline for Current Configuration")
        print(" [2] Run All Scenarios and Classes (6, 4, and 2) in Current Mode")
        print(" [3] Run Complete Benchmark (6, 4, 2 Classes x WITHOUT and WITH SMOTE)")
        print(" [4] Select Class Granularity (2, 4, or 6)")
        print(" [5] Select a Specific Scenario")
        print(" [6] Select a Specific Tree Model")
        print(" [7] Toggle Dataset Mode (WITHOUT SMOTE / WITH SMOTE)")
        print(" [8] Set Number of Optuna Trials (Current: {})".format(n_trials))
        print(" [9] Toggle Multi-Seed Evaluation (Mean +- Std)")
        print("-" * 75)
        print(" [10] Generate Resolution Datasets (2, 4, and 6 Classes from raw data)")
        print(" [11] Generate SMOTE Datasets (Balanced for 2, 4, and 6 Classes)")
        print(" [12] Generate Dataset Characterization and Descriptive Statistics Tables")
        print(" [13] Run Quick Test Run (2 Trials / Confusion Matrix)")
        print(" [0] Exit")
        print("=" * 75)

        choice = input("Enter choice: ").strip()

        if choice == "1":
            run_experiment(
                num_classes=current_num_classes,
                n_trials=n_trials,
                use_smote=use_smote,
                seeds=seeds,
            )
        elif choice == "2":
            for n_cls in [6, 4, 2]:
                run_experiment(
                    num_classes=n_cls,
                    n_trials=n_trials,
                    use_smote=use_smote,
                    seeds=seeds,
                )
        elif choice == "3":
            run_full_benchmark(n_trials=n_trials, seeds=seeds)
        elif choice == "4":
            print("\nSelect class granularity:")
            print(" [1] 2 Classes (SD, HD)")
            print(" [2] 4 Classes (SD, HD, FHD, 4K)")
            print(" [3] 6 Classes (144p, 360p, 480p, 720p, 1080p, 4K)")
            cls_opt = input("Choice: ").strip()
            if cls_opt == "1":
                current_num_classes = 2
            elif cls_opt == "2":
                current_num_classes = 4
            elif cls_opt == "3":
                current_num_classes = 6
            else:
                print("[ERROR] Invalid choice.")
            print(f"Configuration updated to: {current_num_classes} classes.")
        elif choice == "5":
            print("\nAvailable scenarios:")
            datasets_list = list(get_dataset_map(current_num_classes, use_smote).keys())
            for idx, ds in enumerate(datasets_list, 1):
                print(f"  [{idx}] {ds}")
            ds_choice = input("Select scenario number: ").strip()
            if ds_choice.isdigit() and 1 <= int(ds_choice) <= len(datasets_list):
                selected_ds = [datasets_list[int(ds_choice) - 1]]
                run_experiment(
                    num_classes=current_num_classes,
                    datasets_to_run=selected_ds,
                    n_trials=n_trials,
                    use_smote=use_smote,
                    seeds=seeds,
                )
            else:
                print("[ERROR] Invalid scenario choice.")
        elif choice == "6":
            print("\nAvailable tree-based models:")
            for idx, m in enumerate(TREE_MODELS, 1):
                print(f"  [{idx}] {m}")
            m_choice = input("Select model number: ").strip()
            if m_choice.isdigit() and 1 <= int(m_choice) <= len(TREE_MODELS):
                selected_m = [TREE_MODELS[int(m_choice) - 1]]
                run_experiment(
                    num_classes=current_num_classes,
                    models_to_run=selected_m,
                    n_trials=n_trials,
                    use_smote=use_smote,
                    seeds=seeds,
                )
            else:
                print("[ERROR] Invalid model choice.")
        elif choice == "7":
            use_smote = not use_smote
            new_mode = "WITH SMOTE" if use_smote else "WITHOUT SMOTE"
            print(f"Dataset mode switched to: {new_mode}")
        elif choice == "8":
            val = input("Enter new number of Optuna trials: ").strip()
            if val.isdigit() and int(val) > 0:
                n_trials = int(val)
                print(f"Number of trials updated to: {n_trials}")
            else:
                print("[ERROR] Invalid input.")
        elif choice == "9":
            multi_seeds = not multi_seeds
            status_str = "ENABLED (42, 100, 2024)" if multi_seeds else "DISABLED (42)"
            print(f"Multi-seed evaluation: {status_str}")
        elif choice == "10":
            print("\n[GENERATION] Generating resolution datasets (2, 4, and 6 classes) from raw data...")
            generate_all_resolution_datasets()
            print("Datasets generated successfully!")
        elif choice == "11":
            print("\n[SMOTE] Generating balanced SMOTE datasets for 2, 4, and 6 classes...")
            generate_smote_for_all()
            print("SMOTE datasets generated successfully!")
        elif choice == "12":
            generate_dataset_characterization()
        elif choice == "13":
            print("\n[TEST] Running quick validation test run with figure generation...")
            run_experiment(
                num_classes=current_num_classes,
                datasets_to_run=["Amazon Static"],
                models_to_run=["DecisionTree"],
                n_trials=2,
                use_smote=use_smote,
                seeds=[42],
                generate_plots=True,
            )
        elif choice == "0":
            print("Exiting pipeline. Goodbye!")
            sys.exit(0)
        else:
            print("[ERROR] Invalid option. Please try again.")


def main():
    parser = argparse.ArgumentParser(description="Tree-Based ML Pipeline for 5G Video Streaming Quality Prediction")
    parser.add_argument("--test-run", action="store_true", help="Run a quick validation test run.")
    parser.add_argument("--all", action="store_true", help="Run the pipeline for specified classes without interactive menu.")
    parser.add_argument("--classes", type=int, choices=[2, 4, 6], default=6, help="Target number of resolution classes (2, 4, or 6).")
    parser.add_argument("--all-classes", action="store_true", help="Iteratively run across 6, 4, and 2 classes in current mode.")
    parser.add_argument("--full-benchmark", action="store_true", help="Run full benchmark across 6, 4, and 2 classes, WITHOUT and WITH SMOTE.")
    parser.add_argument("--n-trials", type=int, default=15, help="Number of Optuna optimization trials (default: 15).")
    parser.add_argument("--smote", action="store_true", help="Use SMOTE-balanced training datasets.")
    parser.add_argument("--multi-seeds", action="store_true", help="Evaluate across multiple seeds (42, 100, 2024) to report Mean +- Std.")
    parser.add_argument("--generate-datasets", action="store_true", help="Generate 2, 4, and 6 class datasets from raw measurements.")
    parser.add_argument("--generate-smote", action="store_true", help="Generate SMOTE balanced datasets for all granularities.")
    parser.add_argument("--dataset-stats", action="store_true", help="Generate dataset summary and descriptive statistics tables.")

    args = parser.parse_args()
    seeds = [42, 100, 2024] if args.multi_seeds else [42]

    if args.generate_datasets:
        generate_all_resolution_datasets()
    elif args.generate_smote:
        generate_smote_for_all()
    elif args.dataset_stats:
        generate_dataset_characterization()
    elif args.full_benchmark:
        run_full_benchmark(n_trials=args.n_trials, seeds=seeds)
    elif args.test_run:
        print("[TEST] Flag --test-run detected. Running quick validation test...")
        run_experiment(
            num_classes=args.classes,
            datasets_to_run=["Amazon Static"],
            models_to_run=["DecisionTree"],
            n_trials=2,
            use_smote=args.smote,
            seeds=seeds,
            generate_plots=True,
        )
    elif args.all_classes:
        print("[RUN] Executing pipeline for 6, 4, and 2 classes in selected mode...")
        for n_cls in [6, 4, 2]:
            run_experiment(num_classes=n_cls, n_trials=args.n_trials, use_smote=args.smote, seeds=seeds)
    elif args.all:
        print(f"[RUN] Flag --all detected ({args.classes} classes)...")
        run_experiment(num_classes=args.classes, n_trials=args.n_trials, use_smote=args.smote, seeds=seeds)
    else:
        display_menu()


if __name__ == "__main__":
    main()
