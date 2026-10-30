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

# Exclusively Tree-Based Models
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

    mode_str = "COM SMOTE" if use_smote else "SEM SMOTE"
    output_base_dir = Path("Results") / f"classes_{num_classes}" / ("With_SMOTE" if use_smote else "Without_SMOTE")

    print("\n" + "=" * 75)
    print(f" PIPELINE TREE-BASED MODELS - {num_classes} CLASSES ({mode_str})")
    print("=" * 75)
    print(f"Número de Classes     : {num_classes} {CLASSES_BY_CONFIG.get(num_classes)}")
    print(f"Cenários selecionados : {datasets_to_run}")
    print(f"Modelos (Tree-based)  : {models_to_run}")
    print(f"Trials do Optuna      : {n_trials}")
    print(f"Sementes aleatórias   : {seeds}")
    print(f"Modo Dataset          : {mode_str}")
    print("=" * 75)

    for dataset_name in datasets_to_run:
        print(f"\n============================================================")
        print(f" CARREGANDO DATASET ({num_classes} classes | {mode_str}): {dataset_name}")
        print(f"============================================================")

        for model_name in models_to_run:
            if model_name not in MODEL_OPTIMIZERS:
                print(f"[AVISO] Modelo '{model_name}' não suportado. Pulando...")
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
                print(f"[ERRO] Falha ao carregar dataset '{dataset_name}': {e}")
                break

            print(f"\n------------------------------------------------------------")
            print(f"[OPTUNA] Otimizando ({model_name}) para {num_classes} classes ({n_trials} trials)")
            print(f"------------------------------------------------------------")

            optimizer_fn = MODEL_OPTIMIZERS[model_name]
            best_params = optimizer_fn(X_train_opt, y_train_opt, n_trials=n_trials)
            print(f"[OK] Melhores parâmetros [{model_name}]: {best_params}")

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
                # Plot confusion matrix only for first seed to prevent overwriting
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
                mean_res["train_time_seconds"] = df_seeds["train_time_seconds"].mean()
                mean_res["inference_time_seconds"] = df_seeds["inference_time_seconds"].mean()
                mean_res["balanced_accuracy_std"] = df_seeds["balanced_accuracy"].std()
                mean_res["f1_macro_std"] = df_seeds["f1_score_macro"].std()

                global_results.extend(seed_metrics)
                global_results.append(mean_res)

                print(f"   [MÉDIA ± STD] {model_name}:")
                print(f"     Bal Acc : {mean_res['balanced_accuracy']:.4f} ± {mean_res['balanced_accuracy_std']:.4f}")
                print(f"     Macro F1: {mean_res['f1_score_macro']:.4f} ± {mean_res['f1_macro_std']:.4f}")
            else:
                global_results.append(seed_metrics[0])
                res = seed_metrics[0]
                print(f"   Acurácia             : {res['accuracy']:.4f}")
                print(f"   Acurácia Balanceada : {res['balanced_accuracy']:.4f}")
                print(f"   F1-Score (Macro)    : {res['f1_score_macro']:.4f}")
                print(f"   F1-Score (Weighted) : {res['f1_score_weighted']:.4f}")
                print(f"   MCC                  : {res['mcc']:.4f}")
                if not pd.isna(res["roc_auc"]):
                    print(f"   ROC-AUC              : {res['roc_auc']:.4f}")
                print(f"   Tempo Treino (s)    : {res['train_time_seconds']:.2f}s")
                print(f"   Tempo Inferência (s): {res['inference_time_seconds']:.4f}s")

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
    print(f" EXPERIMENTOS ({num_classes} CLASSES | {mode_str}) CONCLUIDOS!")
    print("============================================================\n")


def run_full_benchmark(n_trials: int = 15, seeds: list = None):
    """
    Runs the complete benchmark: all granularities (6, 4, 2)
    both WITHOUT SMOTE and WITH SMOTE across all tree models and scenarios.
    """
    if seeds is None:
        seeds = [42]

    print("\n" + "=" * 75)
    print(" INICIANDO BENCHMARK COMPLETO DO ARTIGO (TUDO)")
    print(" Granularidades : 6, 4 e 2 Classes")
    print(" Modos          : SEM SMOTE e COM SMOTE")
    print(" Modelos        : DecisionTree, RandomForest, ExtraTrees,")
    print("                  GradientBoosting, LightGBM, XGBoost")
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
    print(" BENCHMARK COMPLETO CONCLUÍDO COM SUCESSO!")
    print(" Todos os resultados e figuras foram salvos na pasta 'Results/'")
    print("=" * 75 + "\n")


def display_menu():
    """
    Displays the CLI interactive terminal menu.
    """
    n_trials = 15
    use_smote = False
    current_num_classes = 6
    multi_seeds = False

    while True:
        mode_label = "COM SMOTE" if use_smote else "SEM SMOTE"
        seeds = [42, 100, 2024] if multi_seeds else [42]
        seed_label = f"Múltiplas ({seeds})" if multi_seeds else "Única (42)"

        print("\n" + "=" * 75)
        print("    PIPELINE DE TREINAMENTO 5G - TREE MODELS & NOVO ARTIGO")
        print("=" * 75)
        print(f" Granularidade Atual   : [{current_num_classes} Classes] {CLASSES_BY_CONFIG.get(current_num_classes)}")
        print(f" Modo Atual de Dataset : [{mode_label}]")
        print(f" Trials do Optuna      : [{n_trials}]")
        print(f" Avaliação de Sementes : [{seed_label}]")
        print("-" * 75)
        print(" [1] Executar Pipeline para a configuração atual")
        print(" [2] Executar TODOS os Cenários e Classes (6, 4 e 2) no Modo Atual")
        print(" [3] 🔥 EXECUTAR BENCHMARK COMPLETO (6, 4 e 2 Classes x SEM SMOTE e COM SMOTE)")
        print(" [4] Selecionar Granularidade de Classes (2, 4 ou 6)")
        print(" [5] Selecionar um Cenário Específico")
        print(" [6] Selecionar um Modelo de Árvore Específico")
        print(" [7] Alternar Modo de Dataset (SEM SMOTE / COM SMOTE)")
        print(" [8] Configurar Quantidade de Trials do Optuna (Atual: {})".format(n_trials))
        print(" [9] Alternar Modo de Múltiplas Sementes (Média ± Desvio Padrão)")
        print("-" * 75)
        print(" [10] 🛠️  Gerar Datasets das Classes (2, 4 e 6 Classes a partir dos dados brutos)")
        print(" [11] ⚖️  Gerar Datasets Balanceados com SMOTE (para 2, 4 e 6 Classes)")
        print(" [12] 📊 Gerar Tabelas de Estatísticas e Caracterização do Dataset")
        print(" [13] 🧪 Rodar Teste Rápido de Validação (2 Trials / Matriz de Confusão)")
        print(" [0] Sair")
        print("=" * 75)

        choice = input("Escolha uma opção: ").strip()

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
            print("\nEscolha a quantidade de classes:")
            print(" [1] 2 Classes (SD, HD_Plus)")
            print(" [2] 4 Classes (Low_SD, SD_HD, FHD, 4K)")
            print(" [3] 6 Classes (144p, 360p, 480p, 720p, 1080p, 4K)")
            cls_opt = input("Opção: ").strip()
            if cls_opt == "1":
                current_num_classes = 2
            elif cls_opt == "2":
                current_num_classes = 4
            elif cls_opt == "3":
                current_num_classes = 6
            else:
                print("[ERRO] Opção inválida.")
            print(f"✅ Configuração definida para: {current_num_classes} classes.")
        elif choice == "5":
            print("\nCenários disponíveis:")
            datasets_list = list(get_dataset_map(current_num_classes, use_smote).keys())
            for idx, ds in enumerate(datasets_list, 1):
                print(f"  [{idx}] {ds}")
            ds_choice = input("Selecione o número do cenário: ").strip()
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
                print("[ERRO] Opção de cenário inválida.")
        elif choice == "6":
            print("\nModelos baseados em árvore disponíveis:")
            for idx, m in enumerate(TREE_MODELS, 1):
                print(f"  [{idx}] {m}")
            m_choice = input("Selecione o número do modelo: ").strip()
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
                print("[ERRO] Opção de modelo inválida.")
        elif choice == "7":
            use_smote = not use_smote
            new_mode = "COM SMOTE" if use_smote else "SEM SMOTE"
            print(f"✅ Modo de dataset alterado para: {new_mode}")
        elif choice == "8":
            val = input("Digite o novo número de trials do Optuna: ").strip()
            if val.isdigit() and int(val) > 0:
                n_trials = int(val)
                print(f"✅ Número de trials alterado para: {n_trials}")
            else:
                print("[ERRO] Valor inválido.")
        elif choice == "9":
            multi_seeds = not multi_seeds
            print(f"✅ Múltiplas sementes: {'ATIVADO (42, 100, 2024)' if multi_seeds else 'DESATIVADO (42)'}")
        elif choice == "10":
            print("\n[GERAÇÃO] Gerando datasets para 2, 4 e 6 classes a partir dos dados brutos...")
            generate_all_resolution_datasets()
            print("✅ Datasets gerados com sucesso!")
        elif choice == "11":
            print("\n[SMOTE] Gerando datasets balanceados com SMOTE para 2, 4 e 6 classes...")
            generate_smote_for_all()
            print("✅ Datasets com SMOTE gerados com sucesso!")
        elif choice == "12":
            generate_dataset_characterization()
        elif choice == "13":
            print("\n[TEST] Executando teste rápido de validação com geração de figuras...")
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
            print("Encerrando pipeline. Até logo!")
            sys.exit(0)
        else:
            print("[ERRO] Opção inválida. Tente novamente.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento Tree-Based para Predição de Streaming 5G")
    parser.add_argument("--test-run", action="store_true", help="Executa um teste rápido para verificar funcionamento.")
    parser.add_argument("--all", action="store_true", help="Executa o pipeline completo sem menu.")
    parser.add_argument("--classes", type=int, choices=[2, 4, 6], default=6, help="Número de classes de resolução (2, 4 ou 6).")
    parser.add_argument("--all-classes", action="store_true", help="Executa iterativamente para 6, 4 e 2 classes no modo atual.")
    parser.add_argument("--full-benchmark", action="store_true", help="Executa benchmark completo: 6, 4 e 2 classes, SEM SMOTE e COM SMOTE.")
    parser.add_argument("--n-trials", type=int, default=15, help="Número de trials para Optuna (padrão: 15).")
    parser.add_argument("--smote", action="store_true", help="Utiliza os datasets gerados com SMOTE.")
    parser.add_argument("--multi-seeds", action="store_true", help="Avalia em 3 sementes (42, 100, 2024) calculando média e desvio padrão.")
    parser.add_argument("--generate-datasets", action="store_true", help="Gera os datasets derivados de 2, 4 e 6 classes.")
    parser.add_argument("--generate-smote", action="store_true", help="Gera os datasets balanceados com SMOTE.")
    parser.add_argument("--dataset-stats", action="store_true", help="Gera as tabelas de caracterização do dataset.")

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
        print("[TEST] Flag --test-run detectada. Executando teste de validação...")
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
        print("[RUN] Executando pipeline para 6, 4 e 2 classes no modo selecionado...")
        for n_cls in [6, 4, 2]:
            run_experiment(num_classes=n_cls, n_trials=args.n_trials, use_smote=args.smote, seeds=seeds)
    elif args.all:
        print(f"[RUN] Flag --all detectada ({args.classes} classes)...")
        run_experiment(num_classes=args.classes, n_trials=args.n_trials, use_smote=args.smote, seeds=seeds)
    else:
        display_menu()


if __name__ == "__main__":
    main()
