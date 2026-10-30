import sys
import argparse
import pandas as pd

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
    n_trials: int = 10,
    use_smote: bool = False,
):
    """
    Orchestrates dataset loading, Optuna hyperparameter tuning, model training,
    and performance evaluation across selected tree models and class resolutions.
    """
    dataset_map = get_dataset_map(num_classes=num_classes, use_smote=use_smote)
    if datasets_to_run is None:
        datasets_to_run = list(dataset_map.keys())
    if models_to_run is None:
        models_to_run = TREE_MODELS

    global_results = []
    per_class_results = []
    best_params_list = []

    mode_str = "COM SMOTE" if use_smote else "SEM SMOTE"

    print("\n" + "=" * 70)
    print(f" PIPELINE TREE-BASED MODELS - {num_classes} CLASSES ({mode_str})")
    print("=" * 70)
    print(f"Número de Classes     : {num_classes} {CLASSES_BY_CONFIG.get(num_classes)}")
    print(f"Cenários selecionados : {datasets_to_run}")
    print(f"Modelos (Tree-based)  : {models_to_run}")
    print(f"Trials do Optuna      : {n_trials}")
    print(f"Modo Dataset          : {mode_str}")
    print("=" * 70)

    for dataset_name in datasets_to_run:
        print(f"\n============================================================")
        print(f" CARREGANDO DATASET ({num_classes} classes | {mode_str}): {dataset_name}")
        print(f"============================================================")

        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
                dataset_key=dataset_name,
                num_classes=num_classes,
                use_smote=use_smote,
            )
        except Exception as e:
            print(f"[ERRO] Erro ao carregar dataset '{dataset_name}': {e}")
            continue

        print(f" Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
        print(f" Distribuição de classes (Treino):\n{y_train.value_counts().to_dict()}")

        for model_name in models_to_run:
            if model_name not in MODEL_OPTIMIZERS:
                print(f"[AVISO] Modelo '{model_name}' não suportado. Pulando...")
                continue

            print(f"\n------------------------------------------------------------")
            print(f"[OPTUNA] Otimizando ({model_name}) para {num_classes} classes ({n_trials} trials)")
            print(f"------------------------------------------------------------")

            optimizer_fn = MODEL_OPTIMIZERS[model_name]
            best_params = optimizer_fn(X_train, y_train, n_trials=n_trials)
            print(f"[OK] Melhores parâmetros [{model_name}]: {best_params}")

            best_params_list.append({
                "dataset": dataset_name,
                "model": model_name,
                "num_classes": num_classes,
                "use_smote": use_smote,
                "best_params": str(best_params),
            })

            # Build final model pipeline and evaluate
            final_model = build_final_model(model_name, best_params)
            print(f"[EVAL] Treinando modelo final e calculando métricas...")

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
            )

            global_results.append(global_res)
            per_class_results.extend(per_class_res)

            print(f"   Acurácia             : {global_res['accuracy']:.4f}")
            print(f"   Acurácia Balanceada : {global_res['balanced_accuracy']:.4f}")
            print(f"   F1-Score (Macro)    : {global_res['f1_score_macro']:.4f}")
            print(f"   F1-Score (Weighted) : {global_res['f1_score_weighted']:.4f}")
            print(f"   MCC                  : {global_res['mcc']:.4f}")
            if not pd.isna(global_res["roc_auc"]):
                print(f"   ROC-AUC              : {global_res['roc_auc']:.4f}")
            print(f"   Tempo Treino (s)    : {global_res['train_time_seconds']:.2f}s")
            print(f"   Tempo Inferência (s): {global_res['inference_time_seconds']:.4f}s")

    # Save to CSV in designated subfolders
    save_results(
        global_results=global_results,
        per_class_results=per_class_results,
        best_params_list=best_params_list,
        num_classes=num_classes,
        use_smote=use_smote,
    )

    print("\n============================================================")
    print(f" EXPERIMENTOS ({num_classes} CLASSES | {mode_str}) CONCLUIDOS!")
    print("============================================================\n")


def display_menu():
    """
    Displays the CLI interactive terminal menu.
    """
    n_trials = 10
    use_smote = False
    current_num_classes = 6

    while True:
        mode_label = "COM SMOTE" if use_smote else "SEM SMOTE"
        print("\n" + "=" * 65)
        print("  PIPELINE DE TREINAMENTO 5G - TREE-BASED MODELS")
        print("=" * 65)
        print(f" Granularidade Atual   : [{current_num_classes} Classes] {CLASSES_BY_CONFIG.get(current_num_classes)}")
        print(f" Modo Atual de Dataset : [{mode_label}]")
        print(f" Trials do Optuna      : [{n_trials}]")
        print("-" * 65)
        print(" [1] Executar PIPELINE COMPLETO para configuração atual (4 Cenários x 6 Modelos)")
        print(" [2] Executar TODOS os Cenários e TODAS as Classes (2, 4 e 6 Classes)")
        print(" [3] Selecionar Granularidade de Classes (2, 4 ou 6)")
        print(" [4] Selecionar um Cenário Específico")
        print(" [5] Selecionar um Modelo de Árvore Específico")
        print(" [6] Alternar Modo de Dataset (SEM SMOTE / COM SMOTE)")
        print(" [7] Configurar Quantidade de Trials do Optuna (Atual: {})".format(n_trials))
        print(" [8] Rodar Teste Rápido de Validação (2 Trials / 1 Modelo / 1 Cenário)")
        print(" [0] Sair")
        print("=" * 65)

        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            run_experiment(num_classes=current_num_classes, n_trials=n_trials, use_smote=use_smote)
        elif choice == "2":
            for n_cls in [6, 4, 2]:
                run_experiment(num_classes=n_cls, n_trials=n_trials, use_smote=use_smote)
        elif choice == "3":
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
        elif choice == "4":
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
                )
            else:
                print("[ERRO] Opção de cenário inválida.")
        elif choice == "5":
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
                )
            else:
                print("[ERRO] Opção de modelo inválida.")
        elif choice == "6":
            use_smote = not use_smote
            new_mode = "COM SMOTE" if use_smote else "SEM SMOTE"
            print(f"✅ Modo de dataset alterado para: {new_mode}")
        elif choice == "7":
            val = input("Digite o novo número de trials do Optuna: ").strip()
            if val.isdigit() and int(val) > 0:
                n_trials = int(val)
                print(f"✅ Número de trials alterado para: {n_trials}")
            else:
                print("[ERRO] Valor inválido.")
        elif choice == "8":
            print("\n[TEST] Executando teste rápido de validação...")
            run_experiment(
                num_classes=current_num_classes,
                datasets_to_run=["Amazon Static"],
                models_to_run=["DecisionTree"],
                n_trials=2,
                use_smote=use_smote,
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
    parser.add_argument("--all-classes", action="store_true", help="Executa iterativamente para 6, 4 e 2 classes.")
    parser.add_argument("--n-trials", type=int, default=10, help="Número de trials para Optuna (padrão: 10).")
    parser.add_argument("--smote", action="store_true", help="Utiliza os datasets gerados com SMOTE.")

    args = parser.parse_args()

    if args.test_run:
        print("[TEST] Flag --test-run detectada. Executando teste de validação...")
        run_experiment(
            num_classes=args.classes,
            datasets_to_run=["Amazon Static"],
            models_to_run=["DecisionTree"],
            n_trials=2,
            use_smote=args.smote,
        )
    elif args.all_classes:
        print("[RUN] Executando pipeline para 6, 4 e 2 classes...")
        for n_cls in [6, 4, 2]:
            run_experiment(num_classes=n_cls, n_trials=args.n_trials, use_smote=args.smote)
    elif args.all:
        print(f"[RUN] Flag --all detectada ({args.classes} classes)...")
        run_experiment(num_classes=args.classes, n_trials=args.n_trials, use_smote=args.smote)
    else:
        display_menu()


if __name__ == "__main__":
    main()
