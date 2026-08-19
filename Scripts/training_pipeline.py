import sys
import argparse
import pandas as pd

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from Scripts.data_loader import DATASETS, load_and_preprocess_dataset
from Scripts.models_optuna import MODEL_OPTIMIZERS, build_final_model
from Scripts.evaluation import evaluate_model, save_results

DEFAULT_MODELS = ["KNN", "GaussianNB", "MLP", "RandomForest", "SVM"]


def run_experiment(datasets_to_run=None, models_to_run=None, n_trials=5, use_smote=False):
    """
    Orchestrates dataset loading, Optuna hyperparameter tuning, model training,
    and performance evaluation across selected datasets and models.
    """
    if datasets_to_run is None:
        datasets_to_run = list(DATASETS.keys())
    if models_to_run is None:
        models_to_run = DEFAULT_MODELS

    global_results = []
    per_class_results = []
    best_params_list = []

    print("\n" + "=" * 60)
    print(" INICIANDO EXPERIMENTOS DE TREINAMENTO E OTIMIZACAO (OPTUNA)")
    print("=" * 60)
    print(f"Cenarios selecionados : {datasets_to_run}")
    print(f"Modelos selecionados  : {models_to_run}")
    print(f"Trials do Optuna      : {n_trials}")
    print(f"Uso de SMOTE          : {use_smote}")
    print("=" * 60)

    for dataset_name in datasets_to_run:
        print(f"\n============================================================")
        print(f" CARREGANDO DATASET: {dataset_name}")
        print(f"============================================================")

        try:
            X_train, X_test, y_train, y_test = load_and_preprocess_dataset(
                dataset_key=dataset_name,
                use_smote=use_smote
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
            print(f"[OPTUNA] Otimizando hiperparâmetros para: {model_name} ({n_trials} trials)")
            print(f"------------------------------------------------------------")

            optimizer_fn = MODEL_OPTIMIZERS[model_name]
            best_params = optimizer_fn(X_train, y_train, n_trials=n_trials)
            print(f"[OK] Melhores parâmetros [{model_name}]: {best_params}")

            best_params_list.append({
                "dataset": dataset_name,
                "model": model_name,
                "use_smote": use_smote,
                "best_params": str(best_params)
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
                use_smote=use_smote
            )

            global_results.append(global_res)
            per_class_results.extend(per_class_res)

            print(f"   Acurácia Balanceada : {global_res['balanced_accuracy']:.4f}")
            print(f"   F1-Score (Weighted) : {global_res['f1_score_weighted']:.4f}")
            print(f"   Tempo Treino (s)    : {global_res['train_time_seconds']:.2f}s")

    # Save to CSV
    save_results(
        global_results=global_results,
        per_class_results=per_class_results,
        best_params_list=best_params_list
    )

    print("\n============================================================")
    print(" TODOS OS EXPERIMENTOS FORAM CONCLUIDOS COM SUCESSO!")
    print("============================================================\n")


def display_menu():
    """
    Displays the CLI interactive terminal menu.
    """
    n_trials = 5
    use_smote = False

    while True:
        print("\n" + "=" * 55)
        print("     PIPELINE DE TREINAMENTO 5G - 6 CLASSES MENU ")
        print("=" * 55)
        print(" [1] Executar PIPELINE COMPLETO (4 Cenários x 5 Modelos)")
        print(" [2] Selecionar um Cenário Específico")
        print(" [3] Selecionar um Modelo Específico")
        print(" [4] Configurar Quantidade de Trials do Optuna (Atual: {})".format(n_trials))
        print(" [5] Rodar Teste Rápido de Validação (2 Trials / 1 Cenário)")
        print(" [0] Sair")
        print("=" * 55)

        choice = input("Escolha uma opção: ").strip()

        if choice == "1":
            run_experiment(n_trials=n_trials, use_smote=use_smote)
        elif choice == "2":
            print("\nCenários disponíveis:")
            datasets_list = list(DATASETS.keys())
            for idx, ds in enumerate(datasets_list, 1):
                print(f"  [{idx}] {ds}")
            ds_choice = input("Selecione o número do cenário: ").strip()
            if ds_choice.isdigit() and 1 <= int(ds_choice) <= len(datasets_list):
                selected_ds = [datasets_list[int(ds_choice) - 1]]
                run_experiment(datasets_to_run=selected_ds, n_trials=n_trials, use_smote=use_smote)
            else:
                print("[ERRO] Opção de cenário inválida.")
        elif choice == "3":
            print("\nModelos disponíveis:")
            for idx, m in enumerate(DEFAULT_MODELS, 1):
                print(f"  [{idx}] {m}")
            m_choice = input("Selecione o número do modelo: ").strip()
            if m_choice.isdigit() and 1 <= int(m_choice) <= len(DEFAULT_MODELS):
                selected_m = [DEFAULT_MODELS[int(m_choice) - 1]]
                run_experiment(models_to_run=selected_m, n_trials=n_trials, use_smote=use_smote)
            else:
                print("[ERRO] Opção de modelo inválida.")
        elif choice == "4":
            val = input("Digite o novo número de trials do Optuna: ").strip()
            if val.isdigit() and int(val) > 0:
                n_trials = int(val)
                print(f"[OK] Número de trials alterado para: {n_trials}")
            else:
                print("[ERRO] Valor inválido.")
        elif choice == "5":
            print("\n[TEST] Executando teste rápido de validação...")
            run_experiment(
                datasets_to_run=["Amazon Static"],
                models_to_run=["KNN"],
                n_trials=2,
                use_smote=False
            )
        elif choice == "0":
            print("Encerrando pipeline. Até logo!")
            sys.exit(0)
        else:
            print("[ERRO] Opção inválida. Tente novamente.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento e Otimizacao para Predicao de Qualidade 5G")
    parser.add_argument("--test-run", action="store_true", help="Executa um teste rapido para verificar funcionamento do codigo.")
    parser.add_argument("--all", action="store_true", help="Executa o pipeline completo sem exibir o menu interativo.")
    parser.add_argument("--n-trials", type=int, default=5, help="Numero de trials para otimizacao com Optuna (padrao: 5).")

    args = parser.parse_args()

    if args.test_run:
        print("[TEST] Flag --test-run detectada. Executando teste de validacao...")
        run_experiment(
            datasets_to_run=["Amazon Static"],
            models_to_run=["KNN"],
            n_trials=2,
            use_smote=False
        )
    elif args.all:
        print("[RUN] Flag --all detectada. Executando pipeline completo...")
        run_experiment(n_trials=args.n_trials, use_smote=False)
    else:
        display_menu()


if __name__ == "__main__":
    main()
