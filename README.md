# 📡 Prediction Streaming 5G - Tree-Based Models Suite (2, 4 and 6 Classes)

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-fast%20packaging-purple.svg)](https://github.com/astral-sh/uv)
[![Models](https://img.shields.io/badge/ML-Tree--Based-success.svg)]()
[![Optimization](https://img.shields.io/badge/Optuna-Hyperparameter%20Tuning-orange.svg)](https://optuna.org/)

Este repositório contém a suíte completa de **Machine Learning para predição da qualidade de streaming de vídeo sobre redes 5G reais**, considerando mobilidade (cenários **Estáticos** e **Veiculares**) e os provedores comerciais **Amazon Prime Video** e **Netflix**.

A nova metodologia foca **exclusivamente em modelos baseados em árvores (*Tree-Based Models*)**, investiga o impacto da **granularidade de classes (6, 4 e 2 classes de resolução)**, oferece balanceamento via **SMOTE**, avaliação com **múltiplas sementes aleatórias (Média ± Desvio Padrão)** e geração automatizada de **matrizes de confusão e tabelas estatísticas** prontas para submissão científica.

---

## 📌 Visão Geral & Perguntas de Pesquisa

A proposta do trabalho é estimar proativamente a capacidade de suporte de streaming de vídeo a partir de métricas físicas de radiofrequência (RF) coletadas em smartphone 5G:
- `RSRP` (Reference Signal Received Power)
- `RSRQ` (Reference Signal Received Quality)
- `SNR` (Signal-to-Noise Ratio)
- `CQI` (Channel Quality Indicator)
- `RSSI` (Received Signal Strength Indicator)

### 🎯 Granularidades de Resolução Avaliadas:

| Granularidade | Classes Alvo | Resoluções Agrupadas | Faixa de Bitrate (Proxy YouTube) |
| :--- | :--- | :--- | :--- |
| **6 Classes** | `144p`, `360p`, `480p`, `720p`, `1080p`, `4K` | Individuais por tier | `<0.7`, `0.7–1.1`, `1.1–2.5`, `2.5–5`, `5–20`, `≥20 Mbps` |
| **4 Classes** | `Low_SD`, `SD_HD`, `FHD`, `4K` | • `Low_SD` (`144p` + `360p`)<br>• `SD_HD` (`480p` + `720p`)<br>• `FHD` (`1080p`)<br>• `4K` (`4K`) | `<1.1 Mbps`<br>`1.1 – 5.0 Mbps`<br>`5.0 – 20.0 Mbps`<br>`≥ 20.0 Mbps` |
| **2 Classes** | `SD` vs `HD_Plus` | • `SD` (`144p`, `360p`, `480p`)<br>• `HD_Plus` (`720p`, `1080p`, `4K`) | `<2.5 Mbps` (Definição Padrão)<br>`≥2.5 Mbps` (Alta Definição) |

---

## 🌲 Modelos Baseados em Árvores

A arquitetura do pipeline foca em algoritmos de árvore pela sua alta robustez à colinearidade de features de RF (`RSRP`/`RSRQ`/`RSSI`), capacidade de captura de não-linearidades e excelente eficiência computacional:

1. **Decision Tree** (`DecisionTreeClassifier`) – Baseline interpretável.
2. **Random Forest** (`RandomForestClassifier`) – Ensemble de Bagging clássico.
3. **Extra Trees** (`ExtraTreesClassifier`) – Árvores extremamente aleatórias (baixo viés e variância).
4. **Gradient Boosting** (`GradientBoostingClassifier`) – Boosting sequencial tradicional.
5. **LightGBM** (`LGBMClassifier`) – Histogram-based boosting ultrarrápido com baixo consumo de memória.
6. **XGBoost** (`XGBClassifier`) – Gradient boosting escalável com regularização.

Todos os hiperparâmetros são otimizados dinamicamente via **Optuna** com **Stratified 5-Fold Cross-Validation** maximizando a Acurácia Balanceada e o Macro F1-score.

---

## 📁 Estrutura do Repositório

```text
Prediction_streaming_5G-v2/
├── datasets/                                 # Diretório centralizado de dados
│   ├── Amazon/                               # Dados brutos Amazon (Static e Driving)
│   ├── Netflix/                              # Dados brutos Netflix (Static e Driving)
│   ├── Amazon-{2,4,6}/                       # Datasets gerados por classe (ignorado pelo git)
│   ├── Netflix-{2,4,6}/                      # Datasets gerados por classe (ignorado pelo git)
│   └── *-SMOTE/                              # Datasets balanceados com SMOTE (ignorado pelo git)
├── Scripts/
│   ├── treatment_data.py                     # Geração dos datasets de 2, 4 e 6 classes
│   ├── smote_treatment.py                    # Balanceamento sintético via SMOTE vetorizado
│   ├── dataset_statistics.py                 # Resumo estatístico do dataset, RF e mobilidade
│   ├── data_loader.py                        # Carregamento e split estratificado 80/20
│   ├── models_optuna.py                      # Otimização Bayesiana dos 6 modelos de árvore
│   ├── evaluation.py                         # Métricas (Macro F1, ROC-AUC), Matrizes de Confusão e Gráficos
│   └── training_pipeline.py                  # Orquestrador central e Menu Interativo CLI
├── Results/                                  # Exportação de resultados e figuras
│   ├── dataset_statistics/                   # Tabelas de caracterização do dataset
│   ├── classes_2/                            # Resultados para 2 classes (With/Without SMOTE)
│   ├── classes_4/                            # Resultados para 4 classes (With/Without SMOTE)
│   └── classes_6/                            # Resultados para 6 classes (With/Without SMOTE)
├── pyproject.toml / uv.lock                  # Dependências gerenciadas via uv
└── README.md
```

---

## 🚀 Como Executar

### 1. Instalação do Ambiente
Este projeto utiliza o gerenciador de pacotes **`uv`**. Para instalar e sincronizar o ambiente virtual:
```powershell
uv sync
```

---

### 2. Menu Interativo (Recomendado)
Para acessar todas as operações através de uma interface interativa no terminal:
```powershell
uv run python -m Scripts.training_pipeline
```

#### Menu Principal:
```text
===========================================================================
    PIPELINE DE TREINAMENTO 5G - TREE MODELS & NOVO ARTIGO
===========================================================================
 Granularidade Atual   : [6 Classes] ['144p', '360p', '480p', '720p', '1080p', '4K']
 Modo Atual de Dataset : [SEM SMOTE]
 Trials do Optuna      : [15]
 Avaliação de Sementes : [Única (42)]
---------------------------------------------------------------------------
 [1] Executar Pipeline para a configuração atual
 [2] Executar TODOS os Cenários e Classes (6, 4 e 2) no Modo Atual
 [3] 🔥 EXECUTAR BENCHMARK COMPLETO (6, 4 e 2 Classes x SEM SMOTE e COM SMOTE)
 [4] Selecionar Granularidade de Classes (2, 4 ou 6)
 [5] Selecionar um Cenário Específico
 [6] Selecionar um Modelo de Árvore Específico
 [7] Alternar Modo de Dataset (SEM SMOTE / COM SMOTE)
 [8] Configurar Quantidade de Trials do Optuna
 [9] Alternar Modo de Múltiplas Sementes (Média ± Desvio Padrão)
---------------------------------------------------------------------------
 [10] 🛠️  Gerar Datasets das Classes (2, 4 e 6 Classes a partir dos dados brutos)
 [11] ⚖️  Gerar Datasets Balanceados com SMOTE (para 2, 4 e 6 Classes)
 [12] 📊 Gerar Tabelas de Estatísticas e Caracterização do Dataset
 [13] 🧪 Rodar Teste Rápido de Validação (2 Trials / Matriz de Confusão)
 [0] Sair
===========================================================================
```

---

### 3. Linha de Comando (CLI / Automação)

Você também pode executar diretamente qualquer tarefa via flags de linha de comando:

* **Gerar todos os datasets derivados (2, 4 e 6 classes):**
  ```powershell
  uv run python -m Scripts.training_pipeline --generate-datasets
  ```

* **Gerar os datasets balanceados com SMOTE:**
  ```powershell
  uv run python -m Scripts.training_pipeline --generate-smote
  ```

* **Gerar tabelas de caracterização e estatísticas para o artigo:**
  ```powershell
  uv run python -m Scripts.training_pipeline --dataset-stats
  ```

* **🔥 Executar Benchmark Completo do Artigo (todas as classes $\times$ SEM e COM SMOTE):**
  ```powershell
  uv run python -m Scripts.training_pipeline --full-benchmark --n-trials 15
  ```

* **Benchmark Completo com Múltiplas Sementes (42, 100, 2024 para cálculo de Média $\pm$ Desvio Padrão):**
  ```powershell
  uv run python -m Scripts.training_pipeline --full-benchmark --multi-seeds --n-trials 15
  ```

* **Teste Rápido de Validação (2 trials + geração de figuras):**
  ```powershell
  uv run python -m Scripts.training_pipeline --test-run --classes 4
  ```

---

## 📊 Artefatos e Saídas Geradas

Ao executar os experimentos, as saídas são salvas estruturadas sob `Results/`:

1. **`model_results_global.csv`**:
   - Métricas globais: `accuracy`, `balanced_accuracy`, `f1_score_macro`, `f1_score_weighted`, `precision_weighted`, `recall_weighted`, `mcc`, `roc_auc`, `log_loss`.
   - Custos computacionais: `train_time_seconds`, `inference_time_seconds`, `inference_time_per_sample_seconds`.
   - Estatística amostral (quando ativado `--multi-seeds`): `balanced_accuracy_std`, `f1_macro_std`.

2. **`model_results_per_class.csv`**:
   - Desempenho individual por classe de resolução: `precision`, `recall`, `f1_score`.

3. **`optuna_best_params.csv`**:
   - Melhores hiperparâmetros selecionados pelo Optuna para cada modelo e cenário.

4. **`confusion_matrices/`**:
   - Matrizes de confusão normalizadas prontas para publicação salvas em **PNG** e **PDF vetorial** (`300 DPI`).

5. **`figures/`**:
   - Gráficos de barras agrupados comparando o Macro F1-score entre os modelos.
