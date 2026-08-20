import warnings
import optuna
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_knn(X_train, y_train, n_trials=5):
    def objective(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_gnb(X_train, y_train, n_trials=5):
    def objective(trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB(var_smoothing=var_smoothing))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_mlp(X_train, y_train, n_trials=5):
    def objective(trial):
        hidden_layer_size = trial.suggest_int("hidden_layer_size", 10, 150)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(hidden_layer_size,),
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=500,
                random_state=42
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_rf(X_train, y_train, n_trials=5):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
                n_jobs=-1
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_gradient_boosting(X_train, y_train, n_trials=5):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 400)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 2, 8)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=42
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_adaboost(X_train, y_train, n_trials=5):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 400)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_svm(X_train, y_train, n_trials=5):
    def objective(trial):
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=C, kernel=kernel, gamma=gamma, probability=True, max_iter=5000, cache_size=1000, random_state=42))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_lightgbm(X_train, y_train, n_trials=5):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 5, 64)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


MODEL_OPTIMIZERS = {
    "KNN": optimize_knn,
    "GaussianNB": optimize_gnb,
    "MLP": optimize_mlp,
    "RandomForest": optimize_rf,
    "GradientBoosting": optimize_gradient_boosting,
    "AdaBoost": optimize_adaboost,
    "SVM": optimize_svm,
    "LightGBM": optimize_lightgbm,
}


def build_final_model(model_name: str, best_params: dict):
    """
    Constructs the sklearn Pipeline using the best hyperparams found by Optuna.
    """
    params = best_params.copy()

    if model_name == "KNN":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(**params))
        ])

    elif model_name == "GaussianNB":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB(**params))
        ])

    elif model_name == "MLP":
        hidden_layer_size = params.pop("hidden_layer_size")
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(hidden_layer_size,),
                **params,
                max_iter=1000,
                random_state=42
            ))
        ])

    elif model_name == "RandomForest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(**params, random_state=42, n_jobs=-1))
        ])

    elif model_name == "GradientBoosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", GradientBoostingClassifier(**params, random_state=42))
        ])

    elif model_name == "AdaBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", AdaBoostClassifier(**params, random_state=42))
        ])

    elif model_name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(**params, probability=True, max_iter=5000, cache_size=1000, random_state=42))
        ])

    elif model_name == "LightGBM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1))
        ])

    else:
        raise ValueError(f"Unknown model name: {model_name}")
