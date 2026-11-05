import warnings
import optuna
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Custom Estimators
# ---------------------------------------------------------------------------

class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for XGBoost to handle arbitrary string target labels and maintain sklearn compatibility.
    """
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = XGBClassifier(**xgb_params, random_state=42, eval_metric="mlogloss", n_jobs=-1)
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y):
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        preds_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# Optuna Objective Functions for Tree-Based Models
# ---------------------------------------------------------------------------

def optimize_dt(X_train, y_train, n_trials=10):
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_rf(X_train, y_train, n_trials=10):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
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
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_extra_trees(X_train, y_train, n_trials=10):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42,
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_gradient_boosting(X_train, y_train, n_trials=10):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 250)
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
                random_state=42,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_adaboost(X_train, y_train, n_trials=10):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 1.0, log=True)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", AdaBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=42,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_lightgbm(X_train, y_train, n_trials=10):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 10, 64)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_xgboost(X_train, y_train, n_trials=10):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 5.0)

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_encoded, cv=5, scoring="balanced_accuracy", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------------------------------------------------------------------------
# Model Registry & Pipeline Factory
# ---------------------------------------------------------------------------

MODEL_OPTIMIZERS = {
    "DecisionTree": optimize_dt,
    "RandomForest": optimize_rf,
    "ExtraTrees": optimize_extra_trees,
    "GradientBoosting": optimize_gradient_boosting,
    "AdaBoost": optimize_adaboost,
    "LightGBM": optimize_lightgbm,
    "XGBoost": optimize_xgboost,
}


def build_final_model(model_name: str, best_params: dict):
    """
    Constructs the pipeline using the best hyperparams found by Optuna.
    """
    params = best_params.copy()

    if model_name == "DecisionTree":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(**params, random_state=42))
        ])

    elif model_name == "RandomForest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(**params, random_state=42, n_jobs=-1))
        ])

    elif model_name == "ExtraTrees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(**params, random_state=42, n_jobs=-1))
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

    elif model_name == "LightGBM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(**params, random_state=42, verbosity=-1, n_jobs=-1))
        ])

    elif model_name == "XGBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBoostWrapper(**params))
        ])

    else:
        raise ValueError(f"Unknown model name: {model_name}")
