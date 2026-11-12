import warnings
import optuna
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score
from scipy.optimize import minimize

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Custom Estimators
# ---------------------------------------------------------------------------

class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for XGBoost to handle arbitrary string target labels and maintain sklearn compatibility.
    Uses tree_method='hist' for fast histogram-based execution.
    """
    def __init__(self, **xgb_params):
        self.xgb_params = xgb_params
        self.model = XGBClassifier(
            **xgb_params,
            tree_method="hist",
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y):
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        if len(self.classes_) == 2:
            self.model.set_params(eval_metric="logloss", objective="binary:logistic")
        else:
            self.model.set_params(eval_metric="mlogloss", objective="multi:softprob")
        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        preds_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for CatBoostClassifier with silent output and multi-threading.
    """
    def __init__(self, **cb_params):
        self.cb_params = cb_params
        self.model = CatBoostClassifier(
            **cb_params,
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
        self.classes_ = None

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X).ravel()

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# Optuna Objective Functions for Tree-Based Models (Optimizing f1_macro)
# ---------------------------------------------------------------------------

def optimize_dt(X_train, y_train, n_trials=25):
    def objective(trial):
        max_depth = trial.suggest_int("max_depth", 3, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                class_weight=class_weight,
                random_state=42,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_rf(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 250)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_extra_trees(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 250)
        max_depth = trial.suggest_int("max_depth", 5, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_lightgbm(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int("num_leaves", 15, 64)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                subsample=subsample,
                class_weight=class_weight,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_xgboost(X_train, y_train, n_trials=25):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 250)
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
                tree_method="hist",
                random_state=42,
                eval_metric="mlogloss",
                n_jobs=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_encoded, cv=5, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def optimize_catboost(X_train, y_train, n_trials=25):
    def objective(trial):
        iterations = trial.suggest_int("iterations", 100, 350)
        depth = trial.suggest_int("depth", 4, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 10.0)
        auto_class_weights = trial.suggest_categorical("auto_class_weights", [None, "Balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                l2_leaf_reg=l2_leaf_reg,
                auto_class_weights=auto_class_weights,
                random_seed=42,
                verbose=False,
                thread_count=-1,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
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
    "LightGBM": optimize_lightgbm,
    "XGBoost": optimize_xgboost,
    "CatBoost": optimize_catboost,
}


class ThresholdTunedVotingClassifier(BaseEstimator, ClassifierMixin):
    """Soft VotingClassifier with out-of-fold class-multiplier tuning for Macro F1."""

    def __init__(self, estimators, weights=None, cv=3):
        self.estimators = estimators
        self.weights = weights
        self.cv = cv

    def fit(self, X, y):
        self.voting_ = VotingClassifier(
            estimators=self.estimators,
            voting="soft",
            weights=self.weights,
            n_jobs=-1,
        )
        oof_probs = cross_val_predict(
            self.voting_, X, y, cv=self.cv, method="predict_proba", n_jobs=-1
        )
        self.voting_.fit(X, y)
        self.classes_ = self.voting_.classes_

        n_classes = len(self.classes_)
        def _loss(w):
            preds = self.classes_[np.argmax(oof_probs * w, axis=1)]
            return -f1_score(y, preds, average="macro")

        res = minimize(
            _loss,
            np.ones(n_classes),
            method="Nelder-Mead",
            options={"maxiter": 300},
        )
        self.class_weights_ = res.x
        return self

    def predict_proba(self, X):
        return self.voting_.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs * self.class_weights_, axis=1)]


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

    elif model_name == "CatBoost":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", CatBoostWrapper(**params))
        ])

    elif model_name == "VotingClassifier":
        # params dict contains 'estimators' list of (name, model)
        estimators = params.get("estimators", [])
        weights = params.get("weights", None)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ThresholdTunedVotingClassifier(estimators=estimators, weights=weights))
        ])

    else:
        raise ValueError(f"Unknown model name: {model_name}")
