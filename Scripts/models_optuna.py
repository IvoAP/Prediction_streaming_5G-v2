import warnings
import optuna
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from scipy.optimize import minimize

LAST_OPTUNA_SCORES = {}


def _compute_sample_weights(y, mode):
    """Compute per-sample weights for None, 'balanced', or 'sqrt_balanced'."""
    if mode is None:
        return None
    sw = compute_sample_weight("balanced", y)
    if mode == "balanced":
        return sw
    if mode == "sqrt_balanced":
        return np.sqrt(sw)
    return None


class SampleWeightClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wraps any sklearn/LGBM estimator to support class_weight='sqrt_balanced' via sample_weight."""
    _estimator_type = "classifier"

    def __init__(self, estimator_cls, class_weight=None, **params):
        self.estimator_cls = estimator_cls
        self.class_weight = class_weight
        self.params = params

    def get_params(self, deep=True):
        out = {"estimator_cls": self.estimator_cls, "class_weight": self.class_weight}
        out.update(self.params)
        return out

    def set_params(self, **params):
        if "estimator_cls" in params:
            self.estimator_cls = params.pop("estimator_cls")
        if "class_weight" in params:
            self.class_weight = params.pop("class_weight")
        self.params.update(params)
        return self

    def fit(self, X, y):
        est_params = self.params.copy()
        if self.class_weight == "sqrt_balanced":
            classes = np.unique(y)
            w = compute_class_weight("balanced", classes=classes, y=y)
            est_params["class_weight"] = dict(zip(classes, np.sqrt(w)))
        else:
            est_params["class_weight"] = self.class_weight
        self.model_ = self.estimator_cls(**est_params)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

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

class XGBoostWrapper(ClassifierMixin, BaseEstimator):
    """
    Wrapper for XGBoost to handle arbitrary string target labels and maintain sklearn compatibility.
    Uses tree_method='hist' for fast histogram-based execution and supports class_weight.
    """
    _estimator_type = "classifier"
    def __init__(self, class_weight=None, **xgb_params):
        self.class_weight = class_weight
        self.xgb_params = xgb_params
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def get_params(self, deep=True):
        out = {"class_weight": self.class_weight}
        out.update(self.xgb_params)
        return out

    def set_params(self, **params):
        if "class_weight" in params:
            self.class_weight = params.pop("class_weight")
        self.xgb_params.update(params)
        return self

    def fit(self, X, y):
        y_enc = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        if len(self.classes_) == 2:
            obj, metric = "binary:logistic", "logloss"
        else:
            obj, metric = "multi:softprob", "mlogloss"
        self.model = XGBClassifier(
            **self.xgb_params,
            tree_method="hist",
            random_state=42,
            objective=obj,
            eval_metric=metric,
            n_jobs=-1,
        )
        sw = _compute_sample_weights(y_enc, self.class_weight)
        self.model.fit(X, y_enc, sample_weight=sw)
        return self

    def predict(self, X):
        preds_enc = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds_enc)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class CatBoostWrapper(ClassifierMixin, BaseEstimator):
    """
    Wrapper for CatBoostClassifier with silent output and multi-threading.
    """
    _estimator_type = "classifier"
    def __init__(self, **cb_params):
        self.cb_params = cb_params
        self.classes_ = None

    def get_params(self, deep=True):
        return self.cb_params.copy()

    def set_params(self, **params):
        self.cb_params.update(params)
        return self

    def fit(self, X, y):
        self.model = CatBoostClassifier(
            **self.cb_params,
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
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
        max_depth = trial.suggest_int("max_depth", 10, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 16)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 8)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "sqrt_balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=DecisionTreeClassifier,
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LAST_OPTUNA_SCORES["DecisionTree"] = study.best_value
    return study.best_params


def optimize_rf(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 250)
        max_depth = trial.suggest_int("max_depth", 12, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample", "sqrt_balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=RandomForestClassifier,
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LAST_OPTUNA_SCORES["RandomForest"] = study.best_value
    return study.best_params


def optimize_extra_trees(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 250)
        max_depth = trial.suggest_int("max_depth", 12, 25)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample", "sqrt_balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=ExtraTreesClassifier,
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LAST_OPTUNA_SCORES["ExtraTrees"] = study.best_value
    return study.best_params


def optimize_lightgbm(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 120, 300)
        max_depth = trial.suggest_int("max_depth", 6, 15)
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        num_leaves = trial.suggest_int("num_leaves", 31, 64)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 30)
        subsample = trial.suggest_float("subsample", 0.65, 1.0)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "sqrt_balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=LGBMClassifier,
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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LAST_OPTUNA_SCORES["LightGBM"] = study.best_value
    return study.best_params


def optimize_xgboost(X_train, y_train, n_trials=25):
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 120, 250)
        max_depth = trial.suggest_int("max_depth", 6, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
        subsample = trial.suggest_float("subsample", 0.65, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.65, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 0.5)
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "sqrt_balanced"])

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBoostWrapper(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                class_weight=class_weight,
            ))
        ])

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LAST_OPTUNA_SCORES["XGBoost"] = study.best_value
    return study.best_params


def optimize_catboost(X_train, y_train, n_trials=25):
    def objective(trial):
        iterations = trial.suggest_int("iterations", 180, 350)
        depth = trial.suggest_int("depth", 6, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.05, 0.28, log=True)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 8.0)
        auto_class_weights = trial.suggest_categorical("auto_class_weights", [None, "Balanced", "SqrtBalanced"])

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

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    LAST_OPTUNA_SCORES["CatBoost"] = study.best_value
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


class ThresholdTunedVotingClassifier(ClassifierMixin, BaseEstimator):
    """Soft VotingClassifier with Coordinate-Ascent OOF class-multiplier tuning for Macro F1."""
    _estimator_type = "classifier"

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
        w = np.ones(n_classes, dtype=float)
        best_f1 = f1_score(y, self.classes_[np.argmax(oof_probs * w, axis=1)], average="macro")
        grid = np.linspace(0.2, 2.5, 47)

        for _ in range(3):
            for c in range(n_classes):
                best_c = w[c]
                for val in grid:
                    w[c] = val
                    preds = self.classes_[np.argmax(oof_probs * w, axis=1)]
                    score = f1_score(y, preds, average="macro")
                    if score > best_f1:
                        best_f1 = score
                        best_c = val
                w[c] = best_c

        self.class_weights_ = w
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
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=DecisionTreeClassifier,
                **params,
                random_state=42,
            ))
        ])

    elif model_name == "RandomForest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=RandomForestClassifier,
                **params,
                random_state=42,
                n_jobs=-1,
            ))
        ])

    elif model_name == "ExtraTrees":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=ExtraTreesClassifier,
                **params,
                random_state=42,
                n_jobs=-1,
            ))
        ])

    elif model_name == "LightGBM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", SampleWeightClassifierWrapper(
                estimator_cls=LGBMClassifier,
                **params,
                random_state=42,
                verbosity=-1,
                n_jobs=-1,
            ))
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
