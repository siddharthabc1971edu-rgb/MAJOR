from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

from .metrics import classification_metrics, expected_calibration_error
from .models import make_pipeline


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def _ece_from_proba(y_true, probabilities) -> float:
    values = np.asarray(probabilities)
    if values.ndim == 2:
        values = values[:, 1]
    return expected_calibration_error(y_true, values)


ECE_SCORER = make_scorer(
    _ece_from_proba,
    response_method="predict_proba",
    greater_is_better=False,
)


def compare_models(X: pd.DataFrame, y: pd.Series, numeric: list[str]) -> pd.DataFrame:
    rows = []
    feature_names = list(X.columns)
    for name in ("lr", "lr_weighted", "rf", "xgb", "lightgbm", "catboost"):
        strategy = "class_weight" if name == "xgb" else "none"
        pipeline = make_pipeline(name, strategy, numeric, feature_names)
        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=CV,
            scoring={
                "auc": "roc_auc",
                "pr_auc": "average_precision",
                "brier": "neg_brier_score",
                "recall": "recall",
                "precision": "precision",
                "ece": ECE_SCORER,
            },
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "model": name,
                "auc": float(scores["test_auc"].mean()),
                "pr_auc": float(scores["test_pr_auc"].mean()),
                "brier": float(-scores["test_brier"].mean()),
                "recall": float(scores["test_recall"].mean()),
                "precision": float(scores["test_precision"].mean()),
                "ece": float(-scores["test_ece"].mean()),
                "mean_fit_seconds": float(scores["fit_time"].mean()),
            }
        )
    return pd.DataFrame(rows)


def compare_imbalance(X: pd.DataFrame, y: pd.Series, numeric: list[str]) -> pd.DataFrame:
    rows = []
    feature_names = list(X.columns)
    for strategy in ("none", "class_weight", "smote", "smote_enn"):
        pipeline = make_pipeline("xgb", strategy, numeric, feature_names)
        predicted = cross_val_predict(pipeline, X, y, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
        rows.append({"strategy": strategy, **classification_metrics(y, predicted)})
    return pd.DataFrame(rows)


def fit_oof_and_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric: list[str],
    strategy: str = "class_weight",
) -> tuple[object, np.ndarray]:
    pipeline = make_pipeline("xgb", strategy, numeric, list(X_train.columns))
    oof = cross_val_predict(pipeline, X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
    fitted = clone(pipeline).fit(X_train, y_train)
    return fitted, oof


def timed_predict(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    probabilities = model.predict_proba(X)[:, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000 / max(len(X), 1)
    return probabilities, elapsed_ms
