from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class FoldWeightedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Compute XGBoost's positive-class weight from the current fit partition."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        from xgboost import XGBClassifier

        labels = np.asarray(y)
        positives = int(np.sum(labels == 1))
        negatives = int(np.sum(labels == 0))
        if positives == 0:
            raise ValueError("The training labels contain no positive/default examples.")
        self.estimator_ = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            scale_pos_weight=negatives / positives,
            eval_metric="logloss",
            n_jobs=1,
            random_state=self.random_state,
        )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def make_preprocessor(
    numeric_columns: list[str], feature_names: list[str], scale_numeric: bool = False
) -> ColumnTransformer:
    categorical = [name for name in feature_names if name not in numeric_columns]
    numeric_transformer = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_transformer.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_transformer), numeric_columns),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_estimator(name: str, random_state: int = 42) -> Any:
    if name == "lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight=None, random_state=random_state)
    if name == "lr_weighted":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=random_state
        )
    if name in {"xgb", "xgb_weighted"}:
        if name == "xgb_weighted":
            return FoldWeightedXGBClassifier(random_state=random_state)
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1,
            scale_pos_weight=1,
            eval_metric="logloss",
            n_jobs=1,
            random_state=random_state,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, num_leaves=31, verbosity=-1, n_jobs=-1, random_state=random_state
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.08, verbose=False, random_seed=random_state
        )
    raise ValueError(f"Unknown model {name!r}.")


def make_pipeline(
    model_name: str,
    strategy: str,
    numeric_columns: list[str],
    feature_names: list[str],
) -> Pipeline:
    if strategy not in {"none", "class_weight", "smote", "smote_enn"}:
        raise ValueError(f"Unknown imbalance strategy {strategy!r}.")
    estimator_name = model_name
    if strategy == "class_weight" and model_name == "xgb":
        estimator_name = "xgb_weighted"
    estimator = make_estimator(estimator_name)
    steps: list[tuple[str, Any]] = [
        (
            "preprocess",
            make_preprocessor(
                numeric_columns,
                feature_names,
                scale_numeric=model_name in {"lr", "lr_weighted"},
            ),
        ),
    ]
    if strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=42, k_neighbors=3)))
    elif strategy == "smote_enn":
        steps.append(("sampler", SMOTEENN(random_state=42, smote=SMOTE(random_state=42, k_neighbors=3))))
    steps.append(("model", estimator))
    return Pipeline(steps)


def class_weight_ratio(y: np.ndarray) -> float:
    negatives = int(np.sum(np.asarray(y) == 0))
    positives = int(np.sum(np.asarray(y) == 1))
    if positives == 0:
        raise ValueError("The training labels contain no positive/default examples.")
    return negatives / positives
