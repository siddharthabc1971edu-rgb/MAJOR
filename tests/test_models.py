from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from credit_audit import models


class RecordingSampler(BaseEstimator):
    seen: list[int] = []

    def __init__(self, random_state=42, k_neighbors=3):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(self, X, y):
        type(self).seen.append(len(np.asarray(y)))
        return X, y


def _tiny_frame() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "n1": np.arange(12, dtype=float),
            "n2": np.arange(12, dtype=float) * 2,
            "c": ["a", "b"] * 6,
        }
    )
    y = pd.Series([0, 1] * 6, name="target")
    return X, y


def test_class_weight_ratio_uses_current_fit_partition():
    y = np.array([0, 0, 0, 1, 1])
    assert models.class_weight_ratio(y) == 1.5
    with pytest.raises(ValueError, match="no positive"):
        models.class_weight_ratio(np.array([0, 0, 0]))


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown model"):
        models.make_estimator("not_a_model")


def test_sampler_only_sees_training_fold(monkeypatch):
    RecordingSampler.seen = []
    monkeypatch.setattr(models, "SMOTE", RecordingSampler)
    X, y = _tiny_frame()
    pipeline = models.make_pipeline("lr", "smote", ["n1", "n2"], list(X.columns))

    cross_val_predict(
        pipeline, X, y,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=1),
        method="predict_proba",
    )

    assert len(RecordingSampler.seen) == 3
    assert all(count == 8 for count in RecordingSampler.seen)
