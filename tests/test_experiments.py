from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from credit_audit.experiments import ECE_SCORER


def test_ece_scorer_is_usable_by_cross_validate():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.array([0] * 6 + [1] * 6)

    scores = cross_validate(LogisticRegression(), X, y, cv=2, scoring={"ece": ECE_SCORER})

    assert "test_ece" in scores
    assert np.all(scores["test_ece"] <= 0)
