from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, probabilities, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    error = 0.0
    for index in range(bins):
        in_bin = (p >= edges[index]) & (p < edges[index + 1] if index < bins - 1 else p <= edges[index + 1])
        if np.any(in_bin):
            error += in_bin.mean() * abs(float(p[in_bin].mean()) - float(y[in_bin].mean()))
    return float(error) if total else float("nan")


def classification_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = (p >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }
