from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from credit_audit.audit import (
    apply_calibrator,
    choose_calibrator,
    clear_explainer_cache,
    conformal_p_values,
    explanation_stability,
    group_metrics,
    local_shap_values,
    routing_table,
)
from credit_audit.models import make_preprocessor


def _tiny_tree_model() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "n1": np.linspace(0, 10, 24),
            "n2": np.linspace(10, 0, 24),
            "c": ["a", "b", "c"] * 8,
        }
    )
    y = pd.Series((np.arange(24) % 2), name="target")
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(["n1", "n2"], list(X.columns))),
            ("model", XGBClassifier(
                n_estimators=4, max_depth=2, random_state=3,
                eval_metric="logloss", n_jobs=1,
            )),
        ]
    )
    model.fit(X, y)
    return model, X, y


def test_cached_tree_explainer_matches_a_fresh_explainer():
    import shap

    model, X, _ = _tiny_tree_model()
    sample = X.iloc[:3]
    cached = local_shap_values(model, sample)

    clear_explainer_cache()
    estimator = model.named_steps["model"]
    transformed = model.named_steps["preprocess"].transform(sample)
    fresh = np.asarray(shap.TreeExplainer(estimator)(transformed).values)
    if fresh.ndim == 3:
        fresh = fresh[:, :, -1]

    np.testing.assert_allclose(cached, fresh)


def test_explanation_stability_is_reproducible_for_a_fixed_seed():
    model, X, _ = _tiny_tree_model()
    cases = X.iloc[:4]
    first = explanation_stability(
        model, X, cases, list(X.columns), ["n1", "n2"],
        count=4, perturbations=3, seed=7,
    )
    second = explanation_stability(
        model, X, cases, list(X.columns), ["n1", "n2"],
        count=4, perturbations=3, seed=7,
    )
    pd.testing.assert_frame_equal(first, second)


def test_conformal_observed_mode_uses_true_labels():
    y_cal = np.array([0, 1, 0, 1])
    p_cal = np.array([0.1, 0.9, 0.2, 0.8])
    p_test = np.array([0.15, 0.25])

    observed = conformal_p_values(
        y_cal, p_cal, p_test, label_mode="observed", y_test=np.array([0, 1])
    )
    predicted = conformal_p_values(y_cal, p_cal, p_test, np.array([1, 0]))

    np.testing.assert_allclose(observed, [0.6, 0.2])
    np.testing.assert_allclose(predicted, [0.2, 0.2])
    assert observed[0] != predicted[0]
    with pytest.raises(ValueError, match="label_mode"):
        conformal_p_values(y_cal, p_cal, p_test, np.array([0, 1]), label_mode="wrong")


def test_routing_policies_are_nested():
    y = pd.Series([0, 1, 0, 1])
    probabilities = np.array([0.1, 0.9, 0.4, 0.6])
    stability = np.array([0.9, 0.7, 0.75, 0.95])
    p_values = np.array([0.9, 0.1, 0.5, 0.9])
    _, selected = routing_table(y, probabilities, 0.5, stability, p_values)

    for delta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        p1 = selected[("P1 margin", delta)]
        p2 = selected[("P2 margin + stability", delta)]
        p3 = selected[("P3 + conformal", delta)]
        assert np.all(p3 <= p2)
        assert np.all(p2 <= p1)


def test_group_metrics_reports_expected_confusion_counts():
    groups = pd.Series(["a", "a", "b", "b"])
    result = group_metrics(
        pd.Series([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        groups,
        threshold=0.5,
        stability=np.ones(4),
        review_delta=0.0,
    ).set_index("group")

    assert result.loc["a", "recall"] == 0.0
    assert result.loc["a", "positive_decision_rate"] == 0.0
    assert result.loc["b", "recall"] == 1.0
    assert result.loc["b", "positive_decision_rate"] == 1.0
    assert (result["review_rate"] == 0.0).all()


def test_choose_calibrator_returns_all_candidate_metrics():
    y = pd.Series([0, 0, 1, 1] * 5)
    oof = np.array([0.1, 0.2, 0.8, 0.9] * 5)
    name, calibrator, summary, candidates = choose_calibrator(y, oof)

    assert name in candidates
    assert set(summary) == {"none", "sigmoid", "isotonic"}
    calibrated = apply_calibrator(calibrator, name, oof)
    assert calibrated.shape == oof.shape
