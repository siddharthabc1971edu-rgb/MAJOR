from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest

from credit_audit.config import AuditConfig
from credit_audit.runner import run_audit_experiment


@pytest.mark.slow
def test_run_audit_experiment_writes_expected_artifacts(tmp_path):
    config = AuditConfig(
        stability_cases=5,
        perturbations=2,
        agreement_cases=3,
        lime_samples=50,
    )

    result = run_audit_experiment("german", tmp_path, config)

    report_dir = tmp_path / "german"
    expected_files = (
        "summary.json",
        "explanation_stability.csv",
        "shap_lime_agreement.csv",
        "routing.csv",
        "routing_matched.csv",
        "group_metrics.csv",
        "calibration.csv",
        "predictions.csv",
        "credit_audit_model.joblib",
    )
    for filename in expected_files:
        assert (report_dir / filename).exists(), filename

    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["audit_config"] == config.to_dict()
    assert "environment" in summary
    assert {"auc", "pr_auc", "brier", "ece", "recall", "precision"} <= set(
        summary["baseline_metrics"]
    )

    stability = pd.read_csv(report_dir / "explanation_stability.csv")
    predictions = pd.read_csv(report_dir / "predictions.csv")
    assert len(stability) == 5
    assert len(predictions) == len(result["y_test"])
    assert {"stability", "top5_stability", "displacement"} <= set(stability.columns)

    bundle = joblib.load(report_dir / "credit_audit_model.joblib")
    assert bundle["audit_config"] == config.to_dict()
    assert "X_reference" in bundle
    assert bundle["perturbation_features"] == [
        "duration_months", "credit_amount", "installment_rate",
        "residence_since", "age", "existing_credits", "num_dependents",
    ]
