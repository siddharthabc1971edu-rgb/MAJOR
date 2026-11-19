from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .audit import run_audit
from .config import AuditConfig, DEFAULT_AUDIT_CONFIG
from .data import NUMERIC_BY_DATASET, PERTURBATION_BY_DATASET, load_dataset
from .experiments import compare_imbalance, compare_models
from .paths import ARTIFACTS_DIR


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _environment_info() -> dict[str, object]:
    packages = (
        "numpy", "pandas", "scipy", "scikit-learn", "imbalanced-learn",
        "xgboost", "lightgbm", "catboost", "shap", "lime", "streamlit",
    )
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version.split()[0], "packages": versions}


def run_audit_experiment(
    dataset: str = "german",
    output_dir: Path = ARTIFACTS_DIR,
    config: AuditConfig | None = None,
) -> dict:
    config = config if config is not None else DEFAULT_AUDIT_CONFIG
    X, y, groups = load_dataset(dataset)
    numeric = NUMERIC_BY_DATASET[dataset]
    perturbation = PERTURBATION_BY_DATASET[dataset]
    result = run_audit(X, y, groups, numeric, perturbation, dataset, config)

    report_dir = output_dir / dataset
    report_dir.mkdir(parents=True, exist_ok=True)
    result["stability"].to_csv(report_dir / "explanation_stability.csv", index=False)
    result["agreement"].to_csv(report_dir / "shap_lime_agreement.csv", index=False)
    result["routing"].query("comparison == 'threshold'").to_csv(
        report_dir / "routing.csv", index=False
    )
    result["routing"].query("comparison == 'matched_coverage'").to_csv(
        report_dir / "routing_matched.csv", index=False
    )
    result["group_metrics"].to_csv(report_dir / "group_metrics.csv", index=False)
    result["calibration_metrics"].to_csv(report_dir / "calibration.csv", index=False)
    stability = np.full(len(result["y_test"]), float(result["stability"]["stability"].median()))
    stability[: len(result["stability"])] = result["stability"]["stability"].to_numpy()
    predictions = pd.DataFrame(
        {
            "actual_default": result["y_test"].to_numpy(),
            "default_probability": result["probabilities"],
            "predicted_default": (result["probabilities"] >= result["threshold"]).astype(int),
            "decision_margin": abs(result["probabilities"] - result["threshold"]),
            "stability": stability,
            "conformal_p_value": result["p_values"],
            "group": result["groups_test"].to_numpy(),
        }
    )
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    joblib.dump(
        {
            "model": result["model"],
            "calibrator": result["calibrator"],
            "calibrator_name": result["calibrator_name"],
            "threshold": result["threshold"],
            "feature_names": list(X.columns),
            "numeric_features": numeric,
            "perturbation_features": perturbation,
            "X_reference": result["X_train"],
            "dataset": dataset,
            "audit_config": config.to_dict(),
        },
        report_dir / "credit_audit_model.joblib",
    )
    metadata = {
        "dataset": dataset,
        "n_rows": len(X),
        "n_features": X.shape[1],
        "default_rate": float(y.mean()),
        "selected_calibrator": result["calibrator_name"],
        "threshold": result["threshold"],
        "baseline_metrics": result["baseline_metrics"],
        "calibration_test_metrics": result["calibration_metrics"].to_dict(orient="records"),
        "mean_stability": float(result["stability"]["stability"].mean()),
        "unstable_share": float((result["stability"]["stability"] < 0.8).mean()),
        "mean_shap_lime_top3_jaccard": float(result["agreement"]["jaccard_top3"].mean()),
        "mean_shap_lime_top5_jaccard": float(result["agreement"]["jaccard_top5"].mean()),
        "mean_prediction_ms": result["prediction_ms"],
        "stability_audit_ms_per_case": result["stability_ms_per_case"],
        "shap_lime_audit_ms_per_case": result["shap_lime_audit_ms_per_case"],
        "interpretation_note": (
            "Routing composite risk includes the same stability flag used by P2; "
            "this is not independent evidence that flagged explanations are harmful."
        ),
        "audit_config": config.to_dict(),
        "environment": _environment_info(),
    }
    (report_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def run_full_experiment(
    output_dir: Path = ARTIFACTS_DIR, config: AuditConfig | None = None
) -> dict[str, pd.DataFrame]:
    config = config if config is not None else DEFAULT_AUDIT_CONFIG
    tables: dict[str, pd.DataFrame] = {}
    for dataset in ("german", "australian"):
        X, y, _ = load_dataset(dataset)
        numeric = NUMERIC_BY_DATASET[dataset]
        tables[f"{dataset}_models"] = compare_models(X, y, numeric)
        tables[f"{dataset}_imbalance"] = compare_imbalance(X, y, numeric)
        for suffix, table in (
            ("model_comparison", tables[f"{dataset}_models"]),
            ("imbalance_comparison", tables[f"{dataset}_imbalance"]),
        ):
            _write_frame(table, output_dir / dataset / f"{suffix}.csv")
    run_audit_experiment("german", output_dir, config)
    return tables
