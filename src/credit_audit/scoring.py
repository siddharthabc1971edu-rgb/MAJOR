"""Single-applicant scoring from a saved audit bundle.

Uses exactly the preprocessing, model, calibrator, and threshold produced by
``run_audit_experiment``. Optional on-demand stability uses the same
perturbation routine as the paper audit but is not part of the held-out audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .audit import apply_calibrator, explanation_stability, local_shap_values
from .config import AuditConfig, DEFAULT_AUDIT_CONFIG
from .paths import ARTIFACTS_DIR


REVIEW_DELTA = 0.20
GAMMA = 0.8
REQUIRED_KEYS = {
    "model",
    "calibrator",
    "calibrator_name",
    "threshold",
    "feature_names",
    "numeric_features",
}


def default_bundle_path(dataset: str, artifacts_dir: Path = ARTIFACTS_DIR) -> Path:
    return artifacts_dir / dataset / "credit_audit_model.joblib"


def load_bundle(path: Path) -> dict[str, Any]:
    """Load and validate a fitted audit bundle."""
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model at {path}. Run 'python run_experiment.py --mode audit "
            f"--dataset {path.parent.name}' first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(bundle).__name__}.")
    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Saved bundle is missing required keys: {sorted(missing)}.")
    return bundle


def _coerce_case(case: pd.DataFrame | pd.Series | dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    if isinstance(case, dict):
        frame = pd.DataFrame([case])
    elif isinstance(case, pd.Series):
        frame = case.to_frame().T
    elif isinstance(case, pd.DataFrame):
        frame = case.copy()
    else:
        raise TypeError("case must be a DataFrame, Series, or dict of feature values.")
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature values: {missing}.")
    return frame[feature_names].reset_index(drop=True)


def score_case(
    bundle: dict[str, Any],
    case: pd.DataFrame | pd.Series | dict[str, Any],
    *,
    compute_stability: bool = False,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    """Score one applicant and return decision, routing, and top-5 SHAP reasons."""
    config = config if config is not None else DEFAULT_AUDIT_CONFIG
    X = _coerce_case(case, list(bundle["feature_names"]))
    model = bundle["model"]

    raw = model.predict_proba(X)[:, 1]
    probability = float(
        apply_calibrator(bundle["calibrator"], bundle["calibrator_name"], raw)[0]
    )
    threshold = float(bundle["threshold"])
    predicted_default = int(probability >= threshold)
    margin = float(abs(probability - threshold))

    stability: float | None = None
    stability_available = compute_stability and "X_reference" in bundle
    if stability_available:
        table = explanation_stability(
            model,
            bundle["X_reference"],
            X,
            list(bundle["feature_names"]),
            list(bundle.get("perturbation_features", bundle["numeric_features"])),
            count=1,
            perturbations=config.perturbations,
            seed=config.seed,
        )
        stability = float(table["stability"].iloc[0])

    auto_decision = bool(margin >= REVIEW_DELTA and (stability is None or stability >= GAMMA))
    review_reasons: list[str] = []
    if margin < REVIEW_DELTA:
        review_reasons.append("probability is close to the decision threshold")
    if stability is None:
        if compute_stability:
            review_reasons.append("stability could not be computed for this saved bundle")
    elif stability < GAMMA:
        review_reasons.append("local explanation is unstable under bounded perturbations")

    shap_row = local_shap_values(model, X)[0]
    encoded_names = list(model.named_steps["preprocess"].get_feature_names_out())
    top_indices = np.argsort(np.abs(shap_row))[::-1][:5]
    shap_top5 = [
        {"feature": encoded_names[index], "contribution": float(shap_row[index])}
        for index in top_indices
    ]

    return {
        "probability": probability,
        "threshold": threshold,
        "predicted_default": predicted_default,
        "decision_margin": margin,
        "stability": stability,
        "stability_available": bool(stability_available),
        "auto_decision": auto_decision,
        "review_reasons": review_reasons,
        "shap_top5": shap_top5,
    }
