from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from .config import AuditConfig, DEFAULT_AUDIT_CONFIG
from .experiments import fit_oof_and_final, timed_predict
from .metrics import classification_metrics


_TREE_EXPLAINERS: dict[tuple[int, int], Any] = {}


def clear_explainer_cache() -> None:
    """Release TreeExplainer instances held between audit runs."""
    _TREE_EXPLAINERS.clear()


def _get_tree_explainer(model: Any) -> Any:
    import shap

    estimator = model.named_steps["model"]
    estimator = getattr(estimator, "estimator_", estimator)
    key = (id(model), id(estimator))
    if key not in _TREE_EXPLAINERS:
        _TREE_EXPLAINERS[key] = shap.TreeExplainer(estimator)
    return _TREE_EXPLAINERS[key]


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def choose_calibrator(
    y: pd.Series, oof: np.ndarray
) -> tuple[str, Any, dict[str, dict[str, float]], dict[str, Any]]:
    candidates: dict[str, Any] = {
        "sigmoid": LogisticRegression(max_iter=2000).fit(oof.reshape(-1, 1), y),
        "isotonic": IsotonicRegression(out_of_bounds="clip").fit(oof, y),
    }
    summary = {"none": classification_metrics(y, oof)}
    for name, calibrator in candidates.items():
        calibrated = (
            calibrator.predict_proba(oof.reshape(-1, 1))[:, 1]
            if name == "sigmoid"
            else calibrator.predict(oof)
        )
        summary[name] = classification_metrics(y, calibrated)
    selected_name = min(
        candidates,
        key=lambda name: (summary[name]["brier"], summary[name]["ece"]),
    )
    return selected_name, candidates[selected_name], summary, candidates


def apply_calibrator(calibrator: Any, name: str, probabilities: np.ndarray) -> np.ndarray:
    """Apply a fitted sigmoid/isotonic calibrator to raw probabilities."""
    if name == "sigmoid":
        return calibrator.predict_proba(np.asarray(probabilities).reshape(-1, 1))[:, 1]
    return calibrator.predict(probabilities)


# Backwards-compatible private alias used by existing callers.
_calibrate = apply_calibrator


def youden_threshold(y: pd.Series, probabilities: np.ndarray) -> float:
    thresholds = np.unique(np.r_[0.0, probabilities, 1.0])
    best_threshold = 0.5
    best_j = -np.inf
    y_array = np.asarray(y)
    for threshold in thresholds:
        prediction = probabilities >= threshold
        tp = np.sum(prediction & (y_array == 1))
        fn = np.sum(~prediction & (y_array == 1))
        fp = np.sum(prediction & (y_array == 0))
        tn = np.sum(~prediction & (y_array == 0))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        if sensitivity + specificity - 1 > best_j:
            best_j = sensitivity + specificity - 1
            best_threshold = float(threshold)
    return best_threshold


def conformal_p_values(
    y_cal: pd.Series,
    p_cal: np.ndarray,
    p_test: np.ndarray,
    predicted_labels: np.ndarray | None = None,
    *,
    label_mode: str = "predicted",
    y_test: pd.Series | None = None,
) -> np.ndarray:
    """Split-conformal p-values from calibration residuals.

    ``"predicted"`` (the paper's default) scores test cases against their
    predicted label; ``"observed"`` is an opt-in variant using held-out labels.
    """
    if label_mode not in {"predicted", "observed"}:
        raise ValueError(f"label_mode must be 'predicted' or 'observed', got {label_mode!r}.")
    if label_mode == "predicted":
        if predicted_labels is None:
            raise ValueError("predicted_labels is required when label_mode='predicted'.")
        test_labels = np.asarray(predicted_labels)
    else:
        if y_test is None:
            raise ValueError("y_test is required when label_mode='observed'.")
        test_labels = np.asarray(y_test, dtype=int)

    scores = np.abs(np.asarray(p_cal) - np.asarray(y_cal, dtype=float))
    test_scores = np.abs(np.asarray(p_test) - test_labels)
    return np.array([(1 + np.sum(scores >= score)) / (len(scores) + 1) for score in test_scores])


def _tree_shap(model: Any, X: pd.DataFrame) -> np.ndarray:
    transformed = model.named_steps["preprocess"].transform(X)
    values = np.asarray(_get_tree_explainer(model)(transformed).values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape[1] != X.shape[1]:
        raise ValueError("SHAP output columns do not align with the encoded feature columns.")
    return values


def local_shap_values(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Public accessor for signed local TreeSHAP values."""
    return _tree_shap(model, X)


def _lime_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_cases: pd.DataFrame,
    count: int,
    num_samples: int = 1500,
) -> list[list[str]]:
    from lime.lime_tabular import LimeTabularExplainer

    preprocessor = model.named_steps["preprocess"]
    encoded_train = np.asarray(preprocessor.transform(X_train), dtype=float)
    encoded_cases = np.asarray(preprocessor.transform(X_cases), dtype=float)
    feature_names = list(preprocessor.get_feature_names_out())
    explainer = LimeTabularExplainer(
        encoded_train,
        feature_names=feature_names,
        class_names=["good", "default"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    rows = encoded_cases.iloc[:count] if isinstance(encoded_cases, pd.DataFrame) else encoded_cases[:count]
    results = []
    for row in rows:
        explanation = explainer.explain_instance(
            row,
            lambda encoded: _predict_encoded(model, encoded),
            num_features=min(5, len(feature_names)),
            num_samples=num_samples,
        )
        ranked = sorted(explanation.as_map()[1], key=lambda item: abs(item[1]), reverse=True)
        results.append([feature_names[index] for index, _ in ranked])
    return results


def _predict_encoded(model: Any, encoded: np.ndarray) -> np.ndarray:
    classifier = model.named_steps["model"]
    probabilities = classifier.predict_proba(encoded)[:, 1]
    return np.column_stack([1 - probabilities, probabilities])


def explanation_stability(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_names: list[str],
    perturbation_features: list[str],
    count: int = 150,
    perturbations: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    cases = X_test.iloc[:count].copy()
    base = _tree_shap(model, cases)
    numeric_features = [name for name in perturbation_features if name in feature_names]
    if not numeric_features:
        raise ValueError("No requested numeric features are available for perturbation.")
    quantiles = X_train[numeric_features].quantile([0.05, 0.95])
    spans = X_train[numeric_features].max() - X_train[numeric_features].min()
    scores = np.zeros(len(cases), dtype=float)
    top_scores = np.zeros(len(cases), dtype=float)
    displacement = np.zeros(len(cases), dtype=float)
    for index in range(perturbations):
        changed = cases.copy()
        rng = np.random.default_rng(seed + index)
        for feature in numeric_features:
            noise = rng.uniform(-0.05, 0.05, len(changed)) * spans[feature]
            changed[feature] = (changed[feature].to_numpy(dtype=float) + noise).clip(
                quantiles.loc[0.05, feature], quantiles.loc[0.95, feature]
            )
        perturbed = _tree_shap(model, changed)
        for case_index, (original, noisy) in enumerate(zip(base, perturbed)):
            scores[case_index] += _safe_spearman(np.abs(original), np.abs(noisy)) / perturbations
            top_a = set(np.argsort(np.abs(original))[-5:])
            top_b = set(np.argsort(np.abs(noisy))[-5:])
            top_scores[case_index] += len(top_a & top_b) / max(len(top_a | top_b), 1) / perturbations
            denominator = np.abs(original).sum()
            displacement[case_index] += (
                np.abs(original - noisy).sum() / denominator if denominator > 0 else 0.0
            ) / perturbations
    result = pd.DataFrame(
        {"stability": scores, "top5_stability": top_scores, "displacement": displacement},
        index=cases.index,
    )
    result["decision_margin"] = np.nan
    return result


def shap_lime_agreement(
    model: Any,
    X_train: pd.DataFrame,
    X_cases: pd.DataFrame,
    count: int = 100,
    num_samples: int = 1500,
) -> pd.DataFrame:
    cases = X_cases.iloc[:count]
    shap_values = _tree_shap(model, cases)
    lime_ranked = _lime_explanations(model, X_train, cases, count, num_samples)
    feature_names = list(model.named_steps["preprocess"].get_feature_names_out())
    rows = []
    for row_index, shap_row, lime_features in zip(cases.index, shap_values, lime_ranked):
        shap_ranked = [feature_names[index] for index in np.argsort(np.abs(shap_row))[::-1]]
        shap_top5 = set(shap_ranked[:5])
        lime_top5 = set(lime_features[:5])
        rows.append(
            {
                "case_index": int(row_index),
                "shap_top5": " | ".join(shap_ranked[:5]),
                "lime_top5": " | ".join(lime_features[:5]),
                "jaccard_top3": len(set(shap_ranked[:3]) & set(lime_features[:3]))
                / max(len(set(shap_ranked[:3]) | set(lime_features[:3])), 1),
                "jaccard_top5": len(shap_top5 & lime_top5)
                / max(len(shap_top5 | lime_top5), 1),
            }
        )
    return pd.DataFrame(rows)


def routing_table(
    y: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    stability: np.ndarray,
    p_values: np.ndarray | None = None,
    gamma: float = 0.8,
    alpha: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    y_array = np.asarray(y, dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    errors = predicted != y_array
    margin = np.abs(probabilities - threshold)
    p_values = np.ones(len(y_array)) if p_values is None else np.asarray(p_values)
    risk = errors | (stability < gamma)
    rows = []
    selected = {}
    for delta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        margin_auto = margin >= delta
        stability_auto = margin_auto & (stability >= gamma)
        conformal_auto = stability_auto & (p_values >= alpha)
        for name, auto in (
            ("P1 margin", margin_auto),
            ("P2 margin + stability", stability_auto),
            ("P3 + conformal", conformal_auto),
        ):
            selected[(name, delta)] = auto
            rows.append(
                {
                    "comparison": "threshold",
                    "policy": name,
                    "delta": delta,
                    "target_coverage": np.nan,
                    "coverage": float(auto.mean()),
                    "selective_error": float(errors[auto].mean()) if auto.any() else 0.0,
                    "composite_review_need": float(risk[auto].mean()) if auto.any() else 0.0,
                    "reviewed": int((~auto).sum()),
                }
            )
    for target_coverage in (0.55, 0.75, 0.84):
        desired_count = max(1, int(round(target_coverage * len(y_array))))
        ranked_candidates = {
            "P1 margin": np.arange(len(y_array)),
            "P2 margin + stability": np.flatnonzero(stability >= gamma),
            "P3 + conformal": np.flatnonzero((stability >= gamma) & (p_values >= alpha)),
        }
        for name, candidates in ranked_candidates.items():
            order = candidates[np.argsort(margin[candidates])[::-1]]
            chosen = order[:desired_count]
            auto = np.zeros(len(y_array), dtype=bool)
            auto[chosen] = True
            selected[(name, target_coverage)] = auto
            rows.append(
                {
                    "comparison": "matched_coverage",
                    "policy": name,
                    "delta": float(margin[order[desired_count - 1]]) if len(order) >= desired_count else np.nan,
                    "target_coverage": target_coverage,
                    "coverage": float(auto.mean()),
                    "selective_error": float(errors[auto].mean()) if auto.any() else 0.0,
                    "composite_review_need": float(risk[auto].mean()) if auto.any() else 0.0,
                    "reviewed": int((~auto).sum()),
                }
            )
    return pd.DataFrame(rows), selected


def group_metrics(
    y: pd.Series,
    probabilities: np.ndarray,
    groups: pd.Series,
    threshold: float,
    stability: np.ndarray | None = None,
    review_delta: float = 0.20,
    gamma: float = 0.8,
) -> pd.DataFrame:
    rows = []
    y_array = np.asarray(y, dtype=int)
    predicted = np.asarray(probabilities) >= threshold
    margin = np.abs(np.asarray(probabilities) - threshold)
    stability = np.ones(len(y_array)) if stability is None else np.asarray(stability)
    auto_decision = (margin >= review_delta) & (stability >= gamma)
    for group in sorted(pd.Series(groups).fillna("unknown").unique()):
        mask = np.asarray(pd.Series(groups).fillna("unknown") == group)
        if not mask.any():
            continue
        tn, fp, fn, tp = confusion_matrix(y_array[mask], predicted[mask], labels=[0, 1]).ravel()
        rows.append(
            {
                "group": group,
                "n": int(mask.sum()),
                "recall": tp / max(tp + fn, 1),
                "false_positive_rate": fp / max(fp + tn, 1),
                "positive_decision_rate": float(predicted[mask].mean()),
                "automatic_decision_rate": float(auto_decision[mask].mean()),
                "review_rate": float((~auto_decision[mask]).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_audit(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    numeric: list[str],
    perturbation_features: list[str],
    dataset: str,
    config: AuditConfig | None = None,
) -> dict[str, Any]:
    config = config if config is not None else DEFAULT_AUDIT_CONFIG
    clear_explainer_cache()
    X_train, X_test, y_train, y_test, _, group_test = train_test_split(
        X, y, groups, test_size=0.2, stratify=y, random_state=42
    )
    X_development, X_cal, y_development, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    model, oof = fit_oof_and_final(X_development, y_development, numeric)
    calibrator_name, calibrator, calibration_oof, calibrators = choose_calibrator(y_development, oof)
    raw_probabilities, prediction_ms = timed_predict(model, X_test)
    probabilities = apply_calibrator(calibrator, calibrator_name, raw_probabilities)
    threshold = youden_threshold(
        y_development, apply_calibrator(calibrator, calibrator_name, oof)
    )
    baseline = classification_metrics(y_test, probabilities, threshold)
    calibration_rows = [{"method": "none", **classification_metrics(y_test, raw_probabilities, threshold)}]
    for method, fitted_calibrator in calibrators.items():
        test_calibrated = apply_calibrator(fitted_calibrator, method, raw_probabilities)
        calibration_rows.append(
            {"method": method, **classification_metrics(y_test, test_calibrated, threshold)}
        )

    calibration_probabilities, _ = timed_predict(model, X_cal)
    calibration_probabilities = apply_calibrator(
        calibrator, calibrator_name, calibration_probabilities
    )
    p_values = conformal_p_values(
        y_cal,
        calibration_probabilities,
        probabilities,
        (probabilities >= threshold).astype(int),
        label_mode=config.conformal_label_mode,
        y_test=y_test,
    )
    stability_start = time.perf_counter()
    stability_table = explanation_stability(
        model,
        X_development,
        X_test,
        list(X.columns),
        perturbation_features,
        count=min(config.stability_cases, len(X_test)),
        perturbations=config.perturbations,
        seed=config.seed,
    )
    stability_ms_per_case = (
        (time.perf_counter() - stability_start) * 1000 / max(len(stability_table), 1)
    )
    stability_values = np.full(len(X_test), float(stability_table["stability"].median()))
    stability_values[: len(stability_table)] = stability_table["stability"].to_numpy()
    stability_table["decision_margin"] = np.abs(probabilities[: len(stability_table)] - threshold)
    stability_table["margin_stability_rho"] = _safe_spearman(
        stability_table["stability"].to_numpy(), stability_table["decision_margin"].to_numpy()
    )
    agreement_start = time.perf_counter()
    agreement = shap_lime_agreement(
        model,
        X_development,
        X_test,
        count=min(config.agreement_cases, len(X_test)),
        num_samples=config.lime_samples,
    )
    explanation_audit_ms_per_case = (
        (time.perf_counter() - agreement_start) * 1000 / max(len(agreement), 1)
    )
    routing, _ = routing_table(y_test, probabilities, threshold, stability_values, p_values)
    return {
        "model": model,
        "calibrator": calibrator,
        "calibrator_name": calibrator_name,
        "X_train": X_development,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "groups_test": group_test,
        "probabilities": probabilities,
        "threshold": threshold,
        "baseline_metrics": baseline,
        "calibration_oof_metrics": calibration_oof,
        "calibration_metrics": pd.DataFrame(calibration_rows),
        "stability": stability_table,
        "agreement": agreement,
        "routing": routing,
        "group_metrics": group_metrics(y_test, probabilities, group_test, threshold, stability_values),
        "p_values": p_values,
        "prediction_ms": prediction_ms,
        "stability_ms_per_case": stability_ms_per_case,
        "shap_lime_audit_ms_per_case": explanation_audit_ms_per_case,
        "dataset": dataset,
        "audit_config": config,
    }
