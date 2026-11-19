from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Allow the app to run from the project checkout before the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from credit_audit.config import AuditConfig, DEFAULT_AUDIT_CONFIG
from credit_audit.data import load_dataset
from credit_audit.paths import ARTIFACTS_DIR
from credit_audit.runner import run_audit_experiment
from credit_audit.scoring import default_bundle_path, load_bundle, score_case


@st.cache_data(show_spinner=False)
def cached_load_dataset(dataset: str):
    return load_dataset(dataset)


@st.cache_data(show_spinner=False)
def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def cached_bundle(path: Path):
    return load_bundle(path)


st.set_page_config(page_title="Credit Risk Audit Lab", page_icon="📊", layout="wide")
st.title("Stability-Aware Credit Risk Audit")
st.caption("A classroom reproduction of the paper's audit layer—not a lending decision system.")

with st.sidebar:
    st.header("Experiment")
    dataset = st.selectbox("Benchmark dataset", ["german", "australian"], format_func=str.title)
    st.info("The first run downloads the public UCI dataset and trains the model. Internet access is required.")
    with st.expander("Advanced — audit budget"):
        st.caption("Defaults reproduce the paper protocol exactly.")
        stability_cases = st.number_input(
            "Stability cases", min_value=1, value=DEFAULT_AUDIT_CONFIG.stability_cases,
            key="stability_cases",
        )
        perturbations = st.number_input(
            "Perturbations per case", min_value=1, value=DEFAULT_AUDIT_CONFIG.perturbations,
            key="perturbations",
        )
        agreement_cases = st.number_input(
            "SHAP-LIME cases", min_value=1, value=DEFAULT_AUDIT_CONFIG.agreement_cases,
            key="agreement_cases",
        )
        lime_samples = st.number_input(
            "LIME samples per case", min_value=1, value=DEFAULT_AUDIT_CONFIG.lime_samples,
            key="lime_samples",
        )
        conformal_label_mode = st.selectbox(
            "Conformal labels",
            ["predicted", "observed"],
            index=0,
            help="Predicted is the paper default; observed is an opt-in variant.",
            key="conformal_label_mode",
        )
    run_config = AuditConfig(
        stability_cases=int(stability_cases),
        perturbations=int(perturbations),
        agreement_cases=int(agreement_cases),
        lime_samples=int(lime_samples),
        conformal_label_mode=str(conformal_label_mode),
    )
    if not run_config.is_default:
        st.warning("Non-default audit budget — results are an opt-in variant, not the paper protocol.")
    run = st.button("Train model and run audit", type="primary", use_container_width=True)

st.markdown(
    """
    **What this demo audits:** default-risk probabilities, calibration, TreeSHAP explanations,
    local explanation stability under bounded input noise, SHAP–LIME agreement, and routing
    to an illustrative human-review queue. All resampling happens inside training folds.
    """
)

try:
    X, y, groups = cached_load_dataset(dataset)
    left, middle, right = st.columns(3)
    left.metric("Applicants", f"{len(X):,}")
    middle.metric("Input attributes", X.shape[1])
    right.metric("Positive/default rate", f"{y.mean():.1%}")
    st.subheader("Dataset preview")
    preview = X.copy()
    preview["default"] = y
    st.dataframe(preview.head(8), use_container_width=True)
except Exception as exc:
    st.error(f"Could not load {dataset.title()} Credit data: {exc}")
    st.stop()

if run:
    with st.spinner("Training the model and computing the explanation audit. This can take a few minutes..."):
        try:
            run_audit_experiment(dataset, ARTIFACTS_DIR, run_config)
            st.session_state["audit_result"] = {
                "dataset": dataset,
                "summary_path": str(ARTIFACTS_DIR / dataset / "summary.json"),
            }
            read_json.clear()
            read_csv.clear()
            cached_bundle.clear()
            st.success("Experiment complete. Model and CSV/JSON reports were saved in artifacts/.")
            st.rerun()
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")

state = st.session_state.get("audit_result")
cached_summary = ARTIFACTS_DIR / dataset / "summary.json"
if (not state or state["dataset"] != dataset) and cached_summary.exists():
    state = {"dataset": dataset}
if state and state["dataset"] == dataset:
    report_dir = ARTIFACTS_DIR / dataset
    summary_path = report_dir / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        st.subheader("Held-out test results")
        metrics = summary["baseline_metrics"]
        columns = st.columns(5)
        for column, key, label in zip(
            columns, ("auc", "pr_auc", "brier", "ece", "recall"),
            ("ROC AUC", "PR AUC", "Brier", "ECE", "Recall"),
        ):
            column.metric(label, f"{metrics[key]:.3f}")
        st.write(
            f"Selected calibration: **{summary['selected_calibrator']}** · "
            f"Youden threshold: **{summary['threshold']:.3f}** · "
            f"Mean explanation stability: **{summary['mean_stability']:.3f}** · "
            f"Unstable (S < 0.8): **{summary['unstable_share']:.1%}** · "
            f"Mean top-5 SHAP–LIME Jaccard: **{summary['mean_shap_lime_top5_jaccard']:.3f}**"
        )
        st.caption(
            "Measured latency (ms/case): "
            f"prediction {summary['mean_prediction_ms']:.3f}, "
            f"stability audit {summary['stability_audit_ms_per_case']:.2f}, "
            f"SHAP–LIME audit {summary['shap_lime_audit_ms_per_case']:.2f}"
        )
        st.caption("Held-out comparison of uncalibrated, sigmoid/Platt, and isotonic probabilities")
        st.dataframe(read_csv(report_dir / "calibration.csv"), use_container_width=True, hide_index=True)

        st.subheader("Reliability-guided review routing")
        routing = read_csv(report_dir / "routing.csv")
        st.caption("Composite review-need rate as the margin-review threshold changes")
        chart = routing.pivot(index="delta", columns="policy", values="composite_review_need")
        st.line_chart(chart)
        st.dataframe(routing, use_container_width=True, hide_index=True)
        matched_path = report_dir / "routing_matched.csv"
        if matched_path.exists():
            st.caption("Matched-coverage comparison for P1/P2/P3")
            st.dataframe(read_csv(matched_path), use_container_width=True, hide_index=True)
        st.warning(summary["interpretation_note"])

        st.subheader("Explanation and group audits")
        stability = read_csv(report_dir / "explanation_stability.csv")
        agreement = read_csv(report_dir / "shap_lime_agreement.csv")
        groups_result = read_csv(report_dir / "group_metrics.csv")
        first, second = st.columns(2)
        first.caption("Per-case stability distribution (S(x))")
        stability_counts, stability_edges = np.histogram(stability["stability"], bins=10, range=(0, 1))
        stability_histogram = pd.DataFrame(
            {"cases": stability_counts},
            index=(stability_edges[:-1] + stability_edges[1:]) / 2,
        )
        first.bar_chart(stability_histogram)
        second.caption("Distribution of per-case SHAP–LIME top-5 agreement")
        agreement_counts, agreement_edges = np.histogram(agreement["jaccard_top5"], bins=10, range=(0, 1))
        agreement_histogram = pd.DataFrame(
            {"cases": agreement_counts},
            index=(agreement_edges[:-1] + agreement_edges[1:]) / 2,
        )
        second.bar_chart(agreement_histogram)
        if not agreement.empty:
            selected_case = st.selectbox(
                "Inspect a test applicant's local explanation comparison",
                agreement["case_index"].tolist(),
            )
            example = agreement.loc[agreement["case_index"] == selected_case].iloc[0]
            st.write(f"**Top SHAP features:** {example['shap_top5']}")
            st.write(f"**Top LIME features:** {example['lime_top5']}")
            st.write(
                f"Top-3 Jaccard: {example['jaccard_top3']:.3f} · "
                f"Top-5 Jaccard: {example['jaccard_top5']:.3f}"
            )
        st.dataframe(groups_result, use_container_width=True, hide_index=True)

        with st.expander("Paper scope, caveats, and limitations"):
            st.markdown(
                """
                - The paper evaluates six model families and four imbalance strategies with
                  leakage-free five-fold CV; run `python run_experiment.py --mode full` for those tables.
                - Detailed calibration, explanation, routing, and group analyses use a single
                  stratified 80/20 split. This demo computes stability for up to 150 test cases
                  and assigns the median stability to remaining cases, as described in the paper.
                - Composite review-need includes the same stability threshold used in routing,
                  so the paper explicitly cautions that this comparison is partly definitional.
                - Conformal p-values are included, but the paper found little impact on these datasets.
                - Sex is retained only to reproduce the German benchmark and is not appropriate
                  as a real lending decision feature. This software is for education, not deployment.
                - The paper's asymmetric 5:1 German Credit cost matrix is not modeled. Results
                  also do not establish deployment fairness or explanation faithfulness.
                """
            )
        for filename, title in (
            ("model_comparison.csv", "Five-fold model comparison"),
            ("imbalance_comparison.csv", "Five-fold imbalance comparison"),
        ):
            table_path = report_dir / filename
            if table_path.exists():
                with st.expander(title):
                    st.dataframe(read_csv(table_path), use_container_width=True, hide_index=True)
        st.download_button(
            "Download test predictions",
            read_csv(report_dir / "predictions.csv").to_csv(index=False),
            file_name=f"{dataset}_predictions.csv",
            mime="text/csv",
        )

st.subheader("Score one applicant")
st.caption(
    "Loads the saved audit model and applies the same calibrator and threshold. "
    "This is an educational illustration, not a lending decision."
)
bundle_path = default_bundle_path(dataset, ARTIFACTS_DIR)
if not bundle_path.exists():
    st.info("Run the audit first to create a saved model for this dataset.")
else:
    try:
        bundle = cached_bundle(bundle_path)
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
    else:
        compute_stability = st.checkbox(
            "Compute on-demand stability for this applicant",
            value=False,
            key=f"score_stability_{dataset}",
        )
        example_index = int(
            st.number_input(
                "Start from dataset row",
                min_value=0,
                max_value=len(X) - 1,
                value=0,
                key=f"score_example_{dataset}",
            )
        )
        template = X.iloc[example_index]
        numeric_features = set(bundle["numeric_features"])
        with st.form(f"score_form_{dataset}"):
            columns = st.columns(2)
            case_values: dict[str, object] = {}
            for index, feature in enumerate(bundle["feature_names"]):
                container = columns[index % 2]
                raw_value = template.get(feature)
                if feature in numeric_features:
                    default = float(raw_value) if pd.notna(raw_value) else 0.0
                    case_values[feature] = container.number_input(
                        feature, value=default, format="%.4f", key=f"score_{dataset}_{feature}"
                    )
                else:
                    options = sorted(X[feature].dropna().astype(str).unique().tolist())
                    if not options:
                        options = [str(raw_value)] if pd.notna(raw_value) else ["unknown"]
                    current = str(raw_value) if pd.notna(raw_value) and str(raw_value) in options else options[0]
                    case_values[feature] = container.selectbox(
                        feature, options=options, index=options.index(current),
                        key=f"score_{dataset}_{feature}",
                    )
            submitted = st.form_submit_button("Score applicant", type="primary")

        if submitted:
            try:
                scored = score_case(
                    bundle,
                    case_values,
                    compute_stability=compute_stability,
                    config=run_config,
                )
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")
            else:
                decision_col, probability_col, margin_col = st.columns(3)
                decision_col.metric(
                    "Decision route", "Automatic" if scored["auto_decision"] else "Human review"
                )
                probability_col.metric("Default probability", f"{scored['probability']:.1%}")
                margin_col.metric("Decision margin", f"{scored['decision_margin']:.3f}")
                st.write(
                    f"Predicted class: **{'default' if scored['predicted_default'] else 'good'}** · "
                    f"Youden threshold: **{scored['threshold']:.3f}**"
                )
                if scored["stability"] is not None:
                    st.write(f"On-demand stability S(x): **{scored['stability']:.3f}**")
                if scored["review_reasons"]:
                    st.warning("Review flags: " + "; ".join(scored["review_reasons"]) + ".")
                else:
                    st.success("No review flags under the current margin/stability rules.")
                st.caption("Top-5 TreeSHAP reasons (signed contribution toward default)")
                st.dataframe(
                    pd.DataFrame(scored["shap_top5"]),
                    use_container_width=True,
                    hide_index=True,
                )

with st.expander("What the full paper experiment includes"):
    st.markdown(
        """
        1. German Credit and Australian Credit Approval datasets from UCI.
        2. Logistic Regression, weighted Logistic Regression, Random Forest, XGBoost,
           LightGBM, and CatBoost.
        3. Five-fold stratified CV and XGBoost comparisons for no resampling, class weighting,
           SMOTE, and SMOTE-ENN; samplers are fit only on each training fold.
        4. Sigmoid/Platt and isotonic calibration, ECE, Brier score, and AUC.
        5. Per-case TreeSHAP stability under ten bounded perturbations, top-5 overlap,
           attribution displacement, and SHAP–LIME top-k agreement.
        6. Youden threshold, margin/stability/conformal routing, matched-coverage summaries,
           group metrics, and prediction-time overhead.
        """
    )
