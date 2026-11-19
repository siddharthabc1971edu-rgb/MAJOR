# Architecture

```text
app.py ────────────────┐
run_experiment.py ─────┼──> credit_audit.runner ──> credit_audit.audit ──> experiments / metrics / models
tests/ ────────────────┘            │                      │
                                    │                      └──> shap / lime (cached TreeExplainer)
                                    └──> data / paths / config / scoring
```

## Modules

| Module | Responsibility |
| --- | --- |
| `paths.py` | Project-root-anchored `RAW_DIR` and `ARTIFACTS_DIR`. |
| `config.py` | `AuditConfig` with paper defaults and validation. Non-default values are opt-in. |
| `data.py` | Download/validate UCI ZIPs, map targets, coerce numeric columns, expose dataset-specific feature lists. |
| `models.py` | Estimator factories, fold-local class weighting wrapper, preprocessing, and samplers. |
| `experiments.py` | Five-fold model/imbalance comparisons and OOF fitting for the audit model. |
| `metrics.py` | ROC AUC, PR AUC, Brier, recall, precision, and 10-bin ECE. |
| `audit.py` | Calibrator choice, Youden threshold, conformal p-values, SHAP stability, SHAP–LIME agreement, routing, group metrics, and `run_audit`. |
| `runner.py` | Writes CSV/JSON/joblib artifacts and exposes `run_audit_experiment` / `run_full_experiment`. |
| `scoring.py` | Loads a saved bundle and scores one applicant with the same calibration and threshold. |

## Data flow

1. `load_dataset` returns raw features `X`, binary target `y`, and group labels.
2. `run_audit` makes an 80/20 test split, then a further 80/20 development/calibration split.
3. `fit_oof_and_final` produces out-of-fold probabilities for calibration selection and a final fitted pipeline.
4. `choose_calibrator` selects sigmoid or isotonic by Brier, then ECE, on development OOF predictions.
5. The selected calibrator and a Youden threshold are applied to held-out test probabilities.
6. Explanation audits compute SHAP stability and SHAP–LIME agreement on a configurable number of test cases. Remaining cases receive median stability, as in the paper.
7. `runner` persists reports plus a self-contained model bundle (including `X_reference` for on-demand stability in the dashboard).

## Reproducibility notes

- Defaults in `AuditConfig` are `stability_cases=150`, `perturbations=10`, `agreement_cases=100`, `lime_samples=1500`, `seed=42`, `conformal_label_mode="predicted"`.
- `summary.json` records both `audit_config` and installed package versions.
- TreeExplainer instances are cached per fitted model during an audit and cleared at the start of each new `run_audit` call.
