# Stability-Aware Credit Risk Audit Lab

A classroom-ready implementation of the paper **“Stability-Aware and Confidence-Guided Explainable AI for Credit Risk Assessment: An Audit Layer with Explanation-Reliability Routing”** by Jatin Sachdeva, Siddharth Singh, Gauravi Singh, and Anshika Rai.

This project implements an **audit layer around a credit-risk model**, not an automated lending product. It is intended for demonstration and research reproduction only.

## What is included

- UCI German Credit (1,000 cases) and Australian Credit Approval (690 cases), downloaded on first use and cached under `data/raw/`.
- Six model configurations: Logistic Regression, class-weighted Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost.
- XGBoost comparisons for no resampling, fold-specific class weighting, SMOTE, and SMOTE-ENN.
- Five-fold stratified cross-validation with preprocessing and resampling fitted inside each training fold.
- ROC AUC, PR AUC, recall, precision, Brier score, and 10-bin expected calibration error (ECE).
- Out-of-fold Platt/sigmoid and isotonic calibration, with selection by Brier score then ECE.
- Youden-J thresholding, split-conformal predicted-label p-values, and margin/stability/conformal review routing.
- TreeSHAP local explanations, ten bounded numeric perturbations, rank stability, top-5 overlap, and normalized attribution displacement.
- LIME explanations and top-5 SHAP–LIME Jaccard agreement (1,500 surrogate samples per case).
- Group-level error/decision-rate summaries, per-case predictions, model artifact, and prediction-time measurement.
- Streamlit dashboard for a live classroom demonstration.

The full cross-validation model/imbalance tables are available using the `full` command. The detailed explanation and routing audit follows a stratified 80/20 train/test split; the training partition is further divided to keep a held-out calibration subset for conformal p-values. Explanation stability is measured on up to 150 test cases and the remaining test cases receive the median score, matching the paper's stated protocol and limitation.

## Setup (Windows PowerShell)

From this project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Internet access is needed the first time a UCI dataset is loaded. Later runs use the local cache.

## Run it

Start the demo from the project folder, using the same activated environment where you installed the requirements:

```powershell
python -m streamlit run app.py
```

Using `python -m streamlit` ensures Streamlit uses the selected Python interpreter. The app also adds the project's `src` directory to its import path, so the local `credit_audit` package can be imported directly from a checkout.

After an audit has been run, the dashboard's **Score one applicant** section loads the saved model bundle from `artifacts/<dataset>/`. Pick a dataset row as a starting point, edit the feature values, and optionally compute on-demand local stability for that single case. This view is an educational illustration of the audit layer, not a lending decision tool.

Run the detailed audit directly:

```powershell
python run_experiment.py --mode audit --dataset german
python run_experiment.py --mode audit --dataset australian
```

Run the paper's two-dataset CV comparisons and the German explanation/routing audit:

```powershell
python run_experiment.py --mode full
```

### Customize the audit budget

All defaults below reproduce the paper protocol exactly. They are opt-in switches for faster demos or sensitivity checks:

```powershell
python run_experiment.py --mode audit --dataset german --quick
python run_experiment.py --mode audit --dataset german --stability-cases 30 --perturbations 5 --agreement-cases 20 --lime-samples 500
python run_experiment.py --mode audit --dataset german --conformal-label-mode observed
```

`--quick` is a reduced budget (30 stability cases, 5 perturbations, 20 SHAP–LIME cases, 500 LIME samples) and is explicitly marked as non-paper in the log and in `summary.json`. The Streamlit sidebar exposes the same controls under **Advanced — audit budget**.

The runner writes CSV tables, a JSON summary, per-case predictions, and a fitted model/calibrator to `artifacts/<dataset>/`. Raw benchmark files are cached in `data/raw/`. Both folders are ignored by Git. Every `summary.json` now records the effective `audit_config` and the installed package versions under `environment`, so non-default runs are self-describing.

Run the focused checks:

```powershell
pytest
python -m pytest -m "not slow"
python -m pytest -m slow
```

## Paper protocol and interpretation

- Random seed 42; stratified five-fold CV for model and imbalance comparisons.
- Fixed model settings from the paper; no hyperparameter search.
- Ordinal encoding is used for categorical columns. SMOTE and SMOTE-ENN operate on this representation, as in the paper; interpolating nominal category codes is a known limitation.
- German Credit's raw outcome `2` is mapped to default. For Australian Credit, class `1` is mapped to the 44.5% positive class used by the paper.
- The default detailed audit uses XGBoost with fold-specific positive-class weighting. Calibrators are trained using out-of-fold predictions from the training partition.
- The German benchmark includes sex as a model feature to match the paper. Group metrics are reported, but the protected attribute must not be used as a real lending decision feature.
- The review-need metric includes the same `S < 0.8` condition used by the stability routing rule. Therefore, a lower composite rate for the stability policy is partly definitional—not independent evidence that explanation instability predicts harm.
- The paper reports that the conformal term rarely changes routing on these benchmarks. This implementation includes it for completeness; no benefit under dataset shift is claimed.
- Results can vary slightly with package versions and split details. The paper's own numerical tables are reproduced as targets, not hard-coded as experiment output. No accompanying original code/results were supplied with the paper PDF.
- The paper labels perturbation-width sensitivity, stability-by-model/strategy, and Taiwan-credit replication as planned extensions, not completed experiments. They are documented as future work rather than presented as results here.
- `data/raw/` and `artifacts/` are resolved relative to the project root, so the CLI and dashboard behave the same from any working directory. Cached UCI ZIPs are structurally validated and checksum-checked; a checksum mismatch on a structurally valid archive emits a warning rather than silently changing results.

## Safety note

This project is not validated for credit decisions, fairness compliance, or production use. The datasets are small historical benchmarks, the experiment uses a single held-out split for the audit, and stability is not a measure of explanation faithfulness. Human review remains essential.
