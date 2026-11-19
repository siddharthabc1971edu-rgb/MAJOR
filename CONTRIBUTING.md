# Contributing

Thanks for helping improve the Credit Risk Audit Lab.

## Development setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Checks

```powershell
python -m pip check
python -B -m pytest -m "not slow"
python -B -m pytest -m slow
python -m compileall -q app.py run_experiment.py src
```

`slow` tests fit models and run a reduced audit; they are separated so the fast suite stays quick. CI runs the fast suite on Python 3.10–3.12 on Windows and Ubuntu, plus one slow integration job.

## Ground rules

1. **Preserve paper defaults.** New behavior should be opt-in via `AuditConfig` or a CLI flag. If a default must change, call it out explicitly in the PR and update `README.md`.
2. **Keep preprocessing inside folds.** Samplers and encoders must be fitted only on training partitions. `tests/test_models.py` guards this.
3. **Record reproducibility metadata.** Any new experiment-level option should be added to `AuditConfig` so it appears in `summary.json` and the saved model bundle.
4. **Do not turn this into a lending product.** Changes must keep the educational/audit framing and the safety caveats in `README.md`.

## Useful entry points

- `src/credit_audit/data.py` — dataset download, caching, and validation.
- `src/credit_audit/experiments.py` — cross-validation comparisons.
- `src/credit_audit/audit.py` — calibration, explanations, routing, and the main `run_audit`.
- `src/credit_audit/runner.py` — artifact writing and CLI-facing experiment functions.
- `src/credit_audit/scoring.py` — single-applicant scoring from a saved bundle.
- `app.py` — Streamlit dashboard.
