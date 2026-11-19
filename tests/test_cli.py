from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from credit_audit.config import DEFAULT_AUDIT_CONFIG
from run_experiment import _build_config


def _namespace(**overrides) -> argparse.Namespace:
    values = {
        "mode": "audit",
        "dataset": "german",
        "output": Path("artifacts"),
        "stability_cases": None,
        "perturbations": None,
        "agreement_cases": None,
        "lime_samples": None,
        "seed": None,
        "conformal_label_mode": None,
        "quick": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_build_config_default_matches_paper_protocol():
    assert _build_config(_namespace()) == DEFAULT_AUDIT_CONFIG


def test_build_config_quick_uses_reduced_budget():
    config = _build_config(_namespace(quick=True))
    assert config.stability_cases == 30
    assert config.perturbations == 5
    assert config.agreement_cases == 20
    assert config.lime_samples == 500


def test_explicit_flags_override_quick_preset():
    config = _build_config(_namespace(quick=True, stability_cases=7, seed=9))
    assert config.stability_cases == 7
    assert config.seed == 9
    assert config.agreement_cases == 20


def test_cli_help_lists_new_options():
    completed = subprocess.run(
        [sys.executable, "-B", "run_experiment.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    for flag in ("--quick", "--stability-cases", "--conformal-label-mode"):
        assert flag in completed.stdout
