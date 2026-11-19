from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from credit_audit.config import AuditConfig, DEFAULT_AUDIT_CONFIG
from credit_audit.paths import ARTIFACTS_DIR
from credit_audit.runner import run_audit_experiment, run_full_experiment


LOGGER = logging.getLogger("credit_audit")
QUICK_CONFIG = {
    "stability_cases": 30,
    "perturbations": 5,
    "agreement_cases": 20,
    "lime_samples": 500,
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_config(args: argparse.Namespace) -> AuditConfig:
    values = DEFAULT_AUDIT_CONFIG.to_dict()
    if args.quick:
        values.update(QUICK_CONFIG)
        LOGGER.warning("--quick uses a reduced audit budget; results are not the paper protocol.")
    provided = {
        "stability_cases": args.stability_cases,
        "perturbations": args.perturbations,
        "agreement_cases": args.agreement_cases,
        "lime_samples": args.lime_samples,
        "seed": args.seed,
        "conformal_label_mode": args.conformal_label_mode,
    }
    values.update({key: value for key, value in provided.items() if value is not None})
    return AuditConfig(**values)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run the credit-risk audit experiments.")
    parser.add_argument("--mode", choices=("audit", "full"), default="audit")
    parser.add_argument("--dataset", choices=("german", "australian"), default="german")
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--stability-cases", type=_positive_int, default=None)
    parser.add_argument("--perturbations", type=_positive_int, default=None)
    parser.add_argument("--agreement-cases", type=_positive_int, default=None)
    parser.add_argument("--lime-samples", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--conformal-label-mode",
        choices=("predicted", "observed"),
        default=None,
        help="Predicted is the paper default; observed is an opt-in research variant.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced audit budget for fast demos (not the paper protocol).",
    )
    args = parser.parse_args()
    config = _build_config(args)

    if args.mode == "audit":
        LOGGER.info("Starting %s audit with config %s", args.dataset, config.to_dict())
        result = run_audit_experiment(args.dataset, args.output, config)
        print(json.dumps(result["baseline_metrics"], indent=2))
        LOGGER.info("Saved audit outputs under %s", args.output / args.dataset)
    else:
        LOGGER.info("Starting full experiment with audit config %s", config.to_dict())
        results = run_full_experiment(args.output, config)
        for name, table in results.items():
            print(f"\n{name}\n{table.to_string(index=False)}")
        LOGGER.info("Saved experiment tables under %s", args.output)


if __name__ == "__main__":
    main()
