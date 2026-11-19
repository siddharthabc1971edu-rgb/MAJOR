"""Configuration for the explanation/routing audit.

Defaults mirror the paper's protocol exactly. Any non-default value is an
opt-in variant and is recorded in ``summary.json`` by the runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AuditConfig:
    """Tunable audit costs and conventions.

    Attributes:
        stability_cases: Number of held-out cases used for SHAP stability.
        perturbations: Bounded numeric perturbations per stability case.
        agreement_cases: Number of cases used for SHAP-LIME agreement.
        lime_samples: LIME surrogate samples drawn per agreement case.
        seed: Base random seed; perturbation i uses ``seed + i``.
        conformal_label_mode: ``"predicted"`` (paper default) or ``"observed"``.
    """

    stability_cases: int = 150
    perturbations: int = 10
    agreement_cases: int = 100
    lime_samples: int = 1500
    seed: int = 42
    conformal_label_mode: str = "predicted"

    def __post_init__(self) -> None:
        for name in ("stability_cases", "perturbations", "agreement_cases", "lime_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer, got {self.seed!r}.")
        if self.conformal_label_mode not in {"predicted", "observed"}:
            raise ValueError(
                "conformal_label_mode must be 'predicted' or 'observed', "
                f"got {self.conformal_label_mode!r}."
            )

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @property
    def is_default(self) -> bool:
        return self == AuditConfig()


DEFAULT_AUDIT_CONFIG = AuditConfig()
