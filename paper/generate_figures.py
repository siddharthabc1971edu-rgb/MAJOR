"""Regenerate IEEE paper figures from the experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from credit_audit.paths import ARTIFACTS_DIR


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "figure.dpi": 300,
    }
)

STRATEGIES = ["none", "class_weight", "smote", "smote_enn"]
STRATEGY_LABELS = ["None", "Class wt.", "SMOTE", "SMOTE-ENN"]
MODELS = ["lr", "lr_weighted", "rf", "xgb", "lightgbm", "catboost"]
MODEL_LABELS = ["LR", "LR-cw", "RF", "XGB", "LGBM", "CAT"]
HATCHES = ["", "//", "\\\\", "..", "xx", "--"]


def figure_pipeline(out: Path) -> None:
    """Six-stage audit pipeline as a full-width two-row flow diagram."""
    stages = [
        ("1", "Data and splits",
         r"German / Australian; 80/20 test split;" "\n" r"calibration part.; seed 42"),
        ("2", "Leakage-free development",
         r"5-fold stratified CV; SMOTE and" "\n" r"SMOTE-ENN fitted inside folds only"),
        ("3", "Calibrated predictor",
         r"XGBoost; Platt/isotonic on OOF scores;" "\n" r"Youden threshold $t$"),
        ("4", "Explanation audit",
         r"TreeSHAP + LIME top-$k$ sets;" "\n" r"Jaccard agreement $J_k$"),
        ("5", "Stability score",
         r"$S(x)$ from $K=10$ bounded perturbations" "\n" r"of numeric attributes"),
        ("6", "Review routing",
         r"Auto-decide unless margin $<\Delta$," "\n" r"$S(x)<\gamma$, or $p$-value $<\alpha$"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.29, 0.30
    xs = [0.02, 0.355, 0.69]
    rows = [0.55, 0.10]  # lower-left y of each row
    for index, (number, title, body) in enumerate(stages):
        row, col = divmod(index, 3)
        x, y = xs[col], rows[row]
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#eef2f7",
                                   edgecolor="black", linewidth=0.8))
        ax.text(x + 0.02, y + h - 0.055, number, fontsize=8, fontweight="bold",
                color="white", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.18", fc="black", ec="black"))
        ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="center",
                fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.16, body, ha="center", va="center",
                fontsize=6.6, linespacing=1.3)
        if col < 2:
            ax.annotate("", xy=(xs[col + 1] - 0.006, y + h / 2),
                        xytext=(x + w + 0.006, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
    # Connector from end of row 1 down to start of row 2.
    ax.plot([xs[2] + w / 2, xs[2] + w / 2, xs[0] + w / 2],
            [rows[0], 0.475, 0.475], color="black", lw=1.0)
    ax.annotate("", xy=(xs[0] + w / 2, rows[1] + h + 0.004),
                xytext=(xs[0] + w / 2, 0.475 - 0.004),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color="black"))
    ax.text(0.5, 0.965, "Predictor construction (stages 1--3)",
            ha="center", va="center", fontsize=7, style="italic")
    ax.text(0.5, 0.465, "Explanation audit and routing (stages 4--6)",
            ha="center", va="bottom", fontsize=7, style="italic")
    fig.savefig(out / "fig1_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def _load(artifacts: Path) -> dict[str, pd.DataFrame]:
    german = artifacts / "german"
    australian = artifacts / "australian"
    return {
        "g_imbalance": pd.read_csv(german / "imbalance_comparison.csv"),
        "a_imbalance": pd.read_csv(australian / "imbalance_comparison.csv"),
        "g_models": pd.read_csv(german / "model_comparison.csv"),
        "a_models": pd.read_csv(australian / "model_comparison.csv"),
        "g_calibration": pd.read_csv(german / "calibration.csv"),
        "a_calibration": pd.read_csv(australian / "calibration.csv"),
        "g_stability": pd.read_csv(german / "explanation_stability.csv"),
        "g_agreement": pd.read_csv(german / "shap_lime_agreement.csv"),
        "a_agreement": pd.read_csv(australian / "shap_lime_agreement.csv"),
        "g_routing": pd.read_csv(german / "routing.csv"),
        "g_matched": pd.read_csv(german / "routing_matched.csv"),
    }


def _bar_axis(ax, labels, series, title, ylabel, lower_better=False):
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for index, (name, values, hatch) in enumerate(series):
        positions = x + (index - (len(series) - 1) / 2) * width
        ax.bar(positions, values, width, label=name, hatch=hatch, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    if lower_better:
        ax.set_axisbelow(True)


def figure_imbalance(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for row, (key, dataset) in enumerate((("g_imbalance", "German"), ("a_imbalance", "Australian"))):
        table = data[key].set_index("strategy").loc[STRATEGIES]
        series = [
            ("AUC", table["auc"].to_numpy(), ""),
            ("PR-AUC", table["pr_auc"].to_numpy(), "//"),
            ("Recall", table["recall"].to_numpy(), "\\\\"),
        ]
        _bar_axis(axes[row, 0], STRATEGY_LABELS, series, f"{dataset}: discrimination and recall", "Score")
        _bar_axis(
            axes[row, 1], STRATEGY_LABELS,
            [("Brier", table["brier"].to_numpy(), "")],
            f"{dataset}: Brier score", "Brier", lower_better=True,
        )
    axes[0, 0].legend(loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "fig2_imbalance.pdf")
    plt.close(fig)


def figure_models(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8))
    for index, (metric, label, lower) in enumerate(
        (("auc", "AUC", False), ("pr_auc", "PR-AUC", False), ("brier", "Brier score", True), ("ece", "ECE", True))
    ):
        ax = axes.flat[index]
        german = data["g_models"].set_index("model").loc[MODELS][metric].to_numpy()
        australian = data["a_models"].set_index("model").loc[MODELS][metric].to_numpy()
        x = np.arange(len(MODELS))
        width = 0.36
        ax.bar(x - width / 2, german, width, label="German", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, australian, width, label="Australian", hatch="//", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_LABELS)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        if index == 0:
            ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig3_models.pdf")
    plt.close(fig)


def figure_calibration(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for index, (key, dataset) in enumerate((("g_calibration", "German"), ("a_calibration", "Australian"))):
        table = data[key].set_index("method").loc[["none", "sigmoid", "isotonic"]]
        x = np.arange(3)
        width = 0.36
        axes[index].bar(x - width / 2, table["ece"], width, label="ECE", edgecolor="black", linewidth=0.5)
        axes[index].bar(x + width / 2, table["brier"], width, label="Brier", hatch="//", edgecolor="black", linewidth=0.5)
        axes[index].set_xticks(x)
        axes[index].set_xticklabels(["Uncal.", "Sigmoid", "Isotonic"])
        axes[index].set_title(dataset)
        axes[index].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
        axes[index].set_axisbelow(True)
    axes[0].set_ylabel("Score (lower is better)")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig4_calibration.pdf")
    plt.close(fig)


def figure_explanation(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    stability = data["g_stability"]["stability"].to_numpy()
    axes[0].hist(stability, bins=12, range=(0, 1), edgecolor="black", linewidth=0.5)
    axes[0].axvline(0.8, color="black", linestyle="--", linewidth=1, label=r"$\gamma=0.8$")
    axes[0].set_xlabel("Stability score $S(x)$")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("(a) German stability distribution")
    axes[0].legend(loc="upper left")

    labels = ["German", "Australian"]
    j3 = [data["g_agreement"]["jaccard_top3"].mean(), data["a_agreement"]["jaccard_top3"].mean()]
    j5 = [data["g_agreement"]["jaccard_top5"].mean(), data["a_agreement"]["jaccard_top5"].mean()]
    e3 = [data["g_agreement"]["jaccard_top3"].std(), data["a_agreement"]["jaccard_top3"].std()]
    e5 = [data["g_agreement"]["jaccard_top5"].std(), data["a_agreement"]["jaccard_top5"].std()]
    x = np.arange(2)
    width = 0.34
    axes[1].bar(x - width / 2, j3, width, yerr=e3, label=r"top-3 ($J_3$)", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + width / 2, j5, width, yerr=e5, hatch="//", label=r"top-5 ($J_5$)", edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 0.8)
    axes[1].set_ylabel("Mean Jaccard agreement")
    axes[1].set_title("(b) SHAP--LIME agreement")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig5_explanation.pdf")
    plt.close(fig)


def figure_routing(data: dict[str, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    routing = data["g_routing"].query("comparison == 'threshold'")
    for policy, marker in (("P1 margin", "o"), ("P2 margin + stability", "s")):
        table = routing[routing["policy"] == policy]
        axes[0].plot(table["coverage"], table["composite_review_need"], marker=marker, label=policy)
    axes[0].axhline(0.385, color="black", linestyle="--", linewidth=0.8, label="Unfiltered 0.385")
    axes[0].set_xlabel("Coverage (auto-decided fraction)")
    axes[0].set_ylabel("Composite review-need")
    axes[0].set_title("(a) Threshold-based routing")
    axes[0].legend(loc="lower left", fontsize=6)
    axes[0].grid(linestyle=":", linewidth=0.5, alpha=0.7)

    matched = data["g_matched"].query("target_coverage in [0.55, 0.75, 0.84]")
    coverages = ["55%", "75%", "84%"]
    x = np.arange(3)
    width = 0.25
    for index, (policy, hatch) in enumerate(
        (("P1 margin", ""), ("P2 margin + stability", "//"), ("P3 + conformal", ".."))
    ):
        values = [
            matched[(matched["policy"] == policy) & (matched["target_coverage"] == target)][
                "composite_review_need"
            ].iloc[0]
            for target in (0.55, 0.75, 0.84)
        ]
        axes[1].bar(x + (index - 1) * width, values, width, label=policy, hatch=hatch, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(coverages)
    axes[1].set_ylabel("Composite review-need")
    axes[1].set_title("(b) Matched coverage")
    axes[1].legend(loc="upper left", fontsize=6)
    axes[1].grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out / "fig6_routing.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "paper" / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = _load(args.artifacts)
    figure_pipeline(args.output)
    figure_imbalance(data, args.output)
    figure_models(data, args.output)
    figure_calibration(data, args.output)
    figure_explanation(data, args.output)
    figure_routing(data, args.output)
    print(f"Figures written to {args.output}")


if __name__ == "__main__":
    main()
