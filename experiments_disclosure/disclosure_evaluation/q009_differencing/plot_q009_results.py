"""
Generate plots for the Q009 overlapping-query differencing experiment.

Input
-----
q009_dp_differencing_summary.csv

Outputs
-------
q009_attack_accuracy_vs_total_epsilon.png/pdf
q009_differencing_error_vs_total_epsilon.png/pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q009_differencing"
    / "results"
)

INPUT_FILE = (
    RESULTS_DIR
    / "q009_dp_differencing_summary.csv"
)

PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "cohort_size",
    "epsilon_total",
    "dp_attack_accuracy",
    "dp_attack_accuracy_ci_lower",
    "dp_attack_accuracy_ci_upper",
    "difference_mae",
}

missing = required_columns.difference(
    df.columns
)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


COHORT_SIZES = [
    5,
    10,
    20,
    50,
    100,
]

TOTAL_EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

x_positions = list(
    range(len(TOTAL_EPSILONS))
)


# =====================================================================
# Plot 1
# Differencing attack accuracy
# =====================================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


for cohort_size in COHORT_SIZES:

    subset = (
        df[
            df["cohort_size"]
            == cohort_size
        ]
        .set_index("epsilon_total")
        .reindex(TOTAL_EPSILONS)
        .reset_index()
    )

    y = subset[
        "dp_attack_accuracy"
    ].to_numpy()

    lower = subset[
        "dp_attack_accuracy_ci_lower"
    ].to_numpy()

    upper = subset[
        "dp_attack_accuracy_ci_upper"
    ].to_numpy()

    yerr = [
        y - lower,
        upper - y,
    ]

    ax.errorbar(
        x_positions,
        y,
        yerr=yerr,
        marker="o",
        linewidth=1.8,
        capsize=3,
        label=f"n = {cohort_size}",
    )


# Background baseline
ax.axhline(
    y=0.5,
    linestyle="--",
    linewidth=1.3,
    label="Background only (0.5)",
)

# Exact differencing reference
ax.axhline(
    y=1.0,
    linestyle=":",
    linewidth=1.3,
    label="Exact differencing (1.0)",
)


ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [str(e) for e in TOTAL_EPSILONS]
)

ax.set_xlabel(
    r"Total privacy budget $\epsilon_{\mathrm{total}}$"
)

ax.set_ylabel(
    "Target contribution inference accuracy"
)

ax.set_ylim(
    0.45,
    1.03,
)

ax.set_title(
    "Q009: Differencing Attack vs. Total Privacy Budget"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    frameon=False,
    ncol=2,
)

fig.tight_layout()


attack_png = (
    PLOTS_DIR
    / "q009_attack_accuracy_vs_total_epsilon.png"
)

attack_pdf = (
    PLOTS_DIR
    / "q009_attack_accuracy_vs_total_epsilon.pdf"
)

fig.savefig(
    attack_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    attack_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# Plot 2
# Differencing error
# =====================================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.5)
)


for cohort_size in COHORT_SIZES:

    subset = (
        df[
            df["cohort_size"]
            == cohort_size
        ]
        .set_index("epsilon_total")
        .reindex(TOTAL_EPSILONS)
        .reset_index()
    )

    ax.plot(
        x_positions,
        subset["difference_mae"],
        marker="o",
        linewidth=1.8,
        label=f"n = {cohort_size}",
    )


ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [str(e) for e in TOTAL_EPSILONS]
)

ax.set_xlabel(
    r"Total privacy budget $\epsilon_{\mathrm{total}}$"
)

ax.set_ylabel(
    "Absolute differencing error (MAE)"
)

ax.set_title(
    "Q009: Differencing Uncertainty vs. Total Privacy Budget"
)

ax.set_ylim(
    bottom=0
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()


error_png = (
    PLOTS_DIR
    / "q009_differencing_error_vs_total_epsilon.png"
)

error_pdf = (
    PLOTS_DIR
    / "q009_differencing_error_vs_total_epsilon.pdf"
)

fig.savefig(
    error_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    error_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# Summary
# =====================================================================

print("=" * 72)
print("Q009 PLOTS GENERATED")
print("=" * 72)

print()
print("Attack plot:")
print(f"  {attack_png}")
print(f"  {attack_pdf}")

print()
print("Differencing-error plot:")
print(f"  {error_png}")
print(f"  {error_pdf}")

print()
print(f"Conditions plotted: {len(df)}")

if len(df) == 25:
    print("Expected 25 conditions: PASS")
else:
    print(
        "WARNING: expected 25 cohort-size × epsilon conditions."
    )