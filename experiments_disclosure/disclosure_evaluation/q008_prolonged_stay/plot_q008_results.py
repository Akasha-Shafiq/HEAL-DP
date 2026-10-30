"""
Generate plots for the Q008
controlled attribute-inference experiment.

Inputs
------
q008_attribute_inference_summary.csv

Outputs
-------
q008_attack_accuracy_vs_epsilon.png/pdf
q008_rate_mae_vs_epsilon.png/pdf
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
    / "q008_prolonged_stay"
    / "results"
)

INPUT_FILE = (
    RESULTS_DIR
    / "q008_attribute_inference_summary.csv"
)

PLOTS_DIR = (
    RESULTS_DIR
    / "plots"
)

PLOTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Load and validate
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "cohort_size",
    "epsilon",
    "dp_attack_accuracy",
    "dp_attack_accuracy_ci_lower",
    "dp_attack_accuracy_ci_upper",
    "rate_mae",
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

EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

# Use evenly spaced positions but display actual epsilon values.
x_positions = list(
    range(len(EPSILONS))
)


# =====================================================================
# Plot 1
# Target inference accuracy versus epsilon
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
        .set_index("epsilon")
        .reindex(EPSILONS)
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


# Background-knowledge-only baseline
ax.axhline(
    y=0.5,
    linestyle="--",
    linewidth=1.3,
    label="Background only (0.5)",
)

# Exact aggregate reference
ax.axhline(
    y=1.0,
    linestyle=":",
    linewidth=1.3,
    label="Exact release (1.0)",
)


ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [str(e) for e in EPSILONS]
)

ax.set_xlabel(
    r"Privacy budget $\epsilon$"
)

ax.set_ylabel(
    "Target inference accuracy"
)

ax.set_ylim(
    0.45,
    1.03,
)

ax.set_title(
    "Q008: Target Attribute Inference vs. Privacy Budget"
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
    / "q008_attack_accuracy_vs_epsilon.png"
)

attack_pdf = (
    PLOTS_DIR
    / "q008_attack_accuracy_vs_epsilon.pdf"
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
# Released-rate MAE versus epsilon
# Expressed in percentage points
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
        .set_index("epsilon")
        .reindex(EPSILONS)
        .reset_index()
    )

    # Convert MAE from a fraction to percentage points.
    #
    # Example:
    #   0.38 -> 38 percentage points
    #
    rate_mae_percentage_points = (
        subset["rate_mae"] * 100.0
    )

    ax.plot(
        x_positions,
        rate_mae_percentage_points,
        marker="o",
        linewidth=1.8,
        label=f"n = {cohort_size}",
    )


ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [str(e) for e in EPSILONS]
)

ax.set_xlabel(
    r"Privacy budget $\epsilon$"
)

ax.set_ylabel(
    "Absolute prolonged-stay rate error (percentage points)"
)

ax.set_title(
    "Q008: Released-Rate Utility vs. Privacy Budget"
)

# Start y-axis at zero for easier interpretation.
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


utility_png = (
    PLOTS_DIR
    / "q008_rate_mae_vs_epsilon.png"
)

utility_pdf = (
    PLOTS_DIR
    / "q008_rate_mae_vs_epsilon.pdf"
)

fig.savefig(
    utility_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    utility_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# Summary
# =====================================================================

print("=" * 72)
print("Q008 PLOTS GENERATED")
print("=" * 72)

print()
print("Privacy plot:")
print(f"  {attack_png}")
print(f"  {attack_pdf}")

print()
print("Utility plot:")
print(f"  {utility_png}")
print(f"  {utility_pdf}")

print()
print(f"Conditions plotted: {len(df)}")

if len(df) == 25:
    print("Expected 25 conditions: PASS")
else:
    print(
        "WARNING: expected 25 cohort-size × epsilon conditions."
    )