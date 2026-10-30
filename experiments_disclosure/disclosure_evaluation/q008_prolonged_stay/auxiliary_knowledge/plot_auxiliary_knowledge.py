"""
Plot Q008-B rare-cohort auxiliary-knowledge results.

Main figure:
    n = 5 rare cohort

Shows target inference accuracy as attacker knowledge increases,
for:

    Exact release
    DP epsilon = 0.1
    DP epsilon = 0.5
    DP epsilon = 1
    DP epsilon = 5
    DP epsilon = 10

A second figure shows the 100%-knowledge condition across
cohort sizes n = 5, 10, 20.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[4]

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "auxiliary_knowledge"
    / "results"
)

INPUT_FILE = (
    RESULTS_DIR
    / "q008_auxiliary_knowledge_summary.csv"
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
# Load data
# ---------------------------------------------------------------------

df = pd.read_csv(
    INPUT_FILE
)

required_columns = {
    "cohort_size",
    "actual_knowledge_fraction",
    "epsilon",
    "exact_attack_accuracy",
    "dp_attack_accuracy",
}

missing = required_columns.difference(
    df.columns
)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]


# =====================================================================
# FIGURE 1
# Rare cohort n = 5:
# attack accuracy vs attacker knowledge
# =====================================================================

rare_df = (
    df[
        df["cohort_size"] == 5
    ]
    .copy()
)

knowledge_values = sorted(
    rare_df[
        "actual_knowledge_fraction"
    ].unique()
)

knowledge_percent = [
    value * 100
    for value in knowledge_values
]


fig, ax = plt.subplots(
    figsize=(9, 5.8)
)


# ---------------------------------------------------------------------
# Exact release
#
# Exact accuracy is identical across epsilon for each knowledge level,
# so take one value per knowledge condition.
# ---------------------------------------------------------------------

exact_curve = (
    rare_df
    .groupby(
        "actual_knowledge_fraction"
    )[
        "exact_attack_accuracy"
    ]
    .first()
    .reindex(
        knowledge_values
    )
)


ax.plot(
    knowledge_percent,
    exact_curve,
    marker="o",
    linewidth=2.4,
    linestyle="--",
    label="Exact release",
)


# ---------------------------------------------------------------------
# DP curves
# ---------------------------------------------------------------------

for epsilon in EPSILONS:

    subset = (
        rare_df[
            rare_df["epsilon"]
            == epsilon
        ]
        .set_index(
            "actual_knowledge_fraction"
        )
        .reindex(
            knowledge_values
        )
    )

    ax.plot(
        knowledge_percent,
        subset[
            "dp_attack_accuracy"
        ],
        marker="o",
        linewidth=2.0,
        label=rf"DP $\epsilon$ = {epsilon}",
    )


# Background-only balanced target baseline
ax.axhline(
    y=0.5,
    linestyle=":",
    linewidth=1.5,
    label="Background-only baseline (0.5)",
)


ax.set_xlabel(
    "Attacker knowledge of other patients (%)"
)

ax.set_ylabel(
    "Target inference accuracy"
)

ax.set_ylim(
    0.45,
    1.03,
)

ax.set_xticks(
    knowledge_percent
)

ax.set_title(
    "Q008-B: Rare-Cohort Inference under Auxiliary Knowledge (n = 5)"
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


figure1_png = (
    PLOTS_DIR
    / "q008_auxiliary_knowledge_n5.png"
)

figure1_pdf = (
    PLOTS_DIR
    / "q008_auxiliary_knowledge_n5.pdf"
)

fig.savefig(
    figure1_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    figure1_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# FIGURE 2
# Strongest auxiliary-knowledge condition:
# attacker knows 100% of other patients.
# =====================================================================

full_knowledge_df = (
    df[
        df[
            "actual_knowledge_fraction"
        ]
        == 1.0
    ]
    .copy()
)


fig, ax = plt.subplots(
    figsize=(9, 5.8)
)


for cohort_size in [
    5,
    10,
    20,
]:

    subset = (
        full_knowledge_df[
            full_knowledge_df[
                "cohort_size"
            ]
            == cohort_size
        ]
        .set_index(
            "epsilon"
        )
        .reindex(
            EPSILONS
        )
        .reset_index()
    )

    ax.plot(
        range(
            len(EPSILONS)
        ),
        subset[
            "dp_attack_accuracy"
        ],
        marker="o",
        linewidth=2.2,
        label=f"n = {cohort_size}",
    )


# Exact-release reference
ax.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1.5,
    label="Exact release (1.0)",
)

# Background-only baseline
ax.axhline(
    y=0.5,
    linestyle=":",
    linewidth=1.5,
    label="Background-only baseline (0.5)",
)


ax.set_xticks(
    range(
        len(EPSILONS)
    )
)

ax.set_xticklabels(
    [
        str(epsilon)
        for epsilon in EPSILONS
    ]
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
    "Q008-B: Full Auxiliary Knowledge in Rare Cohorts"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()


figure2_png = (
    PLOTS_DIR
    / "q008_full_auxiliary_knowledge_vs_epsilon.png"
)

figure2_pdf = (
    PLOTS_DIR
    / "q008_full_auxiliary_knowledge_vs_epsilon.pdf"
)

fig.savefig(
    figure2_png,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    figure2_pdf,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================================
# Console summary
# =====================================================================

print("=" * 76)
print("Q008-B AUXILIARY-KNOWLEDGE PLOTS GENERATED")
print("=" * 76)

print()
print("Rare-cohort attacker-knowledge plot:")
print(f"  {figure1_png}")
print(f"  {figure1_pdf}")

print()
print("Full-knowledge epsilon plot:")
print(f"  {figure2_png}")
print(f"  {figure2_pdf}")

print()
print("Plot generation: PASS")