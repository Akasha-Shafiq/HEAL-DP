"""
Create a combined Q008 vs Q009 comparison figure.

Goal
----
Compare:

    Q008:
        Single-release attribute inference.

    Q009:
        Two-release overlapping-query differencing.

For each epsilon, attack accuracy is averaged across:

    n = 5, 10, 20, 50, 100

Inputs
------
Q008:
    q008_attribute_inference_summary.csv

Q009:
    q009_dp_differencing_summary.csv

Outputs
-------
q008_q009_combined_attack_accuracy.png
q008_q009_combined_attack_accuracy.pdf
q008_q009_combined_attack_accuracy_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

Q008_SUMMARY = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "results"
    / "q008_attribute_inference_summary.csv"
)

Q009_SUMMARY = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q009_differencing"
    / "results"
    / "q009_dp_differencing_summary.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "comparison"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PNG = (
    OUTPUT_DIR
    / "q008_q009_combined_attack_accuracy.png"
)

OUTPUT_PDF = (
    OUTPUT_DIR
    / "q008_q009_combined_attack_accuracy.pdf"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "q008_q009_combined_attack_accuracy_summary.csv"
)


# ---------------------------------------------------------------------
# Experimental domain
# ---------------------------------------------------------------------

EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

COHORT_SIZES = [
    5,
    10,
    20,
    50,
    100,
]

x_positions = list(
    range(len(EPSILONS))
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def pick_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Find the first matching column name.
    """

    for column in candidates:

        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find {description}. "
        f"Tried: {candidates}. "
        f"Available columns: "
        f"{list(dataframe.columns)}"
    )


def prepare_curve(
    dataframe: pd.DataFrame,
    epsilon_candidates: list[str],
    accuracy_candidates: list[str],
    source_name: str,
) -> pd.DataFrame:
    """
    Compute mean/min/max/std attack accuracy across cohort sizes
    for each privacy budget.
    """

    epsilon_column = pick_column(
        dataframe,
        epsilon_candidates,
        f"{source_name} epsilon column",
    )

    accuracy_column = pick_column(
        dataframe,
        accuracy_candidates,
        f"{source_name} accuracy column",
    )

    # -------------------------------------------------------------
    # Keep only the expected cohort sizes.
    # -------------------------------------------------------------

    if "cohort_size" in dataframe.columns:

        dataframe = dataframe[
            dataframe["cohort_size"].isin(
                COHORT_SIZES
            )
        ].copy()

    # -------------------------------------------------------------
    # Aggregate across cohort sizes.
    #
    # IMPORTANT:
    # Do not use as_index=False AND reset_index().
    # That was the cause of the previous length-mismatch error.
    # -------------------------------------------------------------

    grouped = (
        dataframe
        .groupby(epsilon_column)[accuracy_column]
        .agg(
            mean_accuracy="mean",
            std_accuracy="std",
            min_accuracy="min",
            max_accuracy="max",
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            epsilon_column: "epsilon"
        }
    )

    grouped["source"] = source_name

    return grouped


def reindex_curve(
    curve: pd.DataFrame,
) -> pd.DataFrame:
    """
    Force the common epsilon ordering.
    """

    curve = (
        curve
        .set_index("epsilon")
        .reindex(EPSILONS)
        .reset_index()
    )

    return curve


# ---------------------------------------------------------------------
# Load summary data
# ---------------------------------------------------------------------

if not Q008_SUMMARY.exists():
    raise FileNotFoundError(
        f"Q008 summary not found: "
        f"{Q008_SUMMARY}"
    )

if not Q009_SUMMARY.exists():
    raise FileNotFoundError(
        f"Q009 summary not found: "
        f"{Q009_SUMMARY}"
    )


q008_df = pd.read_csv(
    Q008_SUMMARY
)

q009_df = pd.read_csv(
    Q009_SUMMARY
)


# ---------------------------------------------------------------------
# Prepare Q008 curve
# ---------------------------------------------------------------------

q008_curve = prepare_curve(
    dataframe=q008_df,
    epsilon_candidates=[
        "epsilon",
        "epsilon_total",
    ],
    accuracy_candidates=[
        "dp_attack_accuracy",
        "attack_accuracy",
    ],
    source_name="Q008",
)


# ---------------------------------------------------------------------
# Prepare Q009 curve
# ---------------------------------------------------------------------

q009_curve = prepare_curve(
    dataframe=q009_df,
    epsilon_candidates=[
        "epsilon_total",
        "epsilon",
    ],
    accuracy_candidates=[
        "dp_attack_accuracy",
        "attack_accuracy",
    ],
    source_name="Q009",
)


q008_curve = reindex_curve(
    q008_curve
)

q009_curve = reindex_curve(
    q009_curve
)


# ---------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------

if q008_curve["mean_accuracy"].isna().any():
    raise RuntimeError(
        "Q008 comparison curve contains missing epsilon values."
    )

if q009_curve["mean_accuracy"].isna().any():
    raise RuntimeError(
        "Q009 comparison curve contains missing epsilon values."
    )


# ---------------------------------------------------------------------
# Save combined summary table
# ---------------------------------------------------------------------

combined_table = pd.concat(
    [
        q008_curve,
        q009_curve,
    ],
    ignore_index=True,
)

combined_table.to_csv(
    OUTPUT_CSV,
    index=False,
)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5.8)
)


# ---------------------------------------------------------------------
# Q008
# ---------------------------------------------------------------------

ax.plot(
    x_positions,
    q008_curve["mean_accuracy"],
    marker="o",
    linewidth=2.4,
    markersize=7,
    label=(
        "Q008: single-release "
        "attribute inference"
    ),
)

ax.fill_between(
    x_positions,
    q008_curve["min_accuracy"],
    q008_curve["max_accuracy"],
    alpha=0.12,
)


# ---------------------------------------------------------------------
# Q009
# ---------------------------------------------------------------------

ax.plot(
    x_positions,
    q009_curve["mean_accuracy"],
    marker="s",
    linewidth=2.4,
    markersize=7,
    label=(
        "Q009: overlapping-release "
        "differencing"
    ),
)

ax.fill_between(
    x_positions,
    q009_curve["min_accuracy"],
    q009_curve["max_accuracy"],
    alpha=0.12,
)


# ---------------------------------------------------------------------
# Reference lines
# ---------------------------------------------------------------------

ax.axhline(
    y=0.5,
    linestyle="--",
    linewidth=1.4,
    label="Background only (0.5)",
)

ax.axhline(
    y=1.0,
    linestyle=":",
    linewidth=1.4,
    label="Exact disclosure (1.0)",
)


# ---------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    [str(value) for value in EPSILONS]
)

ax.set_xlabel(
    r"Privacy budget $\epsilon$"
)

ax.set_ylabel(
    "Mean target inference accuracy"
)

ax.set_ylim(
    0.45,
    1.03,
)

ax.set_title(
    "Q008 vs. Q009: Disclosure Risk under Differential Privacy"
)

ax.grid(
    True,
    alpha=0.25,
)

ax.legend(
    frameon=False,
)

fig.tight_layout()


# ---------------------------------------------------------------------
# Save figures
# ---------------------------------------------------------------------

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ---------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------

print("=" * 76)
print("Q008 vs Q009 COMBINED COMPARISON")
print("=" * 76)

print()
print("Q008 mean attack accuracy:")
print(
    q008_curve[
        [
            "epsilon",
            "mean_accuracy",
            "min_accuracy",
            "max_accuracy",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Q009 mean attack accuracy:")
print(
    q009_curve[
        [
            "epsilon",
            "mean_accuracy",
            "min_accuracy",
            "max_accuracy",
        ]
    ].to_string(
        index=False
    )
)

print()
print("Outputs:")
print(f"PNG: {OUTPUT_PNG}")
print(f"PDF: {OUTPUT_PDF}")
print(f"CSV: {OUTPUT_CSV}")

print()
print("Comparison plot generation: PASS")