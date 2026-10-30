"""
Q009 exact overlapping-query differencing experiment.

The experiment asks two overlapping aggregate queries:

    Q1 = prolonged-stay count over the full cohort D
    Q2 = prolonged-stay count over D with target patient removed

Exact differencing gives:

    Q1 - Q2 = target contribution

Therefore:
    0 -> target not prolonged
    1 -> target prolonged

This is an isolated experimental benchmark only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.query import (
    prolonged_stay_histogram,
)

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.synthetic_data import (
    DEFAULT_RACE,
    DEFAULT_ADMISSION_TYPE,
    TARGET_SUBJECT_ID,
    make_attribute_inference_pair,
)


# ---------------------------------------------------------------------
# Experimental settings
# ---------------------------------------------------------------------

COHORT_SIZES = [
    5,
    10,
    20,
    50,
    100,
]


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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "q009_exact_differencing.csv"
)


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def exact_prolonged_count(
    dataframe: pd.DataFrame,
) -> int:
    """
    Return the exact prolonged count for the controlled subgroup.
    """

    result = prolonged_stay_histogram(
        dataframe
    )

    cell = (
        result["histogram"]
        [DEFAULT_RACE]
        [DEFAULT_ADMISSION_TYPE]
    )

    return int(
        cell["prolonged"]
    )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

rows = []


print("=" * 78)
print("Q009 EXACT OVERLAPPING-QUERY DIFFERENCING")
print("=" * 78)
print()


for cohort_size in COHORT_SIZES:

    short_df, long_df = (
        make_attribute_inference_pair(
            cohort_size=cohort_size
        )
    )

    cases = [
        (
            "SHORT",
            0,
            short_df,
        ),
        (
            "LONG",
            1,
            long_df,
        ),
    ]

    for (
        target_case,
        true_target_contribution,
        full_df,
    ) in cases:

        # ---------------------------------------------------------
        # Query 1:
        # full cohort including target.
        # ---------------------------------------------------------

        q1_full = exact_prolonged_count(
            full_df
        )

        # ---------------------------------------------------------
        # Query 2:
        # identical cohort with only target removed.
        # ---------------------------------------------------------

        without_target_df = (
            full_df[
                full_df["subject_id"]
                != TARGET_SUBJECT_ID
            ]
            .copy()
        )

        q2_without_target = (
            exact_prolonged_count(
                without_target_df
            )
        )

        # ---------------------------------------------------------
        # Differencing attack.
        # ---------------------------------------------------------

        inferred_contribution = (
            q1_full
            - q2_without_target
        )

        attack_success = (
            inferred_contribution
            == true_target_contribution
        )

        if not attack_success:
            raise RuntimeError(
                f"Unexpected exact differencing result for "
                f"n={cohort_size}, case={target_case}."
            )

        rows.append(
            {
                "cohort_size": cohort_size,
                "target_case": target_case,
                "true_target_contribution": (
                    true_target_contribution
                ),
                "q1_full_cohort_count": (
                    q1_full
                ),
                "q2_without_target_count": (
                    q2_without_target
                ),
                "difference": (
                    inferred_contribution
                ),
                "attack_success": (
                    attack_success
                ),
            }
        )

        print(
            f"n={cohort_size:3d} | "
            f"{target_case:5s} | "
            f"Q1={q1_full:3d} | "
            f"Q2={q2_without_target:3d} | "
            f"Q1-Q2={inferred_contribution} | "
            f"PASS"
        )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

df = pd.DataFrame(
    rows
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print()
print("=" * 78)
print("Q009 EXACT DIFFERENCING VALIDATION PASSED")
print("=" * 78)

print()
print(
    f"Cases tested: {len(df)}"
)

print(
    f"Exact attack accuracy: "
    f"{df['attack_success'].mean():.4f}"
)

print()
print(f"Saved: {OUTPUT_FILE}")