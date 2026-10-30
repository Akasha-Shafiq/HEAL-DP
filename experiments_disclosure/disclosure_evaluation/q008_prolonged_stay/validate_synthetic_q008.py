"""
Validate the controlled Q008 synthetic disclosure datasets.

No DP is used here.

The purpose is to verify that:
1. SHORT and LONG datasets contain the same patients.
2. Only the target LOS changes.
3. Q008 counts differ by exactly one prolonged contribution.
"""

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


COHORT_SIZES = [
    5,
    10,
    20,
    50,
    100,
]


REPO_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results = []


print("=" * 72)
print("Q008 SYNTHETIC ATTRIBUTE-INFERENCE VALIDATION")
print("=" * 72)
print()


for cohort_size in COHORT_SIZES:

    short_df, long_df = make_attribute_inference_pair(
        cohort_size=cohort_size
    )

    # -------------------------------------------------------------
    # Check that the patient populations are identical.
    # -------------------------------------------------------------

    short_ids = set(short_df["subject_id"])
    long_ids = set(long_df["subject_id"])

    if short_ids != long_ids:
        raise RuntimeError(
            f"Cohort {cohort_size}: "
            "SHORT/LONG patient IDs differ."
        )

    # -------------------------------------------------------------
    # Check non-target patients are exactly identical.
    # -------------------------------------------------------------

    short_non_target = (
        short_df[
            short_df["subject_id"]
            != TARGET_SUBJECT_ID
        ]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    long_non_target = (
        long_df[
            long_df["subject_id"]
            != TARGET_SUBJECT_ID
        ]
        .sort_values("subject_id")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        short_non_target,
        long_non_target,
    )

    # -------------------------------------------------------------
    # Execute exact Q008 on both datasets.
    # -------------------------------------------------------------

    short_result = prolonged_stay_histogram(
        short_df
    )

    long_result = prolonged_stay_histogram(
        long_df
    )

    short_cell = (
        short_result["histogram"]
        [DEFAULT_RACE]
        [DEFAULT_ADMISSION_TYPE]
    )

    long_cell = (
        long_result["histogram"]
        [DEFAULT_RACE]
        [DEFAULT_ADMISSION_TYPE]
    )

    short_prolonged = int(
        short_cell["prolonged"]
    )

    short_not_prolonged = int(
        short_cell["not_prolonged"]
    )

    long_prolonged = int(
        long_cell["prolonged"]
    )

    long_not_prolonged = int(
        long_cell["not_prolonged"]
    )

    # -------------------------------------------------------------
    # Required controlled difference:
    #
    # LONG prolonged count = SHORT prolonged count + 1
    #
    # LONG not-prolonged = SHORT not-prolonged - 1
    # -------------------------------------------------------------

    if long_prolonged != short_prolonged + 1:
        raise RuntimeError(
            f"Cohort {cohort_size}: "
            "prolonged count did not increase by 1."
        )

    if long_not_prolonged != short_not_prolonged - 1:
        raise RuntimeError(
            f"Cohort {cohort_size}: "
            "not-prolonged count did not decrease by 1."
        )

    if (
        short_result["patient_count"]
        != cohort_size
    ):
        raise RuntimeError(
            f"Cohort {cohort_size}: "
            "SHORT patient count mismatch."
        )

    if (
        long_result["patient_count"]
        != cohort_size
    ):
        raise RuntimeError(
            f"Cohort {cohort_size}: "
            "LONG patient count mismatch."
        )

    known_other_prolonged = (
        short_prolonged
    )

    results.append(
        {
            "cohort_size": cohort_size,
            "known_other_prolonged": (
                known_other_prolonged
            ),
            "short_target_status": (
                "not_prolonged"
            ),
            "short_released_prolonged_count": (
                short_prolonged
            ),
            "long_target_status": (
                "prolonged"
            ),
            "long_released_prolonged_count": (
                long_prolonged
            ),
            "exact_count_difference": (
                long_prolonged
                - short_prolonged
            ),
            "validation": "PASS",
        }
    )

    print(
        f"Cohort n={cohort_size:3d} | "
        f"known prolonged={known_other_prolonged:2d} | "
        f"SHORT={short_prolonged:2d} | "
        f"LONG={long_prolonged:2d} | "
        f"difference={long_prolonged-short_prolonged} | "
        f"PASS"
    )


# ---------------------------------------------------------------------
# Save validation summary.
# ---------------------------------------------------------------------

results_df = pd.DataFrame(results)

output_file = (
    RESULTS_DIR
    / "q008_synthetic_validation.csv"
)

results_df.to_csv(
    output_file,
    index=False,
)


print()
print("=" * 72)
print("ALL SYNTHETIC Q008 VALIDATION CHECKS PASSED")
print("=" * 72)

print()
print(f"Saved: {output_file}")