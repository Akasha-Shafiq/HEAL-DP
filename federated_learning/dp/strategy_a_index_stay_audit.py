"""
Strategy A candidate audit:
one deterministic ICU-stay example per patient.

Candidate rule
--------------
For each subject_id, select the chronologically earliest ICU stay.

Why:
- preserves a real clinical observation;
- retains an unambiguous stay-level LOS target;
- prevents later ICU stays from contributing information to an
  earlier patient representation;
- gives exactly one model example per patient;
- therefore permits later example-level DP-SGD to map cleanly to
  patient-level DP.

This script audits the candidate representation only.
It does NOT train a model and does NOT apply differential privacy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from federated_learning.config import (
    SOURCE_DATA_FILE,
    TRAIN_TEST_ASSIGNMENT_FILE,
    COMPARISON_RESULTS_DIR,
)


# =====================================================================
# Output
# =====================================================================

OUTPUT_DIR = (
    COMPARISON_RESULTS_DIR
    / "patient_contribution"
    / "strategy_a"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Load data
# =====================================================================

print("=" * 80)
print("STRATEGY A — ONE PATIENT / ONE INDEX STAY AUDIT")
print("=" * 80)


data = pd.read_csv(
    SOURCE_DATA_FILE,
    low_memory=False,
)


manifest = pd.read_csv(
    TRAIN_TEST_ASSIGNMENT_FILE
)


required_source_columns = [
    "subject_id",
    "stay_id",
    "intime",
    "los",
    "race",
    "insurance",
]


missing_source_columns = [
    column
    for column in required_source_columns
    if column not in data.columns
]


if missing_source_columns:

    raise ValueError(
        "Required source columns missing: "
        f"{missing_source_columns}"
    )


required_manifest_columns = [
    "subject_id",
    "hospital",
    "split",
    "patient_los_profile",
]


missing_manifest_columns = [
    column
    for column in required_manifest_columns
    if column not in manifest.columns
]


if missing_manifest_columns:

    raise ValueError(
        "Required manifest columns missing: "
        f"{missing_manifest_columns}"
    )


# =====================================================================
# Validate patient manifest
# =====================================================================

if manifest[
    "subject_id"
].duplicated().any():

    raise RuntimeError(
        "Patient manifest contains duplicate subject_id."
    )


# =====================================================================
# Parse ICU admission time
# =====================================================================

data[
    "audit_intime"
] = pd.to_datetime(
    data[
        "intime"
    ],
    errors="coerce",
)


missing_intime_rows = int(
    data[
        "audit_intime"
    ]
    .isna()
    .sum()
)


print()
print("Temporal-field audit")
print("--------------------")

print(
    f"Source column used: intime"
)

print(
    f"Rows with missing/unparseable intime: "
    f"{missing_intime_rows:,}"
)


if missing_intime_rows > 0:

    raise RuntimeError(
        "Cannot define the earliest-stay representation "
        "until missing/unparseable intime values are resolved."
    )


# =====================================================================
# Attach frozen hospital and train/test ownership
# =====================================================================

working = data.merge(
    manifest[
        required_manifest_columns
    ],
    on="subject_id",
    how="left",
    validate="many_to_one",
)


if working[
    "hospital"
].isna().any():

    raise RuntimeError(
        "Some source rows have no hospital assignment."
    )


# =====================================================================
# Check temporal ties
# =====================================================================

minimum_intime = (
    working.groupby(
        "subject_id"
    )[
        "audit_intime"
    ]
    .transform(
        "min"
    )
)


earliest_candidates = (
    working[
        working[
            "audit_intime"
        ]
        == minimum_intime
    ]
)


candidate_counts = (
    earliest_candidates.groupby(
        "subject_id"
    )
    .size()
)


patients_with_earliest_time_tie = int(
    (
        candidate_counts
        > 1
    ).sum()
)


print(
    "Patients with >1 stay sharing earliest intime: "
    f"{patients_with_earliest_time_tie:,}"
)


# =====================================================================
# Deterministic index-stay selection
# =====================================================================

# stay_id is used ONLY as a deterministic tie-breaker when two ICU
# stays have exactly the same earliest intime.

ordered = (
    working.sort_values(
        [
            "subject_id",
            "audit_intime",
            "stay_id",
        ]
    )
)


index_stays = (
    ordered.groupby(
        "subject_id",
        as_index=False,
        sort=False,
    )
    .first()
)


# =====================================================================
# Core validation
# =====================================================================

n_source_patients = int(
    working[
        "subject_id"
    ].nunique()
)


n_index_patients = int(
    index_stays[
        "subject_id"
    ].nunique()
)


duplicate_index_patients = int(
    index_stays[
        "subject_id"
    ]
    .duplicated()
    .sum()
)


source_patient_set = set(
    working[
        "subject_id"
    ]
)


index_patient_set = set(
    index_stays[
        "subject_id"
    ]
)


checks = {
    "one_index_row_per_patient":
        len(
            index_stays
        )
        == n_source_patients,

    "unique_subject_id":
        duplicate_index_patients
        == 0,

    "all_patients_preserved":
        source_patient_set
        == index_patient_set,

    "expected_54314_patients":
        n_index_patients
        == 54314,

    "expected_train_patients":
        index_stays.loc[
            index_stays[
                "split"
            ]
            == "train",
            "subject_id",
        ].nunique()
        == 43450,

    "expected_test_patients":
        index_stays.loc[
            index_stays[
                "split"
            ]
            == "test",
            "subject_id",
        ].nunique()
        == 10864,
}


validation = pd.DataFrame(
    [
        {
            "check": key,
            "pass": bool(
                value
            ),
        }
        for key, value
        in checks.items()
    ]
)


# =====================================================================
# LOS labels
# =====================================================================

working[
    "long_label"
] = (
    working[
        "los"
    ]
    > 4.0
).astype(
    int
)


index_stays[
    "long_label"
] = (
    index_stays[
        "los"
    ]
    > 4.0
).astype(
    int
)


original_long_fraction = float(
    working[
        "long_label"
    ].mean()
)


index_long_fraction = float(
    index_stays[
        "long_label"
    ].mean()
)


# =====================================================================
# Global representation summary
# =====================================================================

global_summary = pd.DataFrame(
    [
        {
            "representation":
                "all_stays",

            "rows":
                len(
                    working
                ),

            "patients":
                working[
                    "subject_id"
                ].nunique(),

            "long_fraction":
                original_long_fraction,
        },

        {
            "representation":
                "strategy_a_earliest_stay",

            "rows":
                len(
                    index_stays
                ),

            "patients":
                index_stays[
                    "subject_id"
                ].nunique(),

            "long_fraction":
                index_long_fraction,
        },
    ]
)


# =====================================================================
# Index-label distribution by patient LOS profile
# =====================================================================

profile_summary = (
    index_stays.groupby(
        "patient_los_profile",
        observed=True,
    )
    .agg(
        patients=(
            "subject_id",
            "nunique",
        ),

        selected_long=(
            "long_label",
            "sum",
        ),

        selected_long_fraction=(
            "long_label",
            "mean",
        ),
    )
    .reset_index()
)


profile_summary[
    "selected_short"
] = (
    profile_summary[
        "patients"
    ]
    - profile_summary[
        "selected_long"
    ]
)


# =====================================================================
# Hospital summary
# =====================================================================

hospital_summary = (
    index_stays.groupby(
        "hospital",
        observed=True,
    )
    .agg(
        patients=(
            "subject_id",
            "nunique",
        ),

        long_patients=(
            "long_label",
            "sum",
        ),

        long_fraction=(
            "long_label",
            "mean",
        ),
    )
    .reset_index()
)


hospital_summary[
    "short_patients"
] = (
    hospital_summary[
        "patients"
    ]
    - hospital_summary[
        "long_patients"
    ]
)


# =====================================================================
# Train/test summary
# =====================================================================

split_summary = (
    index_stays.groupby(
        "split",
        observed=True,
    )
    .agg(
        patients=(
            "subject_id",
            "nunique",
        ),

        long_patients=(
            "long_label",
            "sum",
        ),

        long_fraction=(
            "long_label",
            "mean",
        ),
    )
    .reset_index()
)


split_summary[
    "short_patients"
] = (
    split_summary[
        "patients"
    ]
    - split_summary[
        "long_patients"
    ]
)


# =====================================================================
# Demographic consistency across repeated stays
# =====================================================================

race_working = (
    working[
        [
            "subject_id",
            "race",
        ]
    ]
    .copy()
)


race_working[
    "race"
] = (
    race_working[
        "race"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


insurance_working = (
    working[
        [
            "subject_id",
            "insurance",
        ]
    ]
    .copy()
)


insurance_working[
    "insurance"
] = (
    insurance_working[
        "insurance"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


race_unique_counts = (
    race_working.groupby(
        "subject_id"
    )[
        "race"
    ]
    .nunique()
)


insurance_unique_counts = (
    insurance_working.groupby(
        "subject_id"
    )[
        "insurance"
    ]
    .nunique()
)


demographic_consistency = pd.DataFrame(
    [
        {
            "attribute":
                "race",

            "patients_with_multiple_values":
                int(
                    (
                        race_unique_counts
                        > 1
                    ).sum()
                ),

            "fraction_patients_with_multiple_values":
                float(
                    (
                        race_unique_counts
                        > 1
                    ).mean()
                ),
        },

        {
            "attribute":
                "insurance",

            "patients_with_multiple_values":
                int(
                    (
                        insurance_unique_counts
                        > 1
                    ).sum()
                ),

            "fraction_patients_with_multiple_values":
                float(
                    (
                        insurance_unique_counts
                        > 1
                    ).mean()
                ),
        },
    ]
)


# =====================================================================
# Candidate manifest
# =====================================================================

candidate_manifest = (
    index_stays[
        [
            "subject_id",
            "hospital",
            "split",
            "patient_los_profile",
            "stay_id",
            "intime",
            "los",
            "long_label",
            "race",
            "insurance",
        ]
    ]
    .rename(
        columns={
            "stay_id":
                "index_stay_id",

            "intime":
                "index_intime",

            "los":
                "index_los",
        }
    )
)


# =====================================================================
# Save outputs
# =====================================================================

global_summary.to_csv(
    OUTPUT_DIR
    / "strategy_a_global_summary.csv",
    index=False,
)


profile_summary.to_csv(
    OUTPUT_DIR
    / "strategy_a_los_profile_summary.csv",
    index=False,
)


hospital_summary.to_csv(
    OUTPUT_DIR
    / "strategy_a_hospital_summary.csv",
    index=False,
)


split_summary.to_csv(
    OUTPUT_DIR
    / "strategy_a_split_summary.csv",
    index=False,
)


demographic_consistency.to_csv(
    OUTPUT_DIR
    / "strategy_a_demographic_consistency.csv",
    index=False,
)


candidate_manifest.to_csv(
    OUTPUT_DIR
    / "strategy_a_candidate_index_stays.csv",
    index=False,
)


validation.to_csv(
    OUTPUT_DIR
    / "strategy_a_validation.csv",
    index=False,
)


# =====================================================================
# Console output
# =====================================================================

print()
print("=" * 80)
print("STRATEGY A VALIDATION")
print("=" * 80)

print(
    validation.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("GLOBAL REPRESENTATION EFFECT")
print("=" * 80)

print(
    global_summary.round(
        6
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("EARLIEST-STAY LABEL BY PATIENT LOS PROFILE")
print("=" * 80)

print(
    profile_summary.round(
        6
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("EARLIEST-STAY LABEL BY HOSPITAL")
print("=" * 80)

print(
    hospital_summary.round(
        6
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("TRAIN/TEST REPRESENTATION")
print("=" * 80)

print(
    split_summary.round(
        6
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("DEMOGRAPHIC CONSISTENCY ACROSS STAYS")
print("=" * 80)

print(
    demographic_consistency.round(
        6
    ).to_string(
        index=False
    )
)


print()
print(
    "Candidate manifest saved to:"
)

print(
    "  "
    + str(
        OUTPUT_DIR
        / "strategy_a_candidate_index_stays.csv"
    )
)


print()
print(
    "IMPORTANT: this representation is not yet frozen."
)

print(
    "No model was trained and no DP was applied."
)