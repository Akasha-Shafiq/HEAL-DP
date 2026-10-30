"""
Create the baseline five-hospital patient assignment.

Experimental design
-------------------
All patients are assigned to exactly one simulated hospital.

Requested patient proportions:

    H1 = 20%
    H2 = 20%
    H3 = 20%
    H4 =  5%
    H5 = 35%

Assignment is performed at the subject_id level and stratified by the
patient-level LOS profile established during the source-data audit:

    Short-only
    Long-only
    Mixed

This stage DOES NOT create local train/test splits.

Outputs
-------
Patient assignment:
    data/federated_learning/manifests/
        patient_hospital_assignment.csv

Validation/audit:
    results/federated_learning/partitions/
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =====================================================================
# Configuration
# =====================================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "source"
)

AUDIT_DIR = (
    PROJECT_ROOT
    / "results"
    / "federated_learning"
    / "audit"
)

MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "manifests"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "federated_learning"
    / "partitions"
)

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATA_FILE = (
    SOURCE_DIR
    / "icu_data_no_imputation_atc3_lab5pct.csv"
)

PATIENT_AUDIT_FILE = (
    AUDIT_DIR
    / "patient_audit_manifest.csv"
)


HOSPITAL_PROPORTIONS = {
    "H1": 0.20,
    "H2": 0.20,
    "H3": 0.20,
    "H4": 0.05,
    "H5": 0.35,
}


# =====================================================================
# Validate inputs
# =====================================================================

for path in [
    DATA_FILE,
    PATIENT_AUDIT_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found: {path}"
        )


print("=" * 80)
print("FEDERATED-LEARNING HOSPITAL ASSIGNMENT")
print("=" * 80)

print()
print(
    "Assignment unit: subject_id"
)

print(
    "Stratification: patient_los_profile"
)

print(
    f"Random seed: {SEED}"
)

print()


# =====================================================================
# Load patient-level audit manifest
# =====================================================================

patients = pd.read_csv(
    PATIENT_AUDIT_FILE
)


required_columns = [
    "subject_id",
    "short_stays",
    "long_stays",
    "n_stays",
    "patient_los_profile",
]


missing_columns = [
    column
    for column in required_columns
    if column not in patients.columns
]


if missing_columns:

    raise ValueError(
        "Patient audit manifest is missing "
        f"columns: {missing_columns}"
    )


if patients["subject_id"].duplicated().any():

    raise ValueError(
        "Patient audit manifest contains "
        "duplicate subject_id values."
    )


n_patients = len(
    patients
)


print(
    f"Unique patients available: {n_patients:,}"
)

print()


# =====================================================================
# Calculate exact integer hospital targets
# using the largest-remainder method
# =====================================================================

raw_targets = {
    hospital: (
        n_patients
        * proportion
    )
    for hospital, proportion
    in HOSPITAL_PROPORTIONS.items()
}


hospital_targets = {
    hospital: math.floor(
        raw_target
    )
    for hospital, raw_target
    in raw_targets.items()
}


patients_remaining_to_allocate = (
    n_patients
    - sum(
        hospital_targets.values()
    )
)


remainders = sorted(
    HOSPITAL_PROPORTIONS.keys(),
    key=lambda hospital: (
        raw_targets[hospital]
        - hospital_targets[hospital]
    ),
    reverse=True,
)


for hospital in remainders[
    :patients_remaining_to_allocate
]:

    hospital_targets[
        hospital
    ] += 1


assert (
    sum(
        hospital_targets.values()
    )
    == n_patients
)


print("=" * 80)
print("1. INTEGER HOSPITAL TARGETS")
print("=" * 80)

for hospital in HOSPITAL_PROPORTIONS:

    print(
        f"{hospital}: "
        f"{hospital_targets[hospital]:,} "
        f"patients "
        f"({100 * hospital_targets[hospital] / n_patients:.4f}%)"
    )

print()


# =====================================================================
# Stratified patient assignment
# =====================================================================

# We explicitly allocate H5 and H4 rather than leaving the smallest
# hospital as the residual partition.
#
# This ensures that the 5% hospital also receives a stratified sample.
#
# The final remaining hospital is H3.

allocation_order = [
    "H5",
    "H4",
    "H1",
    "H2",
]

final_hospital = "H3"


remaining = patients.copy()

assigned_parts = []


for index, hospital in enumerate(
    allocation_order
):

    target_n = hospital_targets[
        hospital
    ]

    selected, leftover = (
        train_test_split(
            remaining,
            train_size=target_n,
            random_state=(
                SEED
                + index
            ),
            stratify=remaining[
                "patient_los_profile"
            ],
        )
    )

    selected = selected.copy()

    selected[
        "hospital"
    ] = hospital

    assigned_parts.append(
        selected
    )

    remaining = (
        leftover.copy()
    )


# The final residual partition becomes H3.

if len(
    remaining
) != hospital_targets[
    final_hospital
]:

    raise RuntimeError(
        "Residual hospital size does not "
        "match the H3 target."
    )


remaining = remaining.copy()

remaining[
    "hospital"
] = final_hospital

assigned_parts.append(
    remaining
)


assignment = pd.concat(
    assigned_parts,
    ignore_index=True,
)


# Reorder rows only for readability.
#
# subject_id values are NOT modified.

assignment = assignment.sort_values(
    [
        "hospital",
        "subject_id",
    ]
).reset_index(
    drop=True
)


# =====================================================================
# Core patient-disjoint validation
# =====================================================================

duplicate_patients = int(
    assignment[
        "subject_id"
    ]
    .duplicated()
    .sum()
)


unique_assigned_patients = int(
    assignment[
        "subject_id"
    ]
    .nunique()
)


all_source_patients_assigned = (
    set(
        patients[
            "subject_id"
        ]
    )
    == set(
        assignment[
            "subject_id"
        ]
    )
)


actual_hospital_counts = (
    assignment[
        "hospital"
    ]
    .value_counts()
    .reindex(
        HOSPITAL_PROPORTIONS.keys()
    )
)


exact_target_match = all(
    int(
        actual_hospital_counts[
            hospital
        ]
    )
    == hospital_targets[
        hospital
    ]
    for hospital
    in HOSPITAL_PROPORTIONS
)


# =====================================================================
# Hospital patient summary
# =====================================================================

hospital_patient_summary = (
    assignment.groupby(
        "hospital",
        observed=True,
    )
    .agg(
        patients=(
            "subject_id",
            "nunique",
        ),
        patient_stays=(
            "n_stays",
            "sum",
        ),
        mean_stays_per_patient=(
            "n_stays",
            "mean",
        ),
        maximum_stays_per_patient=(
            "n_stays",
            "max",
        ),
    )
    .reset_index()
)


hospital_patient_summary[
    "patient_fraction"
] = (
    hospital_patient_summary[
        "patients"
    ]
    / n_patients
)


hospital_patient_summary[
    "requested_fraction"
] = (
    hospital_patient_summary[
        "hospital"
    ]
    .map(
        HOSPITAL_PROPORTIONS
    )
)


hospital_patient_summary = (
    hospital_patient_summary[
        [
            "hospital",
            "patients",
            "patient_fraction",
            "requested_fraction",
            "patient_stays",
            "mean_stays_per_patient",
            "maximum_stays_per_patient",
        ]
    ]
)


# =====================================================================
# Patient LOS-profile distribution by hospital
# =====================================================================

profile_counts = (
    assignment.groupby(
        [
            "hospital",
            "patient_los_profile",
        ],
        observed=True,
    )
    .size()
    .rename(
        "patients"
    )
    .reset_index()
)


profile_totals = (
    profile_counts.groupby(
        "hospital"
    )[
        "patients"
    ]
    .transform(
        "sum"
    )
)


profile_counts[
    "fraction_within_hospital"
] = (
    profile_counts[
        "patients"
    ]
    / profile_totals
)


global_profile = (
    patients[
        "patient_los_profile"
    ]
    .value_counts(
        normalize=True
    )
)


profile_counts[
    "global_fraction"
] = (
    profile_counts[
        "patient_los_profile"
    ]
    .map(
        global_profile
    )
)


profile_counts[
    "absolute_deviation"
] = (
    profile_counts[
        "fraction_within_hospital"
    ]
    - profile_counts[
        "global_fraction"
    ]
).abs()


# =====================================================================
# Load only metadata needed for stay-level assignment audit
# =====================================================================

metadata_columns = [
    "subject_id",
    "stay_id",
    "los",
    "race",
    "insurance",
]


if "hadm_id" in pd.read_csv(
    DATA_FILE,
    nrows=0,
).columns:

    metadata_columns.append(
        "hadm_id"
    )


stay_data = pd.read_csv(
    DATA_FILE,
    usecols=metadata_columns,
    low_memory=False,
)


stay_data[
    "los_type"
] = np.where(
    stay_data[
        "los"
    ] > 4.0,
    "Long",
    "Short",
)


stay_assignment = stay_data.merge(
    assignment[
        [
            "subject_id",
            "hospital",
        ]
    ],
    on="subject_id",
    how="left",
    validate="many_to_one",
)


unassigned_stay_rows = int(
    stay_assignment[
        "hospital"
    ]
    .isna()
    .sum()
)


# =====================================================================
# Stay-level hospital summary
# =====================================================================

hospital_stay_summary = (
    stay_assignment.groupby(
        "hospital",
        observed=True,
    )
    .agg(
        patients=(
            "subject_id",
            "nunique",
        ),
        stays=(
            "stay_id",
            "nunique",
        ),
        long_stays=(
            "los_type",
            lambda x: int(
                (
                    x
                    == "Long"
                ).sum()
            ),
        ),
        short_stays=(
            "los_type",
            lambda x: int(
                (
                    x
                    == "Short"
                ).sum()
            ),
        ),
    )
    .reset_index()
)


hospital_stay_summary[
    "long_stay_fraction"
] = (
    hospital_stay_summary[
        "long_stays"
    ]
    / hospital_stay_summary[
        "stays"
    ]
)


# =====================================================================
# Race distribution by hospital
# =====================================================================

stay_assignment[
    "race_audit"
] = (
    stay_assignment[
        "race"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


hospital_race = (
    stay_assignment.groupby(
        [
            "hospital",
            "race_audit",
        ],
        observed=True,
    )
    .size()
    .rename(
        "rows"
    )
    .reset_index()
    .rename(
        columns={
            "race_audit": "race"
        }
    )
)


hospital_race[
    "fraction_within_hospital"
] = (
    hospital_race[
        "rows"
    ]
    / hospital_race.groupby(
        "hospital"
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)


global_race = (
    stay_assignment[
        "race_audit"
    ]
    .value_counts(
        normalize=True
    )
)


hospital_race[
    "global_fraction"
] = (
    hospital_race[
        "race"
    ]
    .map(
        global_race
    )
)


hospital_race[
    "absolute_deviation"
] = (
    hospital_race[
        "fraction_within_hospital"
    ]
    - hospital_race[
        "global_fraction"
    ]
).abs()


# =====================================================================
# Insurance distribution by hospital
# =====================================================================

stay_assignment[
    "insurance_audit"
] = (
    stay_assignment[
        "insurance"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


hospital_insurance = (
    stay_assignment.groupby(
        [
            "hospital",
            "insurance_audit",
        ],
        observed=True,
    )
    .size()
    .rename(
        "rows"
    )
    .reset_index()
    .rename(
        columns={
            "insurance_audit":
                "insurance"
        }
    )
)


hospital_insurance[
    "fraction_within_hospital"
] = (
    hospital_insurance[
        "rows"
    ]
    / hospital_insurance.groupby(
        "hospital"
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)


global_insurance = (
    stay_assignment[
        "insurance_audit"
    ]
    .value_counts(
        normalize=True
    )
)


hospital_insurance[
    "global_fraction"
] = (
    hospital_insurance[
        "insurance"
    ]
    .map(
        global_insurance
    )
)


hospital_insurance[
    "absolute_deviation"
] = (
    hospital_insurance[
        "fraction_within_hospital"
    ]
    - hospital_insurance[
        "global_fraction"
    ]
).abs()


# =====================================================================
# Validation
# =====================================================================

checks = {
    "all_54314_patients_assigned": (
        unique_assigned_patients
        == n_patients
    ),

    "each_patient_assigned_once": (
        duplicate_patients
        == 0
    ),

    "source_and_assignment_patient_sets_match": (
        all_source_patients_assigned
    ),

    "hospital_sizes_match_requested_targets": (
        exact_target_match
    ),

    "all_stay_rows_have_hospital": (
        unassigned_stay_rows
        == 0
    ),

    "patient_count_preserved": (
        hospital_patient_summary[
            "patients"
        ].sum()
        == n_patients
    ),

    "stay_count_preserved": (
        hospital_stay_summary[
            "stays"
        ].sum()
        == stay_data[
            "stay_id"
        ].nunique()
    ),
}


validation_df = pd.DataFrame(
    [
        {
            "check": name,
            "pass": bool(
                result
            ),
        }
        for name, result
        in checks.items()
    ]
)


# =====================================================================
# Print summaries
# =====================================================================

print("=" * 80)
print("2. HOSPITAL PATIENT COUNTS")
print("=" * 80)

print(
    hospital_patient_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("3. PATIENT LOS PROFILE BY HOSPITAL")
print("=" * 80)

print(
    profile_counts.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("4. STAY-LEVEL HOSPITAL SUMMARY")
print("=" * 80)

print(
    hospital_stay_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("5. DISTRIBUTION BALANCE CHECK")
print("=" * 80)

print(
    "Maximum absolute LOS-profile deviation "
    "from global distribution:"
)

print(
    f"  {profile_counts['absolute_deviation'].max():.6f}"
)

print()

print(
    "Maximum absolute race deviation "
    "from global stay-level distribution:"
)

print(
    f"  {hospital_race['absolute_deviation'].max():.6f}"
)

print()

print(
    "Maximum absolute insurance deviation "
    "from global stay-level distribution:"
)

print(
    f"  {hospital_insurance['absolute_deviation'].max():.6f}"
)

print()


print("=" * 80)
print("6. ASSIGNMENT VALIDATION")
print("=" * 80)

print(
    validation_df.to_string(
        index=False
    )
)

print()


# =====================================================================
# Save manifests/results
# =====================================================================

assignment.to_csv(
    MANIFEST_DIR
    / "patient_hospital_assignment.csv",
    index=False,
)


hospital_patient_summary.to_csv(
    RESULTS_DIR
    / "hospital_patient_summary.csv",
    index=False,
)


profile_counts.to_csv(
    RESULTS_DIR
    / "hospital_patient_los_profiles.csv",
    index=False,
)


hospital_stay_summary.to_csv(
    RESULTS_DIR
    / "hospital_stay_summary.csv",
    index=False,
)


hospital_race.to_csv(
    RESULTS_DIR
    / "hospital_race_distribution.csv",
    index=False,
)


hospital_insurance.to_csv(
    RESULTS_DIR
    / "hospital_insurance_distribution.csv",
    index=False,
)


validation_df.to_csv(
    RESULTS_DIR
    / "hospital_assignment_validation.csv",
    index=False,
)


# =====================================================================
# Completion
# =====================================================================

print("=" * 80)
print("HOSPITAL ASSIGNMENT COMPLETE")
print("=" * 80)

print()

print(
    "Patient manifest:"
)

print(
    "  "
    + str(
        MANIFEST_DIR
        / "patient_hospital_assignment.csv"
    )
)

print()

print(
    "Results:"
)

print(
    "  "
    + str(
        RESULTS_DIR
    )
)

print()

print(
    "All validation checks passed:"
    f" {validation_df['pass'].all()}"
)

print()

print(
    "IMPORTANT: no train/test split "
    "has been created yet."
)