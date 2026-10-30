"""
Create patient-disjoint 80/20 train-test splits within each
previously frozen simulated hospital.

Experimental design
-------------------
Hospital ownership is already fixed:

    H1 = 20%
    H2 = 20%
    H3 = 20%
    H4 =  5%
    H5 = 35%

Within EACH hospital:

    80% patients -> train
    20% patients -> test

The split is performed at subject_id level and stratified by
patient_los_profile:

    Short-only
    Long-only
    Mixed

No full feature partition files are created in this stage.

Outputs
-------
Canonical patient split manifest:

    data/federated_learning/manifests/
        patient_hospital_train_test_assignment.csv

Validation summaries:

    results/federated_learning/partitions/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =====================================================================
# Configuration
# =====================================================================

SEED = 42
TEST_FRACTION = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "source"
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

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATA_FILE = (
    SOURCE_DIR
    / "icu_data_no_imputation_atc3_lab5pct.csv"
)

HOSPITAL_ASSIGNMENT_FILE = (
    MANIFEST_DIR
    / "patient_hospital_assignment.csv"
)

OUTPUT_MANIFEST = (
    MANIFEST_DIR
    / "patient_hospital_train_test_assignment.csv"
)


HOSPITALS = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
]


# =====================================================================
# Input validation
# =====================================================================

for path in [
    DATA_FILE,
    HOSPITAL_ASSIGNMENT_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found: {path}"
        )


print("=" * 80)
print("FEDERATED-LEARNING LOCAL TRAIN/TEST SPLIT")
print("=" * 80)

print()
print("Split unit: subject_id")
print("Split performed independently within each hospital")
print(f"Train fraction: {1 - TEST_FRACTION:.2f}")
print(f"Test fraction:  {TEST_FRACTION:.2f}")
print("Stratification: patient_los_profile")
print(f"Random seed base: {SEED}")
print()


# =====================================================================
# Load frozen hospital assignment
# =====================================================================

assignment = pd.read_csv(
    HOSPITAL_ASSIGNMENT_FILE
)


required_assignment_columns = [
    "subject_id",
    "hospital",
    "short_stays",
    "long_stays",
    "n_stays",
    "patient_los_profile",
]


missing_columns = [
    column
    for column in required_assignment_columns
    if column not in assignment.columns
]


if missing_columns:

    raise ValueError(
        "Hospital assignment manifest is missing: "
        f"{missing_columns}"
    )


if assignment["subject_id"].duplicated().any():

    raise ValueError(
        "Hospital assignment contains duplicate subject_id values."
    )


if set(
    assignment["hospital"].unique()
) != set(
    HOSPITALS
):

    raise ValueError(
        "Hospital labels do not match H1-H5."
    )


n_patients = int(
    assignment["subject_id"].nunique()
)


print(
    f"Patients in frozen hospital manifest: {n_patients:,}"
)

print()


# =====================================================================
# Create independent stratified 80/20 split within each hospital
# =====================================================================

split_parts = []


for hospital_index, hospital in enumerate(
    HOSPITALS
):

    hospital_patients = (
        assignment[
            assignment["hospital"]
            == hospital
        ]
        .copy()
    )


    train_part, test_part = train_test_split(
        hospital_patients,
        test_size=TEST_FRACTION,
        random_state=(
            SEED
            + hospital_index
        ),
        stratify=hospital_patients[
            "patient_los_profile"
        ],
    )


    train_part = train_part.copy()
    test_part = test_part.copy()


    train_part["split"] = "train"
    test_part["split"] = "test"


    split_parts.extend(
        [
            train_part,
            test_part,
        ]
    )


patient_split = pd.concat(
    split_parts,
    ignore_index=True,
)


patient_split = patient_split.sort_values(
    [
        "hospital",
        "split",
        "subject_id",
    ]
).reset_index(
    drop=True
)


# =====================================================================
# Core patient-level validation
# =====================================================================

unique_split_patients = int(
    patient_split[
        "subject_id"
    ].nunique()
)


duplicate_patient_rows = int(
    patient_split[
        "subject_id"
    ]
    .duplicated()
    .sum()
)


same_patient_set = (
    set(
        assignment["subject_id"]
    )
    == set(
        patient_split["subject_id"]
    )
)


original_hospital_map = (
    assignment[
        [
            "subject_id",
            "hospital",
        ]
    ]
    .set_index(
        "subject_id"
    )[
        "hospital"
    ]
)


new_hospital_map = (
    patient_split[
        [
            "subject_id",
            "hospital",
        ]
    ]
    .set_index(
        "subject_id"
    )[
        "hospital"
    ]
)


hospital_ownership_unchanged = (
    original_hospital_map
    .sort_index()
    .equals(
        new_hospital_map
        .sort_index()
    )
)


# =====================================================================
# Check train/test patient overlap explicitly
# =====================================================================

train_ids = set(
    patient_split.loc[
        patient_split["split"]
        == "train",
        "subject_id",
    ]
)


test_ids = set(
    patient_split.loc[
        patient_split["split"]
        == "test",
        "subject_id",
    ]
)


train_test_overlap = (
    train_ids
    & test_ids
)


# =====================================================================
# Hospital-level patient counts
# =====================================================================

hospital_split_summary = (
    patient_split.groupby(
        [
            "hospital",
            "split",
        ],
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
    )
    .reset_index()
)


hospital_totals = (
    patient_split.groupby(
        "hospital"
    )[
        "subject_id"
    ]
    .nunique()
)


hospital_split_summary[
    "hospital_total_patients"
] = (
    hospital_split_summary[
        "hospital"
    ]
    .map(
        hospital_totals
    )
)


hospital_split_summary[
    "fraction_within_hospital"
] = (
    hospital_split_summary[
        "patients"
    ]
    / hospital_split_summary[
        "hospital_total_patients"
    ]
)


# =====================================================================
# LOS-profile distribution
# =====================================================================

profile_summary = (
    patient_split.groupby(
        [
            "hospital",
            "split",
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


profile_split_totals = (
    profile_summary.groupby(
        [
            "hospital",
            "split",
        ]
    )[
        "patients"
    ]
    .transform(
        "sum"
    )
)


profile_summary[
    "fraction_within_split"
] = (
    profile_summary[
        "patients"
    ]
    / profile_split_totals
)


hospital_profile_reference = (
    assignment.groupby(
        [
            "hospital",
            "patient_los_profile",
        ]
    )
    .size()
    .rename(
        "patients"
    )
    .reset_index()
)


hospital_profile_reference[
    "hospital_total"
] = (
    hospital_profile_reference.groupby(
        "hospital"
    )[
        "patients"
    ]
    .transform(
        "sum"
    )
)


hospital_profile_reference[
    "hospital_fraction"
] = (
    hospital_profile_reference[
        "patients"
    ]
    / hospital_profile_reference[
        "hospital_total"
    ]
)


reference_lookup = (
    hospital_profile_reference
    .set_index(
        [
            "hospital",
            "patient_los_profile",
        ]
    )[
        "hospital_fraction"
    ]
)


profile_summary[
    "hospital_reference_fraction"
] = [
    reference_lookup.loc[
        (
            row.hospital,
            row.patient_los_profile,
        )
    ]
    for row in profile_summary.itertuples()
]


profile_summary[
    "absolute_deviation"
] = (
    profile_summary[
        "fraction_within_split"
    ]
    - profile_summary[
        "hospital_reference_fraction"
    ]
).abs()


# =====================================================================
# Load stay-level metadata only
# =====================================================================

source_header = pd.read_csv(
    DATA_FILE,
    nrows=0,
)


metadata_columns = [
    "subject_id",
    "stay_id",
    "los",
    "race",
    "insurance",
]


if "hadm_id" in source_header.columns:

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
    stay_data["los"] > 4.0,
    "Long",
    "Short",
)


# =====================================================================
# Attach fixed hospital + split to every stay
# =====================================================================

stay_split = stay_data.merge(
    patient_split[
        [
            "subject_id",
            "hospital",
            "split",
        ]
    ],
    on="subject_id",
    how="left",
    validate="many_to_one",
)


unassigned_stay_rows = int(
    stay_split[
        "hospital"
    ]
    .isna()
    .sum()
)


# =====================================================================
# Stay-level train/test summary
# =====================================================================

stay_summary = (
    stay_split.groupby(
        [
            "hospital",
            "split",
        ],
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
                    x == "Long"
                ).sum()
            ),
        ),
        short_stays=(
            "los_type",
            lambda x: int(
                (
                    x == "Short"
                ).sum()
            ),
        ),
    )
    .reset_index()
)


stay_summary[
    "long_stay_fraction"
] = (
    stay_summary[
        "long_stays"
    ]
    / stay_summary[
        "stays"
    ]
)


# =====================================================================
# Global train/test summary
# =====================================================================

global_split_summary = (
    stay_split.groupby(
        "split",
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
                    x == "Long"
                ).sum()
            ),
        ),
        short_stays=(
            "los_type",
            lambda x: int(
                (
                    x == "Short"
                ).sum()
            ),
        ),
    )
    .reset_index()
)


global_split_summary[
    "patient_fraction"
] = (
    global_split_summary[
        "patients"
    ]
    / n_patients
)


global_split_summary[
    "long_stay_fraction"
] = (
    global_split_summary[
        "long_stays"
    ]
    / global_split_summary[
        "stays"
    ]
)


# =====================================================================
# Race balance audit
# =====================================================================

stay_split[
    "race_audit"
] = (
    stay_split[
        "race"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


race_summary = (
    stay_split.groupby(
        [
            "hospital",
            "split",
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


race_summary[
    "fraction_within_split"
] = (
    race_summary[
        "rows"
    ]
    / race_summary.groupby(
        [
            "hospital",
            "split",
        ]
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)


# Hospital-level reference race distribution

hospital_race_reference = (
    stay_split.groupby(
        [
            "hospital",
            "race_audit",
        ]
    )
    .size()
    .rename(
        "rows"
    )
    .reset_index()
)


hospital_race_reference[
    "hospital_fraction"
] = (
    hospital_race_reference[
        "rows"
    ]
    / hospital_race_reference.groupby(
        "hospital"
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)


race_reference_lookup = (
    hospital_race_reference
    .set_index(
        [
            "hospital",
            "race_audit",
        ]
    )[
        "hospital_fraction"
    ]
)


race_summary[
    "hospital_reference_fraction"
] = [
    race_reference_lookup.loc[
        (
            row.hospital,
            row.race,
        )
    ]
    for row in race_summary.itertuples()
]


race_summary[
    "absolute_deviation"
] = (
    race_summary[
        "fraction_within_split"
    ]
    - race_summary[
        "hospital_reference_fraction"
    ]
).abs()


# =====================================================================
# Insurance balance audit
# =====================================================================

stay_split[
    "insurance_audit"
] = (
    stay_split[
        "insurance"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


insurance_summary = (
    stay_split.groupby(
        [
            "hospital",
            "split",
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


insurance_summary[
    "fraction_within_split"
] = (
    insurance_summary[
        "rows"
    ]
    / insurance_summary.groupby(
        [
            "hospital",
            "split",
        ]
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)


hospital_insurance_reference = (
    stay_split.groupby(
        [
            "hospital",
            "insurance_audit",
        ]
    )
    .size()
    .rename(
        "rows"
    )
    .reset_index()
)


hospital_insurance_reference[
    "hospital_fraction"
] = (
    hospital_insurance_reference[
        "rows"
    ]
    / hospital_insurance_reference.groupby(
        "hospital"
    )[
        "rows"
    ]
    .transform(
        "sum"
    )
)


insurance_reference_lookup = (
    hospital_insurance_reference
    .set_index(
        [
            "hospital",
            "insurance_audit",
        ]
    )[
        "hospital_fraction"
    ]
)


insurance_summary[
    "hospital_reference_fraction"
] = [
    insurance_reference_lookup.loc[
        (
            row.hospital,
            row.insurance,
        )
    ]
    for row in insurance_summary.itertuples()
]


insurance_summary[
    "absolute_deviation"
] = (
    insurance_summary[
        "fraction_within_split"
    ]
    - insurance_summary[
        "hospital_reference_fraction"
    ]
).abs()


# =====================================================================
# Validation
# =====================================================================

hospital_test_fractions = (
    patient_split.groupby(
        "hospital"
    )[
        "split"
    ]
    .apply(
        lambda x: float(
            (
                x == "test"
            ).mean()
        )
    )
)


# Integer rounding means exact 0.20 is not always possible.
# Allow < 0.001 absolute deviation.

test_fraction_valid = bool(
    (
        hospital_test_fractions
        .sub(
            TEST_FRACTION
        )
        .abs()
        < 0.001
    ).all()
)


checks = {
    "all_patients_preserved": (
        unique_split_patients
        == n_patients
    ),

    "each_patient_has_one_split": (
        duplicate_patient_rows
        == 0
    ),

    "source_and_split_patient_sets_match": (
        same_patient_set
    ),

    "hospital_ownership_unchanged": (
        hospital_ownership_unchanged
    ),

    "zero_train_test_patient_overlap": (
        len(
            train_test_overlap
        )
        == 0
    ),

    "all_stay_rows_have_hospital_and_split": (
        unassigned_stay_rows
        == 0
    ),

    "local_test_fraction_approximately_20pct": (
        test_fraction_valid
    ),

    "patient_count_preserved": (
        hospital_split_summary[
            "patients"
        ].sum()
        == n_patients
    ),

    "stay_count_preserved": (
        stay_summary[
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
            "pass": bool(result),
        }
        for name, result
        in checks.items()
    ]
)


# =====================================================================
# Print outputs
# =====================================================================

print("=" * 80)
print("1. HOSPITAL TRAIN/TEST PATIENT COUNTS")
print("=" * 80)

print(
    hospital_split_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("2. GLOBAL TRAIN/TEST SUMMARY")
print("=" * 80)

print(
    global_split_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("3. PATIENT LOS PROFILE BALANCE")
print("=" * 80)

print(
    profile_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("4. STAY-LEVEL HOSPITAL TRAIN/TEST SUMMARY")
print("=" * 80)

print(
    stay_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()


print("=" * 80)
print("5. BALANCE DIAGNOSTICS")
print("=" * 80)

print(
    "Maximum LOS-profile deviation from "
    "the parent hospital:"
)

print(
    f"  {profile_summary['absolute_deviation'].max():.6f}"
)

print()

print(
    "Maximum race deviation from "
    "the parent hospital:"
)

print(
    f"  {race_summary['absolute_deviation'].max():.6f}"
)

print()

print(
    "Maximum insurance deviation from "
    "the parent hospital:"
)

print(
    f"  {insurance_summary['absolute_deviation'].max():.6f}"
)

print()


print("=" * 80)
print("6. TRAIN/TEST VALIDATION")
print("=" * 80)

print(
    validation_df.to_string(
        index=False
    )
)

print()


# =====================================================================
# Save outputs
# =====================================================================

patient_split.to_csv(
    OUTPUT_MANIFEST,
    index=False,
)


hospital_split_summary.to_csv(
    RESULTS_DIR
    / "hospital_train_test_patient_summary.csv",
    index=False,
)


global_split_summary.to_csv(
    RESULTS_DIR
    / "global_train_test_summary.csv",
    index=False,
)


profile_summary.to_csv(
    RESULTS_DIR
    / "hospital_train_test_los_profiles.csv",
    index=False,
)


stay_summary.to_csv(
    RESULTS_DIR
    / "hospital_train_test_stay_summary.csv",
    index=False,
)


race_summary.to_csv(
    RESULTS_DIR
    / "hospital_train_test_race_distribution.csv",
    index=False,
)


insurance_summary.to_csv(
    RESULTS_DIR
    / "hospital_train_test_insurance_distribution.csv",
    index=False,
)


validation_df.to_csv(
    RESULTS_DIR
    / "train_test_split_validation.csv",
    index=False,
)


# =====================================================================
# Completion
# =====================================================================

print("=" * 80)
print("LOCAL TRAIN/TEST SPLIT COMPLETE")
print("=" * 80)

print()

print("Canonical split manifest:")

print(
    f"  {OUTPUT_MANIFEST}"
)

print()

print(
    "Train patients:"
    f" {len(train_ids):,}"
)

print(
    "Test patients:"
    f"  {len(test_ids):,}"
)

print(
    "Train/test overlap:"
    f" {len(train_test_overlap):,}"
)

print()

print(
    "All validation checks passed:"
    f" {validation_df['pass'].all()}"
)

print()

print(
    "IMPORTANT: full feature partition files "
    "have NOT been materialized yet."
)