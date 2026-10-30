"""
Federated-learning source-data audit.

Purpose
-------
Audit the frozen modelling dataset before any hospital assignment,
train/test split, federated training, or differential privacy.

This script DOES NOT modify the source dataset.

It validates:

1. Dataset size and identifiers.
2. Patient/stay multiplicity.
3. Patient-level Short/Long/Mixed LOS profiles.
4. Overall LOS class balance.
5. Race distribution.
6. Insurance distribution.
7. Frozen 351-feature representation.
8. Binary/continuous preprocessing specification.
9. Missing/non-finite feature values.
10. Duplicate stay identifiers.
11. Candidate patient-level stratification variables.

Outputs are written only to:

    results/federated_learning/audit/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# Paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "source"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "federated_learning"
    / "audit"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATA_FILE = (
    SOURCE_DIR
    / "icu_data_no_imputation_atc3_lab5pct.csv"
)

FEATURE_FILE = (
    SOURCE_DIR
    / "final_351_feature_list.csv"
)

PREPROCESSING_FILE = (
    SOURCE_DIR
    / "a3_preprocessing_specification.csv"
)


# =====================================================================
# Constants
# =====================================================================

EXPECTED_FEATURES = 351
EXPECTED_BINARY_FEATURES = 193
EXPECTED_CONTINUOUS_FEATURES = 158

LOS_THRESHOLD_DAYS = 4.0


# =====================================================================
# Validate input files
# =====================================================================

for path in [
    DATA_FILE,
    FEATURE_FILE,
    PREPROCESSING_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen input not found: {path}"
        )


print("=" * 80)
print("FEDERATED-LEARNING SOURCE DATA AUDIT")
print("=" * 80)

print()
print("Dataset:")
print(f"  {DATA_FILE}")

print("Feature list:")
print(f"  {FEATURE_FILE}")

print("Preprocessing specification:")
print(f"  {PREPROCESSING_FILE}")

print()


# =====================================================================
# Load frozen metadata
# =====================================================================

feature_df = pd.read_csv(
    FEATURE_FILE
)

preprocessing_df = pd.read_csv(
    PREPROCESSING_FILE
)


if "feature" not in feature_df.columns:
    raise ValueError(
        "Feature list must contain a 'feature' column."
    )

if (
    "feature" not in preprocessing_df.columns
    or "feature_type" not in preprocessing_df.columns
):
    raise ValueError(
        "Preprocessing specification must contain "
        "'feature' and 'feature_type' columns."
    )


feature_columns = (
    feature_df["feature"]
    .astype(str)
    .tolist()
)


binary_features = (
    preprocessing_df.loc[
        preprocessing_df["feature_type"]
        .astype(str)
        .str.lower()
        == "binary",
        "feature",
    ]
    .astype(str)
    .tolist()
)


continuous_features = (
    preprocessing_df.loc[
        preprocessing_df["feature_type"]
        .astype(str)
        .str.lower()
        == "continuous",
        "feature",
    ]
    .astype(str)
    .tolist()
)


# =====================================================================
# Load source dataset
# =====================================================================

print("Loading frozen modelling dataset...")

data = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print("Dataset loaded.")
print()


# =====================================================================
# Required metadata columns
# =====================================================================

required_metadata = [
    "subject_id",
    "stay_id",
    "los",
    "race",
    "insurance",
]

missing_metadata = [
    column
    for column in required_metadata
    if column not in data.columns
]

if missing_metadata:
    raise ValueError(
        "Required metadata columns are missing: "
        f"{missing_metadata}"
    )


# =====================================================================
# Recreate canonical LOS label
# =====================================================================

# Do not alter the source CSV.
# This creates only an in-memory derived column.

data["audit_los_type"] = np.where(
    data["los"] > LOS_THRESHOLD_DAYS,
    "Long",
    "Short",
)


# =====================================================================
# Dataset summary
# =====================================================================

n_rows = int(
    len(data)
)

n_patients = int(
    data["subject_id"].nunique()
)

n_stays = int(
    data["stay_id"].nunique()
)

n_hadm = (
    int(data["hadm_id"].nunique())
    if "hadm_id" in data.columns
    else np.nan
)


duplicate_stay_ids = int(
    data["stay_id"].duplicated(
        keep=False
    ).sum()
)

duplicate_subject_stay_pairs = int(
    data.duplicated(
        subset=[
            "subject_id",
            "stay_id",
        ],
        keep=False,
    ).sum()
)


dataset_summary = pd.DataFrame(
    [
        {
            "rows": n_rows,
            "unique_patients": n_patients,
            "unique_stays": n_stays,
            "unique_admissions": n_hadm,
            "columns": int(data.shape[1]),
            "los_threshold_days": LOS_THRESHOLD_DAYS,
            "duplicate_stay_id_rows": duplicate_stay_ids,
            "duplicate_subject_stay_rows": (
                duplicate_subject_stay_pairs
            ),
        }
    ]
)


print("=" * 80)
print("1. DATASET SUMMARY")
print("=" * 80)

print(
    f"Rows / ICU-stay observations: {n_rows:,}"
)

print(
    f"Unique patients:             {n_patients:,}"
)

print(
    f"Unique stay_id:              {n_stays:,}"
)

if not pd.isna(n_hadm):
    print(
        f"Unique hadm_id:              {int(n_hadm):,}"
    )

print(
    f"Dataset columns:             {data.shape[1]:,}"
)

print(
    f"Duplicate stay_id rows:      {duplicate_stay_ids:,}"
)

print(
    "Duplicate subject/stay rows: "
    f"{duplicate_subject_stay_pairs:,}"
)

print()


# =====================================================================
# Patient stay multiplicity
# =====================================================================

patient_stay_counts = (
    data.groupby(
        "subject_id"
    )
    .size()
    .rename(
        "n_stays"
    )
    .reset_index()
)


stay_frequency = (
    patient_stay_counts[
        "n_stays"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "stays_per_patient"
    )
    .reset_index(
        name="patients"
    )
)

stay_frequency["patient_fraction"] = (
    stay_frequency["patients"]
    / n_patients
)


n_one_stay = int(
    (
        patient_stay_counts["n_stays"]
        == 1
    ).sum()
)

n_two_stays = int(
    (
        patient_stay_counts["n_stays"]
        == 2
    ).sum()
)

n_three_plus = int(
    (
        patient_stay_counts["n_stays"]
        >= 3
    ).sum()
)

max_stays = int(
    patient_stay_counts[
        "n_stays"
    ].max()
)

median_stays = float(
    patient_stay_counts[
        "n_stays"
    ].median()
)

mean_stays = float(
    patient_stay_counts[
        "n_stays"
    ].mean()
)


print("=" * 80)
print("2. PATIENT STAY MULTIPLICITY")
print("=" * 80)

print(
    f"Patients with exactly 1 stay: "
    f"{n_one_stay:,} "
    f"({100*n_one_stay/n_patients:.2f}%)"
)

print(
    f"Patients with exactly 2 stays: "
    f"{n_two_stays:,} "
    f"({100*n_two_stays/n_patients:.2f}%)"
)

print(
    f"Patients with 3+ stays:         "
    f"{n_three_plus:,} "
    f"({100*n_three_plus/n_patients:.2f}%)"
)

print(
    f"Mean stays/patient:             "
    f"{mean_stays:.3f}"
)

print(
    f"Median stays/patient:           "
    f"{median_stays:.1f}"
)

print(
    f"Maximum stays/patient:          "
    f"{max_stays}"
)

print()


# =====================================================================
# Stay-level label distribution
# =====================================================================

label_distribution = (
    data["audit_los_type"]
    .value_counts()
    .rename_axis(
        "los_type"
    )
    .reset_index(
        name="rows"
    )
)

label_distribution[
    "fraction"
] = (
    label_distribution["rows"]
    / n_rows
)


print("=" * 80)
print("3. STAY-LEVEL LOS CLASS DISTRIBUTION")
print("=" * 80)

for _, row in label_distribution.iterrows():

    print(
        f"{row['los_type']:>5}: "
        f"{int(row['rows']):,} "
        f"({100*row['fraction']:.2f}%)"
    )

print()


# =====================================================================
# Patient-level Short / Long / Mixed profiles
# =====================================================================

patient_label_counts = (
    data.groupby(
        "subject_id"
    )[
        "audit_los_type"
    ]
    .value_counts()
    .unstack(
        fill_value=0
    )
)


if "Short" not in patient_label_counts.columns:
    patient_label_counts[
        "Short"
    ] = 0

if "Long" not in patient_label_counts.columns:
    patient_label_counts[
        "Long"
    ] = 0


patient_label_counts = (
    patient_label_counts[
        [
            "Short",
            "Long",
        ]
    ]
    .rename(
        columns={
            "Short": "short_stays",
            "Long": "long_stays",
        }
    )
    .reset_index()
)


patient_label_counts[
    "n_stays"
] = (
    patient_label_counts[
        "short_stays"
    ]
    + patient_label_counts[
        "long_stays"
    ]
)


def classify_patient_profile(
    row,
):

    if (
        row["short_stays"] > 0
        and row["long_stays"] == 0
    ):
        return "Short-only"

    if (
        row["long_stays"] > 0
        and row["short_stays"] == 0
    ):
        return "Long-only"

    return "Mixed"


patient_label_counts[
    "patient_los_profile"
] = patient_label_counts.apply(
    classify_patient_profile,
    axis=1,
)


patient_profile_distribution = (
    patient_label_counts[
        "patient_los_profile"
    ]
    .value_counts()
    .rename_axis(
        "patient_los_profile"
    )
    .reset_index(
        name="patients"
    )
)


patient_profile_distribution[
    "fraction"
] = (
    patient_profile_distribution[
        "patients"
    ]
    / n_patients
)


print("=" * 80)
print("4. PATIENT-LEVEL LOS PROFILE")
print("=" * 80)

for _, row in (
    patient_profile_distribution
    .iterrows()
):

    print(
        f"{row['patient_los_profile']:>10}: "
        f"{int(row['patients']):,} "
        f"({100*row['fraction']:.2f}%)"
    )

print()


# =====================================================================
# Race distribution
# =====================================================================

race_distribution = (
    data["race"]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
    .value_counts()
    .rename_axis(
        "race"
    )
    .reset_index(
        name="rows"
    )
)

race_distribution[
    "fraction"
] = (
    race_distribution["rows"]
    / n_rows
)


print("=" * 80)
print("5. RACE DISTRIBUTION — TOP 15")
print("=" * 80)

print(
    race_distribution
    .head(15)
    .to_string(
        index=False
    )
)

print()


# =====================================================================
# Insurance distribution
# =====================================================================

insurance_distribution = (
    data["insurance"]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
    .value_counts()
    .rename_axis(
        "insurance"
    )
    .reset_index(
        name="rows"
    )
)

insurance_distribution[
    "fraction"
] = (
    insurance_distribution["rows"]
    / n_rows
)


print("=" * 80)
print("6. INSURANCE DISTRIBUTION")
print("=" * 80)

print(
    insurance_distribution
    .to_string(
        index=False
    )
)

print()


# =====================================================================
# Frozen feature representation validation
# =====================================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in data.columns
]


extra_preprocessing_features = sorted(
    set(
        preprocessing_df[
            "feature"
        ].astype(str)
    )
    - set(
        feature_columns
    )
)


missing_preprocessing_features = sorted(
    set(
        feature_columns
    )
    - set(
        preprocessing_df[
            "feature"
        ].astype(str)
    )
)


feature_validation = {
    "feature_list_count": (
        len(feature_columns)
    ),

    "unique_feature_list_count": (
        len(
            set(
                feature_columns
            )
        )
    ),

    "binary_feature_count": (
        len(binary_features)
    ),

    "continuous_feature_count": (
        len(continuous_features)
    ),

    "features_missing_from_dataset": (
        len(missing_features)
    ),

    "features_missing_from_preprocessing_spec": (
        len(
            missing_preprocessing_features
        )
    ),

    "extra_preprocessing_features": (
        len(
            extra_preprocessing_features
        )
    ),
}


print("=" * 80)
print("7. FROZEN FEATURE REPRESENTATION")
print("=" * 80)

print(
    "Features in frozen list:       "
    f"{len(feature_columns)}"
)

print(
    "Unique frozen features:        "
    f"{len(set(feature_columns))}"
)

print(
    "Binary features:               "
    f"{len(binary_features)}"
)

print(
    "Continuous features:           "
    f"{len(continuous_features)}"
)

print(
    "Missing from dataset:          "
    f"{len(missing_features)}"
)

print(
    "Missing from preprocessing:    "
    f"{len(missing_preprocessing_features)}"
)

print(
    "Extra preprocessing features:  "
    f"{len(extra_preprocessing_features)}"
)

print()


# =====================================================================
# Missing and non-finite feature audit
# =====================================================================

feature_audit_rows = []


print("=" * 80)
print("8. FEATURE QUALITY AUDIT")
print("=" * 80)

print(
    "Checking missing and non-finite "
    "values across the 351 frozen features..."
)


for feature in feature_columns:

    if feature not in data.columns:

        feature_audit_rows.append(
            {
                "feature": feature,
                "present": False,
                "feature_type": None,
                "missing_count": np.nan,
                "infinite_count": np.nan,
                "minimum": np.nan,
                "maximum": np.nan,
            }
        )

        continue

    series = pd.to_numeric(
        data[feature],
        errors="coerce",
    )

    values = series.to_numpy(
        dtype=np.float64,
        copy=False,
    )

    missing_count = int(
        np.isnan(
            values
        ).sum()
    )

    infinite_count = int(
        np.isinf(
            values
        ).sum()
    )

    finite_values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        finite_values
    ) > 0:

        minimum = float(
            finite_values.min()
        )

        maximum = float(
            finite_values.max()
        )

    else:

        minimum = np.nan
        maximum = np.nan


    if feature in binary_features:

        feature_type = "binary"

    elif feature in continuous_features:

        feature_type = "continuous"

    else:

        feature_type = "unknown"


    feature_audit_rows.append(
        {
            "feature": feature,
            "present": True,
            "feature_type": feature_type,
            "missing_count": missing_count,
            "infinite_count": infinite_count,
            "minimum": minimum,
            "maximum": maximum,
        }
    )


feature_audit = pd.DataFrame(
    feature_audit_rows
)


total_missing = int(
    feature_audit[
        "missing_count"
    ]
    .fillna(0)
    .sum()
)

total_infinite = int(
    feature_audit[
        "infinite_count"
    ]
    .fillna(0)
    .sum()
)


binary_invalid = []

for feature in binary_features:

    if feature not in data.columns:
        continue

    values = (
        data[feature]
        .dropna()
        .unique()
    )

    invalid_values = [
        value
        for value in values
        if value not in {
            0,
            0.0,
            1,
            1.0,
        }
    ]

    if invalid_values:

        binary_invalid.append(
            {
                "feature": feature,
                "invalid_values": (
                    str(
                        invalid_values[
                            :10
                        ]
                    )
                ),
            }
        )


binary_invalid_df = pd.DataFrame(
    binary_invalid
)


print(
    f"Total missing feature values:  "
    f"{total_missing:,}"
)

print(
    f"Total infinite feature values: "
    f"{total_infinite:,}"
)

print(
    "Binary features with non-0/1 "
    f"values: {len(binary_invalid_df)}"
)

print()


# =====================================================================
# Patient-level audit manifest
# =====================================================================

# Preserve one row per patient for the NEXT stage.
#
# No hospital assignment is made here.

patient_audit_manifest = (
    patient_label_counts
    .merge(
        patient_stay_counts,
        on="subject_id",
        how="left",
        suffixes=(
            "",
            "_check",
        ),
    )
)


if (
    "n_stays_check"
    in patient_audit_manifest.columns
):

    assert (
        patient_audit_manifest[
            "n_stays"
        ]
        == patient_audit_manifest[
            "n_stays_check"
        ]
    ).all()

    patient_audit_manifest = (
        patient_audit_manifest.drop(
            columns=[
                "n_stays_check"
            ]
        )
    )


# =====================================================================
# Final validation checks
# =====================================================================

checks = {
    "expected_351_features": (
        len(
            feature_columns
        )
        == EXPECTED_FEATURES
    ),

    "expected_193_binary": (
        len(
            binary_features
        )
        == EXPECTED_BINARY_FEATURES
    ),

    "expected_158_continuous": (
        len(
            continuous_features
        )
        == EXPECTED_CONTINUOUS_FEATURES
    ),

    "all_features_present": (
        len(
            missing_features
        )
        == 0
    ),

    "all_features_have_preprocessing_type": (
        len(
            missing_preprocessing_features
        )
        == 0
    ),

    "no_extra_preprocessing_features": (
        len(
            extra_preprocessing_features
        )
        == 0
    ),

    "no_duplicate_stay_ids": (
        duplicate_stay_ids
        == 0
    ),

    "no_duplicate_subject_stay_pairs": (
        duplicate_subject_stay_pairs
        == 0
    ),

    "no_missing_feature_values": (
        total_missing
        == 0
    ),

    "no_infinite_feature_values": (
        total_infinite
        == 0
    ),

    "binary_features_are_0_or_1": (
        len(
            binary_invalid_df
        )
        == 0
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


print("=" * 80)
print("9. AUDIT VALIDATION")
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

dataset_summary.to_csv(
    RESULTS_DIR
    / "dataset_summary.csv",
    index=False,
)

stay_frequency.to_csv(
    RESULTS_DIR
    / "patient_stay_frequency.csv",
    index=False,
)

patient_stay_counts.to_csv(
    RESULTS_DIR
    / "patient_stay_counts.csv",
    index=False,
)

label_distribution.to_csv(
    RESULTS_DIR
    / "stay_label_distribution.csv",
    index=False,
)

patient_profile_distribution.to_csv(
    RESULTS_DIR
    / "patient_los_profile_distribution.csv",
    index=False,
)

patient_audit_manifest.to_csv(
    RESULTS_DIR
    / "patient_audit_manifest.csv",
    index=False,
)

race_distribution.to_csv(
    RESULTS_DIR
    / "race_distribution.csv",
    index=False,
)

insurance_distribution.to_csv(
    RESULTS_DIR
    / "insurance_distribution.csv",
    index=False,
)

feature_audit.to_csv(
    RESULTS_DIR
    / "feature_quality_audit.csv",
    index=False,
)

validation_df.to_csv(
    RESULTS_DIR
    / "audit_validation.csv",
    index=False,
)

if len(
    binary_invalid_df
) > 0:

    binary_invalid_df.to_csv(
        RESULTS_DIR
        / "binary_feature_invalid_values.csv",
        index=False,
    )


# =====================================================================
# Completion
# =====================================================================

print("=" * 80)
print("FEDERATED-LEARNING DATA AUDIT COMPLETE")
print("=" * 80)

print()
print(
    f"Results directory:\n  {RESULTS_DIR}"
)

print()

print(
    "All validation checks passed:"
    f" {validation_df['pass'].all()}"
)

print()

print(
    "IMPORTANT: no hospital assignment "
    "or train/test split has been created."
)