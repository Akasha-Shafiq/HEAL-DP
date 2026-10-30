"""
Strategy B candidate audit:
retain multiple ICU stays per patient subject to a deterministic
contribution bound K.

Primary candidate:
    K = 5

Sensitivity candidates:
    K = 3
    K = 10

Only the TRAINING population is contribution-bounded.
The frozen held-out test population remains unchanged.

Selection rule
--------------
For each training patient, retain the chronologically earliest
K ICU stays, ordered by:

    intime
    stay_id (deterministic tie breaker)

No model is trained and no differential privacy is applied here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from federated_learning.config import (
    SOURCE_DATA_FILE,
    TRAIN_TEST_ASSIGNMENT_FILE,
    COMPARISON_RESULTS_DIR,
)


# =====================================================================
# Configuration
# =====================================================================

K_VALUES = [
    3,
    5,
    10,
]

PRIMARY_K = 5


OUTPUT_DIR = (
    COMPARISON_RESULTS_DIR
    / "patient_contribution"
    / "strategy_b"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Load source data and frozen patient assignment
# =====================================================================

print("=" * 80)
print("STRATEGY B — BOUNDED MULTI-STAY TRAINING AUDIT")
print("=" * 80)


data = pd.read_csv(
    SOURCE_DATA_FILE,
    low_memory=False,
)


manifest = pd.read_csv(
    TRAIN_TEST_ASSIGNMENT_FILE
)


required_manifest_columns = [
    "subject_id",
    "hospital",
    "split",
    "patient_los_profile",
]


missing = [
    column
    for column in required_manifest_columns
    if column not in manifest.columns
]


if missing:

    raise ValueError(
        f"Missing manifest columns: {missing}"
    )


if manifest[
    "subject_id"
].duplicated().any():

    raise RuntimeError(
        "Patient manifest contains duplicate subject_id."
    )


# =====================================================================
# Attach frozen hospital/split assignment
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
        "Some source rows have no frozen hospital assignment."
    )


# =====================================================================
# Training population only
# =====================================================================

train = (
    working.loc[
        working[
            "split"
        ]
        == "train"
    ]
    .copy()
)


assert train[
    "subject_id"
].nunique() == 43450


assert len(
    train
) == 59522


print()
print("Frozen training population")
print("--------------------------")

print(
    f"Training patients: "
    f"{train['subject_id'].nunique():,}"
)

print(
    f"Training stays:    "
    f"{len(train):,}"
)


# =====================================================================
# Parse temporal field
# =====================================================================

train[
    "audit_intime"
] = pd.to_datetime(
    train[
        "intime"
    ],
    errors="coerce",
)


missing_intime = int(
    train[
        "audit_intime"
    ]
    .isna()
    .sum()
)


print(
    f"Missing/unparseable intime: "
    f"{missing_intime:,}"
)


if missing_intime:

    raise RuntimeError(
        "Cannot define bounded chronological contribution."
    )


# =====================================================================
# Label
# =====================================================================

train[
    "long_label"
] = (
    train[
        "los"
    ]
    > 4.0
).astype(
    int
)


# =====================================================================
# Deterministic contribution order
# =====================================================================

ordered = (
    train.sort_values(
        [
            "subject_id",
            "audit_intime",
            "stay_id",
        ]
    )
    .copy()
)


ordered[
    "patient_contribution_rank"
] = (
    ordered.groupby(
        "subject_id"
    )
    .cumcount()
    + 1
)


original_rows = int(
    len(
        ordered
    )
)


original_patients = int(
    ordered[
        "subject_id"
    ].nunique()
)


original_long = int(
    ordered[
        "long_label"
    ].sum()
)


original_short = (
    original_rows
    - original_long
)


original_long_fraction = (
    original_long
    / original_rows
)


# =====================================================================
# Global K audit
# =====================================================================

global_rows = []


for K in K_VALUES:

    retained = (
        ordered.loc[
            ordered[
                "patient_contribution_rank"
            ]
            <= K
        ]
        .copy()
    )


    removed = (
        ordered.loc[
            ordered[
                "patient_contribution_rank"
            ]
            > K
        ]
        .copy()
    )


    patient_counts = (
        ordered.groupby(
            "subject_id"
        )
        .size()
    )


    patients_affected = int(
        (
            patient_counts
            > K
        ).sum()
    )


    retained_long = int(
        retained[
            "long_label"
        ].sum()
    )


    retained_short = int(
        len(
            retained
        )
        - retained_long
    )


    removed_long = int(
        removed[
            "long_label"
        ].sum()
    )


    removed_short = int(
        len(
            removed
        )
        - removed_long
    )


    global_rows.append(
        {
            "K":
                K,

            "patients":
                original_patients,

            "patients_affected":
                patients_affected,

            "fraction_patients_affected":
                patients_affected
                / original_patients,

            "original_stays":
                original_rows,

            "retained_stays":
                len(
                    retained
                ),

            "removed_stays":
                len(
                    removed
                ),

            "fraction_stays_retained":
                len(
                    retained
                )
                / original_rows,

            "original_long_fraction":
                original_long_fraction,

            "retained_long_fraction":
                retained_long
                / len(
                    retained
                ),

            "long_fraction_change":
                (
                    retained_long
                    / len(
                        retained
                    )
                )
                - original_long_fraction,

            "removed_long":
                removed_long,

            "removed_short":
                removed_short,

            "removed_long_fraction":
                (
                    removed_long
                    / len(
                        removed
                    )
                    if len(
                        removed
                    ) > 0
                    else np.nan
                ),
        }
    )


global_audit = pd.DataFrame(
    global_rows
)


global_audit.to_csv(
    OUTPUT_DIR
    / "strategy_b_global_k_audit.csv",
    index=False,
)


# =====================================================================
# Hospital-specific effect
# =====================================================================

hospital_rows = []


for hospital, hospital_data in (
    ordered.groupby(
        "hospital",
        observed=True,
    )
):

    hospital_patient_counts = (
        hospital_data.groupby(
            "subject_id"
        )
        .size()
    )


    for K in K_VALUES:

        retained = hospital_data.loc[
            hospital_data[
                "patient_contribution_rank"
            ]
            <= K
        ]


        removed = hospital_data.loc[
            hospital_data[
                "patient_contribution_rank"
            ]
            > K
        ]


        retained_long = int(
            retained[
                "long_label"
            ].sum()
        )


        original_long_fraction_h = float(
            hospital_data[
                "long_label"
            ].mean()
        )


        retained_long_fraction_h = float(
            retained[
                "long_label"
            ].mean()
        )


        hospital_rows.append(
            {
                "hospital":
                    hospital,

                "K":
                    K,

                "patients":
                    hospital_data[
                        "subject_id"
                    ].nunique(),

                "patients_affected":
                    int(
                        (
                            hospital_patient_counts
                            > K
                        ).sum()
                    ),

                "original_stays":
                    len(
                        hospital_data
                    ),

                "retained_stays":
                    len(
                        retained
                    ),

                "removed_stays":
                    len(
                        removed
                    ),

                "fraction_stays_retained":
                    len(
                        retained
                    )
                    / len(
                        hospital_data
                    ),

                "original_long_fraction":
                    original_long_fraction_h,

                "retained_long_fraction":
                    retained_long_fraction_h,

                "long_fraction_change":
                    retained_long_fraction_h
                    - original_long_fraction_h,

                "removed_long":
                    int(
                        removed[
                            "long_label"
                        ].sum()
                    ),

                "removed_short":
                    int(
                        len(
                            removed
                        )
                        - removed[
                            "long_label"
                        ].sum()
                    ),
            }
        )


hospital_audit = pd.DataFrame(
    hospital_rows
)


hospital_audit.to_csv(
    OUTPUT_DIR
    / "strategy_b_hospital_k_audit.csv",
    index=False,
)


# =====================================================================
# LOS-profile effect
# =====================================================================

profile_rows = []


for profile, profile_data in (
    ordered.groupby(
        "patient_los_profile",
        observed=True,
    )
):

    patient_counts = (
        profile_data.groupby(
            "subject_id"
        )
        .size()
    )


    for K in K_VALUES:

        retained = profile_data.loc[
            profile_data[
                "patient_contribution_rank"
            ]
            <= K
        ]


        removed = profile_data.loc[
            profile_data[
                "patient_contribution_rank"
            ]
            > K
        ]


        profile_rows.append(
            {
                "patient_los_profile":
                    profile,

                "K":
                    K,

                "patients":
                    profile_data[
                        "subject_id"
                    ].nunique(),

                "patients_affected":
                    int(
                        (
                            patient_counts
                            > K
                        ).sum()
                    ),

                "fraction_patients_affected":
                    float(
                        (
                            patient_counts
                            > K
                        ).mean()
                    ),

                "original_stays":
                    len(
                        profile_data
                    ),

                "retained_stays":
                    len(
                        retained
                    ),

                "removed_stays":
                    len(
                        removed
                    ),

                "fraction_stays_retained":
                    len(
                        retained
                    )
                    / len(
                        profile_data
                    ),

                "original_long_fraction":
                    float(
                        profile_data[
                            "long_label"
                        ].mean()
                    ),

                "retained_long_fraction":
                    float(
                        retained[
                            "long_label"
                        ].mean()
                    ),

                "removed_long":
                    int(
                        removed[
                            "long_label"
                        ].sum()
                    ),

                "removed_short":
                    int(
                        len(
                            removed
                        )
                        - removed[
                            "long_label"
                        ].sum()
                    ),
            }
        )


profile_audit = pd.DataFrame(
    profile_rows
)


profile_audit.to_csv(
    OUTPUT_DIR
    / "strategy_b_los_profile_k_audit.csv",
    index=False,
)


# =====================================================================
# K=5 candidate training manifest
# =====================================================================

primary = (
    ordered.loc[
        ordered[
            "patient_contribution_rank"
        ]
        <= PRIMARY_K,
        [
            "subject_id",
            "hospital",
            "stay_id",
            "intime",
            "los",
            "long_label",
            "patient_los_profile",
            "patient_contribution_rank",
        ],
    ]
    .copy()
)


# =====================================================================
# Validate candidate
# =====================================================================

candidate_patient_counts = (
    primary.groupby(
        "subject_id"
    )
    .size()
)


validation_checks = {
    "all_43450_training_patients_preserved":
        primary[
            "subject_id"
        ].nunique()
        == 43450,

    "max_5_stays_per_patient":
        candidate_patient_counts.max()
        <= 5,

    "minimum_1_stay_per_patient":
        candidate_patient_counts.min()
        >= 1,

    "all_rows_are_training_rows":
        set(
            primary[
                "subject_id"
            ]
        ).issubset(
            set(
                manifest.loc[
                    manifest[
                        "split"
                    ]
                    == "train",
                    "subject_id",
                ]
            )
        ),
}


validation = pd.DataFrame(
    [
        {
            "check":
                name,

            "pass":
                bool(
                    result
                ),
        }
        for name, result
        in validation_checks.items()
    ]
)


primary.to_csv(
    OUTPUT_DIR
    / "strategy_b_candidate_k5_training_stays.csv",
    index=False,
)


validation.to_csv(
    OUTPUT_DIR
    / "strategy_b_candidate_k5_validation.csv",
    index=False,
)


# =====================================================================
# Console output
# =====================================================================

print()
print("=" * 80)
print("GLOBAL STRATEGY B EFFECT")
print("=" * 80)

print(
    global_audit[
        [
            "K",
            "patients_affected",
            "fraction_patients_affected",
            "retained_stays",
            "removed_stays",
            "fraction_stays_retained",
            "original_long_fraction",
            "retained_long_fraction",
            "long_fraction_change",
            "removed_long",
            "removed_short",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("HOSPITAL-SPECIFIC EFFECT")
print("=" * 80)

print(
    hospital_audit[
        [
            "hospital",
            "K",
            "patients_affected",
            "retained_stays",
            "removed_stays",
            "fraction_stays_retained",
            "original_long_fraction",
            "retained_long_fraction",
            "long_fraction_change",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("LOS-PROFILE-SPECIFIC EFFECT")
print("=" * 80)

print(
    profile_audit[
        [
            "patient_los_profile",
            "K",
            "patients_affected",
            "fraction_patients_affected",
            "retained_stays",
            "removed_stays",
            "fraction_stays_retained",
            "original_long_fraction",
            "retained_long_fraction",
            "removed_long",
            "removed_short",
        ]
    ]
    .round(
        6
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("PRIMARY K=5 VALIDATION")
print("=" * 80)

print(
    validation.to_string(
        index=False
    )
)


print()
print()
print(
    "Primary Strategy B candidate:"
)

print(
    f"  K = {PRIMARY_K}"
)

print(
    "  Selection = earliest K training ICU stays per patient"
)

print()

print(
    "Candidate manifest saved to:"
)

print(
    "  "
    + str(
        OUTPUT_DIR
        / "strategy_b_candidate_k5_training_stays.csv"
    )
)


print()
print(
    "No model was trained and no DP was applied."
)