"""
Patient-contribution audit for the future patient-level DP experiments.

This script DOES NOT apply DP and DOES NOT modify any dataset.

It quantifies the empirical effect of bounding each patient's number
of ICU-stay contributions at:

    K = 1, 2, 3, 5, 10

The audit is performed:

    1. globally;
    2. separately for each simulated hospital;
    3. separately for train/test populations;
    4. by patient LOS profile.

The goal is to inform the patient-level DP contribution design before
choosing a bound K.
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
# Configuration
# =====================================================================

K_VALUES = [
    1,
    2,
    3,
    5,
    10,
]


OUTPUT_DIR = (
    COMPARISON_RESULTS_DIR
    / "patient_contribution"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Load frozen source and patient manifest
# =====================================================================

print("=" * 80)
print("PATIENT-CONTRIBUTION AUDIT")
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
        f"Manifest missing columns: {missing}"
    )


if manifest[
    "subject_id"
].duplicated().any():

    raise ValueError(
        "Patient manifest contains duplicate subject_id."
    )


# =====================================================================
# Attach frozen hospital/split ownership
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
        "Some source rows do not have a frozen hospital assignment."
    )


# =====================================================================
# Per-patient stay counts
# =====================================================================

patient_counts = (
    working.groupby(
        [
            "subject_id",
            "hospital",
            "split",
            "patient_los_profile",
        ],
        observed=True,
    )
    .agg(
        n_stays=(
            "stay_id",
            "nunique",
        ),
        long_stays=(
            "los",
            lambda x: int(
                (
                    x > 4.0
                ).sum()
            ),
        ),
        short_stays=(
            "los",
            lambda x: int(
                (
                    x <= 4.0
                ).sum()
            ),
        ),
    )
    .reset_index()
)


n_patients = int(
    patient_counts[
        "subject_id"
    ].nunique()
)


n_stays = int(
    patient_counts[
        "n_stays"
    ].sum()
)


print()
print(
    f"Patients: {n_patients:,}"
)

print(
    f"Stays:    {n_stays:,}"
)

print(
    f"Maximum stays/patient: "
    f"{patient_counts['n_stays'].max()}"
)


# =====================================================================
# Exact stay-count distribution
# =====================================================================

stay_count_distribution = (
    patient_counts[
        "n_stays"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "n_stays"
    )
    .reset_index(
        name="patients"
    )
)


stay_count_distribution[
    "patient_fraction"
] = (
    stay_count_distribution[
        "patients"
    ]
    / n_patients
)


stay_count_distribution[
    "cumulative_patient_fraction"
] = (
    stay_count_distribution[
        "patient_fraction"
    ]
    .cumsum()
)


stay_count_distribution.to_csv(
    OUTPUT_DIR
    / "patient_stay_count_distribution.csv",
    index=False,
)


# =====================================================================
# Global K audit
# =====================================================================

global_rows = []


for K in K_VALUES:

    affected = (
        patient_counts[
            "n_stays"
        ]
        > K
    )


    patients_affected = int(
        affected.sum()
    )


    patients_unaffected = int(
        (
            ~affected
        ).sum()
    )


    stays_retained = int(
        np.minimum(
            patient_counts[
                "n_stays"
            ],
            K,
        ).sum()
    )


    stays_removed = (
        n_stays
        - stays_retained
    )


    global_rows.append(
        {
            "K":
                K,

            "patients_total":
                n_patients,

            "patients_unaffected":
                patients_unaffected,

            "patients_affected":
                patients_affected,

            "fraction_patients_unaffected":
                patients_unaffected
                / n_patients,

            "fraction_patients_affected":
                patients_affected
                / n_patients,

            "stays_original":
                n_stays,

            "stays_retained":
                stays_retained,

            "stays_removed":
                stays_removed,

            "fraction_stays_retained":
                stays_retained
                / n_stays,

            "fraction_stays_removed":
                stays_removed
                / n_stays,
        }
    )


global_k_audit = pd.DataFrame(
    global_rows
)


global_k_audit.to_csv(
    OUTPUT_DIR
    / "global_contribution_bound_audit.csv",
    index=False,
)


# =====================================================================
# Hospital-specific K audit
# =====================================================================

hospital_rows = []


for hospital, subset in (
    patient_counts.groupby(
        "hospital",
        observed=True,
    )
):

    hospital_patients = int(
        len(
            subset
        )
    )

    hospital_stays = int(
        subset[
            "n_stays"
        ].sum()
    )


    for K in K_VALUES:

        patients_affected = int(
            (
                subset[
                    "n_stays"
                ]
                > K
            ).sum()
        )


        stays_retained = int(
            np.minimum(
                subset[
                    "n_stays"
                ],
                K,
            ).sum()
        )


        hospital_rows.append(
            {
                "hospital":
                    hospital,

                "K":
                    K,

                "patients":
                    hospital_patients,

                "patients_affected":
                    patients_affected,

                "fraction_patients_affected":
                    patients_affected
                    / hospital_patients,

                "stays_original":
                    hospital_stays,

                "stays_retained":
                    stays_retained,

                "stays_removed":
                    hospital_stays
                    - stays_retained,

                "fraction_stays_retained":
                    stays_retained
                    / hospital_stays,
            }
        )


hospital_k_audit = pd.DataFrame(
    hospital_rows
)


hospital_k_audit.to_csv(
    OUTPUT_DIR
    / "hospital_contribution_bound_audit.csv",
    index=False,
)


# =====================================================================
# Train/test-specific K audit
# =====================================================================

split_rows = []


for split, subset in (
    patient_counts.groupby(
        "split",
        observed=True,
    )
):

    split_patients = int(
        len(
            subset
        )
    )

    split_stays = int(
        subset[
            "n_stays"
        ].sum()
    )


    for K in K_VALUES:

        affected = int(
            (
                subset[
                    "n_stays"
                ]
                > K
            ).sum()
        )


        retained = int(
            np.minimum(
                subset[
                    "n_stays"
                ],
                K,
            ).sum()
        )


        split_rows.append(
            {
                "split":
                    split,

                "K":
                    K,

                "patients":
                    split_patients,

                "patients_affected":
                    affected,

                "fraction_patients_affected":
                    affected
                    / split_patients,

                "stays_original":
                    split_stays,

                "stays_retained":
                    retained,

                "stays_removed":
                    split_stays
                    - retained,

                "fraction_stays_retained":
                    retained
                    / split_stays,
            }
        )


split_k_audit = pd.DataFrame(
    split_rows
)


split_k_audit.to_csv(
    OUTPUT_DIR
    / "split_contribution_bound_audit.csv",
    index=False,
)


# =====================================================================
# LOS-profile-specific audit
# =====================================================================

profile_rows = []


for profile, subset in (
    patient_counts.groupby(
        "patient_los_profile",
        observed=True,
    )
):

    profile_patients = int(
        len(
            subset
        )
    )

    profile_stays = int(
        subset[
            "n_stays"
        ].sum()
    )


    for K in K_VALUES:

        affected = int(
            (
                subset[
                    "n_stays"
                ]
                > K
            ).sum()
        )


        retained = int(
            np.minimum(
                subset[
                    "n_stays"
                ],
                K,
            ).sum()
        )


        profile_rows.append(
            {
                "patient_los_profile":
                    profile,

                "K":
                    K,

                "patients":
                    profile_patients,

                "patients_affected":
                    affected,

                "fraction_patients_affected":
                    affected
                    / profile_patients,

                "stays_original":
                    profile_stays,

                "stays_retained":
                    retained,

                "stays_removed":
                    profile_stays
                    - retained,

                "fraction_stays_retained":
                    retained
                    / profile_stays,
            }
        )


profile_k_audit = pd.DataFrame(
    profile_rows
)


profile_k_audit.to_csv(
    OUTPUT_DIR
    / "los_profile_contribution_bound_audit.csv",
    index=False,
)


# =====================================================================
# Quantiles
# =====================================================================

quantile_rows = []


for q in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    1.0,
]:

    quantile_rows.append(
        {
            "quantile":
                q,

            "n_stays":
                float(
                    patient_counts[
                        "n_stays"
                    ].quantile(
                        q
                    )
                ),
        }
    )


quantiles = pd.DataFrame(
    quantile_rows
)


quantiles.to_csv(
    OUTPUT_DIR
    / "patient_stay_count_quantiles.csv",
    index=False,
)


# =====================================================================
# Console output
# =====================================================================

print()
print("=" * 80)
print("GLOBAL CONTRIBUTION-BOUND EFFECT")
print("=" * 80)

print(
    global_k_audit[
        [
            "K",
            "patients_affected",
            "fraction_patients_affected",
            "stays_retained",
            "stays_removed",
            "fraction_stays_retained",
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
print("PATIENT STAY-COUNT QUANTILES")
print("=" * 80)

print(
    quantiles.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("HOSPITAL-SPECIFIC K EFFECT")
print("=" * 80)

print(
    hospital_k_audit[
        [
            "hospital",
            "K",
            "patients_affected",
            "fraction_patients_affected",
            "stays_removed",
            "fraction_stays_retained",
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
print("LOS-PROFILE-SPECIFIC K EFFECT")
print("=" * 80)

print(
    profile_k_audit[
        [
            "patient_los_profile",
            "K",
            "patients_affected",
            "fraction_patients_affected",
            "stays_removed",
            "fraction_stays_retained",
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
print(
    f"Results saved to:\n"
    f"  {OUTPUT_DIR}"
)

print()

print(
    "No patient representation was changed and no DP was applied."
)