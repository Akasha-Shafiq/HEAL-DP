"""
Freeze and materialize the accepted patient-contribution strategies.

Strategy A
----------
Exactly one training stay per patient:
the chronologically earliest ICU stay.

Strategy B
----------
Up to K=5 training stays per patient:
the chronologically earliest five ICU stays.

IMPORTANT
---------
Only TRAINING data are changed.

The original frozen H1-H5 test partitions remain untouched and are
used for evaluating F1, Strategy A, Strategy B, and future DP models.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd

from federated_learning.config import (
    PROJECT_ROOT,
    TRAIN_FILES,
    HOSPITALS,
)


# =====================================================================
# Paths
# =====================================================================

COMPARISON_DIR = (
    PROJECT_ROOT
    / "results"
    / "federated_learning"
    / "comparison"
    / "patient_contribution"
)

A_CANDIDATE = (
    COMPARISON_DIR
    / "strategy_a"
    / "strategy_a_candidate_index_stays.csv"
)

B_CANDIDATE = (
    COMPARISON_DIR
    / "strategy_b"
    / "strategy_b_candidate_k5_training_stays.csv"
)


MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "manifests"
    / "patient_contribution"
)

PARTITION_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "partitions"
    / "patient_contribution"
)

A_PARTITION_DIR = (
    PARTITION_DIR
    / "strategy_a"
)

B_PARTITION_DIR = (
    PARTITION_DIR
    / "strategy_b_k5"
)


for directory in [
    MANIFEST_DIR,
    A_PARTITION_DIR,
    B_PARTITION_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# Load accepted candidate manifests
# =====================================================================

a = pd.read_csv(
    A_CANDIDATE
)

b = pd.read_csv(
    B_CANDIDATE
)


# Strategy A audit included train and test index stays.
# Only TRAIN rows become part of the training representation.

a_train = (
    a.loc[
        a["split"] == "train"
    ]
    .copy()
)


# =====================================================================
# Canonical manifest validation
# =====================================================================

assert len(a_train) == 43450

assert (
    a_train[
        "subject_id"
    ].nunique()
    == 43450
)

assert (
    a_train[
        "index_stay_id"
    ].nunique()
    == 43450
)


assert (
    b[
        "subject_id"
    ].nunique()
    == 43450
)

assert len(b) == 58228

assert (
    b.groupby(
        "subject_id"
    )
    .size()
    .max()
    <= 5
)

assert (
    b.groupby(
        "subject_id"
    )
    .size()
    .min()
    >= 1
)


# =====================================================================
# Write canonical manifests
# =====================================================================

A_MANIFEST = (
    MANIFEST_DIR
    / "strategy_a_earliest_stay_training_manifest.csv"
)

B_MANIFEST = (
    MANIFEST_DIR
    / "strategy_b_k5_training_manifest.csv"
)


a_manifest = (
    a_train[
        [
            "subject_id",
            "hospital",
            "index_stay_id",
            "index_intime",
            "index_los",
            "long_label",
            "patient_los_profile",
        ]
    ]
    .rename(
        columns={
            "index_stay_id":
                "stay_id",

            "index_intime":
                "intime",

            "index_los":
                "los",
        }
    )
)


b_manifest = (
    b[
        [
            "subject_id",
            "hospital",
            "stay_id",
            "intime",
            "los",
            "long_label",
            "patient_los_profile",
            "patient_contribution_rank",
        ]
    ]
    .copy()
)


a_manifest.to_csv(
    A_MANIFEST,
    index=False,
)

b_manifest.to_csv(
    B_MANIFEST,
    index=False,
)


# =====================================================================
# Materialize hospital-specific training partitions
# =====================================================================

audit_rows = []


for hospital in HOSPITALS:

    original = pd.read_csv(
        TRAIN_FILES[
            hospital
        ],
        low_memory=False,
    )


    # -------------------------------------------------------------
    # Strategy A
    # -------------------------------------------------------------

    a_ids = set(
        a_manifest.loc[
            a_manifest[
                "hospital"
            ]
            == hospital,
            "stay_id",
        ]
    )


    a_partition = (
        original.loc[
            original[
                "stay_id"
            ].isin(
                a_ids
            )
        ]
        .copy()
    )


    expected_a_patients = int(
        a_manifest.loc[
            a_manifest[
                "hospital"
            ]
            == hospital,
            "subject_id",
        ]
        .nunique()
    )


    assert len(
        a_partition
    ) == expected_a_patients

    assert (
        a_partition[
            "subject_id"
        ].nunique()
        == expected_a_patients
    )

    assert not (
        a_partition[
            "subject_id"
        ].duplicated()
    ).any()


    a_path = (
        A_PARTITION_DIR
        / f"{hospital}_train.csv.gz"
    )


    a_partition.to_csv(
        a_path,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )


    # -------------------------------------------------------------
    # Strategy B
    # -------------------------------------------------------------

    b_ids = set(
        b_manifest.loc[
            b_manifest[
                "hospital"
            ]
            == hospital,
            "stay_id",
        ]
    )


    b_partition = (
        original.loc[
            original[
                "stay_id"
            ].isin(
                b_ids
            )
        ]
        .copy()
    )


    expected_b_rows = int(
        (
            b_manifest[
                "hospital"
            ]
            == hospital
        ).sum()
    )


    assert len(
        b_partition
    ) == expected_b_rows


    b_counts = (
        b_partition.groupby(
            "subject_id"
        )
        .size()
    )


    assert b_counts.max() <= 5

    assert b_counts.min() >= 1


    b_path = (
        B_PARTITION_DIR
        / f"{hospital}_train.csv.gz"
    )


    b_partition.to_csv(
        b_path,
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )


    audit_rows.append(
        {
            "hospital":
                hospital,

            "A_rows":
                len(
                    a_partition
                ),

            "A_patients":
                a_partition[
                    "subject_id"
                ].nunique(),

            "B_rows":
                len(
                    b_partition
                ),

            "B_patients":
                b_partition[
                    "subject_id"
                ].nunique(),

            "B_max_stays_per_patient":
                int(
                    b_counts.max()
                ),
        }
    )


audit = pd.DataFrame(
    audit_rows
)


# =====================================================================
# Global validation
# =====================================================================

assert audit[
    "A_rows"
].sum() == 43450

assert audit[
    "A_patients"
].sum() == 43450

assert audit[
    "B_rows"
].sum() == 58228

assert audit[
    "B_patients"
].sum() == 43450


audit.to_csv(
    MANIFEST_DIR
    / "patient_strategy_partition_audit.csv",
    index=False,
)


# =====================================================================
# Hash canonical manifests
# =====================================================================

def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest().upper()


hash_rows = []


for name, path in [
    (
        "strategy_a",
        A_MANIFEST,
    ),
    (
        "strategy_b_k5",
        B_MANIFEST,
    ),
]:

    hash_rows.append(
        {
            "strategy":
                name,

            "file_name":
                path.name,

            "size_bytes":
                path.stat().st_size,

            "sha256":
                sha256_file(
                    path
                ),
        }
    )


hash_manifest = pd.DataFrame(
    hash_rows
)


hash_manifest.to_csv(
    MANIFEST_DIR
    / "patient_strategy_manifest_hashes.csv",
    index=False,
)


# =====================================================================
# Console output
# =====================================================================

print("=" * 80)
print("PATIENT-STRATEGY MATERIALIZATION")
print("=" * 80)

print()

print(
    audit.to_string(
        index=False
    )
)

print()

print(
    "Strategy A global rows/patients:"
)

print(
    f"  {audit['A_rows'].sum():,} / "
    f"{audit['A_patients'].sum():,}"
)

print()

print(
    "Strategy B K=5 global rows/patients:"
)

print(
    f"  {audit['B_rows'].sum():,} / "
    f"{audit['B_patients'].sum():,}"
)

print()

print(
    "Canonical manifest hashes"
)

print(
    "-------------------------"
)

print(
    hash_manifest.to_string(
        index=False
    )
)

print()

print(
    "Strategy A partitions:"
)

print(
    f"  {A_PARTITION_DIR}"
)

print()

print(
    "Strategy B partitions:"
)

print(
    f"  {B_PARTITION_DIR}"
)

print()

print(
    "Original H1-H5 test partitions were not modified."
)