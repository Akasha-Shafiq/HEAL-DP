"""
Materialize the frozen federated-learning hospital partitions.

This script reads:

    1. the frozen source modelling dataset;
    2. the frozen patient hospital/train-test manifest.

It creates exactly ten local partition files:

    H1_train / H1_test
    H2_train / H2_test
    H3_train / H3_test
    H4_train / H4_test
    H5_train / H5_test

No new splitting is performed here.

The existing patient assignment is treated as immutable.

Global train/test datasets are NOT duplicated on disk.
They are defined as the union of the corresponding five local files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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

MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "manifests"
)

PARTITION_DIR = (
    PROJECT_ROOT
    / "data"
    / "federated_learning"
    / "partitions"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "federated_learning"
    / "partitions"
)

PARTITION_DIR.mkdir(
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

SPLIT_MANIFEST = (
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

SPLITS = [
    "train",
    "test",
]


# =====================================================================
# Helper
# =====================================================================

def sha256_file(path: Path) -> str:
    """
    Calculate SHA-256 without loading the whole file into memory.
    """

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


# =====================================================================
# Validate input
# =====================================================================

for path in [
    DATA_FILE,
    SPLIT_MANIFEST,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found: {path}"
        )


print("=" * 80)
print("FEDERATED-LEARNING PARTITION MATERIALIZATION")
print("=" * 80)

print()
print(f"Source dataset:\n  {DATA_FILE}")
print()
print(f"Frozen split manifest:\n  {SPLIT_MANIFEST}")
print()


# =====================================================================
# Load frozen inputs
# =====================================================================

print("Loading source dataset...")

data = pd.read_csv(
    DATA_FILE,
    low_memory=False,
)

print(
    f"Source rows:     {len(data):,}"
)

print(
    f"Source patients: "
    f"{data['subject_id'].nunique():,}"
)

print(
    f"Source stays:    "
    f"{data['stay_id'].nunique():,}"
)

print()


split_manifest = pd.read_csv(
    SPLIT_MANIFEST
)


required_manifest_columns = [
    "subject_id",
    "hospital",
    "split",
]


missing_manifest_columns = [
    column
    for column in required_manifest_columns
    if column not in split_manifest.columns
]


if missing_manifest_columns:

    raise ValueError(
        "Split manifest is missing columns: "
        f"{missing_manifest_columns}"
    )


if split_manifest[
    "subject_id"
].duplicated().any():

    raise ValueError(
        "The frozen split manifest contains "
        "duplicate subject_id values."
    )


if set(
    split_manifest[
        "hospital"
    ].unique()
) != set(
    HOSPITALS
):

    raise ValueError(
        "Unexpected hospital labels."
    )


if set(
    split_manifest[
        "split"
    ].unique()
) != set(
    SPLITS
):

    raise ValueError(
        "Unexpected split labels."
    )


# =====================================================================
# Attach hospital and split to every source row
# =====================================================================

assignment_columns = (
    split_manifest[
        [
            "subject_id",
            "hospital",
            "split",
        ]
    ]
)


partitioned_data = data.merge(
    assignment_columns,
    on="subject_id",
    how="left",
    validate="many_to_one",
)


missing_assignments = int(
    partitioned_data[
        "hospital"
    ]
    .isna()
    .sum()
)


if missing_assignments != 0:

    raise RuntimeError(
        f"{missing_assignments:,} source rows "
        "have no hospital/split assignment."
    )


# Put traceability columns first.

original_columns = list(
    data.columns
)

partitioned_data = partitioned_data[
    [
        "hospital",
        "split",
        *original_columns,
    ]
]


# =====================================================================
# Pre-materialization validation
# =====================================================================

source_patients = set(
    data[
        "subject_id"
    ].unique()
)

partition_patients = set(
    partitioned_data[
        "subject_id"
    ].unique()
)


source_stays = set(
    data[
        "stay_id"
    ].unique()
)

partition_stays = set(
    partitioned_data[
        "stay_id"
    ].unique()
)


if source_patients != partition_patients:

    raise RuntimeError(
        "Patient set changed during manifest merge."
    )


if source_stays != partition_stays:

    raise RuntimeError(
        "Stay set changed during manifest merge."
    )


if len(
    partitioned_data
) != len(
    data
):

    raise RuntimeError(
        "Row count changed during manifest merge."
    )


# =====================================================================
# Materialize ten local partitions
# =====================================================================

summary_rows = []
file_manifest_rows = []


for hospital in HOSPITALS:

    for split in SPLITS:

        local = (
            partitioned_data[
                (
                    partitioned_data[
                        "hospital"
                    ]
                    == hospital
                )
                &
                (
                    partitioned_data[
                        "split"
                    ]
                    == split
                )
            ]
            .copy()
        )


        if len(local) == 0:

            raise RuntimeError(
                f"Empty partition: "
                f"{hospital}_{split}"
            )


        output_file = (
            PARTITION_DIR
            / f"{hospital}_{split}.csv.gz"
        )


        print(
            f"Writing "
            f"{hospital}_{split}: "
            f"{len(local):,} rows, "
            f"{local['subject_id'].nunique():,} patients"
        )


        local.to_csv(
            output_file,
            index=False,
            compression={
                "method": "gzip",
                "compresslevel": 6,
                "mtime": 0,
            },
        )


        summary_rows.append(
            {
                "hospital": hospital,
                "split": split,
                "patients": int(
                    local[
                        "subject_id"
                    ].nunique()
                ),
                "rows": int(
                    len(local)
                ),
                "stays": int(
                    local[
                        "stay_id"
                    ].nunique()
                ),
                "long_stays": int(
                    (
                        local[
                            "los"
                        ]
                        > 4.0
                    ).sum()
                ),
                "short_stays": int(
                    (
                        local[
                            "los"
                        ]
                        <= 4.0
                    ).sum()
                ),
            }
        )


        file_manifest_rows.append(
            {
                "hospital": hospital,
                "split": split,
                "file_name": (
                    output_file.name
                ),
                "size_bytes": (
                    output_file.stat()
                    .st_size
                ),
                "sha256": (
                    sha256_file(
                        output_file
                    )
                ),
            }
        )


# =====================================================================
# Summaries
# =====================================================================

partition_summary = pd.DataFrame(
    summary_rows
)


partition_summary[
    "long_stay_fraction"
] = (
    partition_summary[
        "long_stays"
    ]
    / partition_summary[
        "stays"
    ]
)


file_manifest = pd.DataFrame(
    file_manifest_rows
)


# =====================================================================
# Post-materialization validation
# =====================================================================

materialized_rows = int(
    partition_summary[
        "rows"
    ].sum()
)


materialized_stays = int(
    partition_summary[
        "stays"
    ].sum()
)


materialized_patients_by_split = (
    partition_summary.groupby(
        "split"
    )[
        "patients"
    ]
    .sum()
)


expected_train_patients = int(
    (
        split_manifest[
            "split"
        ]
        == "train"
    ).sum()
)


expected_test_patients = int(
    (
        split_manifest[
            "split"
        ]
        == "test"
    ).sum()
)


# Explicitly check patient overlap between
# every physical partition.

patient_sets = {}


for hospital in HOSPITALS:

    for split in SPLITS:

        key = (
            hospital,
            split,
        )

        patient_sets[
            key
        ] = set(
            split_manifest.loc[
                (
                    split_manifest[
                        "hospital"
                    ]
                    == hospital
                )
                &
                (
                    split_manifest[
                        "split"
                    ]
                    == split
                ),
                "subject_id",
            ]
        )


overlap_pairs = []


keys = list(
    patient_sets.keys()
)


for i in range(
    len(keys)
):

    for j in range(
        i + 1,
        len(keys),
    ):

        overlap = (
            patient_sets[
                keys[i]
            ]
            &
            patient_sets[
                keys[j]
            ]
        )

        if overlap:

            overlap_pairs.append(
                {
                    "partition_1":
                        f"{keys[i][0]}_{keys[i][1]}",
                    "partition_2":
                        f"{keys[j][0]}_{keys[j][1]}",
                    "overlap_patients":
                        len(overlap),
                }
            )


checks = {
    "source_row_count_preserved": (
        materialized_rows
        == len(data)
    ),

    "source_stay_count_preserved": (
        materialized_stays
        == data[
            "stay_id"
        ].nunique()
    ),

    "train_patient_count_matches_manifest": (
        int(
            materialized_patients_by_split[
                "train"
            ]
        )
        == expected_train_patients
    ),

    "test_patient_count_matches_manifest": (
        int(
            materialized_patients_by_split[
                "test"
            ]
        )
        == expected_test_patients
    ),

    "ten_partition_files_created": (
        len(
            file_manifest
        )
        == 10
    ),

    "zero_patient_overlap_between_partitions": (
        len(
            overlap_pairs
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


# =====================================================================
# Save audit outputs
# =====================================================================

partition_summary.to_csv(
    RESULTS_DIR
    / "materialized_partition_summary.csv",
    index=False,
)


file_manifest.to_csv(
    RESULTS_DIR
    / "materialized_partition_file_manifest.csv",
    index=False,
)


validation_df.to_csv(
    RESULTS_DIR
    / "materialized_partition_validation.csv",
    index=False,
)


if overlap_pairs:

    pd.DataFrame(
        overlap_pairs
    ).to_csv(
        RESULTS_DIR
        / "partition_patient_overlap_errors.csv",
        index=False,
    )


# =====================================================================
# Print final results
# =====================================================================

print()
print("=" * 80)
print("PARTITION SUMMARY")
print("=" * 80)

print(
    partition_summary.round(
        5
    ).to_string(
        index=False
    )
)

print()

print("=" * 80)
print("MATERIALIZATION VALIDATION")
print("=" * 80)

print(
    validation_df.to_string(
        index=False
    )
)

print()

print("=" * 80)
print("PARTITION MATERIALIZATION COMPLETE")
print("=" * 80)

print()

print(
    f"Partition directory:\n  "
    f"{PARTITION_DIR}"
)

print()

print(
    "Physical partition files:"
    f" {len(file_manifest)}"
)

print(
    "Total rows materialized:"
    f" {materialized_rows:,}"
)

print(
    "Total stays materialized:"
    f" {materialized_stays:,}"
)

print(
    "Train patients:"
    f" {expected_train_patients:,}"
)

print(
    "Test patients:"
    f"  {expected_test_patients:,}"
)

print()

print(
    "All validation checks passed:"
    f" {validation_df['pass'].all()}"
)

print()

print(
    "No new patient assignment or "
    "train/test split was generated."
)