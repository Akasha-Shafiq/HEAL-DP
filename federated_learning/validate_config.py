"""
Validate the frozen federated-learning configuration.

No data are modified.
"""

from __future__ import annotations

import pandas as pd

from federated_learning.config import (
    FEATURE_LIST_FILE,
    PREPROCESSING_SPEC_FILE,
    TRAIN_TEST_ASSIGNMENT_FILE,
    TRAIN_FILES,
    TEST_FILES,
    HOSPITALS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_BINARY_FEATURE_COUNT,
    EXPECTED_CONTINUOUS_FEATURE_COUNT,
)


print("=" * 80)
print("FEDERATED-LEARNING CONFIGURATION VALIDATION")
print("=" * 80)


# ---------------------------------------------------------------------
# Frozen features
# ---------------------------------------------------------------------

features = pd.read_csv(
    FEATURE_LIST_FILE
)

preprocessing = pd.read_csv(
    PREPROCESSING_SPEC_FILE
)


feature_columns = (
    features["feature"]
    .astype(str)
    .tolist()
)


binary_features = (
    preprocessing.loc[
        preprocessing["feature_type"]
        .astype(str)
        .str.lower()
        == "binary",
        "feature",
    ]
    .astype(str)
    .tolist()
)


continuous_features = (
    preprocessing.loc[
        preprocessing["feature_type"]
        .astype(str)
        .str.lower()
        == "continuous",
        "feature",
    ]
    .astype(str)
    .tolist()
)


# ---------------------------------------------------------------------
# Frozen patient manifest
# ---------------------------------------------------------------------

manifest = pd.read_csv(
    TRAIN_TEST_ASSIGNMENT_FILE
)


# ---------------------------------------------------------------------
# Physical partitions
# ---------------------------------------------------------------------

missing_partition_files = []


for hospital in HOSPITALS:

    for path in [
        TRAIN_FILES[hospital],
        TEST_FILES[hospital],
    ]:

        if not path.exists():

            missing_partition_files.append(
                str(path)
            )


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------

checks = {
    "351_features": (
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT
    ),

    "193_binary_features": (
        len(binary_features)
        == EXPECTED_BINARY_FEATURE_COUNT
    ),

    "158_continuous_features": (
        len(continuous_features)
        == EXPECTED_CONTINUOUS_FEATURE_COUNT
    ),

    "feature_sets_match": (
        set(feature_columns)
        == set(
            binary_features
            + continuous_features
        )
    ),

    "54314_patients_in_manifest": (
        manifest[
            "subject_id"
        ].nunique()
        == 54314
    ),

    "43450_train_patients": (
        manifest.loc[
            manifest["split"]
            == "train",
            "subject_id",
        ].nunique()
        == 43450
    ),

    "10864_test_patients": (
        manifest.loc[
            manifest["split"]
            == "test",
            "subject_id",
        ].nunique()
        == 10864
    ),

    "all_10_partition_files_exist": (
        len(
            missing_partition_files
        )
        == 0
    ),
}


validation = pd.DataFrame(
    [
        {
            "check": key,
            "pass": bool(value),
        }
        for key, value
        in checks.items()
    ]
)


print()
print(
    validation.to_string(
        index=False
    )
)

print()


if missing_partition_files:

    print(
        "Missing partition files:"
    )

    for path in missing_partition_files:

        print(
            f"  {path}"
        )

    print()


print(
    "All configuration checks passed:"
    f" {validation['pass'].all()}"
)