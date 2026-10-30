
"""
Shared configuration for the MIMIC-DP federated-learning experiments.

This file defines the frozen experimental inputs used by:

    F0 - centralized non-private baseline
    F1 - federated non-private baseline
    F2 - federated DP training

The patient/hospital/train-test assignments must not be regenerated
inside model-specific scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


# =====================================================================
# Reproducibility
# =====================================================================

SEED = 42

LOS_THRESHOLD_DAYS = 4.0
CLASSIFICATION_THRESHOLD = 0.5


# =====================================================================
#  paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FL_DATA_DIR = Path(
    os.environ.get(
        "PET_FL_DATA_DIR",
        str(PROJECT_ROOT / "data" / "federated_learning"),
    )
).expanduser()

SOURCE_DIR = (
    FL_DATA_DIR
    / "source"
)

MANIFEST_DIR = (
    FL_DATA_DIR
    / "manifests"
)

PARTITION_DIR = (
    FL_DATA_DIR
    / "partitions"
)


# Experimental results can be stored outside the source repository.

RESULTS_ROOT = Path(
    os.environ.get(
        "PET_FL_RESULTS_DIR",
        str(PROJECT_ROOT / "results" / "federated_learning"),
    )
).expanduser()

AUDIT_RESULTS_DIR = (
    RESULTS_ROOT
    / "audit"
)

PARTITION_RESULTS_DIR = (
    RESULTS_ROOT
    / "partitions"
)

F0_RESULTS_DIR = (
    RESULTS_ROOT
    / "f0_centralized"
)

F1_RESULTS_DIR = (
    RESULTS_ROOT
    / "f1_federated"
)

F2_RESULTS_DIR = (
    RESULTS_ROOT
    / "f2_dp"
)

COMPARISON_RESULTS_DIR = (
    RESULTS_ROOT
    / "comparison"
)


for directory in [
    F0_RESULTS_DIR,
    F1_RESULTS_DIR,
    F2_RESULTS_DIR,
    COMPARISON_RESULTS_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# Frozen source artifacts
# =====================================================================

SOURCE_DATA_FILE = (
    SOURCE_DIR
    / "icu_data_no_imputation_atc3_lab5pct.csv"
)

FEATURE_LIST_FILE = (
    SOURCE_DIR
    / "final_351_feature_list.csv"
)

PREPROCESSING_SPEC_FILE = (
    SOURCE_DIR
    / "a3_preprocessing_specification.csv"
)


# =====================================================================
# Frozen patient manifests
# =====================================================================

HOSPITAL_ASSIGNMENT_FILE = (
    MANIFEST_DIR
    / "patient_hospital_assignment.csv"
)

TRAIN_TEST_ASSIGNMENT_FILE = (
    MANIFEST_DIR
    / "patient_hospital_train_test_assignment.csv"
)


# =====================================================================
# Federated hospitals
# =====================================================================

HOSPITALS = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
)


HOSPITAL_PROPORTIONS = {
    "H1": 0.20,
    "H2": 0.20,
    "H3": 0.20,
    "H4": 0.05,
    "H5": 0.35,
}


# =====================================================================
# Physical local partitions
# =====================================================================

TRAIN_FILES = {
    hospital: (
        PARTITION_DIR
        / f"{hospital}_train.csv.gz"
    )
    for hospital in HOSPITALS
}


TEST_FILES = {
    hospital: (
        PARTITION_DIR
        / f"{hospital}_test.csv.gz"
    )
    for hospital in HOSPITALS
}


# =====================================================================
# Frozen model representation
# =====================================================================

EXPECTED_FEATURE_COUNT = 351
EXPECTED_BINARY_FEATURE_COUNT = 193
EXPECTED_CONTINUOUS_FEATURE_COUNT = 158


# =====================================================================
# Initial F0 model settings
# =====================================================================

# These preserve the established A3 logistic-regression configuration
# for the first centralized comparison.

F0_EPOCHS = 30
F0_BATCH_SIZE = 256
F0_LEARNING_RATE = 0.05


# =====================================================================
# Fairness / subgroup evaluation
# =====================================================================

FAIRNESS_ATTRIBUTES = (
    "race",
    "insurance",
)


# Minimum subgroup support used for comparative disparity summaries.
# Smaller groups can still be reported but should not drive strong
# comparative conclusions.

MIN_SUBGROUP_SUPPORT = 100


# =====================================================================
# F1 federated non-private configuration
# =====================================================================

F1_NUM_ROUNDS = 30

F1_LOCAL_EPOCHS = 1

F1_BATCH_SIZE = 256

F1_LEARNING_RATE = 0.05

# Baseline experiment:
# all five hospitals participate in every federated round.

F1_FRACTION_TRAIN = 1.0

F1_MIN_TRAIN_NODES = 5

F1_MIN_AVAILABLE_NODES = 5
