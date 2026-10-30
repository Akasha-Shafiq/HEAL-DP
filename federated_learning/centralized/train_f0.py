"""
F0: Centralized non-private logistic-regression baseline.

Training population
-------------------
Union of:

    H1_train
    H2_train
    H3_train
    H4_train
    H5_train

Test population
---------------
Union of:

    H1_test
    H2_test
    H3_test
    H4_test
    H5_test

The hospital-first patient-disjoint split is frozen and is NOT
regenerated here.

Model and preprocessing reproduce the established A3 centralized
logistic-regression configuration.
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from federated_learning.config import (
    SEED,
    HOSPITALS,
    TRAIN_FILES,
    TEST_FILES,
    F0_RESULTS_DIR,
    F0_EPOCHS,
    F0_BATCH_SIZE,
    F0_LEARNING_RATE,
    CLASSIFICATION_THRESHOLD,
    EXPECTED_FEATURE_COUNT,
    MIN_SUBGROUP_SUPPORT,
)

from federated_learning.preprocessing import (
    load_feature_specification,
    transform_features,
    make_binary_labels,
)

from federated_learning.models import (
    LogisticRegressionModel,
)

from federated_learning.evaluation import (
    binary_metrics,
    evaluate_groups,
)


# =====================================================================
# Reproducibility
# =====================================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================================
# Load frozen hospital partitions
# =====================================================================

print("=" * 80)
print("F0 CENTRALIZED NON-PRIVATE BASELINE")
print("=" * 80)

print()
print(
    f"Device: {device}"
)

print(
    "Loading frozen hospital partitions..."
)


train_parts = []

test_parts = []


for hospital in HOSPITALS:

    train_df = pd.read_csv(
        TRAIN_FILES[
            hospital
        ],
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_FILES[
            hospital
        ],
        low_memory=False,
    )

    # Defensive validation.

    assert (
        train_df[
            "hospital"
        ]
        == hospital
    ).all()

    assert (
        train_df[
            "split"
        ]
        == "train"
    ).all()

    assert (
        test_df[
            "hospital"
        ]
        == hospital
    ).all()

    assert (
        test_df[
            "split"
        ]
        == "test"
    ).all()

    train_parts.append(
        train_df
    )

    test_parts.append(
        test_df
    )


train_data = pd.concat(
    train_parts,
    ignore_index=True,
)


test_data = pd.concat(
    test_parts,
    ignore_index=True,
)


print()
print("Frozen F0 populations")
print("---------------------")

print(
    "Training rows/stays:"
    f" {len(train_data):,}"
)

print(
    "Training patients:"
    f" {train_data['subject_id'].nunique():,}"
)

print(
    "Test rows/stays:"
    f"     {len(test_data):,}"
)

print(
    "Test patients:"
    f"       {test_data['subject_id'].nunique():,}"
)


# =====================================================================
# Patient leakage validation
# =====================================================================

train_patients = set(
    train_data[
        "subject_id"
    ].unique()
)

test_patients = set(
    test_data[
        "subject_id"
    ].unique()
)


patient_overlap = (
    train_patients
    & test_patients
)


if patient_overlap:

    raise RuntimeError(
        "Patient leakage detected between "
        "training and test data."
    )


assert len(
    train_patients
) == 43450

assert len(
    test_patients
) == 10864


print(
    "Train/test patient overlap:"
    f" {len(patient_overlap)}"
)


# =====================================================================
# Frozen feature representation
# =====================================================================

(
    feature_columns,
    binary_features,
    continuous_features,
) = load_feature_specification()


assert (
    len(
        feature_columns
    )
    == EXPECTED_FEATURE_COUNT
)


print()
print("Frozen representation")
print("---------------------")

print(
    "Features:"
    f"   {len(feature_columns)}"
)

print(
    "Binary:"
    f"     {len(binary_features)}"
)

print(
    "Continuous:"
    f" {len(continuous_features)}"
)


# =====================================================================
# Preprocessing
# =====================================================================

print()
print(
    "Applying frozen A3 preprocessing..."
)


X_train = transform_features(
    train_data,
    feature_columns,
    continuous_features,
)


X_test = transform_features(
    test_data,
    feature_columns,
    continuous_features,
)


y_train = make_binary_labels(
    train_data
)


y_test = make_binary_labels(
    test_data
)


assert X_train.shape == (
    59522,
    351,
)

assert X_test.shape == (
    14979,
    351,
)


print(
    f"X_train shape: {X_train.shape}"
)

print(
    f"X_test shape:  {X_test.shape}"
)

print(
    "Training Long proportion:"
    f" {y_train.mean():.5f}"
)

print(
    "Test Long proportion:"
    f"     {y_test.mean():.5f}"
)


# =====================================================================
# PyTorch tensors
# =====================================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32,
)


y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32,
).unsqueeze(
    1
)


X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)


# =====================================================================
# Model
# =====================================================================

model = LogisticRegressionModel(
    n_features=len(
        feature_columns
    )
).to(
    device
)


n_parameters = sum(
    parameter.numel()
    for parameter
    in model.parameters()
    if parameter.requires_grad
)


assert (
    n_parameters
    == 352
)


print()
print("Model")
print("-----")

print(
    model
)

print(
    "Trainable parameters:"
    f" {n_parameters}"
)


# =====================================================================
# Training configuration
# =====================================================================

generator = torch.Generator()

generator.manual_seed(
    SEED
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=F0_BATCH_SIZE,
    shuffle=True,
    generator=generator,
)


criterion = (
    nn.BCEWithLogitsLoss()
)


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=F0_LEARNING_RATE,
)


print()
print("Training configuration")
print("----------------------")

print(
    "Epochs:"
    f"        {F0_EPOCHS}"
)

print(
    "Batch size:"
    f"    {F0_BATCH_SIZE}"
)

print(
    "Learning rate:"
    f" {F0_LEARNING_RATE}"
)

print(
    "Threshold:"
    f"     {CLASSIFICATION_THRESHOLD}"
)


# =====================================================================
# Train F0
# =====================================================================

training_history = []


print()
print("Training F0")
print("-----------")


for epoch in range(
    1,
    F0_EPOCHS + 1,
):

    model.train()

    total_loss = 0.0

    total_examples = 0


    for (
        X_batch,
        y_batch,
    ) in train_loader:

        X_batch = X_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        )


        optimizer.zero_grad()


        logits = model(
            X_batch
        )


        loss = criterion(
            logits,
            y_batch,
        )


        loss.backward()

        optimizer.step()


        batch_size = int(
            X_batch.size(
                0
            )
        )


        total_loss += (
            loss.item()
            * batch_size
        )

        total_examples += (
            batch_size
        )


    mean_loss = (
        total_loss
        / total_examples
    )


    training_history.append(
        {
            "epoch": epoch,
            "loss": mean_loss,
        }
    )


    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == F0_EPOCHS
    ):

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{F0_EPOCHS}"
            f" - loss: "
            f"{mean_loss:.6f}"
        )


training_history_df = pd.DataFrame(
    training_history
)


training_history_df.to_csv(
    F0_RESULTS_DIR
    / "f0_training_history.csv",
    index=False,
)


# =====================================================================
# Global held-out prediction
# =====================================================================

model.eval()


with torch.no_grad():

    logits = model(
        X_test_tensor.to(
            device
        )
    )


    probabilities = (
        torch.sigmoid(
            logits
        )
        .cpu()
        .numpy()
        .ravel()
    )


predictions = (
    probabilities
    > CLASSIFICATION_THRESHOLD
).astype(
    int
)


global_metrics = binary_metrics(
    y_test,
    probabilities,
    predictions,
)


global_metrics.update(
    {
        "model":
            "F0_Centralized_Logistic_Regression",

        "features":
            len(
                feature_columns
            ),

        "trainable_parameters":
            n_parameters,

        "epochs":
            F0_EPOCHS,

        "batch_size":
            F0_BATCH_SIZE,

        "learning_rate":
            F0_LEARNING_RATE,

        "classification_threshold":
            CLASSIFICATION_THRESHOLD,

        "training_rows":
            len(
                train_data
            ),

        "training_patients":
            len(
                train_patients
            ),

        "test_rows":
            len(
                test_data
            ),

        "test_patients":
            len(
                test_patients
            ),
    }
)


global_metrics_df = pd.DataFrame(
    [
        global_metrics
    ]
)


global_metrics_df.to_csv(
    F0_RESULTS_DIR
    / "f0_global_metrics.csv",
    index=False,
)


# =====================================================================
# Preserve row-level held-out predictions
# =====================================================================

prediction_columns = [
    "subject_id",
    "stay_id",
    "hospital",
    "race",
    "insurance",
    "los",
]


predictions_df = (
    test_data[
        prediction_columns
    ]
    .copy()
)


predictions_df[
    "y_true"
] = y_test.astype(
    int
)


predictions_df[
    "probability"
] = probabilities


predictions_df[
    "y_pred"
] = predictions


predictions_df.to_csv(
    F0_RESULTS_DIR
    / "f0_test_predictions.csv.gz",
    index=False,
    compression={
        "method": "gzip",
        "compresslevel": 6,
        "mtime": 0,
    },
)


# =====================================================================
# Hospital evaluation
# =====================================================================

hospital_metrics = evaluate_groups(
    predictions_df,
    [
        "hospital",
    ],
)


hospital_metrics.to_csv(
    F0_RESULTS_DIR
    / "f0_hospital_metrics.csv",
    index=False,
)


# =====================================================================
# Race evaluation
# =====================================================================

race_predictions = (
    predictions_df.copy()
)


race_predictions[
    "race"
] = (
    race_predictions[
        "race"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


race_metrics = evaluate_groups(
    race_predictions,
    [
        "race",
    ],
)


race_metrics[
    "eligible_n_ge_100"
] = (
    race_metrics[
        "n"
    ]
    >= MIN_SUBGROUP_SUPPORT
)


race_metrics.to_csv(
    F0_RESULTS_DIR
    / "f0_race_metrics.csv",
    index=False,
)


# =====================================================================
# Insurance evaluation
# =====================================================================

insurance_predictions = (
    predictions_df.copy()
)


insurance_predictions[
    "insurance"
] = (
    insurance_predictions[
        "insurance"
    ]
    .fillna(
        "<MISSING>"
    )
    .astype(str)
)


insurance_metrics = evaluate_groups(
    insurance_predictions,
    [
        "insurance",
    ],
)


insurance_metrics[
    "eligible_n_ge_100"
] = (
    insurance_metrics[
        "n"
    ]
    >= MIN_SUBGROUP_SUPPORT
)


insurance_metrics.to_csv(
    F0_RESULTS_DIR
    / "f0_insurance_metrics.csv",
    index=False,
)


# =====================================================================
# Hospital × Race
# =====================================================================

hospital_race_metrics = (
    evaluate_groups(
        race_predictions,
        [
            "hospital",
            "race",
        ],
    )
)


hospital_race_metrics[
    "eligible_n_ge_100"
] = (
    hospital_race_metrics[
        "n"
    ]
    >= MIN_SUBGROUP_SUPPORT
)


hospital_race_metrics.to_csv(
    F0_RESULTS_DIR
    / "f0_hospital_race_metrics.csv",
    index=False,
)


# =====================================================================
# Hospital × Insurance
# =====================================================================

hospital_insurance_metrics = (
    evaluate_groups(
        insurance_predictions,
        [
            "hospital",
            "insurance",
        ],
    )
)


hospital_insurance_metrics[
    "eligible_n_ge_100"
] = (
    hospital_insurance_metrics[
        "n"
    ]
    >= MIN_SUBGROUP_SUPPORT
)


hospital_insurance_metrics.to_csv(
    F0_RESULTS_DIR
    / "f0_hospital_insurance_metrics.csv",
    index=False,
)


# =====================================================================
# Disparity summaries
# =====================================================================

def disparity_summary(
    dataframe,
    attribute,
):
    """
    Descriptive range among groups with n >= minimum support.
    """

    eligible = dataframe[
        dataframe[
            "n"
        ]
        >= MIN_SUBGROUP_SUPPORT
    ].copy()

    return {
        "attribute":
            attribute,

        "minimum_support":
            MIN_SUBGROUP_SUPPORT,

        "groups_included":
            len(
                eligible
            ),

        "long_recall_spread":
            float(
                eligible[
                    "long_recall"
                ].max()
                - eligible[
                    "long_recall"
                ].min()
            ),

        "FPR_spread":
            float(
                eligible[
                    "FPR"
                ].max()
                - eligible[
                    "FPR"
                ].min()
            ),
    }


disparity_df = pd.DataFrame(
    [
        disparity_summary(
            race_metrics,
            "race",
        ),

        disparity_summary(
            insurance_metrics,
            "insurance",
        ),
    ]
)


disparity_df.to_csv(
    F0_RESULTS_DIR
    / "f0_subgroup_disparity_summary.csv",
    index=False,
)


# =====================================================================
# Save model
# =====================================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "feature_columns":
            feature_columns,

        "model_type":
            "logistic_regression",

        "n_features":
            len(
                feature_columns
            ),

        "classification_threshold":
            CLASSIFICATION_THRESHOLD,

        "seed":
            SEED,

        "epochs":
            F0_EPOCHS,

        "batch_size":
            F0_BATCH_SIZE,

        "learning_rate":
            F0_LEARNING_RATE,
    },
    F0_RESULTS_DIR
    / "f0_model.pt",
)


# =====================================================================
# Save experiment metadata
# =====================================================================

metadata = {
    "experiment":
        "F0",

    "architecture":
        "centralized_non_private",

    "model":
        "PyTorch logistic regression",

    "privacy":
        "none",

    "patient_disjoint_train_test":
        True,

    "hospital_allocation":
        {
            "H1": 0.20,
            "H2": 0.20,
            "H3": 0.20,
            "H4": 0.05,
            "H5": 0.35,
        },

    "test_population":
        "union of H1-H5 local held-out test partitions",

    "feature_count":
        351,

    "decision_threshold":
        CLASSIFICATION_THRESHOLD,
}


with open(
    F0_RESULTS_DIR
    / "f0_experiment_metadata.json",
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        metadata,
        handle,
        indent=2,
    )


# =====================================================================
# Console summary
# =====================================================================

print()
print("=" * 80)
print("F0 GLOBAL HELD-OUT RESULTS")
print("=" * 80)

print(
    global_metrics_df[
        [
            "roc_auc",
            "pr_auc",
            "accuracy",
            "long_precision",
            "long_recall",
            "long_f1",
            "FPR",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(
        5
    )
    .to_string(
        index=False
    )
)


print()
print("Global confusion matrix")
print("-----------------------")

print(
    np.array(
        [
            [
                global_metrics[
                    "TN"
                ],
                global_metrics[
                    "FP"
                ],
            ],
            [
                global_metrics[
                    "FN"
                ],
                global_metrics[
                    "TP"
                ],
            ],
        ]
    )
)


print()
print("=" * 80)
print("F0 HOSPITAL RESULTS")
print("=" * 80)

print(
    hospital_metrics[
        [
            "hospital",
            "n",
            "roc_auc",
            "pr_auc",
            "accuracy",
            "long_recall",
            "FPR",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(
        5
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 80)
print("F0 SUBGROUP DISPARITY SUMMARY")
print("=" * 80)

print(
    disparity_df.round(
        5
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("F0 COMPLETE")
print("=" * 80)

print()

print(
    f"Results directory:\n"
    f"  {F0_RESULTS_DIR}"
)

print()

print(
    "No DP and no federated aggregation "
    "were used in F0."
)