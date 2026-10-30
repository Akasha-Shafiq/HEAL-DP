"""
Shared F1 Flower task utilities.

The functions in this module:

- map Flower partition IDs to the five frozen hospitals;
- load the existing H1-H5 partitions;
- apply the frozen 351-feature A3 preprocessing;
- construct deterministic PyTorch DataLoaders;
- perform local SGD;
- support centralized evaluation of the global FedAvg model.

No patient assignment or train/test splitting occurs here.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from federated_learning.models import (
    LogisticRegressionModel,
)

from federated_learning.preprocessing import (
    transform_features,
)

from federated_learning.evaluation import (
    binary_metrics,
)


HOSPITALS = (
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
)

LOS_THRESHOLD_DAYS = 4.0

EXPECTED_FEATURE_COUNT = 351


# ---------------------------------------------------------------------
# Small process-local cache
# ---------------------------------------------------------------------
#
# ClientApps can be ephemeral, but Ray workers may be reused.
# Caching avoids unnecessarily re-reading and re-transforming the
# same hospital CSV if the same worker later serves the same partition.
#

_DATA_CACHE = {}


# =====================================================================
# Reproducibility
# =====================================================================

def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =====================================================================
# Project paths
# =====================================================================

def get_project_root(
    run_config,
) -> Path:

    project_root = Path(
        str(
            run_config[
                "project-root"
            ]
        )
    )

    if not project_root.exists():

        raise FileNotFoundError(
            "Configured project root does not exist: "
            f"{project_root}"
        )

    return project_root


def get_partition_file(
    project_root: Path,
    hospital: str,
    split: str,
) -> Path:

    path = (
        project_root
        / "data"
        / "federated_learning"
        / "partitions"
        / f"{hospital}_{split}.csv.gz"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing frozen partition: {path}"
        )

    return path


# =====================================================================
# Frozen feature specification
# =====================================================================

def load_feature_specification(
    project_root: Path,
):

    feature_file = (
        project_root
        / "data"
        / "federated_learning"
        / "source"
        / "final_351_feature_list.csv"
    )

    preprocessing_file = (
        project_root
        / "data"
        / "federated_learning"
        / "source"
        / "a3_preprocessing_specification.csv"
    )

    feature_df = pd.read_csv(
        feature_file
    )

    preprocessing_df = pd.read_csv(
        preprocessing_file
    )

    feature_columns = (
        feature_df[
            "feature"
        ]
        .astype(str)
        .tolist()
    )

    binary_features = (
        preprocessing_df.loc[
            preprocessing_df[
                "feature_type"
            ]
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
            preprocessing_df[
                "feature_type"
            ]
            .astype(str)
            .str.lower()
            == "continuous",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    if len(
        feature_columns
    ) != EXPECTED_FEATURE_COUNT:

        raise RuntimeError(
            "Frozen feature count is not 351."
        )

    if len(
        binary_features
    ) != 193:

        raise RuntimeError(
            "Frozen binary feature count is not 193."
        )

    if len(
        continuous_features
    ) != 158:

        raise RuntimeError(
            "Frozen continuous feature count is not 158."
        )

    if set(
        feature_columns
    ) != set(
        binary_features
        + continuous_features
    ):

        raise RuntimeError(
            "Feature and preprocessing specifications disagree."
        )

    return (
        feature_columns,
        binary_features,
        continuous_features,
    )


# =====================================================================
# Labels
# =====================================================================

def make_labels(
    dataframe: pd.DataFrame,
) -> np.ndarray:

    return (
        dataframe[
            "los"
        ]
        .gt(
            LOS_THRESHOLD_DAYS
        )
        .astype(
            np.float32
        )
        .to_numpy()
    )


# =====================================================================
# Load one hospital partition
# =====================================================================

def load_hospital_partition(
    project_root: Path,
    hospital: str,
    split: str,
):

    cache_key = (
        str(
            project_root
        ),
        hospital,
        split,
    )

    if cache_key in _DATA_CACHE:

        return _DATA_CACHE[
            cache_key
        ]


    path = get_partition_file(
        project_root,
        hospital,
        split,
    )


    dataframe = pd.read_csv(
        path,
        low_memory=False,
    )


    if not (
        dataframe[
            "hospital"
        ]
        == hospital
    ).all():

        raise RuntimeError(
            f"Hospital mismatch in {path}"
        )


    if not (
        dataframe[
            "split"
        ]
        == split
    ).all():

        raise RuntimeError(
            f"Split mismatch in {path}"
        )


    (
        feature_columns,
        _,
        continuous_features,
    ) = load_feature_specification(
        project_root
    )


    X = transform_features(
        dataframe,
        feature_columns,
        continuous_features,
    )


    y = make_labels(
        dataframe
    )


    output = (
        dataframe,
        X,
        y,
        feature_columns,
    )


    _DATA_CACHE[
        cache_key
    ] = output


    return output


# =====================================================================
# Flower partition -> hospital mapping
# =====================================================================

def hospital_from_context(
    context,
) -> tuple[int, str]:

    partition_id = int(
        context.node_config[
            "partition-id"
        ]
    )

    num_partitions = int(
        context.node_config[
            "num-partitions"
        ]
    )


    if num_partitions != 5:

        raise RuntimeError(
            "F1 requires exactly 5 Flower SuperNodes, "
            f"but num-partitions={num_partitions}."
        )


    if not (
        0
        <= partition_id
        < len(
            HOSPITALS
        )
    ):

        raise RuntimeError(
            f"Invalid partition-id: {partition_id}"
        )


    return (
        partition_id,
        HOSPITALS[
            partition_id
        ],
    )


# =====================================================================
# Client training data
# =====================================================================

def make_train_loader(
    context,
    batch_size: int,
    server_round: int,
):

    project_root = get_project_root(
        context.run_config
    )

    (
        partition_id,
        hospital,
    ) = hospital_from_context(
        context
    )


    (
        dataframe,
        X,
        y,
        feature_columns,
    ) = load_hospital_partition(
        project_root,
        hospital,
        "train",
    )


    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )


    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    ).unsqueeze(
        1
    )


    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )


    generator = (
        torch.Generator()
    )

    # Stable but round-specific local shuffling.
    generator.manual_seed(
        42
        + (
            partition_id
            * 10_000
        )
        + server_round
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


    return (
        loader,
        hospital,
        len(
            dataframe
        ),
        len(
            feature_columns
        ),
    )


# =====================================================================
# Local SGD
# =====================================================================

def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    local_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:

    model.train()

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    optimizer = (
        torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
        )
    )


    total_loss = 0.0
    total_examples = 0


    for _ in range(
        local_epochs
    ):

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


            batch_n = int(
                X_batch.size(
                    0
                )
            )


            total_loss += (
                float(
                    loss.item()
                )
                * batch_n
            )

            total_examples += (
                batch_n
            )


    return (
        total_loss
        / total_examples
    )


# =====================================================================
# Global test population
# =====================================================================

def load_global_test(
    project_root: Path,
):

    frames = []


    for hospital in HOSPITALS:

        path = get_partition_file(
            project_root,
            hospital,
            "test",
        )

        frames.append(
            pd.read_csv(
                path,
                low_memory=False,
            )
        )


    test_data = pd.concat(
        frames,
        ignore_index=True,
    )


    (
        feature_columns,
        _,
        continuous_features,
    ) = load_feature_specification(
        project_root
    )


    X_test = transform_features(
        test_data,
        feature_columns,
        continuous_features,
    )


    y_test = make_labels(
        test_data
    )


    return (
        test_data,
        X_test,
        y_test,
        feature_columns,
    )


# =====================================================================
# Prediction
# =====================================================================

def predict_model(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
):

    model.eval()


    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )


    with torch.no_grad():

        logits = model(
            X_tensor.to(
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


    return probabilities