"""
Shared utilities for the non-private patient-representation
federated experiments.

Supported strategies
--------------------
strategy_a:
    exactly one earliest ICU training stay per patient

strategy_b_k5:
    earliest up to five ICU training stays per patient

The original frozen H1-H5 test partitions remain unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from federated_learning.preprocessing import (
    transform_features,
)

from federated_learning.federated.task import (
    get_project_root,
    hospital_from_context,
    load_feature_specification,
    make_labels,
    train_local_model,
    load_global_test,
    predict_model,
    set_seed,
)


SUPPORTED_STRATEGIES = (
    "strategy_a",
    "strategy_b_k5",
)


_CACHE = {}


def get_training_strategy(
    context,
) -> str:

    strategy = str(
        context.run_config[
            "training-strategy"
        ]
    )

    if strategy not in SUPPORTED_STRATEGIES:

        raise ValueError(
            "Unsupported training strategy: "
            f"{strategy}"
        )

    return strategy


def get_strategy_train_file(
    project_root,
    hospital,
    strategy,
):

    if strategy == "strategy_a":

        directory = (
            project_root
            / "data"
            / "federated_learning"
            / "partitions"
            / "patient_contribution"
            / "strategy_a"
        )

    elif strategy == "strategy_b_k5":

        directory = (
            project_root
            / "data"
            / "federated_learning"
            / "partitions"
            / "patient_contribution"
            / "strategy_b_k5"
        )

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )


    path = (
        directory
        / f"{hospital}_train.csv.gz"
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Missing frozen strategy partition: {path}"
        )


    return path


def load_strategy_training_partition(
    context,
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

    strategy = get_training_strategy(
        context
    )


    cache_key = (
        str(project_root),
        hospital,
        strategy,
    )


    if cache_key in _CACHE:

        return _CACHE[
            cache_key
        ]


    path = get_strategy_train_file(
        project_root,
        hospital,
        strategy,
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
        == "train"
    ).all():

        raise RuntimeError(
            f"Non-training rows found in {path}"
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


    result = (
        dataframe,
        X,
        y,
        feature_columns,
        hospital,
        partition_id,
        strategy,
    )


    _CACHE[
        cache_key
    ] = result


    return result


def make_strategy_train_loader(
    context,
    batch_size,
    server_round,
):

    (
        dataframe,
        X,
        y,
        feature_columns,
        hospital,
        partition_id,
        strategy,
    ) = load_strategy_training_partition(
        context
    )


    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            y,
            dtype=torch.float32,
        ).unsqueeze(
            1
        ),
    )


    generator = torch.Generator()

    generator.manual_seed(
        42
        + partition_id * 10_000
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
        len(dataframe),
        dataframe[
            "subject_id"
        ].nunique(),
        len(feature_columns),
        strategy,
    )