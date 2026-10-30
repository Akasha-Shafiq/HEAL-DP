"""
Shared preprocessing for F0, F1 and F2.

The feature representation and transformations are frozen from A3.
No feature selection is performed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from federated_learning.config import (
    FEATURE_LIST_FILE,
    PREPROCESSING_SPEC_FILE,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_BINARY_FEATURE_COUNT,
    EXPECTED_CONTINUOUS_FEATURE_COUNT,
    LOS_THRESHOLD_DAYS,
)


def load_feature_specification():
    """
    Load the frozen A3 feature list and preprocessing specification.
    """

    feature_df = pd.read_csv(
        FEATURE_LIST_FILE
    )

    preprocessing_df = pd.read_csv(
        PREPROCESSING_SPEC_FILE
    )

    feature_columns = (
        feature_df["feature"]
        .astype(str)
        .tolist()
    )

    binary_features = (
        preprocessing_df.loc[
            preprocessing_df[
                "feature_type"
            ].astype(str).str.lower()
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
            ].astype(str).str.lower()
            == "continuous",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    assert (
        len(feature_columns)
        == EXPECTED_FEATURE_COUNT
    )

    assert (
        len(binary_features)
        == EXPECTED_BINARY_FEATURE_COUNT
    )

    assert (
        len(continuous_features)
        == EXPECTED_CONTINUOUS_FEATURE_COUNT
    )

    assert set(
        feature_columns
    ) == set(
        binary_features
        + continuous_features
    )

    return (
        feature_columns,
        binary_features,
        continuous_features,
    )


def transform_features(
    data: pd.DataFrame,
    feature_columns: list[str],
    continuous_features: list[str],
) -> np.ndarray:
    """
    Apply the frozen A3 transformation.

    Binary features:
        unchanged 0/1 values

    Continuous features:
        u = sign(x) * log(1 + |x|)
        x' = u / (1 + |u|)
    """

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing frozen model features: "
            f"{missing_features[:10]}"
        )

    X = (
        data[
            feature_columns
        ]
        .copy()
        .astype(
            np.float32
        )
    )

    values = (
        X[
            continuous_features
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    signed_log = (
        np.sign(values)
        * np.log1p(
            np.abs(values)
        )
    )

    transformed = (
        signed_log
        / (
            1.0
            + np.abs(
                signed_log
            )
        )
    )

    X.loc[
        :,
        continuous_features,
    ] = transformed.astype(
        np.float32
    )

    output = X.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(
        output
    ).all():
        raise ValueError(
            "Non-finite values found "
            "after preprocessing."
        )

    return output


def make_binary_labels(
    data: pd.DataFrame,
) -> np.ndarray:
    """
    Reproduce the original implemented LOS rule:

        Short = LOS <= 4 days
        Long  = LOS > 4 days

    Short -> 0
    Long  -> 1
    """

    if "los" not in data.columns:
        raise ValueError(
            "Column 'los' is required."
        )

    return (
        data[
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