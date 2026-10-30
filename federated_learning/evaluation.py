"""
Shared binary-classification and subgroup evaluation utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true,
    probabilities,
    predictions,
) -> dict:
    """
    Calculate the common F0/F1/F2 evaluation metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    specificity = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn
            + fp
        ) > 0
        else np.nan
    )

    fpr = (
        fp
        / (
            fp
            + tn
        )
        if (
            fp
            + tn
        ) > 0
        else np.nan
    )

    result = {
        "n": int(
            len(
                y_true
            )
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "long_precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "long_recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "long_f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "FPR": float(
            fpr
        ),
    }

    # AUC metrics require both classes.

    if len(
        np.unique(
            y_true
        )
    ) == 2:

        result[
            "roc_auc"
        ] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        result[
            "pr_auc"
        ] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:

        result[
            "roc_auc"
        ] = np.nan

        result[
            "pr_auc"
        ] = np.nan

    return result


def evaluate_groups(
    predictions_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Evaluate predictions separately for arbitrary subgroup columns.

    Examples:

        ["hospital"]
        ["race"]
        ["insurance"]
        ["hospital", "race"]
    """

    rows = []

    grouped = predictions_df.groupby(
        group_columns,
        dropna=False,
        observed=True,
    )

    for group_values, subset in grouped:

        if not isinstance(
            group_values,
            tuple,
        ):
            group_values = (
                group_values,
            )

        metrics = binary_metrics(
            subset[
                "y_true"
            ],
            subset[
                "probability"
            ],
            subset[
                "y_pred"
            ],
        )

        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                group_values,
            )
        }

        row.update(
            metrics
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )