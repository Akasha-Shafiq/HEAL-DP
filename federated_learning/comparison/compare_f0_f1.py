"""
Formal comparison of:

    F0 = centralized non-private logistic regression
    F1 = non-private five-client FedAvg logistic regression

The test population, representation, label, and decision threshold
are identical. Therefore differences describe the empirical effect
associated with moving from centralized optimization to FedAvg under
the frozen approximately-IID, unequal-client-size partition.
"""

from __future__ import annotations

import pandas as pd

from federated_learning.config import (
    F0_RESULTS_DIR,
    F1_RESULTS_DIR,
    COMPARISON_RESULTS_DIR,
)


COMPARISON_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "long_precision",
    "long_recall",
    "long_f1",
    "specificity",
    "FPR",
]


# =====================================================================
# Global comparison
# =====================================================================

f0_global = pd.read_csv(
    F0_RESULTS_DIR
    / "f0_global_metrics.csv"
).iloc[0]


f1_global = pd.read_csv(
    F1_RESULTS_DIR
    / "f1_global_metrics.csv"
).iloc[0]


global_rows = []


for metric in METRICS:

    f0_value = float(
        f0_global[
            metric
        ]
    )

    f1_value = float(
        f1_global[
            metric
        ]
    )

    global_rows.append(
        {
            "metric": metric,
            "F0": f0_value,
            "F1": f1_value,
            "delta_F1_minus_F0":
                f1_value
                - f0_value,
        }
    )


global_comparison = pd.DataFrame(
    global_rows
)


global_comparison.to_csv(
    COMPARISON_RESULTS_DIR
    / "f0_vs_f1_global_metrics.csv",
    index=False,
)


# =====================================================================
# Confusion-matrix comparison
# =====================================================================

cm_rows = []


for cell in [
    "TN",
    "FP",
    "FN",
    "TP",
]:

    f0_value = int(
        f0_global[
            cell
        ]
    )

    f1_value = int(
        f1_global[
            cell
        ]
    )

    cm_rows.append(
        {
            "cell": cell,
            "F0": f0_value,
            "F1": f1_value,
            "delta_F1_minus_F0":
                f1_value
                - f0_value,
        }
    )


confusion_comparison = pd.DataFrame(
    cm_rows
)


confusion_comparison.to_csv(
    COMPARISON_RESULTS_DIR
    / "f0_vs_f1_confusion_matrix.csv",
    index=False,
)


# =====================================================================
# Generic grouped comparison
# =====================================================================

def compare_grouped(
    f0_file,
    f1_file,
    keys,
    output_file,
):

    f0 = pd.read_csv(
        f0_file
    )

    f1 = pd.read_csv(
        f1_file
    )


    merged = f0.merge(
        f1,
        on=keys,
        how="inner",
        suffixes=(
            "_F0",
            "_F1",
        ),
        validate="one_to_one",
    )


    if (
        "n_F0"
        in merged.columns
        and "n_F1"
        in merged.columns
    ):

        if not (
            merged[
                "n_F0"
            ]
            == merged[
                "n_F1"
            ]
        ).all():

            raise RuntimeError(
                f"Support mismatch for {output_file}"
            )


    for metric in METRICS:

        f0_column = (
            f"{metric}_F0"
        )

        f1_column = (
            f"{metric}_F1"
        )

        if (
            f0_column
            in merged.columns
            and f1_column
            in merged.columns
        ):

            merged[
                f"delta_{metric}"
            ] = (
                merged[
                    f1_column
                ]
                - merged[
                    f0_column
                ]
            )


    merged.to_csv(
        COMPARISON_RESULTS_DIR
        / output_file,
        index=False,
    )


    return merged


# =====================================================================
# Hospital comparison
# =====================================================================

hospital_comparison = (
    compare_grouped(
        F0_RESULTS_DIR
        / "f0_hospital_metrics.csv",

        F1_RESULTS_DIR
        / "f1_hospital_metrics.csv",

        [
            "hospital",
        ],

        "f0_vs_f1_hospital_metrics.csv",
    )
)


# =====================================================================
# Race comparison
# =====================================================================

race_comparison = (
    compare_grouped(
        F0_RESULTS_DIR
        / "f0_race_metrics.csv",

        F1_RESULTS_DIR
        / "f1_race_metrics.csv",

        [
            "race",
        ],

        "f0_vs_f1_race_metrics.csv",
    )
)


# =====================================================================
# Insurance comparison
# =====================================================================

insurance_comparison = (
    compare_grouped(
        F0_RESULTS_DIR
        / "f0_insurance_metrics.csv",

        F1_RESULTS_DIR
        / "f1_insurance_metrics.csv",

        [
            "insurance",
        ],

        "f0_vs_f1_insurance_metrics.csv",
    )
)


# =====================================================================
# Hospital x Race
# =====================================================================

compare_grouped(
    F0_RESULTS_DIR
    / "f0_hospital_race_metrics.csv",

    F1_RESULTS_DIR
    / "f1_hospital_race_metrics.csv",

    [
        "hospital",
        "race",
    ],

    "f0_vs_f1_hospital_race_metrics.csv",
)


# =====================================================================
# Hospital x Insurance
# =====================================================================

compare_grouped(
    F0_RESULTS_DIR
    / "f0_hospital_insurance_metrics.csv",

    F1_RESULTS_DIR
    / "f1_hospital_insurance_metrics.csv",

    [
        "hospital",
        "insurance",
    ],

    "f0_vs_f1_hospital_insurance_metrics.csv",
)


# =====================================================================
# Subgroup spread sensitivity
# =====================================================================

def subgroup_spread(
    dataframe,
    group_column,
    model_name,
    minimum_support,
):

    eligible = dataframe[
        dataframe[
            "n"
        ]
        >= minimum_support
    ].copy()


    if len(
        eligible
    ) < 2:

        return {
            "model":
                model_name,

            "attribute":
                group_column,

            "minimum_support":
                minimum_support,

            "groups_included":
                len(
                    eligible
                ),

            "long_recall_spread":
                float("nan"),

            "FPR_spread":
                float("nan"),
        }


    return {
        "model":
            model_name,

        "attribute":
            group_column,

        "minimum_support":
            minimum_support,

        "groups_included":
            len(
                eligible
            ),

        "long_recall_spread":
            (
                eligible[
                    "long_recall"
                ].max()
                - eligible[
                    "long_recall"
                ].min()
            ),

        "FPR_spread":
            (
                eligible[
                    "FPR"
                ].max()
                - eligible[
                    "FPR"
                ].min()
            ),
    }


f0_race = pd.read_csv(
    F0_RESULTS_DIR
    / "f0_race_metrics.csv"
)

f1_race = pd.read_csv(
    F1_RESULTS_DIR
    / "f1_race_metrics.csv"
)

f0_insurance = pd.read_csv(
    F0_RESULTS_DIR
    / "f0_insurance_metrics.csv"
)

f1_insurance = pd.read_csv(
    F1_RESULTS_DIR
    / "f1_insurance_metrics.csv"
)


spread_rows = []


for threshold in [
    100,
    500,
]:

    for (
        dataframe,
        attribute,
        model,
    ) in [
        (
            f0_race,
            "race",
            "F0",
        ),
        (
            f1_race,
            "race",
            "F1",
        ),
        (
            f0_insurance,
            "insurance",
            "F0",
        ),
        (
            f1_insurance,
            "insurance",
            "F1",
        ),
    ]:

        spread_rows.append(
            subgroup_spread(
                dataframe,
                attribute,
                model,
                threshold,
            )
        )


spread_comparison = pd.DataFrame(
    spread_rows
)


spread_comparison.to_csv(
    COMPARISON_RESULTS_DIR
    / "f0_vs_f1_subgroup_spread_sensitivity.csv",
    index=False,
)


# =====================================================================
# Print core comparison
# =====================================================================

print("=" * 80)
print("F0 -> F1 GLOBAL EFFECT")
print("=" * 80)

print(
    global_comparison.round(
        6
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("F0 -> F1 CONFUSION MATRIX EFFECT")
print("=" * 80)

print(
    confusion_comparison.to_string(
        index=False
    )
)


print()
print("=" * 80)
print("F0 -> F1 HOSPITAL EFFECT")
print("=" * 80)

print(
    hospital_comparison[
        [
            "hospital",
            "n_F0",
            "delta_roc_auc",
            "delta_pr_auc",
            "delta_accuracy",
            "delta_long_recall",
            "delta_FPR",
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
print("SUBGROUP SPREAD SENSITIVITY")
print("=" * 80)

print(
    spread_comparison.round(
        6
    ).to_string(
        index=False
    )
)


print()
print(
    "Comparison results saved to:"
)

print(
    f"  {COMPARISON_RESULTS_DIR}"
)