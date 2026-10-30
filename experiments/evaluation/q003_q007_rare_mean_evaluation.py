import argparse
import math
from pathlib import Path

import pandas as pd

from coordinator.aggregator import (
    aggregate_q003,
    aggregate_q007,
)
from coordinator.mechanisms import privatize_bounded_mean


EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]

LOWER = 0.0
UPPER = 30.0

SUM_BUDGET_FRACTION = 0.5


def release_bounded_mean(
    exact_sum: float,
    exact_count: int,
    epsilon: float,
) -> float:
    """
    Apply the same bounded sum-and-count DP mechanism
    used by the platform for Q003 and Q007.
    """

    result = privatize_bounded_mean(
        exact_sum=exact_sum,
        exact_count=exact_count,
        epsilon=epsilon,
        lower=LOWER,
        upper=UPPER,
        sum_budget_fraction=SUM_BUDGET_FRACTION,
    )

    return float(result["dp_mean"])


def main(repetitions: int) -> None:

    # ---------------------------------------------------------
    # 1. Obtain exact distributed baselines once
    # ---------------------------------------------------------

    q003 = aggregate_q003()["global_result"]
    q007 = aggregate_q007()["global_result"]

    queries = {
        "Q003": {
            "description": "Global bounded mean LOS",
            "exact_sum": float(q003["bounded_sum"]),
            "exact_count": int(q003["count"]),
            "exact_mean": float(q003["bounded_mean"]),
        },
        "Q007": {
            "description": (
                "Rare subgroup bounded mean LOS: "
                "ASIAN - KOREAN and URGENT"
            ),
            "exact_sum": float(q007["bounded_sum"]),
            "exact_count": int(q007["count"]),
            "exact_mean": float(q007["bounded_mean"]),
        },
    }

    print("EXACT BASELINES")
    print("===============")

    for query_id, info in queries.items():
        print(
            f"{query_id}: "
            f"count={info['exact_count']}, "
            f"sum={info['exact_sum']}, "
            f"mean={info['exact_mean']}"
        )

    print()

    # ---------------------------------------------------------
    # 2. Monte Carlo releases
    # ---------------------------------------------------------

    raw_rows = []

    for query_id, info in queries.items():

        for epsilon in EPSILONS:

            print(
                f"Running {query_id}, "
                f"epsilon={epsilon}, "
                f"repetitions={repetitions}"
            )

            for repetition in range(1, repetitions + 1):

                released_mean = release_bounded_mean(
                    exact_sum=info["exact_sum"],
                    exact_count=info["exact_count"],
                    epsilon=epsilon,
                )

                error = (
                    released_mean
                    - info["exact_mean"]
                )

                absolute_error = abs(error)

                relative_absolute_error = (
                    absolute_error
                    / info["exact_mean"]
                )

                raw_rows.append(
                    {
                        "query_id": query_id,
                        "description": info["description"],
                        "epsilon": epsilon,
                        "repetition": repetition,
                        "exact_count": info["exact_count"],
                        "exact_sum": info["exact_sum"],
                        "exact_mean": info["exact_mean"],
                        "released_mean": released_mean,
                        "error": error,
                        "absolute_error": absolute_error,
                        "squared_error": error ** 2,
                        "relative_absolute_error": (
                            relative_absolute_error
                        ),
                        "at_lower_bound": (
                            released_mean <= LOWER
                        ),
                        "at_upper_bound": (
                            released_mean >= UPPER
                        ),
                    }
                )

    raw_df = pd.DataFrame(raw_rows)

    # ---------------------------------------------------------
    # 3. Summarize utility
    # ---------------------------------------------------------

    summary_rows = []

    grouped = raw_df.groupby(
        [
            "query_id",
            "description",
            "epsilon",
        ],
        sort=True,
    )

    for (
        query_id,
        description,
        epsilon,
    ), group in grouped:

        exact_count = int(
            group["exact_count"].iloc[0]
        )

        exact_mean = float(
            group["exact_mean"].iloc[0]
        )

        bias = group["error"].mean()

        mae = group[
            "absolute_error"
        ].mean()

        rmse = math.sqrt(
            group[
                "squared_error"
            ].mean()
        )

        std = group[
            "released_mean"
        ].std(ddof=1)

        mean_release = group[
            "released_mean"
        ].mean()

        mean_relative_error = group[
            "relative_absolute_error"
        ].mean()

        median_absolute_error = group[
            "absolute_error"
        ].median()

        lower_bound_rate = group[
            "at_lower_bound"
        ].mean()

        upper_bound_rate = group[
            "at_upper_bound"
        ].mean()

        summary_rows.append(
            {
                "query_id": query_id,
                "description": description,
                "epsilon": epsilon,
                "repetitions": len(group),
                "exact_count": exact_count,
                "exact_mean": exact_mean,
                "mean_released_mean": mean_release,
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "std_released_mean": std,
                "median_absolute_error": (
                    median_absolute_error
                ),
                "mean_relative_absolute_error_percent": (
                    mean_relative_error * 100.0
                ),
                "lower_bound_release_rate": (
                    lower_bound_rate
                ),
                "upper_bound_release_rate": (
                    upper_bound_rate
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["epsilon", "query_id"]
    )

    # ---------------------------------------------------------
    # 4. Direct Q003 vs Q007 comparison
    # ---------------------------------------------------------

    q003_summary = (
        summary_df[
            summary_df["query_id"] == "Q003"
        ]
        .set_index("epsilon")
    )

    q007_summary = (
        summary_df[
            summary_df["query_id"] == "Q007"
        ]
        .set_index("epsilon")
    )

    comparison_rows = []

    for epsilon in EPSILONS:

        q003_mae = q003_summary.loc[
            epsilon,
            "mae",
        ]

        q007_mae = q007_summary.loc[
            epsilon,
            "mae",
        ]

        q003_rel = q003_summary.loc[
            epsilon,
            "mean_relative_absolute_error_percent",
        ]

        q007_rel = q007_summary.loc[
            epsilon,
            "mean_relative_absolute_error_percent",
        ]

        comparison_rows.append(
            {
                "epsilon": epsilon,
                "q003_count": int(
                    q003_summary.loc[
                        epsilon,
                        "exact_count",
                    ]
                ),
                "q007_count": int(
                    q007_summary.loc[
                        epsilon,
                        "exact_count",
                    ]
                ),
                "q003_exact_mean": q003_summary.loc[
                    epsilon,
                    "exact_mean",
                ],
                "q007_exact_mean": q007_summary.loc[
                    epsilon,
                    "exact_mean",
                ],
                "q003_mae": q003_mae,
                "q007_mae": q007_mae,
                "q007_vs_q003_mae_ratio": (
                    q007_mae / q003_mae
                    if q003_mae > 0
                    else float("inf")
                ),
                "q003_relative_error_percent": (
                    q003_rel
                ),
                "q007_relative_error_percent": (
                    q007_rel
                ),
                "q007_vs_q003_relative_error_ratio": (
                    q007_rel / q003_rel
                    if q003_rel > 0
                    else float("inf")
                ),
                "q007_lower_bound_rate": (
                    q007_summary.loc[
                        epsilon,
                        "lower_bound_release_rate",
                    ]
                ),
                "q007_upper_bound_rate": (
                    q007_summary.loc[
                        epsilon,
                        "upper_bound_release_rate",
                    ]
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    # ---------------------------------------------------------
    # 5. Save results
    # ---------------------------------------------------------

    output_dir = Path(
        "results/evaluation/rare_mean"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "q003_q007_rare_mean_raw.csv"
    )

    summary_path = (
        output_dir
        / "q003_q007_rare_mean_summary.csv"
    )

    comparison_path = (
        output_dir
        / "q003_q007_rare_mean_comparison.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Print concise results
    # ---------------------------------------------------------

    print()
    print("SUMMARY")
    print("=======")

    print(
        summary_df[
            [
                "query_id",
                "epsilon",
                "exact_count",
                "exact_mean",
                "bias",
                "mae",
                "rmse",
                "mean_relative_absolute_error_percent",
                "lower_bound_release_rate",
                "upper_bound_release_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("RARE-MEAN COMPARISON")
    print("====================")

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(raw_path)
    print(summary_path)
    print(comparison_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    main(
        repetitions=args.repetitions
    )