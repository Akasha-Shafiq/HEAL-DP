import argparse
import math
from pathlib import Path

import pandas as pd

from coordinator.aggregator import (
    aggregate_q001,
    aggregate_q006,
)
from coordinator.mechanisms import privatize_count


EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]
SENSITIVITY = 1.0


def release_count(exact_count: int, epsilon: float) -> int:
    """
    Apply exactly the same count mechanism and post-processing
    used by the platform API.
    """

    noisy_count = privatize_count(
        exact_count=exact_count,
        epsilon=epsilon,
        sensitivity=SENSITIVITY,
    )

    return max(
        0,
        int(round(noisy_count)),
    )


def main(repetitions: int) -> None:

    # 1. Obtain exact distributed baselines once
    # ---------------------------------------------------------

    q001_result = aggregate_q001()
    q006_result = aggregate_q006()

    q001_exact = int(
        q001_result["global_result"]
    )

    q006_exact = int(
        q006_result["global_result"]["count"]
    )

    print("Exact baselines")
    print("----------------")
    print(f"Q001 exact count: {q001_exact}")
    print(f"Q006 exact count: {q006_exact}")
    print()

    queries = {
        "Q001": {
            "description": "All unique patients",
            "exact_count": q001_exact,
        },
        "Q006": {
            "description": (
                "ASIAN - KOREAN and URGENT"
            ),
            "exact_count": q006_exact,
        },
    }

    raw_rows = []

    
    # 2. Monte Carlo DP releases
    # ---------------------------------------------------------

    for query_id, query_info in queries.items():

        exact_count = query_info["exact_count"]

        for epsilon in EPSILONS:

            print(
                f"Running {query_id}, "
                f"epsilon={epsilon}, "
                f"repetitions={repetitions}"
            )

            for repetition in range(1, repetitions + 1):

                released_count = release_count(
                    exact_count=exact_count,
                    epsilon=epsilon,
                )

                error = released_count - exact_count
                abs_error = abs(error)
                squared_error = error ** 2

                relative_error = (
                    abs_error / exact_count
                    if exact_count > 0
                    else float("nan")
                )

                raw_rows.append(
                    {
                        "query_id": query_id,
                        "description": query_info["description"],
                        "epsilon": epsilon,
                        "repetition": repetition,
                        "exact_count": exact_count,
                        "released_count": released_count,
                        "error": error,
                        "absolute_error": abs_error,
                        "squared_error": squared_error,
                        "relative_absolute_error": relative_error,
                    }
                )

    raw_df = pd.DataFrame(raw_rows)


    # 3. Aggregate evaluation metrics
    # ---------------------------------------------------------

    summary_rows = []

    grouped = raw_df.groupby(
        ["query_id", "description", "epsilon"],
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
            "released_count"
        ].std(ddof=1)

        mean_release = group[
            "released_count"
        ].mean()

        mean_relative_error = group[
            "relative_absolute_error"
        ].mean()

        median_absolute_error = group[
            "absolute_error"
        ].median()

        zero_release_rate = (
            group["released_count"]
            .eq(0)
            .mean()
        )

        summary_rows.append(
            {
                "query_id": query_id,
                "description": description,
                "epsilon": epsilon,
                "repetitions": len(group),
                "exact_count": exact_count,
                "mean_released_count": mean_release,
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "std_released_count": std,
                "median_absolute_error": median_absolute_error,
                "mean_relative_absolute_error": (
                    mean_relative_error
                ),
                "mean_relative_absolute_error_percent": (
                    mean_relative_error * 100.0
                ),
                "zero_release_rate": zero_release_rate,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        ["epsilon", "query_id"]
    )

    
    # 4. Compare rare vs common relative error directly
    # ---------------------------------------------------------

    q001_summary = (
        summary_df[
            summary_df["query_id"] == "Q001"
        ]
        .set_index("epsilon")
    )

    q006_summary = (
        summary_df[
            summary_df["query_id"] == "Q006"
        ]
        .set_index("epsilon")
    )

    comparison_rows = []

    for epsilon in EPSILONS:

        q001_mae = q001_summary.loc[
            epsilon,
            "mae",
        ]

        q006_mae = q006_summary.loc[
            epsilon,
            "mae",
        ]

        q001_rel = q001_summary.loc[
            epsilon,
            "mean_relative_absolute_error_percent",
        ]

        q006_rel = q006_summary.loc[
            epsilon,
            "mean_relative_absolute_error_percent",
        ]

        relative_error_ratio = (
            q006_rel / q001_rel
            if q001_rel > 0
            else float("inf")
        )

        comparison_rows.append(
            {
                "epsilon": epsilon,
                "q001_exact_count": q001_exact,
                "q006_exact_count": q006_exact,
                "q001_mae": q001_mae,
                "q006_mae": q006_mae,
                "q001_mean_relative_error_percent": q001_rel,
                "q006_mean_relative_error_percent": q006_rel,
                "q006_vs_q001_relative_error_ratio": (
                    relative_error_ratio
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    
    # 5. Save results
    # ---------------------------------------------------------

    output_dir = Path(
        "results/evaluation/rarity"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        output_dir
        / "q001_q006_rarity_raw.csv"
    )

    summary_path = (
        output_dir
        / "q001_q006_rarity_summary.csv"
    )

    comparison_path = (
        output_dir
        / "q001_q006_rarity_comparison.csv"
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

    print()
    print("SUMMARY")
    print("=======")

    print(
        summary_df[
            [
                "query_id",
                "epsilon",
                "exact_count",
                "bias",
                "mae",
                "rmse",
                "mean_relative_absolute_error_percent",
                "zero_release_rate",
            ]
        ].to_string(index=False)
    )

    print()
    print("RARITY COMPARISON")
    print("=================")

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