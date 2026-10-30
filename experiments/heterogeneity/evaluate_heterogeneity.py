from pathlib import Path
import argparse
import math

import pandas as pd

from node.queries import (
    rare_subgroup_count,
    rare_subgroup_bounded_los_sum_and_count,
)

from coordinator.mechanisms import (
    privatize_count,
    privatize_bounded_mean,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "heterogeneity"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "heterogeneity"
)

SCENARIOS = [
    "h5_retention_100",
    "h5_retention_75",
    "h5_retention_50",
    "h5_retention_25",
    "h5_retention_0",
]

HOSPITAL_IDS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5",
]

EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]


# ---------------------------------------------------------------------
# Load one scenario
# ---------------------------------------------------------------------

def load_scenario(
    scenario_name: str,
) -> dict[str, pd.DataFrame]:

    hospital_data = {}

    for hospital_id in HOSPITAL_IDS:

        path = (
            SCENARIO_ROOT
            / scenario_name
            / hospital_id
            / "admissions.csv.gz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        hospital_data[hospital_id] = (
            pd.read_csv(path)
        )

    return hospital_data


# ---------------------------------------------------------------------
# Compute exact distributed Q006/Q007 statistics
# ---------------------------------------------------------------------

def compute_exact_statistics(
    hospital_data: dict[str, pd.DataFrame],
) -> dict:

    # ---------------------------------------------------------
    # Q006
    # ---------------------------------------------------------

    q006_local = {}

    for hospital_id in HOSPITAL_IDS:

        q006_local[hospital_id] = (
            rare_subgroup_count(
                hospital_data[hospital_id]
            )
        )

    q006_global = sum(
        q006_local.values()
    )

    # ---------------------------------------------------------
    # Q007
    # ---------------------------------------------------------

    q007_local = {}

    global_sum = 0.0
    global_count = 0

    for hospital_id in HOSPITAL_IDS:

        result = (
            rare_subgroup_bounded_los_sum_and_count(
                hospital_data[hospital_id],
                lower=0.0,
                upper=30.0,
            )
        )

        bounded_sum = float(
            result["bounded_sum"]
        )

        count = int(
            result["count"]
        )

        q007_local[hospital_id] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

        global_sum += bounded_sum
        global_count += count

    if global_count <= 0:
        raise ValueError(
            "Q007 global count must be positive."
        )

    global_mean = (
        global_sum
        / global_count
    )

    return {
        "q006_local": q006_local,
        "q006_global_count": q006_global,
        "q007_local": q007_local,
        "q007_global_sum": global_sum,
        "q007_global_count": global_count,
        "q007_global_mean": global_mean,
    }


# ---------------------------------------------------------------------
# Q006 DP release
# ---------------------------------------------------------------------

def release_q006(
    exact_count: int,
    epsilon: float,
) -> int:

    noisy_count = privatize_count(
        exact_count=exact_count,
        epsilon=epsilon,
        sensitivity=1.0,
    )

    # Match the Q006 API post-processing.
    released_count = max(
        0,
        int(round(noisy_count)),
    )

    return released_count


# ---------------------------------------------------------------------
# Q007 DP release
# ---------------------------------------------------------------------

def release_q007(
    exact_sum: float,
    exact_count: int,
    epsilon: float,
) -> float:

    result = privatize_bounded_mean(
        exact_sum=exact_sum,
        exact_count=exact_count,
        epsilon=epsilon,
        lower=0.0,
        upper=30.0,
        sum_budget_fraction=0.5,
    )

    return float(
        result["dp_mean"]
    )



# Main experiment
# ---------------------------------------------------------------------

def main(
    repetitions: int,
) -> None:

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_rows = []
    exact_rows = []

    for scenario_name in SCENARIOS:

        print()
        print("=" * 72)
        print(
            f"Scenario: {scenario_name}"
        )
        print("=" * 72)

        hospital_data = load_scenario(
            scenario_name
        )

        exact = compute_exact_statistics(
            hospital_data
        )

        print(
            "Q006 local counts:",
            exact["q006_local"],
        )

        print(
            "Q006 global count:",
            exact["q006_global_count"],
        )

        print(
            "Q007 global:",
            f"sum={exact['q007_global_sum']}, "
            f"count={exact['q007_global_count']}, "
            f"mean={exact['q007_global_mean']}",
        )

        exact_rows.append(
            {
                "scenario": scenario_name,
                "h5_q006_count": (
                    exact[
                        "q006_local"
                    ]["hospital_5"]
                ),
                "q006_global_count": (
                    exact[
                        "q006_global_count"
                    ]
                ),
                "q007_global_sum": (
                    exact[
                        "q007_global_sum"
                    ]
                ),
                "q007_global_count": (
                    exact[
                        "q007_global_count"
                    ]
                ),
                "q007_global_mean": (
                    exact[
                        "q007_global_mean"
                    ]
                ),
            }
        )

        for epsilon in EPSILONS:

            print(
                f"  epsilon={epsilon}, "
                f"repetitions={repetitions}"
            )

            for repetition in range(
                1,
                repetitions + 1,
            ):

                # ---------------------------------------------
                # Q006
                # ---------------------------------------------

                q006_release = release_q006(
                    exact_count=(
                        exact[
                            "q006_global_count"
                        ]
                    ),
                    epsilon=epsilon,
                )

                q006_error = (
                    q006_release
                    - exact[
                        "q006_global_count"
                    ]
                )

                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "query_id": "Q006",
                        "epsilon": epsilon,
                        "repetition": repetition,
                        "exact_value": (
                            exact[
                                "q006_global_count"
                            ]
                        ),
                        "released_value": (
                            q006_release
                        ),
                        "error": q006_error,
                        "absolute_error": abs(
                            q006_error
                        ),
                        "squared_error": (
                            q006_error ** 2
                        ),
                        "relative_absolute_error": (
                            abs(q006_error)
                            /
                            exact[
                                "q006_global_count"
                            ]
                        ),
                        "at_lower_bound": (
                            q006_release == 0
                        ),
                        "at_upper_bound": False,
                    }
                )

                # ---------------------------------------------
                # Q007
                # ---------------------------------------------

                q007_release = release_q007(
                    exact_sum=(
                        exact[
                            "q007_global_sum"
                        ]
                    ),
                    exact_count=(
                        exact[
                            "q007_global_count"
                        ]
                    ),
                    epsilon=epsilon,
                )

                q007_error = (
                    q007_release
                    - exact[
                        "q007_global_mean"
                    ]
                )

                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "query_id": "Q007",
                        "epsilon": epsilon,
                        "repetition": repetition,
                        "exact_value": (
                            exact[
                                "q007_global_mean"
                            ]
                        ),
                        "released_value": (
                            q007_release
                        ),
                        "error": q007_error,
                        "absolute_error": abs(
                            q007_error
                        ),
                        "squared_error": (
                            q007_error ** 2
                        ),
                        "relative_absolute_error": (
                            abs(q007_error)
                            /
                            exact[
                                "q007_global_mean"
                            ]
                        ),
                        "at_lower_bound": (
                            q007_release <= 0.0
                        ),
                        "at_upper_bound": (
                            q007_release >= 30.0
                        ),
                    }
                )

    # Raw results
    # -----------------------------------------------------------------

    raw_df = pd.DataFrame(
        raw_rows
    )

    exact_df = pd.DataFrame(
        exact_rows
    )

    # Summary
    # -----------------------------------------------------------------

    summary_rows = []

    grouped = raw_df.groupby(
        [
            "scenario",
            "query_id",
            "epsilon",
        ],
        sort=True,
    )

    for (
        scenario,
        query_id,
        epsilon,
    ), group in grouped:

        exact_value = float(
            group["exact_value"].iloc[0]
        )

        bias = float(
            group["error"].mean()
        )

        mae = float(
            group[
                "absolute_error"
            ].mean()
        )

        rmse = math.sqrt(
            float(
                group[
                    "squared_error"
                ].mean()
            )
        )

        std = float(
            group[
                "released_value"
            ].std(ddof=1)
        )

        relative_error = float(
            group[
                "relative_absolute_error"
            ].mean()
            * 100.0
        )

        lower_rate = float(
            group[
                "at_lower_bound"
            ].mean()
        )

        upper_rate = float(
            group[
                "at_upper_bound"
            ].mean()
        )

        summary_rows.append(
            {
                "scenario": scenario,
                "query_id": query_id,
                "epsilon": epsilon,
                "repetitions": len(group),
                "exact_value": exact_value,
                "mean_released_value": float(
                    group[
                        "released_value"
                    ].mean()
                ),
                "bias": bias,
                "mae": mae,
                "rmse": rmse,
                "std_released_value": std,
                "mean_relative_absolute_error_percent": (
                    relative_error
                ),
                "lower_bound_rate": lower_rate,
                "upper_bound_rate": upper_rate,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "query_id",
            "epsilon",
            "scenario",
        ]
    )

    # Save
    # -----------------------------------------------------------------

    exact_path = (
        RESULT_ROOT
        / "heterogeneity_exact_statistics.csv"
    )

    raw_path = (
        RESULT_ROOT
        / "heterogeneity_dp_raw.csv"
    )

    summary_path = (
        RESULT_ROOT
        / "heterogeneity_dp_summary.csv"
    )

    exact_df.to_csv(
        exact_path,
        index=False,
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    
    # Print concise summary
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print("EXACT INVARIANCE")
    print("=" * 72)

    print(
        exact_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("DP SUMMARY")
    print("=" * 72)

    print(
        summary_df[
            [
                "scenario",
                "query_id",
                "epsilon",
                "exact_value",
                "bias",
                "mae",
                "rmse",
                "mean_relative_absolute_error_percent",
                "lower_bound_rate",
                "upper_bound_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(exact_path)
    print(raw_path)
    print(summary_path)


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