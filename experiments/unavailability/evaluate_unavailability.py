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
    / "unavailability"
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "unavailability"
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
# Load data
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
# Exact distributed statistics
# ---------------------------------------------------------------------

def compute_exact_statistics(
    hospital_data: dict[str, pd.DataFrame],
) -> dict:

    # Q006
    q006_global = 0

    for hospital_id in HOSPITAL_IDS:

        q006_global += rare_subgroup_count(
            hospital_data[hospital_id]
        )

    # Q007
    q007_sum = 0.0
    q007_count = 0

    for hospital_id in HOSPITAL_IDS:

        result = (
            rare_subgroup_bounded_los_sum_and_count(
                hospital_data[hospital_id],
                lower=0.0,
                upper=30.0,
            )
        )

        q007_sum += float(
            result["bounded_sum"]
        )

        q007_count += int(
            result["count"]
        )

    if q007_count <= 0:
        raise ValueError(
            "Q007 global count must be positive."
        )

    q007_mean = (
        q007_sum
        / q007_count
    )

    return {
        "q006_count": q006_global,
        "q007_sum": q007_sum,
        "q007_count": q007_count,
        "q007_mean": q007_mean,
    }


# ---------------------------------------------------------------------
# DP releases
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

    return max(
        0,
        int(round(noisy_count)),
    )


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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main(
    repetitions: int,
) -> None:

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Calculate exact values for every availability scenario.
    # -------------------------------------------------------------

    exact_by_scenario = {}

    for scenario_name in SCENARIOS:

        data = load_scenario(
            scenario_name
        )

        exact_by_scenario[
            scenario_name
        ] = compute_exact_statistics(
            data
        )

    # Full-availability reference.
    baseline = exact_by_scenario[
        "h5_retention_100"
    ]

    baseline_q006 = float(
        baseline["q006_count"]
    )

    baseline_q007 = float(
        baseline["q007_mean"]
    )

    # -------------------------------------------------------------
    # Exact availability-shift table
    # -------------------------------------------------------------

    exact_rows = []

    for scenario_name in SCENARIOS:

        exact = exact_by_scenario[
            scenario_name
        ]

        q006_availability_error = (
            exact["q006_count"]
            - baseline_q006
        )

        q007_availability_error = (
            exact["q007_mean"]
            - baseline_q007
        )

        exact_rows.append(
            {
                "scenario": scenario_name,

                "q006_exact_count":
                    exact["q006_count"],

                "q006_availability_error":
                    q006_availability_error,

                "q006_availability_absolute_error":
                    abs(q006_availability_error),

                "q006_availability_error_percent":
                    (
                        abs(q006_availability_error)
                        / baseline_q006
                        * 100.0
                    ),

                "q007_exact_count":
                    exact["q007_count"],

                "q007_exact_sum":
                    exact["q007_sum"],

                "q007_exact_mean":
                    exact["q007_mean"],

                "q007_availability_error":
                    q007_availability_error,

                "q007_availability_absolute_error":
                    abs(q007_availability_error),

                "q007_availability_error_percent":
                    (
                        abs(q007_availability_error)
                        / baseline_q007
                        * 100.0
                    ),
            }
        )

    exact_df = pd.DataFrame(
        exact_rows
    )

    # -------------------------------------------------------------
    # Monte Carlo DP evaluation
    # -------------------------------------------------------------

    raw_rows = []

    for scenario_name in SCENARIOS:

        exact = exact_by_scenario[
            scenario_name
        ]

        print()
        print("=" * 72)
        print(
            f"Scenario: {scenario_name}"
        )
        print("=" * 72)

        print(
            f"Q006 exact count: "
            f"{exact['q006_count']}"
        )

        print(
            f"Q007 exact: "
            f"sum={exact['q007_sum']}, "
            f"count={exact['q007_count']}, "
            f"mean={exact['q007_mean']}"
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

                # =================================================
                # Q006
                # =================================================

                released_q006 = release_q006(
                    exact_count=(
                        exact["q006_count"]
                    ),
                    epsilon=epsilon,
                )

                # Error due to DP relative to
                # currently available data.
                dp_error_q006 = (
                    released_q006
                    - exact["q006_count"]
                )

                # End-to-end error relative to the
                # original full-data truth.
                total_error_q006 = (
                    released_q006
                    - baseline_q006
                )

                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "query_id": "Q006",
                        "epsilon": epsilon,
                        "repetition": repetition,

                        "full_data_value":
                            baseline_q006,

                        "scenario_exact_value":
                            exact["q006_count"],

                        "released_value":
                            released_q006,

                        "availability_error":
                            (
                                exact["q006_count"]
                                - baseline_q006
                            ),

                        "dp_error":
                            dp_error_q006,

                        "dp_absolute_error":
                            abs(dp_error_q006),

                        "dp_squared_error":
                            dp_error_q006 ** 2,

                        "total_error":
                            total_error_q006,

                        "total_absolute_error":
                            abs(total_error_q006),

                        "total_squared_error":
                            total_error_q006 ** 2,
                    }
                )

                # =================================================
                # Q007
                # =================================================

                released_q007 = release_q007(
                    exact_sum=(
                        exact["q007_sum"]
                    ),
                    exact_count=(
                        exact["q007_count"]
                    ),
                    epsilon=epsilon,
                )

                dp_error_q007 = (
                    released_q007
                    - exact["q007_mean"]
                )

                total_error_q007 = (
                    released_q007
                    - baseline_q007
                )

                raw_rows.append(
                    {
                        "scenario": scenario_name,
                        "query_id": "Q007",
                        "epsilon": epsilon,
                        "repetition": repetition,

                        "full_data_value":
                            baseline_q007,

                        "scenario_exact_value":
                            exact["q007_mean"],

                        "released_value":
                            released_q007,

                        "availability_error":
                            (
                                exact["q007_mean"]
                                - baseline_q007
                            ),

                        "dp_error":
                            dp_error_q007,

                        "dp_absolute_error":
                            abs(dp_error_q007),

                        "dp_squared_error":
                            dp_error_q007 ** 2,

                        "total_error":
                            total_error_q007,

                        "total_absolute_error":
                            abs(total_error_q007),

                        "total_squared_error":
                            total_error_q007 ** 2,
                    }
                )

    raw_df = pd.DataFrame(
        raw_rows
    )

    # -------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------

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

        availability_error = float(
            group[
                "availability_error"
            ].iloc[0]
        )

        dp_mae = float(
            group[
                "dp_absolute_error"
            ].mean()
        )

        dp_rmse = math.sqrt(
            float(
                group[
                    "dp_squared_error"
                ].mean()
            )
        )

        total_mae = float(
            group[
                "total_absolute_error"
            ].mean()
        )

        total_rmse = math.sqrt(
            float(
                group[
                    "total_squared_error"
                ].mean()
            )
        )

        summary_rows.append(
            {
                "scenario": scenario,
                "query_id": query_id,
                "epsilon": epsilon,
                "repetitions": len(group),

                "full_data_value": float(
                    group[
                        "full_data_value"
                    ].iloc[0]
                ),

                "scenario_exact_value": float(
                    group[
                        "scenario_exact_value"
                    ].iloc[0]
                ),

                "availability_error":
                    availability_error,

                "availability_absolute_error":
                    abs(availability_error),

                "dp_bias": float(
                    group[
                        "dp_error"
                    ].mean()
                ),

                "dp_mae":
                    dp_mae,

                "dp_rmse":
                    dp_rmse,

                "total_bias": float(
                    group[
                        "total_error"
                    ].mean()
                ),

                "total_mae":
                    total_mae,

                "total_rmse":
                    total_rmse,
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

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    exact_path = (
        RESULT_ROOT
        / "unavailability_exact_shift.csv"
    )

    raw_path = (
        RESULT_ROOT
        / "unavailability_dp_raw.csv"
    )

    summary_path = (
        RESULT_ROOT
        / "unavailability_dp_summary.csv"
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

    # -------------------------------------------------------------
    # Print
    # -------------------------------------------------------------

    print()
    print("=" * 72)
    print("EXACT DATA-AVAILABILITY SHIFT")
    print("=" * 72)

    print(
        exact_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("DP + AVAILABILITY SUMMARY")
    print("=" * 72)

    print(
        summary_df[
            [
                "scenario",
                "query_id",
                "epsilon",
                "full_data_value",
                "scenario_exact_value",
                "availability_error",
                "dp_mae",
                "dp_rmse",
                "total_mae",
                "total_rmse",
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