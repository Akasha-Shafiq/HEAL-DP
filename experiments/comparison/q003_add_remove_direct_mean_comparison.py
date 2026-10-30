import argparse
import math
from pathlib import Path

import pandas as pd
import opendp.prelude as dp

from coordinator.aggregator import aggregate_q003
from coordinator.mechanisms import privatize_bounded_mean


dp.enable_features("contrib")


EPSILONS = (
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
)


def privatize_direct_mean_add_remove(
    exact_mean: float,
    epsilon: float,
    lower: float = 0.0,
    upper: float = 30.0,
) -> dict:
    """
    Experimental direct bounded-mean mechanism.

    This mechanism uses the SAME add/remove-one-patient
    adjacency model as the main platform.

    The mean query is considered over non-empty datasets.

    For a bounded mean on [lower, upper], the global
    sensitivity under add/remove adjacency over non-empty
    neighboring datasets is:

        (upper - lower) / 2

    For Q003 with bounds [0, 30]:

        sensitivity = 15

    This mechanism is for comparison only. It does not alter
    the main platform implementation.
    """

    exact_mean = float(exact_mean)
    epsilon = float(epsilon)
    lower = float(lower)
    upper = float(upper)

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    if lower >= upper:
        raise ValueError(
            "lower must be smaller than upper."
        )

    # ---------------------------------------------------------
    # Add/remove-one-patient global sensitivity
    # for a bounded mean over non-empty datasets.
    # ---------------------------------------------------------

    sensitivity = (
        upper - lower
    ) / 2.0

    scale = (
        sensitivity / epsilon
    )

    # ---------------------------------------------------------
    # Scalar Laplace mechanism
    # ---------------------------------------------------------

    mechanism = dp.m.make_laplace(
        dp.atom_domain(
            T=float,
            nan=False,
        ),
        dp.absolute_distance(
            T=float,
        ),
        scale=scale,
    )

    noisy_mean = float(
        mechanism(exact_mean)
    )

    # ---------------------------------------------------------
    # DP-safe post-processing
    # ---------------------------------------------------------

    released_mean = min(
        upper,
        max(
            lower,
            noisy_mean,
        ),
    )

    return {
        "dp_mean": float(released_mean),
        "sensitivity": sensitivity,
        "scale": scale,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--output-dir",
        default="results/comparison/q003_add_remove",
    )

    args = parser.parse_args()

    if args.repetitions <= 0:
        raise ValueError(
            "repetitions must be positive."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Obtain the already validated distributed Q003 baseline.
    # ---------------------------------------------------------

    aggregate = aggregate_q003()[
        "global_result"
    ]

    exact_sum = float(
        aggregate["bounded_sum"]
    )

    exact_count = int(
        aggregate["count"]
    )

    exact_mean = float(
        aggregate["bounded_mean"]
    )

    lower = float(
        aggregate["lower"]
    )

    upper = float(
        aggregate["upper"]
    )

    print()
    print("Q003 SAME-ADJACENCY COMPARISON")
    print("==========================================")
    print("Adjacency          : add/remove patient")
    print("Exact bounded sum  :", exact_sum)
    print("Exact count        :", exact_count)
    print("Exact bounded mean :", exact_mean)
    print("Bounds             :", lower, upper)
    print(
        "Direct sensitivity:",
        (upper - lower) / 2.0,
    )
    print("Repetitions        :", args.repetitions)
    print()

    records = []

    # ---------------------------------------------------------
    # Monte Carlo experiment
    # ---------------------------------------------------------

    for epsilon in EPSILONS:

        print(
            f"Running epsilon={epsilon} ..."
        )

        for repetition in range(
            1,
            args.repetitions + 1,
        ):

            # =================================================
            # METHOD A
            #
            # Existing platform:
            # noisy bounded sum + noisy count
            # =================================================

            current = privatize_bounded_mean(
                exact_sum=exact_sum,
                exact_count=exact_count,
                epsilon=epsilon,
                lower=lower,
                upper=upper,
                sum_budget_fraction=0.5,
            )

            current_mean = float(
                current["dp_mean"]
            )

            current_error = (
                current_mean - exact_mean
            )

            records.append(
                {
                    "method":
                        "current_sum_count",
                    "adjacency":
                        "add_remove_patient",
                    "epsilon":
                        epsilon,
                    "repetition":
                        repetition,
                    "released_mean":
                        current_mean,
                    "exact_mean":
                        exact_mean,
                    "error":
                        current_error,
                    "absolute_error":
                        abs(current_error),
                    "squared_error":
                        current_error ** 2,
                }
            )

            # =================================================
            # METHOD B
            #
            # Experimental direct mean:
            # same add/remove adjacency
            # =================================================

            direct = (
                privatize_direct_mean_add_remove(
                    exact_mean=exact_mean,
                    epsilon=epsilon,
                    lower=lower,
                    upper=upper,
                )
            )

            direct_mean = float(
                direct["dp_mean"]
            )

            direct_error = (
                direct_mean - exact_mean
            )

            records.append(
                {
                    "method":
                        "direct_mean_add_remove",
                    "adjacency":
                        "add_remove_patient",
                    "epsilon":
                        epsilon,
                    "repetition":
                        repetition,
                    "released_mean":
                        direct_mean,
                    "exact_mean":
                        exact_mean,
                    "error":
                        direct_error,
                    "absolute_error":
                        abs(direct_error),
                    "squared_error":
                        direct_error ** 2,
                }
            )

    raw = pd.DataFrame(
        records
    )

    # ---------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------

    summary_rows = []

    grouped = raw.groupby(
        [
            "method",
            "adjacency",
            "epsilon",
        ]
    )

    for (
        method,
        adjacency,
        epsilon,
    ), group in grouped:

        mean_error = float(
            group["error"].mean()
        )

        mae = float(
            group["absolute_error"].mean()
        )

        rmse = math.sqrt(
            float(
                group["squared_error"].mean()
            )
        )

        output_std = float(
            group["released_mean"].std()
        )

        summary_rows.append(
            {
                "method": method,
                "adjacency": adjacency,
                "epsilon": epsilon,
                "repetitions": len(group),
                "exact_mean": exact_mean,
                "mean_error_bias": mean_error,
                "mae": mae,
                "rmse": rmse,
                "released_mean_std": output_std,
            }
        )

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        [
            "epsilon",
            "method",
        ]
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    raw_path = (
        output_dir
        / "q003_add_remove_comparison_raw.csv"
    )

    summary_path = (
        output_dir
        / "q003_add_remove_comparison_summary.csv"
    )

    raw.to_csv(
        raw_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("SUMMARY")
    print(
        "=========================================================="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Raw results saved to:",
        raw_path,
    )

    print(
        "Summary saved to:",
        summary_path,
    )


if __name__ == "__main__":
    main()