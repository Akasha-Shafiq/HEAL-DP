"""
Paired CDP vs HE+DP utility-equivalence experiment for Q006 and Q007.

Purpose
-------
Determine whether inserting homomorphic encryption before the already
validated central-DP mechanism materially changes DP utility.

This is an OFFLINE evaluation:
- it does not call the production /dp API;
- it does not modify the privacy ledger;
- it does not consume privacy budget.

Design
------
For every Monte Carlo repetition, the CDP and HE+DP paths receive the
same Laplace-noise realization.

Therefore:

    CDP output     = plaintext statistic + DP noise
    HE+DP output  = HE-decrypted statistic + SAME DP noise

Any paired difference is therefore attributable to the HE representation,
rather than to independent DP randomness.

Q006 and Q007 plaintext/HE-decrypted reference statistics are supplied
by the researcher via a local JSON file (not committed to the repository).

The DP construction matches the validated platform:
- Q006: Laplace count, sensitivity 1.
- Q007: total epsilon split 50/50 between bounded sum and count.
- Q007 bounded-sum sensitivity = 30.
- Q007 count sensitivity = 1.
- Q007 released mean clipped to [0, 30].

NumPy is used only to generate reproducible offline Laplace draws.
The production system continues to use OpenDP.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

EPSILONS = (
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
)

N_RUNS = 1000

RANDOM_SEED = 20260921

OUTPUT_DIR = Path(
    "results/he/equivalence"
)

HE_EQUIVALENCE_TOLERANCE = 1e-4


# ---------------------------------------------------------------------
# Public mechanism settings (not dataset-specific oracle values)
# ---------------------------------------------------------------------

Q006_SENSITIVITY = 1.0
Q007_LOWER = 0.0
Q007_UPPER = 30.0

Q007_SUM_SENSITIVITY = 30.0
Q007_COUNT_SENSITIVITY = 1.0

Q007_SUM_BUDGET_FRACTION = 0.5


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def calculate_metrics(
    releases: np.ndarray,
    truth: float,
) -> dict[str, float]:
    """
    Calculate standard utility metrics relative to the true answer.
    """

    releases = np.asarray(
        releases,
        dtype=float,
    )

    errors = (
        releases
        - float(truth)
    )

    absolute_errors = np.abs(
        errors
    )

    bias = float(
        np.mean(
            errors
        )
    )

    mae = float(
        np.mean(
            absolute_errors
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    if truth == 0:
        relative_error_percent = float(
            "nan"
        )
    else:
        relative_error_percent = float(
            np.mean(
                absolute_errors
                / abs(
                    float(truth)
                )
            )
            * 100.0
        )

    return {
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "mean_relative_absolute_error_percent": (
            relative_error_percent
        ),
    }


def paired_metrics(
    cdp: np.ndarray,
    he_dp: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, float]:
    """
    Measure the difference between paired CDP and HE+DP releases.
    """

    cdp = np.asarray(
        cdp,
        dtype=float,
    )

    he_dp = np.asarray(
        he_dp,
        dtype=float,
    )

    difference = (
        he_dp
        - cdp
    )

    absolute_difference = np.abs(
        difference
    )

    return {
        "paired_mean_difference": float(
            np.mean(
                difference
            )
        ),
        "paired_mean_absolute_difference": float(
            np.mean(
                absolute_difference
            )
        ),
        "paired_max_absolute_difference": float(
            np.max(
                absolute_difference
            )
        ),
        "exact_match_rate_percent": float(
            np.mean(
                cdp == he_dp
            )
            * 100.0
        ),
        "within_tolerance_rate_percent": float(
            np.mean(
                absolute_difference
                <= tolerance
            )
            * 100.0
        ),
    }


# ---------------------------------------------------------------------
# Q006 experiment
# ---------------------------------------------------------------------

def run_q006(
    rng: np.random.Generator,
    *,
    true_count: int,
    he_decrypted_count: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Paired Q006 CDP vs HE+DP experiment.
    """

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    if (
        true_count
        != he_decrypted_count
    ):
        raise AssertionError(
            "Q006 BFV-decrypted count does not "
            "match the plaintext oracle."
        )

    for epsilon in EPSILONS:

        scale = (
            Q006_SENSITIVITY
            / epsilon
        )

        count_noise = rng.laplace(
            loc=0.0,
            scale=scale,
            size=N_RUNS,
        )

        cdp_unrounded = (
            float(
                true_count
            )
            + count_noise
        )

        he_dp_unrounded = (
            float(
                he_decrypted_count
            )
            + count_noise
        )

        # Production Q006 post-processing:
        #
        # max(0, int(round(noisy_count)))
        #
        # np.rint implements round-to-nearest;
        # exact half-way cases have probability zero for
        # continuous Laplace noise.
        cdp_release = np.maximum(
            0,
            np.rint(
                cdp_unrounded
            ),
        ).astype(
            int
        )

        he_dp_release = np.maximum(
            0,
            np.rint(
                he_dp_unrounded
            ),
        ).astype(
            int
        )

        cdp_metrics = calculate_metrics(
            cdp_release,
            true_count,
        )

        he_dp_metrics = calculate_metrics(
            he_dp_release,
            true_count,
        )

        comparison = paired_metrics(
            cdp_release,
            he_dp_release,
            tolerance=0.0,
        )

        zero_rate_cdp = float(
            np.mean(
                cdp_release == 0
            )
            * 100.0
        )

        zero_rate_he_dp = float(
            np.mean(
                he_dp_release == 0
            )
            * 100.0
        )

        summary_rows.append(
            {
                "query_id": "Q006",
                "epsilon": epsilon,
                "runs": N_RUNS,
                "laplace_scale": scale,
                "true_value": (
                    true_count
                ),
                "he_decrypted_value": (
                    he_decrypted_count
                ),
                "he_pre_dp_absolute_error": abs(
                    he_decrypted_count
                    - true_count
                ),
                "cdp_bias": (
                    cdp_metrics[
                        "bias"
                    ]
                ),
                "he_dp_bias": (
                    he_dp_metrics[
                        "bias"
                    ]
                ),
                "cdp_mae": (
                    cdp_metrics[
                        "mae"
                    ]
                ),
                "he_dp_mae": (
                    he_dp_metrics[
                        "mae"
                    ]
                ),
                "cdp_rmse": (
                    cdp_metrics[
                        "rmse"
                    ]
                ),
                "he_dp_rmse": (
                    he_dp_metrics[
                        "rmse"
                    ]
                ),
                "cdp_relative_error_percent": (
                    cdp_metrics[
                        "mean_relative_absolute_error_percent"
                    ]
                ),
                "he_dp_relative_error_percent": (
                    he_dp_metrics[
                        "mean_relative_absolute_error_percent"
                    ]
                ),
                "cdp_zero_rate_percent": (
                    zero_rate_cdp
                ),
                "he_dp_zero_rate_percent": (
                    zero_rate_he_dp
                ),
                **comparison,
            }
        )

        for run_index in range(
            N_RUNS
        ):
            raw_rows.append(
                {
                    "query_id": "Q006",
                    "epsilon": epsilon,
                    "run": (
                        run_index + 1
                    ),
                    "count_noise": float(
                        count_noise[
                            run_index
                        ]
                    ),
                    "cdp_release": int(
                        cdp_release[
                            run_index
                        ]
                    ),
                    "he_dp_release": int(
                        he_dp_release[
                            run_index
                        ]
                    ),
                    "paired_absolute_difference": float(
                        abs(
                            he_dp_release[
                                run_index
                            ]
                            - cdp_release[
                                run_index
                            ]
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(
            raw_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# ---------------------------------------------------------------------
# Q007 experiment
# ---------------------------------------------------------------------

def run_q007(
    rng: np.random.Generator,
    *,
    true_sum: float,
    he_decrypted_sum: float,
    true_count: int,
    he_decrypted_count: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Paired Q007 CDP vs HE+DP experiment.
    """

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []

    if (
        true_count
        != he_decrypted_count
    ):
        raise AssertionError(
            "Q007 BFV-decrypted count does not "
            "match the plaintext oracle."
        )

    true_mean = (
        true_sum
        / true_count
    )

    he_pre_dp_mean = (
        he_decrypted_sum
        / he_decrypted_count
    )

    he_sum_error = abs(
        he_decrypted_sum
        - true_sum
    )

    he_mean_error = abs(
        he_pre_dp_mean
        - true_mean
    )

    for epsilon in EPSILONS:

        epsilon_sum = (
            epsilon
            * Q007_SUM_BUDGET_FRACTION
        )

        epsilon_count = (
            epsilon
            * (
                1.0
                - Q007_SUM_BUDGET_FRACTION
            )
        )

        sum_scale = (
            Q007_SUM_SENSITIVITY
            / epsilon_sum
        )

        count_scale = (
            Q007_COUNT_SENSITIVITY
            / epsilon_count
        )

        sum_noise = rng.laplace(
            loc=0.0,
            scale=sum_scale,
            size=N_RUNS,
        )

        count_noise = rng.laplace(
            loc=0.0,
            scale=count_scale,
            size=N_RUNS,
        )

        cdp_noisy_sum = (
            true_sum
            + sum_noise
        )

        he_dp_noisy_sum = (
            he_decrypted_sum
            + sum_noise
        )

        # Count is exact under BFV, so both paths
        # begin from the same count and receive the
        # same paired DP noise.
        cdp_noisy_count = (
            float(
                true_count
            )
            + count_noise
        )

        he_dp_noisy_count = (
            float(
                he_decrypted_count
            )
            + count_noise
        )

        # Production bounded-mean post-processing.
        cdp_safe_count = np.maximum(
            cdp_noisy_count,
            1.0,
        )

        he_dp_safe_count = np.maximum(
            he_dp_noisy_count,
            1.0,
        )

        cdp_unclipped_mean = (
            cdp_noisy_sum
            / cdp_safe_count
        )

        he_dp_unclipped_mean = (
            he_dp_noisy_sum
            / he_dp_safe_count
        )

        cdp_release = np.clip(
            cdp_unclipped_mean,
            Q007_LOWER,
            Q007_UPPER,
        )

        he_dp_release = np.clip(
            he_dp_unclipped_mean,
            Q007_LOWER,
            Q007_UPPER,
        )

        cdp_metrics = calculate_metrics(
            cdp_release,
            true_mean,
        )

        he_dp_metrics = calculate_metrics(
            he_dp_release,
            true_mean,
        )

        comparison = paired_metrics(
            cdp_release,
            he_dp_release,
            tolerance=(
                HE_EQUIVALENCE_TOLERANCE
            ),
        )

        cdp_lower_rate = float(
            np.mean(
                cdp_release
                == Q007_LOWER
            )
            * 100.0
        )

        cdp_upper_rate = float(
            np.mean(
                cdp_release
                == Q007_UPPER
            )
            * 100.0
        )

        he_dp_lower_rate = float(
            np.mean(
                he_dp_release
                == Q007_LOWER
            )
            * 100.0
        )

        he_dp_upper_rate = float(
            np.mean(
                he_dp_release
                == Q007_UPPER
            )
            * 100.0
        )

        summary_rows.append(
            {
                "query_id": "Q007",
                "epsilon": epsilon,
                "runs": N_RUNS,
                "epsilon_sum": epsilon_sum,
                "epsilon_count": epsilon_count,
                "sum_laplace_scale": (
                    sum_scale
                ),
                "count_laplace_scale": (
                    count_scale
                ),
                "true_sum": (
                    true_sum
                ),
                "he_decrypted_sum": (
                    he_decrypted_sum
                ),
                "he_sum_absolute_error": (
                    he_sum_error
                ),
                "true_mean": (
                    true_mean
                ),
                "he_pre_dp_mean": (
                    he_pre_dp_mean
                ),
                "he_mean_absolute_error": (
                    he_mean_error
                ),
                "cdp_bias": (
                    cdp_metrics[
                        "bias"
                    ]
                ),
                "he_dp_bias": (
                    he_dp_metrics[
                        "bias"
                    ]
                ),
                "cdp_mae": (
                    cdp_metrics[
                        "mae"
                    ]
                ),
                "he_dp_mae": (
                    he_dp_metrics[
                        "mae"
                    ]
                ),
                "cdp_rmse": (
                    cdp_metrics[
                        "rmse"
                    ]
                ),
                "he_dp_rmse": (
                    he_dp_metrics[
                        "rmse"
                    ]
                ),
                "cdp_relative_error_percent": (
                    cdp_metrics[
                        "mean_relative_absolute_error_percent"
                    ]
                ),
                "he_dp_relative_error_percent": (
                    he_dp_metrics[
                        "mean_relative_absolute_error_percent"
                    ]
                ),
                "cdp_lower_boundary_rate_percent": (
                    cdp_lower_rate
                ),
                "he_dp_lower_boundary_rate_percent": (
                    he_dp_lower_rate
                ),
                "cdp_upper_boundary_rate_percent": (
                    cdp_upper_rate
                ),
                "he_dp_upper_boundary_rate_percent": (
                    he_dp_upper_rate
                ),
                **comparison,
            }
        )

        for run_index in range(
            N_RUNS
        ):
            raw_rows.append(
                {
                    "query_id": "Q007",
                    "epsilon": epsilon,
                    "run": (
                        run_index + 1
                    ),
                    "sum_noise": float(
                        sum_noise[
                            run_index
                        ]
                    ),
                    "count_noise": float(
                        count_noise[
                            run_index
                        ]
                    ),
                    "cdp_release": float(
                        cdp_release[
                            run_index
                        ]
                    ),
                    "he_dp_release": float(
                        he_dp_release[
                            run_index
                        ]
                    ),
                    "paired_absolute_difference": float(
                        abs(
                            he_dp_release[
                                run_index
                            ]
                            - cdp_release[
                                run_index
                            ]
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(
            raw_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def print_q006_summary(
    summary: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("Q006: CDP VS HE+DP PAIRED UTILITY EQUIVALENCE")
    print("=" * 78)

    columns = [
        "epsilon",
        "cdp_mae",
        "he_dp_mae",
        "cdp_relative_error_percent",
        "he_dp_relative_error_percent",
        "cdp_zero_rate_percent",
        "he_dp_zero_rate_percent",
        "paired_max_absolute_difference",
        "exact_match_rate_percent",
    ]

    print(
        summary[
            columns
        ].to_string(
            index=False
        )
    )


def print_q007_summary(
    summary: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("Q007: CDP VS HE+DP PAIRED UTILITY EQUIVALENCE")
    print("=" * 78)

    columns = [
        "epsilon",
        "cdp_mae",
        "he_dp_mae",
        "cdp_relative_error_percent",
        "he_dp_relative_error_percent",
        "cdp_lower_boundary_rate_percent",
        "he_dp_lower_boundary_rate_percent",
        "cdp_upper_boundary_rate_percent",
        "he_dp_upper_boundary_rate_percent",
        "paired_mean_absolute_difference",
        "paired_max_absolute_difference",
        "within_tolerance_rate_percent",
    ]

    print(
        summary[
            columns
        ].to_string(
            index=False
        )
    )



def load_oracle(path: Path) -> dict:
    """Load local, researcher-owned exact/HE reference values.

    The JSON file and any generated output may contain restricted derived
    statistics. Keep both outside source control and review before sharing.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Oracle file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Oracle JSON must be a mapping.")
    required = {
        "q006": ("true_count", "he_decrypted_count"),
        "q007": ("true_sum", "he_decrypted_sum", "true_count", "he_decrypted_count"),
    }
    output = {}
    for query, keys in required.items():
        value = payload.get(query)
        if not isinstance(value, dict):
            raise ValueError(f"Oracle JSON must contain an object for {query}.")
        output[query] = {}
        for key in keys:
            if key not in value:
                raise ValueError(f"Oracle JSON is missing {query}.{key}.")
            number = value[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise ValueError(f"{query}.{key} must be numeric.")
            number = float(number)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{query}.{key} must be finite and non-negative.")
            if key.endswith("count"):
                if not number.is_integer():
                    raise ValueError(f"{query}.{key} must be an integer.")
                number = int(number)
            output[query][key] = number
    if output["q007"]["true_count"] <= 0 or output["q007"]["he_decrypted_count"] <= 0:
        raise ValueError("Q007 bounded means require nonzero counts.")
    # A shared Q006/Q007 predicate means their exact counts must agree.
    if output["q006"]["true_count"] != output["q007"]["true_count"]:
        raise ValueError("Q006 and Q007 plaintext subgroup counts must agree.")
    if output["q006"]["he_decrypted_count"] != output["q007"]["he_decrypted_count"]:
        raise ValueError("Q006 and Q007 HE-decrypted subgroup counts must agree.")
    # 30-day clipped length-of-stay sum constraint.
    for kind in ("true", "he_decrypted"):
        count = output["q007"][f"{kind}_count"]
        total = output["q007"][f"{kind}_sum"]
        if total > Q007_UPPER * count + HE_EQUIVALENCE_TOLERANCE:
            raise ValueError(f"Q007 {kind} sum is outside bounded domain.")
    return output

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired offline CDP vs HE+DP evaluation for Q006/Q007."
    )
    parser.add_argument(
        "--oracle-json", required=True, type=Path,
        help="Path to local researcher-owned reference statistics (do not commit)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Local destination for CSV results; keep generated results out of Git."
    )
    args = parser.parse_args()
    oracle = load_oracle(args.oracle_json)
    q006 = oracle["q006"]
    q007 = oracle["q007"]

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("PAIRED CDP VS HE+DP UTILITY-EQUIVALENCE EXPERIMENT")
    print("=" * 78)

    print()
    print(
        f"Runs per epsilon: {N_RUNS}"
    )

    print(
        f"Epsilons: {EPSILONS}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    print(
        "Production privacy ledger: NOT USED"
    )

    print(
        "OpenDP production API: NOT CALLED"
    )

    print()
    print("Validated HE baseline:")

    print(
        "  Q006 count:"
        f" plaintext={q006['true_count']},"
        f" HE={q006['he_decrypted_count']}"
    )

    print(
        "  Q007 sum:"
        f" plaintext={q007['true_sum']},"
        f" HE={q007['he_decrypted_sum']}"
    )

    print(
        "  Q007 sum abs error:"
        f" {abs(q007['he_decrypted_sum'] - q007['true_sum'])}"
    )

    print(
        "  HE equivalence tolerance:"
        f" {HE_EQUIVALENCE_TOLERANCE}"
    )

    # Separate deterministic RNG streams keep the
    # two query experiments reproducible.
    q006_rng = np.random.default_rng(
        RANDOM_SEED
    )

    q007_rng = np.random.default_rng(
        RANDOM_SEED + 1
    )

    (
        q006_raw,
        q006_summary,
    ) = run_q006(
        q006_rng,
        **q006,
    )

    (
        q007_raw,
        q007_summary,
    ) = run_q007(
        q007_rng,
        **q007,
    )

    q006_raw_path = (
        args.output_dir
        / "q006_cdp_vs_he_dp_paired_runs.csv"
    )

    q006_summary_path = (
        args.output_dir
        / "q006_cdp_vs_he_dp_summary.csv"
    )

    q007_raw_path = (
        args.output_dir
        / "q007_cdp_vs_he_dp_paired_runs.csv"
    )

    q007_summary_path = (
        args.output_dir
        / "q007_cdp_vs_he_dp_summary.csv"
    )

    q006_raw.to_csv(
        q006_raw_path,
        index=False,
    )

    q006_summary.to_csv(
        q006_summary_path,
        index=False,
    )

    q007_raw.to_csv(
        q007_raw_path,
        index=False,
    )

    q007_summary.to_csv(
        q007_summary_path,
        index=False,
    )

    print_q006_summary(
        q006_summary
    )

    print_q007_summary(
        q007_summary
    )

    print()
    print("=" * 78)
    print("AUTOMATED EQUIVALENCE CHECKS")
    print("=" * 78)

    q006_max_difference = float(
        q006_summary[
            "paired_max_absolute_difference"
        ].max()
    )

    q006_exact_match = float(
        q006_summary[
            "exact_match_rate_percent"
        ].min()
    )

    q007_max_difference = float(
        q007_summary[
            "paired_max_absolute_difference"
        ].max()
    )

    q007_min_tolerance_rate = float(
        q007_summary[
            "within_tolerance_rate_percent"
        ].min()
    )

    q006_pass = (
        math.isclose(
            q006_max_difference,
            0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            q006_exact_match,
            100.0,
            abs_tol=0.0,
        )
    )

    q007_pass = (
        q007_max_difference
        <= HE_EQUIVALENCE_TOLERANCE
        and math.isclose(
            q007_min_tolerance_rate,
            100.0,
            abs_tol=1e-12,
        )
    )

    print(
        "Q006 exact paired equivalence:",
        "PASS" if q006_pass else "FAIL",
    )

    print(
        "Q007 paired equivalence "
        f"(tolerance={HE_EQUIVALENCE_TOLERANCE}):",
        "PASS" if q007_pass else "FAIL",
    )

    print()

    print(
        "Q006 maximum paired difference:",
        q006_max_difference,
    )

    print(
        "Q007 maximum paired difference:",
        q007_max_difference,
    )

    print()

    print("Saved files:")

    for path in (
        q006_raw_path,
        q006_summary_path,
        q007_raw_path,
        q007_summary_path,
    ):
        print(
            f"  {path}"
        )

    if not q006_pass:
        raise AssertionError(
            "Q006 CDP and HE+DP releases are not "
            "exactly equivalent under paired noise."
        )

    if not q007_pass:
        raise AssertionError(
            "Q007 HE+DP difference exceeds the "
            "configured HE-equivalence tolerance."
        )

    print()
    print(
        "Status: PASS"
    )


if __name__ == "__main__":
    main()