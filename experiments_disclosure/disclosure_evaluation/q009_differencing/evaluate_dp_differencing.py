"""
Q009 DP overlapping-query differencing experiment.

Two overlapping count queries are released:

    Q1 = prolonged-stay count over full cohort D
    Q2 = prolonged-stay count over D with target excluded

Exact differencing:
    Q1 - Q2 = target contribution

DP experiment:
    privatize Q1 and Q2 independently using the existing
    scalar Laplace count mechanism.

Privacy accounting:
    epsilon_total is split equally across the two releases:

        epsilon_query = epsilon_total / 2

    Sequential composition therefore gives:

        epsilon_query + epsilon_query = epsilon_total

Experimental benchmark only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from coordinator.mechanisms import (
    privatize_count,
)

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.query import (
    prolonged_stay_histogram,
)

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.synthetic_data import (
    DEFAULT_RACE,
    DEFAULT_ADMISSION_TYPE,
    TARGET_SUBJECT_ID,
    make_attribute_inference_pair,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

COHORT_SIZES = [
    5,
    10,
    20,
    50,
    100,
]

TOTAL_EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

# Start with 50 for smoke test.
# Change to 1000 after validation.
REPETITIONS_PER_CASE = 1000


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q009_differencing"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_OUTPUT = (
    RESULTS_DIR
    / "q009_dp_differencing_raw.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "q009_dp_differencing_summary.csv"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def exact_prolonged_count(
    dataframe: pd.DataFrame,
) -> int:
    """
    Return exact prolonged-stay count for the controlled subgroup.
    """

    result = prolonged_stay_histogram(
        dataframe
    )

    cell = (
        result["histogram"]
        [DEFAULT_RACE]
        [DEFAULT_ADMISSION_TYPE]
    )

    return int(
        cell["prolonged"]
    )


def calculate_auc(
    labels,
    scores,
) -> float:
    """
    ROC-AUC using rank-sum formulation.

    label:
        0 = SHORT target
        1 = LONG target

    score:
        noisy Q1 - noisy Q2

    Higher score means stronger evidence that the target
    contributed to the prolonged-stay count.
    """

    frame = pd.DataFrame(
        {
            "label": labels,
            "score": scores,
        }
    )

    positives = int(
        (frame["label"] == 1).sum()
    )

    negatives = int(
        (frame["label"] == 0).sum()
    )

    if positives == 0 or negatives == 0:
        return float("nan")

    ranks = frame["score"].rank(
        method="average"
    )

    positive_rank_sum = float(
        ranks[
            frame["label"] == 1
        ].sum()
    )

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2
    ) / (
        positives * negatives
    )

    return float(auc)


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

raw_rows = []
summary_rows = []


print("=" * 80)
print("Q009 DP OVERLAPPING-QUERY DIFFERENCING EXPERIMENT")
print("=" * 80)

print()
print(
    f"Repetitions per target state: "
    f"{REPETITIONS_PER_CASE}"
)

expected_trials = (
    len(COHORT_SIZES)
    * len(TOTAL_EPSILONS)
    * 2
    * REPETITIONS_PER_CASE
)

print(
    f"Expected attack trials: "
    f"{expected_trials}"
)

print()


for cohort_size in COHORT_SIZES:

    # -------------------------------------------------------------
    # Controlled SHORT / LONG datasets
    # -------------------------------------------------------------

    short_df, long_df = (
        make_attribute_inference_pair(
            cohort_size=cohort_size
        )
    )

    cases = [
        {
            "case": "SHORT",
            "label": 0,
            "true_contribution": 0,
            "dataframe": short_df,
        },
        {
            "case": "LONG",
            "label": 1,
            "true_contribution": 1,
            "dataframe": long_df,
        },
    ]

    print("-" * 80)
    print(
        f"Cohort n={cohort_size}"
    )

    # -------------------------------------------------------------
    # epsilon sweep
    # -------------------------------------------------------------

    for epsilon_total in TOTAL_EPSILONS:

        epsilon_per_query = (
            epsilon_total / 2.0
        )

        condition_rows = []

        for case in cases:

            full_df = case[
                "dataframe"
            ]

            without_target_df = (
                full_df[
                    full_df["subject_id"]
                    != TARGET_SUBJECT_ID
                ]
                .copy()
            )

            # -----------------------------------------------------
            # Exact Q1 and Q2
            # -----------------------------------------------------

            q1_exact = (
                exact_prolonged_count(
                    full_df
                )
            )

            q2_exact = (
                exact_prolonged_count(
                    without_target_df
                )
            )

            exact_difference = (
                q1_exact
                - q2_exact
            )

            if (
                exact_difference
                != case["true_contribution"]
            ):
                raise RuntimeError(
                    "Exact differencing validation failed."
                )

            # -----------------------------------------------------
            # Monte Carlo DP releases
            # -----------------------------------------------------

            for repetition in range(
                REPETITIONS_PER_CASE
            ):

                # Each release receives half the total epsilon.

                q1_dp = privatize_count(
                    exact_count=q1_exact,
                    epsilon=epsilon_per_query,
                    sensitivity=1.0,
                )

                q2_dp = privatize_count(
                    exact_count=q2_exact,
                    epsilon=epsilon_per_query,
                    sensitivity=1.0,
                )

                # -------------------------------------------------
                # Differencing attack
                # -------------------------------------------------

                noisy_difference = (
                    q1_dp
                    - q2_dp
                )

                # Exact hypotheses:
                #
                # SHORT -> difference = 0
                # LONG  -> difference = 1
                #
                # With equal prior probability and symmetric noise,
                # use midpoint threshold 0.5.
                #
                if noisy_difference > 0.5:
                    predicted_label = 1

                elif noisy_difference < 0.5:
                    predicted_label = 0

                else:
                    predicted_label = None

                if predicted_label is None:

                    correctness = 0.5

                elif (
                    predicted_label
                    == case["label"]
                ):

                    correctness = 1.0

                else:

                    correctness = 0.0

                q1_absolute_error = abs(
                    q1_dp
                    - q1_exact
                )

                q2_absolute_error = abs(
                    q2_dp
                    - q2_exact
                )

                difference_absolute_error = abs(
                    noisy_difference
                    - case["true_contribution"]
                )

                row = {
                    "cohort_size": cohort_size,

                    "epsilon_total": (
                        epsilon_total
                    ),

                    "epsilon_per_query": (
                        epsilon_per_query
                    ),

                    "composed_epsilon": (
                        epsilon_per_query
                        + epsilon_per_query
                    ),

                    "case": case["case"],

                    "label": case["label"],

                    "true_target_contribution": (
                        case[
                            "true_contribution"
                        ]
                    ),

                    "repetition": (
                        repetition + 1
                    ),

                    "q1_exact": q1_exact,
                    "q2_exact": q2_exact,

                    "exact_difference": (
                        exact_difference
                    ),

                    "q1_dp": q1_dp,
                    "q2_dp": q2_dp,

                    "noisy_difference": (
                        noisy_difference
                    ),

                    "attack_prediction": (
                        predicted_label
                    ),

                    "attack_correctness": (
                        correctness
                    ),

                    "q1_absolute_error": (
                        q1_absolute_error
                    ),

                    "q2_absolute_error": (
                        q2_absolute_error
                    ),

                    "difference_absolute_error": (
                        difference_absolute_error
                    ),
                }

                raw_rows.append(row)
                condition_rows.append(row)

        # ---------------------------------------------------------
        # Summarise cohort × total epsilon
        # ---------------------------------------------------------

        condition_df = pd.DataFrame(
            condition_rows
        )

        accuracy = float(
            condition_df[
                "attack_correctness"
            ].mean()
        )

        auc = calculate_auc(
            condition_df[
                "label"
            ].tolist(),
            condition_df[
                "noisy_difference"
            ].tolist(),
        )

        advantage = (
            accuracy - 0.5
        )

        n_trials = int(
            len(condition_df)
        )

        correctness_sd = float(
            condition_df[
                "attack_correctness"
            ].std(ddof=1)
        )

        standard_error = (
            correctness_sd
            / math.sqrt(n_trials)
        )

        ci_lower = max(
            0.0,
            accuracy
            - 1.96 * standard_error,
        )

        ci_upper = min(
            1.0,
            accuracy
            + 1.96 * standard_error,
        )

        mean_difference_error = float(
            condition_df[
                "difference_absolute_error"
            ].mean()
        )

        mean_q1_error = float(
            condition_df[
                "q1_absolute_error"
            ].mean()
        )

        mean_q2_error = float(
            condition_df[
                "q2_absolute_error"
            ].mean()
        )

        summary_rows.append(
            {
                "cohort_size": (
                    cohort_size
                ),

                "epsilon_total": (
                    epsilon_total
                ),

                "epsilon_per_query": (
                    epsilon_per_query
                ),

                "composed_epsilon": (
                    epsilon_per_query
                    + epsilon_per_query
                ),

                "trials": n_trials,

                "background_only_accuracy": (
                    0.5
                ),

                "exact_release_accuracy": (
                    1.0
                ),

                "dp_attack_accuracy": (
                    accuracy
                ),

                "dp_attack_accuracy_ci_lower": (
                    ci_lower
                ),

                "dp_attack_accuracy_ci_upper": (
                    ci_upper
                ),

                "dp_attack_advantage": (
                    advantage
                ),

                "dp_attack_auc": auc,

                "difference_mae": (
                    mean_difference_error
                ),

                "q1_mae": (
                    mean_q1_error
                ),

                "q2_mae": (
                    mean_q2_error
                ),

                "sensitivity_per_query": (
                    1.0
                ),

                "adjacency_model": (
                    "add_remove_patient"
                ),

                "composition": (
                    "sequential_two_releases"
                ),
            }
        )

        print(
            f"  total epsilon={epsilon_total:<4} | "
            f"per query={epsilon_per_query:<5} | "
            f"attack acc={accuracy:.4f} | "
            f"AUC={auc:.4f} | "
            f"advantage={advantage:.4f} | "
            f"diff MAE={mean_difference_error:.4f}"
        )


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

raw_df = pd.DataFrame(
    raw_rows
)

summary_df = pd.DataFrame(
    summary_rows
)

raw_df.to_csv(
    RAW_OUTPUT,
    index=False,
)

summary_df.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


print()
print("=" * 80)
print("Q009 DP DIFFERENCING EXPERIMENT COMPLETE")
print("=" * 80)

print()
print(
    f"Raw trials: "
    f"{len(raw_df)}"
)

print(
    f"Summary conditions: "
    f"{len(summary_df)}"
)

print()
print(f"Raw:     {RAW_OUTPUT}")
print(f"Summary: {SUMMARY_OUTPUT}")