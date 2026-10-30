"""
Q008 controlled attribute-inference experiment.

Question:
    Given the public DP histogram and knowledge of all other
    patients' prolonged-stay statuses, how well can an attacker
    infer whether the target patient had:

        SHORT LOS = 7 days
        LONG  LOS = 20 days

The experiment compares:

    background knowledge only -> 50% balanced baseline
    exact aggregate           -> 100% inference
    DP aggregate              -> measured inference

Important:
The SHORT/LONG comparison is a replacement of one patient's
category. Under the platform's add/remove patient adjacency,
this corresponds to two add/remove changes.

This is an isolated experimental benchmark only.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.dp import (
    privatize_prolonged_histogram,
)

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.query import (
    prolonged_stay_histogram,
)

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.synthetic_data import (
    DEFAULT_RACE,
    DEFAULT_ADMISSION_TYPE,
    make_attribute_inference_pair,
)


# ---------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------

COHORT_SIZES = [
    5,
    10,
    20,
    50,
    100,
]

EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

# Independent Monte Carlo repetitions PER hidden target state.
#
# Therefore each (cohort_size, epsilon) condition contains:
#
#     1000 SHORT
#     1000 LONG
#
# = 2000 trials.

REPETITIONS_PER_CASE = 1000


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RAW_OUTPUT = (
    RESULTS_DIR
    / "q008_attribute_inference_raw.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "q008_attribute_inference_summary.csv"
)


# ---------------------------------------------------------------------
# AUC without requiring scikit-learn
# ---------------------------------------------------------------------

def calculate_auc(
    labels,
    scores,
) -> float:
    """
    Compute ROC-AUC using the rank-sum definition.

    label:
        0 = SHORT target
        1 = LONG target

    Higher attack scores indicate stronger evidence for LONG.
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
# Store raw and summary results
# ---------------------------------------------------------------------

raw_rows = []
summary_rows = []


print("=" * 78)
print("Q008 CONTROLLED ATTRIBUTE-INFERENCE EXPERIMENT")
print("=" * 78)

print()
print(
    f"Repetitions per target state: "
    f"{REPETITIONS_PER_CASE}"
)

print(
    f"Total expected DP trials: "
    f"{len(COHORT_SIZES) * len(EPSILONS) * 2 * REPETITIONS_PER_CASE}"
)

print()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

for cohort_size in COHORT_SIZES:

    # -------------------------------------------------------------
    # Construct the controlled SHORT / LONG pair once.
    # -------------------------------------------------------------

    short_df, long_df = (
        make_attribute_inference_pair(
            cohort_size=cohort_size
        )
    )

    short_exact = prolonged_stay_histogram(
        short_df
    )

    long_exact = prolonged_stay_histogram(
        long_df
    )

    short_cell = (
        short_exact["histogram"]
        [DEFAULT_RACE]
        [DEFAULT_ADMISSION_TYPE]
    )

    long_cell = (
        long_exact["histogram"]
        [DEFAULT_RACE]
        [DEFAULT_ADMISSION_TYPE]
    )

    short_not = int(
        short_cell["not_prolonged"]
    )

    short_prolonged = int(
        short_cell["prolonged"]
    )

    long_not = int(
        long_cell["not_prolonged"]
    )

    long_prolonged = int(
        long_cell["prolonged"]
    )

    # -------------------------------------------------------------
    # Known background contribution.
    #
    # SHORT:
    #   background not-prolonged + target short
    #
    # LONG:
    #   background prolonged + target long
    #
    # Therefore:
    #
    # background prolonged     = SHORT prolonged
    # background not-prolonged = LONG not-prolonged
    # -------------------------------------------------------------

    background_prolonged = (
        short_prolonged
    )

    background_not = (
        long_not
    )

    # Controlled-pair validation.

    if (
        long_prolonged
        != background_prolonged + 1
    ):
        raise RuntimeError(
            "Unexpected LONG prolonged count."
        )

    if (
        short_not
        != background_not + 1
    ):
        raise RuntimeError(
            "Unexpected SHORT not-prolonged count."
        )

    if (
        short_prolonged
        != background_prolonged
    ):
        raise RuntimeError(
            "Unexpected SHORT prolonged count."
        )

    if (
        long_not
        != background_not
    ):
        raise RuntimeError(
            "Unexpected LONG not-prolonged count."
        )

    short_exact_rate = (
        short_prolonged / cohort_size
    )

    long_exact_rate = (
        long_prolonged / cohort_size
    )

    print("-" * 78)

    print(
        f"Cohort n={cohort_size}"
    )

    print(
        f"Known background: "
        f"not={background_not}, "
        f"prolonged={background_prolonged}"
    )

    print(
        f"SHORT exact: "
        f"[not={short_not}, "
        f"prolonged={short_prolonged}]"
    )

    print(
        f"LONG exact:  "
        f"[not={long_not}, "
        f"prolonged={long_prolonged}]"
    )

    print()

    # -------------------------------------------------------------
    # epsilon sweep
    # -------------------------------------------------------------

    for epsilon in EPSILONS:

        condition_rows = []

        # ---------------------------------------------------------
        # Balanced hidden-state experiment.
        # ---------------------------------------------------------

        cases = [
            {
                "case": "SHORT",
                "label": 0,
                "exact_not": short_not,
                "exact_prolonged": short_prolonged,
                "exact_rate": short_exact_rate,
            },
            {
                "case": "LONG",
                "label": 1,
                "exact_not": long_not,
                "exact_prolonged": long_prolonged,
                "exact_rate": long_exact_rate,
            },
        ]

        for case in cases:

            for repetition in range(
                REPETITIONS_PER_CASE
            ):

                released = (
                    privatize_prolonged_histogram(
                        not_prolonged=case[
                            "exact_not"
                        ],
                        prolonged=case[
                            "exact_prolonged"
                        ],
                        epsilon=epsilon,
                    )
                )

                released_not = int(
                    released["not_prolonged"]
                )

                released_prolonged = int(
                    released["prolonged"]
                )

                # -------------------------------------------------
                # Attack score.
                #
                # Attacker knows the background contributions.
                #
                # Under exact release:
                #
                # SHORT score = -1
                # LONG  score = +1
                #
                # Positive -> evidence for LONG
                # Negative -> evidence for SHORT
                # -------------------------------------------------

                attack_score = (
                    (
                        released_prolonged
                        - background_prolonged
                    )
                    -
                    (
                        released_not
                        - background_not
                    )
                )

                if attack_score > 0:
                    predicted_label = 1
                    tie = False

                elif attack_score < 0:
                    predicted_label = 0
                    tie = False

                else:
                    predicted_label = None
                    tie = True

                # For a balanced prior, a score tie carries
                # no directional evidence. Give it 0.5 expected
                # correctness rather than arbitrarily selecting
                # one class.

                if tie:
                    correctness = 0.5

                elif (
                    predicted_label
                    == case["label"]
                ):
                    correctness = 1.0

                else:
                    correctness = 0.0

                released_rate = (
                    released["prolonged_rate"]
                )

                if released_rate is None:
                    rate_error = None

                else:
                    rate_error = abs(
                        released_rate
                        - case["exact_rate"]
                    )

                row = {
                    "cohort_size": cohort_size,
                    "epsilon": epsilon,
                    "case": case["case"],
                    "label": case["label"],
                    "repetition": repetition + 1,

                    "background_not_prolonged": (
                        background_not
                    ),

                    "background_prolonged": (
                        background_prolonged
                    ),

                    "exact_not_prolonged": (
                        case["exact_not"]
                    ),

                    "exact_prolonged": (
                        case["exact_prolonged"]
                    ),

                    "exact_rate": (
                        case["exact_rate"]
                    ),

                    "released_not_prolonged": (
                        released_not
                    ),

                    "released_prolonged": (
                        released_prolonged
                    ),

                    "released_total": (
                        released["total"]
                    ),

                    "released_rate": (
                        released_rate
                    ),

                    "attack_score": (
                        attack_score
                    ),

                    "attack_prediction": (
                        predicted_label
                    ),

                    "attack_tie": tie,

                    "attack_correctness": (
                        correctness
                    ),

                    "rate_absolute_error": (
                        rate_error
                    ),
                }

                raw_rows.append(row)
                condition_rows.append(row)

        # ---------------------------------------------------------
        # Summarise this cohort × epsilon condition.
        # ---------------------------------------------------------

        condition_df = pd.DataFrame(
            condition_rows
        )

        labels = condition_df[
            "label"
        ].tolist()

        scores = condition_df[
            "attack_score"
        ].tolist()

        auc = calculate_auc(
            labels,
            scores,
        )

        accuracy = float(
            condition_df[
                "attack_correctness"
            ].mean()
        )

        advantage = (
            accuracy - 0.5
        )

        tie_rate = float(
            condition_df[
                "attack_tie"
            ].mean()
        )

        correctness_sd = float(
            condition_df[
                "attack_correctness"
            ].std(ddof=1)
        )

        n_trials = int(
            len(condition_df)
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

        invalid_rate_fraction = float(
            condition_df[
                "released_rate"
            ].isna().mean()
        )

        valid_rate_errors = (
            condition_df[
                "rate_absolute_error"
            ]
            .dropna()
        )

        if len(valid_rate_errors) > 0:

            rate_mae = float(
                valid_rate_errors.mean()
            )

        else:
            rate_mae = None

        summary_rows.append(
            {
                "cohort_size": cohort_size,
                "epsilon": epsilon,

                "trials": n_trials,

                "background_only_accuracy": 0.5,

                "exact_release_accuracy": 1.0,

                "dp_attack_accuracy": accuracy,

                "dp_attack_accuracy_ci_lower": (
                    ci_lower
                ),

                "dp_attack_accuracy_ci_upper": (
                    ci_upper
                ),

                "dp_attack_advantage_over_background": (
                    advantage
                ),

                "dp_attack_auc": auc,

                "attack_tie_rate": tie_rate,

                "rate_mae": rate_mae,

                "invalid_rate_fraction": (
                    invalid_rate_fraction
                ),

                "adjacency_model": (
                    "add_remove_patient"
                ),

                "attribute_pair_relation": (
                    "two_step_replacement"
                ),
            }
        )

        print(
            f"  epsilon={epsilon:<4} | "
            f"attack acc={accuracy:.4f} | "
            f"AUC={auc:.4f} | "
            f"advantage={advantage:.4f} | "
            f"ties={tie_rate:.4f} | "
            f"rate MAE="
            f"{rate_mae if rate_mae is not None else 'NA'}"
        )


# ---------------------------------------------------------------------
# Save all results
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
print("=" * 78)
print("Q008 ATTRIBUTE-INFERENCE EXPERIMENT COMPLETE")
print("=" * 78)

print(
    f"Raw trials: {len(raw_df)}"
)

print(
    f"Summary conditions: "
    f"{len(summary_df)}"
)

print()
print(f"Raw:     {RAW_OUTPUT}")
print(f"Summary: {SUMMARY_OUTPUT}")