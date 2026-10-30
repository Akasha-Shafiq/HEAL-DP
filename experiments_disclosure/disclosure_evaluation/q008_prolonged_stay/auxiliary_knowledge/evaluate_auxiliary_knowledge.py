"""
Q008-B: Rare-cohort auxiliary-knowledge experiment.

Research question
-----------------
When a subgroup is small and an attacker already knows some of the
other patients' prolonged-stay statuses, how much additional
information does an exact or DP release reveal about one target?

Cohort sizes:
    n = 5, 10, 20

Attacker knowledge:
    0%, 25%, 50%, 75%, 100%
    of the OTHER patients' prolonged-stay statuses.

Target:
    one patient whose prolonged-stay status is unknown.

DP mechanism:
    same Q008 two-cell vector Laplace mechanism already validated.

The experiment compares:

    Background knowledge only
        -> 50% balanced target prior

    Background + exact release
        -> measured attack accuracy

    Background + DP release
        -> measured attack accuracy at each epsilon

The background prolonged-stay probability is read from the
previously validated Q008 exact result so that the controlled
experiment is informed by the observed workload.

This is an isolated disclosure experiment only.
It does not modify the main platform.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.dp import (
    privatize_prolonged_histogram,
)


# =====================================================================
# Configuration
# =====================================================================

COHORT_SIZES = [
    5,
    10,
    20,
]

KNOWLEDGE_FRACTIONS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

EPSILONS = [
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
]

# Start with 50.
# After successful smoke test change to 1000.
REPETITIONS_PER_CASE = 1000

RANDOM_SEED = 20260928


# =====================================================================
# Paths
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[4]

Q008_EXACT_FILE = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "results"
    / "q008_exact_results.json"
)

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "auxiliary_knowledge"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_OUTPUT = (
    RESULTS_DIR
    / "q008_auxiliary_knowledge_raw.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR
    / "q008_auxiliary_knowledge_summary.csv"
)


# =====================================================================
# Load empirical prolonged-stay prevalence
# =====================================================================

if not Q008_EXACT_FILE.exists():
    raise FileNotFoundError(
        f"Q008 exact result not found: {Q008_EXACT_FILE}"
    )


with open(
    Q008_EXACT_FILE,
    "r",
    encoding="utf-8",
) as file:

    exact_q008 = json.load(file)


BACKGROUND_PROLONGED_PROBABILITY = float(
    exact_q008["overall_prolonged_rate"]
)


if not (
    0.0
    < BACKGROUND_PROLONGED_PROBABILITY
    < 1.0
):
    raise ValueError(
        "Invalid prolonged-stay prevalence."
    )


# =====================================================================
# Random generator
# =====================================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


# =====================================================================
# Helper functions
# =====================================================================

def calculate_auc(
    labels,
    scores,
) -> float:
    """
    Compute ROC-AUC using the rank-sum definition.
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


def classify_score(
    score: float,
) -> tuple[int | None, float]:
    """
    Positive score -> target prolonged.
    Negative score -> target not prolonged.
    Zero -> tie, worth 0.5 expected accuracy.
    """

    if score > 0:
        return 1, 1.0

    if score < 0:
        return 0, 1.0

    return None, 0.5


def correctness_from_score(
    score: float,
    true_label: int,
) -> tuple[int | None, float]:
    """
    Convert an attack score into prediction and correctness.
    """

    prediction, _ = classify_score(
        score
    )

    if prediction is None:
        return None, 0.5

    if prediction == true_label:
        return prediction, 1.0

    return prediction, 0.0


# =====================================================================
# Experiment storage
# =====================================================================

raw_rows = []
summary_rows = []


print("=" * 84)
print("Q008-B RARE-COHORT AUXILIARY-KNOWLEDGE EXPERIMENT")
print("=" * 84)

print()
print(
    "Observed prolonged-stay prevalence used as "
    f"background prior: "
    f"{100 * BACKGROUND_PROLONGED_PROBABILITY:.4f}%"
)

print(
    f"Repetitions per target state: "
    f"{REPETITIONS_PER_CASE}"
)

expected_dp_trials = (
    len(COHORT_SIZES)
    * len(KNOWLEDGE_FRACTIONS)
    * len(EPSILONS)
    * 2
    * REPETITIONS_PER_CASE
)

print(
    f"Expected DP trials: "
    f"{expected_dp_trials}"
)

print()


# =====================================================================
# Main experiment
# =====================================================================

for cohort_size in COHORT_SIZES:

    number_of_other_patients = (
        cohort_size - 1
    )

    print("-" * 84)
    print(
        f"Rare cohort n={cohort_size}"
    )

    for knowledge_fraction in KNOWLEDGE_FRACTIONS:

        known_other_count = int(
            round(
                number_of_other_patients
                * knowledge_fraction
            )
        )

        known_other_count = min(
            known_other_count,
            number_of_other_patients,
        )

        unknown_other_count = (
            number_of_other_patients
            - known_other_count
        )

        actual_knowledge_fraction = (
            known_other_count
            / number_of_other_patients
        )

        # ---------------------------------------------------------
        # The attacker does not know the remaining patients'
        # statuses, but knows the background prolonged-stay rate.
        #
        # Expected unknown background contribution:
        # ---------------------------------------------------------

        expected_unknown_prolonged = (
            unknown_other_count
            * BACKGROUND_PROLONGED_PROBABILITY
        )

        expected_unknown_not_prolonged = (
            unknown_other_count
            * (
                1.0
                - BACKGROUND_PROLONGED_PROBABILITY
            )
        )

        # ---------------------------------------------------------
        # Generate a fixed trial bank for this
        # cohort × knowledge condition.
        #
        # The SAME underlying trials are used for every epsilon.
        # ---------------------------------------------------------

        base_trials = []

        for target_label in [
            0,
            1,
        ]:

            for repetition in range(
                REPETITIONS_PER_CASE
            ):

                # -------------------------------------------------
                # Attacker-known patients.
                #
                # We model each background patient's prolonged
                # status using the empirically observed Q008
                # prolonged-stay prevalence.
                # -------------------------------------------------

                known_prolonged = int(
                    rng.binomial(
                        known_other_count,
                        BACKGROUND_PROLONGED_PROBABILITY,
                    )
                )

                known_not_prolonged = (
                    known_other_count
                    - known_prolonged
                )

                # -------------------------------------------------
                # Patients whose status is unknown to the attacker.
                # -------------------------------------------------

                unknown_prolonged = int(
                    rng.binomial(
                        unknown_other_count,
                        BACKGROUND_PROLONGED_PROBABILITY,
                    )
                )

                unknown_not_prolonged = (
                    unknown_other_count
                    - unknown_prolonged
                )

                # -------------------------------------------------
                # Complete exact Q008 histogram.
                #
                # target_label:
                #   0 -> target not prolonged
                #   1 -> target prolonged
                # -------------------------------------------------

                exact_prolonged = (
                    known_prolonged
                    + unknown_prolonged
                    + target_label
                )

                exact_not_prolonged = (
                    known_not_prolonged
                    + unknown_not_prolonged
                    + (1 - target_label)
                )

                if (
                    exact_prolonged
                    + exact_not_prolonged
                    != cohort_size
                ):
                    raise RuntimeError(
                        "Synthetic cohort count mismatch."
                    )

                # -------------------------------------------------
                # Exact-release attack score.
                #
                # First remove what the attacker already knows.
                #
                # Then subtract the EXPECTED contribution of the
                # still-unknown background patients.
                #
                # What remains is evidence about the target.
                #
                # Expected:
                #
                # target not prolonged -> score around -1
                # target prolonged     -> score around +1
                # -------------------------------------------------

                exact_attack_score = (
                    (
                        exact_prolonged
                        - known_prolonged
                        - expected_unknown_prolonged
                    )
                    -
                    (
                        exact_not_prolonged
                        - known_not_prolonged
                        - expected_unknown_not_prolonged
                    )
                )

                (
                    exact_prediction,
                    exact_correctness,
                ) = correctness_from_score(
                    exact_attack_score,
                    target_label,
                )

                base_trials.append(
                    {
                        "target_label": (
                            target_label
                        ),

                        "repetition": (
                            repetition + 1
                        ),

                        "known_prolonged": (
                            known_prolonged
                        ),

                        "known_not_prolonged": (
                            known_not_prolonged
                        ),

                        "unknown_prolonged": (
                            unknown_prolonged
                        ),

                        "unknown_not_prolonged": (
                            unknown_not_prolonged
                        ),

                        "exact_prolonged": (
                            exact_prolonged
                        ),

                        "exact_not_prolonged": (
                            exact_not_prolonged
                        ),

                        "exact_attack_score": (
                            exact_attack_score
                        ),

                        "exact_prediction": (
                            exact_prediction
                        ),

                        "exact_correctness": (
                            exact_correctness
                        ),
                    }
                )

        # ---------------------------------------------------------
        # Exact-release metrics are independent of epsilon.
        # ---------------------------------------------------------

        base_df = pd.DataFrame(
            base_trials
        )

        exact_accuracy = float(
            base_df[
                "exact_correctness"
            ].mean()
        )

        exact_auc = calculate_auc(
            base_df[
                "target_label"
            ].tolist(),
            base_df[
                "exact_attack_score"
            ].tolist(),
        )

        # ---------------------------------------------------------
        # Apply DP at each epsilon to the SAME underlying trials.
        # ---------------------------------------------------------

        for epsilon in EPSILONS:

            condition_rows = []

            for trial in base_trials:

                released = (
                    privatize_prolonged_histogram(
                        not_prolonged=trial[
                            "exact_not_prolonged"
                        ],
                        prolonged=trial[
                            "exact_prolonged"
                        ],
                        epsilon=epsilon,
                        sensitivity=1.0,
                    )
                )

                released_prolonged = int(
                    released[
                        "prolonged"
                    ]
                )

                released_not_prolonged = int(
                    released[
                        "not_prolonged"
                    ]
                )

                # -------------------------------------------------
                # Same attacker.
                #
                # They know:
                #
                #   known patient statuses
                #   number of unknown patients
                #   background prevalence
                #   public DP output
                #
                # They do NOT know:
                #
                #   unknown patients' actual statuses
                #   target status
                #   sampled DP noise
                # -------------------------------------------------

                dp_attack_score = (
                    (
                        released_prolonged
                        - trial[
                            "known_prolonged"
                        ]
                        - expected_unknown_prolonged
                    )
                    -
                    (
                        released_not_prolonged
                        - trial[
                            "known_not_prolonged"
                        ]
                        - expected_unknown_not_prolonged
                    )
                )

                (
                    dp_prediction,
                    dp_correctness,
                ) = correctness_from_score(
                    dp_attack_score,
                    trial[
                        "target_label"
                    ],
                )

                exact_rate = (
                    trial[
                        "exact_prolonged"
                    ]
                    / cohort_size
                )

                released_rate = (
                    released[
                        "prolonged_rate"
                    ]
                )

                if released_rate is None:

                    rate_error = None

                else:

                    rate_error = abs(
                        released_rate
                        - exact_rate
                    )

                row = {
                    "cohort_size": (
                        cohort_size
                    ),

                    "requested_knowledge_fraction": (
                        knowledge_fraction
                    ),

                    "actual_knowledge_fraction": (
                        actual_knowledge_fraction
                    ),

                    "known_other_patients": (
                        known_other_count
                    ),

                    "unknown_other_patients": (
                        unknown_other_count
                    ),

                    "epsilon": epsilon,

                    "background_prolonged_probability": (
                        BACKGROUND_PROLONGED_PROBABILITY
                    ),

                    "target_label": (
                        trial[
                            "target_label"
                        ]
                    ),

                    "repetition": (
                        trial[
                            "repetition"
                        ]
                    ),

                    "known_prolonged": (
                        trial[
                            "known_prolonged"
                        ]
                    ),

                    "known_not_prolonged": (
                        trial[
                            "known_not_prolonged"
                        ]
                    ),

                    "unknown_prolonged": (
                        trial[
                            "unknown_prolonged"
                        ]
                    ),

                    "unknown_not_prolonged": (
                        trial[
                            "unknown_not_prolonged"
                        ]
                    ),

                    "exact_prolonged": (
                        trial[
                            "exact_prolonged"
                        ]
                    ),

                    "exact_not_prolonged": (
                        trial[
                            "exact_not_prolonged"
                        ]
                    ),

                    "exact_attack_score": (
                        trial[
                            "exact_attack_score"
                        ]
                    ),

                    "exact_correctness": (
                        trial[
                            "exact_correctness"
                        ]
                    ),

                    "released_prolonged": (
                        released_prolonged
                    ),

                    "released_not_prolonged": (
                        released_not_prolonged
                    ),

                    "dp_attack_score": (
                        dp_attack_score
                    ),

                    "dp_prediction": (
                        dp_prediction
                    ),

                    "dp_correctness": (
                        dp_correctness
                    ),

                    "rate_absolute_error": (
                        rate_error
                    ),
                }

                raw_rows.append(
                    row
                )

                condition_rows.append(
                    row
                )

            # -----------------------------------------------------
            # Summarise this condition.
            # -----------------------------------------------------

            condition_df = pd.DataFrame(
                condition_rows
            )

            dp_accuracy = float(
                condition_df[
                    "dp_correctness"
                ].mean()
            )

            dp_auc = calculate_auc(
                condition_df[
                    "target_label"
                ].tolist(),
                condition_df[
                    "dp_attack_score"
                ].tolist(),
            )

            # DP protection relative to exact release.
            #
            # Positive value means DP reduced attack accuracy.
            #
            attack_accuracy_reduction = (
                exact_accuracy
                - dp_accuracy
            )

            dp_advantage_over_background = (
                dp_accuracy
                - 0.5
            )

            exact_advantage_over_background = (
                exact_accuracy
                - 0.5
            )

            correctness_sd = float(
                condition_df[
                    "dp_correctness"
                ].std(
                    ddof=1
                )
            )

            n_trials = int(
                len(
                    condition_df
                )
            )

            standard_error = (
                correctness_sd
                / math.sqrt(
                    n_trials
                )
            )

            ci_lower = max(
                0.0,
                dp_accuracy
                - 1.96 * standard_error,
            )

            ci_upper = min(
                1.0,
                dp_accuracy
                + 1.96 * standard_error,
            )

            valid_rate_errors = (
                condition_df[
                    "rate_absolute_error"
                ]
                .dropna()
            )

            if (
                len(
                    valid_rate_errors
                )
                > 0
            ):

                rate_mae = float(
                    valid_rate_errors.mean()
                )

            else:

                rate_mae = None

            summary_rows.append(
                {
                    "cohort_size": (
                        cohort_size
                    ),

                    "requested_knowledge_fraction": (
                        knowledge_fraction
                    ),

                    "actual_knowledge_fraction": (
                        actual_knowledge_fraction
                    ),

                    "known_other_patients": (
                        known_other_count
                    ),

                    "unknown_other_patients": (
                        unknown_other_count
                    ),

                    "epsilon": (
                        epsilon
                    ),

                    "trials": (
                        n_trials
                    ),

                    "background_only_accuracy": (
                        0.5
                    ),

                    "exact_attack_accuracy": (
                        exact_accuracy
                    ),

                    "exact_attack_auc": (
                        exact_auc
                    ),

                    "dp_attack_accuracy": (
                        dp_accuracy
                    ),

                    "dp_attack_auc": (
                        dp_auc
                    ),

                    "dp_attack_accuracy_ci_lower": (
                        ci_lower
                    ),

                    "dp_attack_accuracy_ci_upper": (
                        ci_upper
                    ),

                    "exact_advantage_over_background": (
                        exact_advantage_over_background
                    ),

                    "dp_advantage_over_background": (
                        dp_advantage_over_background
                    ),

                    "attack_accuracy_reduction_due_to_dp": (
                        attack_accuracy_reduction
                    ),

                    "rate_mae": (
                        rate_mae
                    ),

                    "background_prolonged_probability": (
                        BACKGROUND_PROLONGED_PROBABILITY
                    ),
                }
            )

            print(
                f"  knowledge="
                f"{100 * actual_knowledge_fraction:5.1f}% "
                f"({known_other_count}/{number_of_other_patients}) | "
                f"epsilon={epsilon:<4} | "
                f"exact={exact_accuracy:.4f} | "
                f"DP={dp_accuracy:.4f} | "
                f"reduction={attack_accuracy_reduction:.4f}"
            )


# =====================================================================
# Save results
# =====================================================================

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


# =====================================================================
# Final output
# =====================================================================

print()
print("=" * 84)
print("Q008-B AUXILIARY-KNOWLEDGE EXPERIMENT COMPLETE")
print("=" * 84)

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