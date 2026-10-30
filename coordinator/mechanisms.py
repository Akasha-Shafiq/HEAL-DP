""" 
differential privacy mechanism for MIMIC-DP platform 

The module contains the mechanism used to transform aggregate statistics in DP releases

The mechanism used is OpenDp and the accounting is handled by the diffprivlib accounting seprately

"""

import opendp.prelude as dp

# OpenDP 0.15.x requires explicit opt-in for this API.

dp.enable_features("contrib")


def privatize_count(
    exact_count: int,
    epsilon: float,
    sensitivity: float = 1.0,
) -> float:



    """
    Apply the Laplace mechanism to a scalar count.

    Parameters
    ----------
    exact_count:
        Exact aggregate count computed internally by the coordinator.

    epsilon:
        Privacy parameter epsilon. Must be strictly positive.

    sensitivity:
        L1 sensitivity of the count query. For Q001 under
        add/remove-one-patient adjacency, sensitivity is 1.

    Returns
    -------
    float
        Differentially private noisy count.
    """

    epsilon = float(epsilon)
    sensitivity = float(sensitivity)

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    if sensitivity <= 0:
        raise ValueError("sensitivity must be greater than zero.")

    scale = sensitivity / epsilon

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

    noisy_count = mechanism(
        float(exact_count)
    )

    return float(noisy_count)


# ---------------------------------------------------------------------
# Q002 — Admission-type histogram
# ---------------------------------------------------------------------

Q002_CATEGORIES = (
    "AMBULATORY OBSERVATION",
    "DIRECT EMER.",
    "DIRECT OBSERVATION",
    "ELECTIVE",
    "EU OBSERVATION",
    "EW EMER.",
    "OBSERVATION ADMIT",
    "SURGICAL SAME DAY ADMISSION",
    "URGENT",
)


Q004_RACE_CATEGORIES = (
    "AMERICAN INDIAN/ALASKA NATIVE",
    "ASIAN",
    "ASIAN - ASIAN INDIAN",
    "ASIAN - CHINESE",
    "ASIAN - KOREAN",
    "ASIAN - SOUTH EAST ASIAN",
    "BLACK/AFRICAN",
    "BLACK/AFRICAN AMERICAN",
    "BLACK/CAPE VERDEAN",
    "BLACK/CARIBBEAN ISLAND",
    "HISPANIC OR LATINO",
    "HISPANIC/LATINO - CENTRAL AMERICAN",
    "HISPANIC/LATINO - COLUMBIAN",
    "HISPANIC/LATINO - CUBAN",
    "HISPANIC/LATINO - DOMINICAN",
    "HISPANIC/LATINO - GUATEMALAN",
    "HISPANIC/LATINO - HONDURAN",
    "HISPANIC/LATINO - MEXICAN",
    "HISPANIC/LATINO - PUERTO RICAN",
    "HISPANIC/LATINO - SALVADORAN",
    "MULTIPLE RACE/ETHNICITY",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
    "OTHER",
    "PATIENT DECLINED TO ANSWER",
    "PORTUGUESE",
    "SOUTH AMERICAN",
    "UNABLE TO OBTAIN",
    "UNKNOWN",
    "WHITE",
    "WHITE - BRAZILIAN",
    "WHITE - EASTERN EUROPEAN",
    "WHITE - OTHER EUROPEAN",
    "WHITE - RUSSIAN",
)

Q004_ADMISSION_TYPES = (
    "AMBULATORY OBSERVATION",
    "DIRECT EMER.",
    "DIRECT OBSERVATION",
    "ELECTIVE",
    "EU OBSERVATION",
    "EW EMER.",
    "OBSERVATION ADMIT",
    "SURGICAL SAME DAY ADMISSION",
    "URGENT",
)
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def privatize_histogram(
    exact_histogram: dict[str, int],
    epsilon: float,
    sensitivity: float = 1.0,
) -> dict[str, int]:
    """
    Apply a vector-valued Laplace mechanism to Q002.

    The complete admission-type histogram is treated as one
    vector-valued query.

    Under add/remove-one-patient adjacency, and because each patient
    contributes to exactly one histogram bin, the L1 sensitivity of
    the complete histogram vector is 1.

    Parameters
    ----------
    exact_histogram:
        Exact global admission-type histogram.

    epsilon:
        Privacy parameter epsilon. Must be strictly positive.

    sensitivity:
        L1 sensitivity of the histogram vector. For Q002 this is 1.

    Returns
    -------
    dict[str, int]
        Differentially private histogram after safe post-processing.
    """

    epsilon = float(epsilon)
    sensitivity = float(sensitivity)

    if epsilon <= 0:
        raise ValueError("epsilon must be greater than zero.")

    if sensitivity <= 0:
        raise ValueError("sensitivity must be greater than zero.")

    expected_categories = set(Q002_CATEGORIES)
    observed_categories = set(exact_histogram)

    if observed_categories != expected_categories:
        missing = expected_categories - observed_categories
        unexpected = observed_categories - expected_categories

        raise ValueError(
            "Q002 histogram categories do not match the "
            f"approved domain. Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    exact_vector = [
        float(exact_histogram[category])
        for category in Q002_CATEGORIES
    ]

    scale = sensitivity / epsilon

    mechanism = dp.m.make_laplace(
        dp.vector_domain(
            dp.atom_domain(
                T=float,
                nan=False,
            ),
            size=len(Q002_CATEGORIES),
        ),
        dp.l1_distance(
            T=float,
        ),
        scale=scale,
    )

    noisy_vector = mechanism(
        exact_vector
    )

    # Safe DP post-processing:
    # histogram counts are represented as non-negative integers.
    return {
        category: max(
            0,
            int(round(noisy_value)),
        )
        for category, noisy_value
        in zip(Q002_CATEGORIES, noisy_vector)
    }
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def privatize_bounded_mean(
    exact_sum: float,
    exact_count: int,
    epsilon: float,
    lower: float = 0.0,
    upper: float = 30.0,
    sum_budget_fraction: float = 0.5,
) -> dict:
    """
    Privatize a bounded mean using noisy bounded sum and noisy count.

    The total privacy budget is divided between:
        1. the bounded sum, and
        2. the patient count.

    The released mean is then computed entirely by DP-safe
    post-processing.

    Parameters
    ----------
    exact_sum:
        Exact global sum of bounded LOS values.

    exact_count:
        Exact number of contributing patients.

    epsilon:
        Total privacy budget assigned to Q003.

    lower, upper:
        Approved bounds for each patient's LOS contribution.

    sum_budget_fraction:
        Fraction of epsilon allocated to the bounded sum.
        The remainder is allocated to the count.

    Returns
    -------
    dict
        Differentially private bounded mean and internal mechanism
        metadata.
    """

    exact_sum = float(exact_sum)
    exact_count = int(exact_count)
    epsilon = float(epsilon)
    lower = float(lower)
    upper = float(upper)
    sum_budget_fraction = float(sum_budget_fraction)

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    if exact_count <= 0:
        raise ValueError(
            "exact_count must be greater than zero."
        )

    if lower >= upper:
        raise ValueError(
            "lower must be smaller than upper."
        )

    if not 0.0 < sum_budget_fraction < 1.0:
        raise ValueError(
            "sum_budget_fraction must lie strictly between 0 and 1."
        )

    epsilon_sum = (
        epsilon * sum_budget_fraction
    )

    epsilon_count = (
        epsilon * (1.0 - sum_budget_fraction)
    )

    # Under add/remove-one-patient adjacency:
    # bounded-sum sensitivity = max(|lower|, |upper|)
    # For the approved Q003 bounds [0, 30], this is 30.
    sum_sensitivity = max(
        abs(lower),
        abs(upper),
    )

    count_sensitivity = 1.0

    sum_scale = (
        sum_sensitivity / epsilon_sum
    )

    count_scale = (
        count_sensitivity / epsilon_count
    )

    sum_mechanism = dp.m.make_laplace(
        dp.atom_domain(
            T=float,
            nan=False,
        ),
        dp.absolute_distance(
            T=float,
        ),
        scale=sum_scale,
    )

    count_mechanism = dp.m.make_laplace(
        dp.atom_domain(
            T=float,
            nan=False,
        ),
        dp.absolute_distance(
            T=float,
        ),
        scale=count_scale,
    )

    noisy_sum = float(
        sum_mechanism(exact_sum)
    )

    noisy_count = float(
        count_mechanism(float(exact_count))
    )

    # Avoid an unstable or invalid denominator.
    # This is DP-safe post-processing.
    safe_noisy_count = max(
        noisy_count,
        1.0,
    )

    noisy_mean = (
        noisy_sum / safe_noisy_count
    )

    # A bounded mean should remain within the approved domain.
    # Clipping uses only the DP output and is therefore post-processing.
    released_mean = min(
        upper,
        max(
            lower,
            noisy_mean,
        ),
    )

    return {
        "dp_mean": float(released_mean),
        "epsilon_total": epsilon,
        "epsilon_sum": epsilon_sum,
        "epsilon_count": epsilon_count,
        "sum_sensitivity": sum_sensitivity,
        "count_sensitivity": count_sensitivity,
    }
    
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------    
def privatize_joint_histogram(
    exact_table: dict,
    epsilon: float,
    sensitivity: float = 1.0,
) -> dict:
    """
    Privatize the Q004 race × admission-type contingency table.

    Q004 is treated as one vector-valued histogram over the fixed
    33 × 9 = 297 cell domain.

    Under add/remove-one-patient adjacency, with each patient
    contributing to exactly one joint cell, the L1 sensitivity of
    the complete histogram is 1.

    Parameters
    ----------
    exact_table:
        Nested mapping:
            race -> admission_type -> exact patient count

    epsilon:
        Total privacy budget for the complete 297-cell histogram.

    sensitivity:
        L1 sensitivity of the full histogram vector.
        For Q004 this is 1.

    Returns
    -------
    dict
        Differentially private contingency table and mechanism
        metadata.
    """

    epsilon = float(epsilon)
    sensitivity = float(sensitivity)

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    if sensitivity <= 0:
        raise ValueError(
            "sensitivity must be greater than zero."
        )

    # -------------------------------------------------------------
    # Validate the complete fixed race domain
    # -------------------------------------------------------------

    if set(exact_table.keys()) != set(Q004_RACE_CATEGORIES):
        raise ValueError(
            "Q004 race domain does not match the approved domain."
        )

    for race in Q004_RACE_CATEGORIES:

        if set(exact_table[race].keys()) != set(
            Q004_ADMISSION_TYPES
        ):
            raise ValueError(
                f"Q004 admission-type domain is invalid "
                f"for race '{race}'."
            )

    # -------------------------------------------------------------
    # Flatten the 33 × 9 table into one fixed 297-element vector
    # -------------------------------------------------------------

    exact_vector = []

    for race in Q004_RACE_CATEGORIES:
        for admission_type in Q004_ADMISSION_TYPES:

            count = int(
                exact_table[race][admission_type]
            )

            if count < 0:
                raise ValueError(
                    "Q004 exact histogram contains a negative count."
                )

            exact_vector.append(
                float(count)
            )

    expected_size = (
        len(Q004_RACE_CATEGORIES)
        * len(Q004_ADMISSION_TYPES)
    )

    if len(exact_vector) != expected_size:
        raise ValueError(
            "Q004 histogram does not contain exactly 297 cells."
        )

    # -------------------------------------------------------------
    # Laplace mechanism for one vector-valued histogram
    # -------------------------------------------------------------

    scale = sensitivity / epsilon

    mechanism = dp.m.make_laplace(
        dp.vector_domain(
            dp.atom_domain(
                T=float,
                nan=False,
            ),
            size=expected_size,
        ),
        dp.l1_distance(
            T=float,
        ),
        scale=scale,
    )

    noisy_vector = mechanism(
        exact_vector
    )

    # -------------------------------------------------------------
    # DP-safe post-processing
    # Histogram counts cannot be negative, so round and clamp
    # each noisy count to a non-negative integer.
    # -------------------------------------------------------------

    released_vector = [
        max(
            0,
            int(round(float(value))),
        )
        for value in noisy_vector
    ]

    # -------------------------------------------------------------
    # Reconstruct the fixed 33 × 9 nested table
    # -------------------------------------------------------------

    dp_table = {}

    index = 0

    for race in Q004_RACE_CATEGORIES:

        dp_table[race] = {}

        for admission_type in Q004_ADMISSION_TYPES:

            dp_table[race][admission_type] = (
                released_vector[index]
            )

            index += 1

    return {
        "dp_table": dp_table,
        "epsilon": epsilon,
        "sensitivity": sensitivity,
        "scale": scale,
        "cells": expected_size,
    }
    
    

# -------------------------------------------------------------
# -------------------------------------------------------------
    
def privatize_grouped_bounded_means(
    exact_groups: dict,
    epsilon: float,
    lower: float = 0.0,
    upper: float = 30.0,
    sum_budget_fraction: float = 0.5,
) -> dict:
    """
    Privatize Q005 bounded mean LOS by race.

    The 33 grouped bounded sums are treated as one vector-valued
    query, and the 33 grouped counts are treated as a second
    vector-valued query.

    Because each patient belongs to exactly one race group:

        L1 sensitivity of grouped sum vector   = 30
        L1 sensitivity of grouped count vector = 1

    The requested epsilon is split between the two vector queries.
    Per-race DP means are then computed entirely by post-processing.
    """

    epsilon = float(epsilon)
    lower = float(lower)
    upper = float(upper)
    sum_budget_fraction = float(sum_budget_fraction)

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    if lower >= upper:
        raise ValueError(
            "lower must be smaller than upper."
        )

    if not 0.0 < sum_budget_fraction < 1.0:
        raise ValueError(
            "sum_budget_fraction must lie strictly between 0 and 1."
        )

    # -------------------------------------------------------------
    # Validate fixed race domain
    # -------------------------------------------------------------

    if set(exact_groups.keys()) != set(Q004_RACE_CATEGORIES):
        raise ValueError(
            "Q005 race domain does not match the approved domain."
        )

    # -------------------------------------------------------------
    # Build fixed sum and count vectors
    # -------------------------------------------------------------

    sum_vector = []
    count_vector = []

    for race in Q004_RACE_CATEGORIES:

        bounded_sum = float(
            exact_groups[race]["bounded_sum"]
        )

        count = int(
            exact_groups[race]["count"]
        )

        if bounded_sum < 0:
            raise ValueError(
                "Q005 contains a negative bounded sum."
            )

        if count < 0:
            raise ValueError(
                "Q005 contains a negative count."
            )

        sum_vector.append(
            bounded_sum
        )

        count_vector.append(
            float(count)
        )

    # -------------------------------------------------------------
    # Split total epsilon
    # -------------------------------------------------------------

    epsilon_sum = (
        epsilon * sum_budget_fraction
    )

    epsilon_count = (
        epsilon * (1.0 - sum_budget_fraction)
    )

    # Under add/remove-one-patient adjacency.
    sum_sensitivity = max(
        abs(lower),
        abs(upper),
    )

    count_sensitivity = 1.0

    sum_scale = (
        sum_sensitivity / epsilon_sum
    )

    count_scale = (
        count_sensitivity / epsilon_count
    )

    # -------------------------------------------------------------
    # DP mechanism for complete 33-element sum vector
    # -------------------------------------------------------------

    sum_mechanism = dp.m.make_laplace(
        dp.vector_domain(
            dp.atom_domain(
                T=float,
                nan=False,
            ),
            size=len(Q004_RACE_CATEGORIES),
        ),
        dp.l1_distance(
            T=float,
        ),
        scale=sum_scale,
    )

    noisy_sums = sum_mechanism(
        sum_vector
    )

    # -------------------------------------------------------------
    # DP mechanism for complete 33-element count vector
    # -------------------------------------------------------------

    count_mechanism = dp.m.make_laplace(
        dp.vector_domain(
            dp.atom_domain(
                T=float,
                nan=False,
            ),
            size=len(Q004_RACE_CATEGORIES),
        ),
        dp.l1_distance(
            T=float,
        ),
        scale=count_scale,
    )

    noisy_counts = count_mechanism(
        count_vector
    )

    # -------------------------------------------------------------
    # DP-safe post-processing
    # -------------------------------------------------------------

    dp_groups = {}

    for index, race in enumerate(
        Q004_RACE_CATEGORIES
    ):

        noisy_sum = float(
            noisy_sums[index]
        )

        noisy_count = float(
            noisy_counts[index]
        )

        # Protect against zero or negative noisy denominator.
        safe_count = max(
            noisy_count,
            1.0,
        )

        noisy_mean = (
            noisy_sum / safe_count
        )

        # A bounded mean must remain within [lower, upper].
        released_mean = min(
            upper,
            max(
                lower,
                noisy_mean,
            ),
        )

        dp_groups[race] = {
            "bounded_mean": float(
                released_mean
            )
        }

    return {
        "dp_groups": dp_groups,
        "epsilon_total": epsilon,
        "epsilon_sum": epsilon_sum,
        "epsilon_count": epsilon_count,
        "sum_sensitivity": sum_sensitivity,
        "count_sensitivity": count_sensitivity,
        "sum_scale": sum_scale,
        "count_scale": count_scale,
        "groups": len(Q004_RACE_CATEGORIES),
    }