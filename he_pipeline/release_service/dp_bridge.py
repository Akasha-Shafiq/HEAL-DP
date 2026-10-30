"""
Bridge between homomorphic aggregation and the existing
central-DP mechanisms.

This module:

    Q001:
        encrypted global count
            ->
        trusted BFV decryption
            ->
        existing OpenDP count mechanism

    Q002:
        encrypted global 9-bin histogram
            ->
        trusted BFV decryption
            ->
        reconstruct canonical histogram
            ->
        existing OpenDP vector-valued histogram mechanism

    Q003:
        encrypted global bounded LOS sum + patient count
            ->
        trusted CKKS decryption of bounded sum
        trusted BFV decryption of patient count
            ->
        existing OpenDP bounded-mean mechanism
            ->
        DP bounded mean

    Q004:
        encrypted global 297-cell race x admission-type histogram
            ->
        trusted BFV decryption
            ->
        reconstruct canonical 33 x 9 contingency table
            ->
        existing OpenDP joint-histogram mechanism
            ->
        DP contingency table

    Q005:
        encrypted global grouped bounded-LOS sufficient statistics
            ->
        trusted CKKS decryption of 33 bounded sums
        trusted BFV decryption of 33 counts
            ->
        reconstruct canonical race-group sufficient statistics
            ->
        existing OpenDP grouped bounded-means mechanism
            ->
        33 DP bounded means by race

    Q006:
        encrypted global rare-subgroup count
            ->
        trusted BFV decryption
            ->
        existing OpenDP count mechanism
            ->
        DP rare-subgroup count

    Q007:
        encrypted global rare-subgroup bounded LOS sum + count
            ->
        trusted CKKS decryption of bounded sum
        trusted BFV decryption of count
            ->
        existing OpenDP bounded-mean mechanism
            ->
        DP rare-subgroup bounded mean

It does NOT:

    - manage the privacy budget,
    - commit privacy spend,
    - perform high-spend confirmation,
    - write audit events.

Those remain coordinator responsibilities.
"""

import math
import os
from pathlib import Path

from coordinator.mechanisms import (
    Q002_CATEGORIES,
    privatize_bounded_mean,
    privatize_count,
    privatize_histogram,
    privatize_joint_histogram,
    privatize_grouped_bounded_means,
)

from he_pipeline.common.serialization import (
    base64_to_bytes,
)

from he_pipeline.crypto.interface import (
    decrypt_bfv,
    decrypt_ckks,
)

from he_pipeline.key_service.key_manager import (
    load_context,
)

from he_pipeline.node_service.query_adapter import (
    Q004_ADMISSION_TYPE_ORDER,
    Q004_RACE_ORDER,
    Q004_VECTOR_LENGTH,
    Q005_LOWER_BOUND,
    Q005_RACE_ORDER,
    Q005_UPPER_BOUND,
    Q005_VECTOR_LENGTH,
    Q006_ADMISSION_TYPE,
    Q006_RACE,
    Q007_ADMISSION_TYPE,
    Q007_LOWER_BOUND,
    Q007_RACE,
    Q007_UPPER_BOUND,
    q004_vector_to_table,
    q005_vectors_to_groups,
)



# -------------------------------------------------------------
# Private HE context configuration
# -------------------------------------------------------------
# Private contexts are accessible only within the trusted
# release service and must not be included in the repository.

PRIVATE_BFV_CONTEXT_PATH = Path(
    os.getenv(
        "HE_PRIVATE_CONTEXT_PATH",
        ".he_runtime/private/bfv_private_context.bin",
    )
)

PRIVATE_CKKS_CONTEXT_PATH = Path(
    os.getenv(
        "HE_PRIVATE_CKKS_CONTEXT_PATH",
        ".he_runtime/private/ckks_private_context.bin",
    )
)



# Backward-compatible alias retained for existing Q001/Q002 code/tests.
PRIVATE_CONTEXT_PATH = PRIVATE_BFV_CONTEXT_PATH


# ---------------------------------------------------------------------
# Approved Q003 contribution bounds
# ---------------------------------------------------------------------

Q003_LOWER_BOUND = 0.0
Q003_UPPER_BOUND = 30.0

Q003_SUM_BUDGET_FRACTION = 0.5


# ---------------------------------------------------------------------
# Approved Q005 contribution bounds / privacy split
# ---------------------------------------------------------------------

Q005_SUM_BUDGET_FRACTION = 0.5

Q005_CKKS_ABSOLUTE_TOLERANCE = 1e-4


# ---------------------------------------------------------------------
# Approved Q006 / Q007 rare-subgroup privacy parameters
# ---------------------------------------------------------------------

Q006_SENSITIVITY = 1.0

Q007_SUM_BUDGET_FRACTION = 0.5
Q007_CKKS_ABSOLUTE_TOLERANCE = 1e-4


# =====================================================================
# Q001
# =====================================================================

def privatize_encrypted_q001(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q001 count and apply the existing
    OpenDP Laplace mechanism.

    Q001 sensitivity = 1 under add/remove-one-patient adjacency.
    """

    private_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_context.is_private():
        raise RuntimeError(
            "Q001 DP release requires the private BFV context."
        )

    ciphertext = base64_to_bytes(
        ciphertext_base64
    )

    decrypted = decrypt_bfv(
        private_context,
        ciphertext,
    )

    if len(decrypted) != 1:
        raise ValueError(
            "Q001 encrypted statistic must contain exactly "
            "one BFV slot."
        )

    exact_count = int(
        decrypted[0]
    )

    noisy_count = privatize_count(
        exact_count=exact_count,
        epsilon=epsilon,
        sensitivity=1.0,
    )

    # Same DP-safe post-processing as the validated CDP endpoint.
    released_count = max(
        0,
        int(round(noisy_count)),
    )

    return {
        "query_id": "Q001",
        "mechanism": "laplace",
        "epsilon": float(epsilon),
        "delta": 0.0,
        "sensitivity": 1.0,
        "dp_result": released_count,
    }


# =====================================================================
# Q002
# =====================================================================

def privatize_encrypted_q002(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q002 histogram and apply the
    existing validated OpenDP vector-valued Laplace mechanism.

    Q002 is represented as one nine-element vector in
    ``Q002_CATEGORIES`` order.

    Under add/remove-one-patient adjacency, each patient contributes
    to exactly one histogram bin, so the L1 sensitivity of the entire
    vector is 1.
    """

    private_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_context.is_private():
        raise RuntimeError(
            "Q002 DP release requires the private BFV context."
        )

    ciphertext = base64_to_bytes(
        ciphertext_base64
    )

    decrypted = decrypt_bfv(
        private_context,
        ciphertext,
    )

    expected_length = len(
        Q002_CATEGORIES
    )

    if len(decrypted) != expected_length:
        raise ValueError(
            "Q002 encrypted statistic has an invalid vector "
            f"length. Expected {expected_length}, "
            f"received {len(decrypted)}."
        )

    exact_vector = [
        int(value)
        for value in decrypted
    ]

    if any(
        value < 0
        for value in exact_vector
    ):
        raise ValueError(
            "Q002 decrypted histogram contains a negative count."
        )

    exact_histogram = {
        category: count
        for category, count in zip(
            Q002_CATEGORIES,
            exact_vector,
        )
    }

    released_histogram = privatize_histogram(
        exact_histogram=exact_histogram,
        epsilon=epsilon,
        sensitivity=1.0,
    )

    return {
        "query_id": "Q002",
        "mechanism": "laplace_vector",
        "epsilon": float(epsilon),
        "delta": 0.0,
        "sensitivity": 1.0,
        "vector_length": expected_length,
        "category_order": list(
            Q002_CATEGORIES
        ),
        "dp_result": released_histogram,
    }


# =====================================================================
# Q003
# =====================================================================

def privatize_encrypted_q003(
    bounded_sum_ciphertext_base64: str,
    count_ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q003 sufficient statistics and
    apply the existing validated bounded-mean DP mechanism.

    Hybrid HE representation:

        bounded LOS sum -> CKKS
        patient count   -> BFV

    Approved Q003 semantics:

        lower bound = 0 days
        upper bound = 30 days

    The total epsilon is split equally between:

        1. bounded LOS sum
        2. patient count

    The DP bounded mean is then obtained entirely through
    post-processing of the two noisy sufficient statistics.

    Exact decrypted statistics are never returned.
    """

    # -----------------------------------------------------------------
    # Load trusted private contexts.
    # -----------------------------------------------------------------

    private_ckks_context = load_context(
        PRIVATE_CKKS_CONTEXT_PATH
    )

    private_bfv_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_ckks_context.is_private():
        raise RuntimeError(
            "Q003 DP release requires the private CKKS context."
        )

    if not private_bfv_context.is_private():
        raise RuntimeError(
            "Q003 DP release requires the private BFV context."
        )


    # -----------------------------------------------------------------
    # Decode encrypted global sufficient statistics.
    # -----------------------------------------------------------------

    bounded_sum_ciphertext = base64_to_bytes(
        bounded_sum_ciphertext_base64
    )

    count_ciphertext = base64_to_bytes(
        count_ciphertext_base64
    )


    # -----------------------------------------------------------------
    # Trusted CKKS decryption of bounded sum.
    # -----------------------------------------------------------------

    decrypted_sum = decrypt_ckks(
        private_ckks_context,
        bounded_sum_ciphertext,
    )

    if len(decrypted_sum) != 1:
        raise ValueError(
            "Q003 encrypted bounded sum must contain exactly "
            "one CKKS slot."
        )

    exact_sum = float(
        decrypted_sum[0]
    )


    # -----------------------------------------------------------------
    # Trusted BFV decryption of patient count.
    # -----------------------------------------------------------------

    decrypted_count = decrypt_bfv(
        private_bfv_context,
        count_ciphertext,
    )

    if len(decrypted_count) != 1:
        raise ValueError(
            "Q003 encrypted patient count must contain exactly "
            "one BFV slot."
        )

    exact_count = int(
        decrypted_count[0]
    )


    # -----------------------------------------------------------------
    # Validate decrypted sufficient statistics.
    # -----------------------------------------------------------------

    if not math.isfinite(
        exact_sum
    ):
        raise ValueError(
            "Q003 decrypted bounded sum is not finite."
        )

    if exact_sum < 0.0:
        raise ValueError(
            "Q003 decrypted bounded sum cannot be negative."
        )

    if exact_count <= 0:
        raise ValueError(
            "Q003 decrypted patient count must be greater than zero."
        )

    maximum_possible_sum = (
        Q003_UPPER_BOUND
        * exact_count
    )

    # CKKS arithmetic is approximate, so retain a tiny tolerance
    # around the theoretical maximum.
    if exact_sum > (
        maximum_possible_sum
        + 1e-4
    ):
        raise ValueError(
            "Q003 decrypted bounded sum exceeds the "
            "approved contribution bound."
        )


    # -----------------------------------------------------------------
    # Apply the EXISTING validated OpenDP bounded-mean mechanism.
    #
    # The mechanism handles:
    #
    #   epsilon_sum   = 0.5 * epsilon
    #   epsilon_count = 0.5 * epsilon
    #
    #   sum sensitivity   = 30
    #   count sensitivity = 1
    #
    # The released mean is DP-safe post-processing.
    # -----------------------------------------------------------------

    dp_result = privatize_bounded_mean(
        exact_sum=exact_sum,
        exact_count=exact_count,
        epsilon=epsilon,
        lower=Q003_LOWER_BOUND,
        upper=Q003_UPPER_BOUND,
        sum_budget_fraction=(
            Q003_SUM_BUDGET_FRACTION
        ),
    )


    # -----------------------------------------------------------------
    # Return only DP-safe information.
    #
    # Do NOT return:
    #
    #   exact_sum
    #   exact_count
    #   exact_mean
    #   noisy_sum
    #   noisy_count
    # -----------------------------------------------------------------

    return {
        "query_id": "Q003",
        "mechanism": "laplace_sum_and_count",
        "epsilon": float(epsilon),
        "delta": 0.0,

        "bounds": {
            "lower": Q003_LOWER_BOUND,
            "upper": Q003_UPPER_BOUND,
            "unit": "days",
        },

        "budget_allocation": {
            "epsilon_sum": float(
                dp_result["epsilon_sum"]
            ),
            "epsilon_count": float(
                dp_result["epsilon_count"]
            ),
        },

        "sensitivities": {
            "sum": float(
                dp_result["sum_sensitivity"]
            ),
            "count": float(
                dp_result["count_sensitivity"]
            ),
        },

        "dp_result": {
            "bounded_mean": float(
                dp_result["dp_mean"]
            ),
            "unit": "days",
        },
    }

# =====================================================================
# Q004
# =====================================================================

def privatize_encrypted_q004(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q004 contingency-table vector and
    apply the existing validated OpenDP joint-histogram mechanism.

    Q004 HE representation:

        canonical 297-element integer vector -> BFV

    Canonical structure:

        33 race categories
            x
        9 admission types
            =
        297 joint histogram cells

    Each patient contributes to exactly one joint cell, determined by
    the patient's earliest admission. Under add/remove-one-patient
    adjacency, the L1 sensitivity of the complete 297-cell vector is 1.

    Exact decrypted counts are used only inside this trusted release
    boundary and are never returned.
    """

    # -----------------------------------------------------------------
    # Load trusted private BFV context.
    # -----------------------------------------------------------------

    private_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_context.is_private():
        raise RuntimeError(
            "Q004 DP release requires the private BFV context."
        )


    # -----------------------------------------------------------------
    # Decode and decrypt the encrypted global 297-cell vector.
    # -----------------------------------------------------------------

    ciphertext = base64_to_bytes(
        ciphertext_base64
    )

    decrypted = decrypt_bfv(
        private_context,
        ciphertext,
    )


    # -----------------------------------------------------------------
    # Validate the canonical decrypted representation.
    # -----------------------------------------------------------------

    if len(decrypted) != Q004_VECTOR_LENGTH:
        raise ValueError(
            "Q004 encrypted statistic has an invalid vector "
            f"length. Expected {Q004_VECTOR_LENGTH}, "
            f"received {len(decrypted)}."
        )

    exact_vector = [
        int(value)
        for value in decrypted
    ]

    if any(
        value < 0
        for value in exact_vector
    ):
        raise ValueError(
            "Q004 decrypted contingency vector contains "
            "a negative count."
        )

    # Reconstruct the exact canonical 33 x 9 table inside the trusted
    # boundary. This helper also validates the fixed slot semantics.
    exact_table = q004_vector_to_table(
        exact_vector
    )


    # -----------------------------------------------------------------
    # Apply the EXISTING validated OpenDP joint-histogram mechanism.
    #
    # Q004 is one vector-valued query with:
    #
    #     cells          = 297
    #     L1 sensitivity = 1
    #
    # One total epsilon is used for the whole vector-valued release.
    # -----------------------------------------------------------------

    dp_result = privatize_joint_histogram(
        exact_table=exact_table,
        epsilon=epsilon,
        sensitivity=1.0,
    )


    # -----------------------------------------------------------------
    # Validate mechanism output metadata.
    # -----------------------------------------------------------------

    required_fields = {
        "sensitivity",
        "cells",
        "dp_table",
    }

    missing_fields = (
        required_fields
        - set(dp_result.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q004 DP mechanism returned incomplete output: "
            f"{sorted(missing_fields)}"
        )

    if int(dp_result["cells"]) != Q004_VECTOR_LENGTH:
        raise ValueError(
            "Q004 DP mechanism returned an unexpected "
            "number of cells."
        )


    # -----------------------------------------------------------------
    # Return only DP-safe information.
    #
    # Do NOT return:
    #
    #     exact_vector
    #     exact_table
    #     exact cell counts
    #     plaintext global patient total
    # -----------------------------------------------------------------

    return {
        "query_id": "Q004",
        "mechanism": "laplace_joint_histogram",
        "epsilon": float(epsilon),
        "delta": 0.0,
        "sensitivity": float(
            dp_result["sensitivity"]
        ),
        "cells": int(
            dp_result["cells"]
        ),
        "dimensions": {
            "race_categories": len(
                Q004_RACE_ORDER
            ),
            "admission_types": len(
                Q004_ADMISSION_TYPE_ORDER
            ),
        },
        "race_order": list(
            Q004_RACE_ORDER
        ),
        "admission_type_order": list(
            Q004_ADMISSION_TYPE_ORDER
        ),
        "dp_result": {
            "contingency_table": (
                dp_result["dp_table"]
            ),
        },
    }

# =====================================================================
# Q005
# =====================================================================

def privatize_encrypted_q005(
    bounded_sum_ciphertext_base64: str,
    count_ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q005 grouped sufficient statistics and
    apply the existing validated OpenDP grouped bounded-means mechanism.

    Hybrid HE representation
    ------------------------
    Q005 uses two canonical 33-element vectors in exactly the same race
    order:

        bounded LOS sums -> CKKS
        patient counts   -> BFV

    Approved Q005 semantics
    -----------------------
        lower bound = 0 days
        upper bound = 30 days
        groups      = 33 race categories

    Privacy semantics
    -----------------
    The requested epsilon is split equally between:

        1. the complete 33-element bounded-sum vector
        2. the complete 33-element count vector

    Under add/remove-one-patient adjacency:

        sum-vector L1 sensitivity   = 30
        count-vector L1 sensitivity = 1

    Exact decrypted sufficient statistics remain inside this trusted
    boundary and are never returned.
    """

    # -----------------------------------------------------------------
    # Load trusted PRIVATE HE contexts.
    # -----------------------------------------------------------------

    private_ckks_context = load_context(
        PRIVATE_CKKS_CONTEXT_PATH
    )

    private_bfv_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_ckks_context.is_private():
        raise RuntimeError(
            "Q005 DP release requires the private CKKS context."
        )

    if not private_bfv_context.is_private():
        raise RuntimeError(
            "Q005 DP release requires the private BFV context."
        )


    # -----------------------------------------------------------------
    # Decode encrypted global sufficient statistics.
    # -----------------------------------------------------------------

    bounded_sum_ciphertext = base64_to_bytes(
        bounded_sum_ciphertext_base64
    )

    count_ciphertext = base64_to_bytes(
        count_ciphertext_base64
    )


    # -----------------------------------------------------------------
    # Trusted CKKS decryption of the 33 bounded-sum slots.
    # -----------------------------------------------------------------

    decrypted_sums = decrypt_ckks(
        private_ckks_context,
        bounded_sum_ciphertext,
    )

    if len(
        decrypted_sums
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 encrypted bounded-sum statistic has an invalid "
            f"vector length. Expected {Q005_VECTOR_LENGTH}, "
            f"received {len(decrypted_sums)}."
        )

    exact_sums = [
        float(
            value
        )
        for value in decrypted_sums
    ]


    # -----------------------------------------------------------------
    # Trusted BFV decryption of the 33 count slots.
    # -----------------------------------------------------------------

    decrypted_counts = decrypt_bfv(
        private_bfv_context,
        count_ciphertext,
    )

    if len(
        decrypted_counts
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 encrypted count statistic has an invalid "
            f"vector length. Expected {Q005_VECTOR_LENGTH}, "
            f"received {len(decrypted_counts)}."
        )

    exact_counts = []

    for value in decrypted_counts:

        if isinstance(
            value,
            bool,
        ):
            raise ValueError(
                "Q005 decrypted counts must be integers, "
                "not booleans."
            )

        count = int(
            value
        )

        if count != value:
            raise ValueError(
                "Q005 decrypted counts must be exact integers."
            )

        exact_counts.append(
            count
        )


    # -----------------------------------------------------------------
    # Strict validation of decrypted sufficient statistics.
    #
    # CKKS is approximate, so a small tolerance is retained around
    # theoretical lower/upper sum bounds.
    # -----------------------------------------------------------------

    validated_sums = []

    for index, race in enumerate(
        Q005_RACE_ORDER
    ):

        exact_sum = exact_sums[
            index
        ]

        exact_count = exact_counts[
            index
        ]

        if not math.isfinite(
            exact_sum
        ):
            raise ValueError(
                "Q005 decrypted bounded sum is not finite for "
                f"race {race!r}."
            )

        if exact_count < 0:
            raise ValueError(
                "Q005 decrypted count cannot be negative for "
                f"race {race!r}."
            )

        minimum_possible_sum = (
            Q005_LOWER_BOUND
            * exact_count
        )

        maximum_possible_sum = (
            Q005_UPPER_BOUND
            * exact_count
        )

        if exact_sum < (
            minimum_possible_sum
            - Q005_CKKS_ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                "Q005 decrypted bounded sum is below the "
                "approved contribution bound for "
                f"race {race!r}."
            )

        if exact_sum > (
            maximum_possible_sum
            + Q005_CKKS_ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                "Q005 decrypted bounded sum exceeds the "
                "approved contribution bound for "
                f"race {race!r}."
            )

        if (
            exact_count == 0
            and abs(
                exact_sum
            ) > Q005_CKKS_ABSOLUTE_TOLERANCE
        ):
            raise ValueError(
                "Q005 zero-count race group has a non-zero "
                f"bounded sum for race {race!r}."
            )

        # CKKS is approximate. Once the decrypted value has passed the
        # tolerance check above, clamp only tiny numerical excursions
        # back onto the theoretically valid bounded-sum interval.
        bounded_sum = min(
            maximum_possible_sum,
            max(
                minimum_possible_sum,
                exact_sum,
            ),
        )

        validated_sums.append(
            float(
                bounded_sum
            )
        )


    # -----------------------------------------------------------------
    # Reconstruct canonical grouped sufficient statistics.
    #
    # This helper binds slot i in both vectors to the same race.
    # -----------------------------------------------------------------

    exact_groups = q005_vectors_to_groups(
        validated_sums,
        exact_counts,
    )


    # -----------------------------------------------------------------
    # Apply the EXISTING validated OpenDP grouped bounded-means
    # mechanism.
    #
    # The mechanism handles:
    #
    #     epsilon_sum   = 0.5 * epsilon
    #     epsilon_count = 0.5 * epsilon
    #
    #     sum-vector L1 sensitivity   = 30
    #     count-vector L1 sensitivity = 1
    #
    # The per-race means are DP-safe post-processing.
    # -----------------------------------------------------------------

    dp_result = privatize_grouped_bounded_means(
        exact_groups=exact_groups,
        epsilon=epsilon,
        lower=Q005_LOWER_BOUND,
        upper=Q005_UPPER_BOUND,
        sum_budget_fraction=(
            Q005_SUM_BUDGET_FRACTION
        ),
    )


    # -----------------------------------------------------------------
    # Validate mechanism output before crossing the release boundary.
    # -----------------------------------------------------------------

    required_fields = {
        "dp_groups",
        "epsilon_total",
        "epsilon_sum",
        "epsilon_count",
        "sum_sensitivity",
        "count_sensitivity",
        "sum_scale",
        "count_scale",
        "groups",
    }

    missing_fields = (
        required_fields
        - set(dp_result.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q005 DP mechanism returned incomplete output: "
            f"{sorted(missing_fields)}"
        )

    if int(
        dp_result[
            "groups"
        ]
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 DP mechanism returned an unexpected "
            "number of race groups."
        )

    returned_dp_groups = dp_result[
        "dp_groups"
    ]

    if not isinstance(
        returned_dp_groups,
        dict,
    ):
        raise ValueError(
            "Q005 DP mechanism returned an invalid group object."
        )

    if set(
        returned_dp_groups.keys()
    ) != set(
        Q005_RACE_ORDER
    ):
        raise ValueError(
            "Q005 DP mechanism returned an unexpected race domain."
        )

    released_groups = {}

    for race in Q005_RACE_ORDER:

        group_result = returned_dp_groups[
            race
        ]

        if not isinstance(
            group_result,
            dict,
        ):
            raise ValueError(
                "Q005 DP group result must be a dictionary."
            )

        if set(
            group_result.keys()
        ) != {
            "bounded_mean",
        }:
            raise ValueError(
                "Q005 DP group result contains unexpected fields "
                f"for race {race!r}."
            )

        bounded_mean = float(
            group_result[
                "bounded_mean"
            ]
        )

        if not math.isfinite(
            bounded_mean
        ):
            raise ValueError(
                "Q005 DP mechanism returned a non-finite "
                f"bounded mean for race {race!r}."
            )

        if not (
            Q005_LOWER_BOUND
            <= bounded_mean
            <= Q005_UPPER_BOUND
        ):
            raise ValueError(
                "Q005 DP mechanism returned a bounded mean "
                f"outside [0, 30] for race {race!r}."
            )

        released_groups[
            race
        ] = {
            "bounded_mean": bounded_mean,
        }


    # -----------------------------------------------------------------
    # Return only DP-safe information.
    #
    # Do NOT return:
    #
    #     exact_sums
    #     exact_counts
    #     exact_groups
    #     exact per-race means
    #     noisy sum vector
    #     noisy count vector
    # -----------------------------------------------------------------

    return {
        "query_id": "Q005",
        "mechanism": (
            "laplace_grouped_sum_and_count"
        ),
        "epsilon": float(
            epsilon
        ),
        "delta": 0.0,

        "bounds": {
            "lower": Q005_LOWER_BOUND,
            "upper": Q005_UPPER_BOUND,
            "unit": "days",
        },

        "groups": Q005_VECTOR_LENGTH,

        "race_order": list(
            Q005_RACE_ORDER
        ),

        "budget_allocation": {
            "epsilon_sum": float(
                dp_result[
                    "epsilon_sum"
                ]
            ),
            "epsilon_count": float(
                dp_result[
                    "epsilon_count"
                ]
            ),
        },

        "sensitivities": {
            "sum_vector": float(
                dp_result[
                    "sum_sensitivity"
                ]
            ),
            "count_vector": float(
                dp_result[
                    "count_sensitivity"
                ]
            ),
        },

        "dp_result": {
            "bounded_mean_by_race": (
                released_groups
            ),
            "unit": "days",
        },
    }

# =====================================================================
# Q006
# =====================================================================

def privatize_encrypted_q006(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q006 rare-subgroup count and apply the
    existing validated OpenDP Laplace count mechanism.

    Approved predicate
    ------------------
        race = ASIAN - KOREAN
        admission_type = URGENT

    Under add/remove-one-patient adjacency, each patient contributes at
    most one count after the earliest-admission rule, so sensitivity = 1.

    The exact decrypted subgroup count remains inside this trusted release
    boundary and is never returned.
    """

    private_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_context.is_private():
        raise RuntimeError(
            "Q006 DP release requires the private BFV context."
        )

    ciphertext = base64_to_bytes(
        ciphertext_base64
    )

    decrypted = decrypt_bfv(
        private_context,
        ciphertext,
    )

    if len(decrypted) != 1:
        raise ValueError(
            "Q006 encrypted statistic must contain exactly "
            "one BFV slot."
        )

    raw_count = decrypted[0]

    if isinstance(
        raw_count,
        bool,
    ):
        raise ValueError(
            "Q006 decrypted count must be an integer, not a boolean."
        )

    exact_count = int(
        raw_count
    )

    if exact_count != raw_count:
        raise ValueError(
            "Q006 decrypted count must be an exact integer."
        )

    if exact_count < 0:
        raise ValueError(
            "Q006 decrypted count cannot be negative."
        )

    noisy_count = privatize_count(
        exact_count=exact_count,
        epsilon=epsilon,
        sensitivity=Q006_SENSITIVITY,
    )

    released_count = max(
        0,
        int(
            round(
                noisy_count
            )
        ),
    )

    return {
        "query_id": "Q006",
        "mechanism": "laplace",
        "epsilon": float(
            epsilon
        ),
        "delta": 0.0,
        "sensitivity": Q006_SENSITIVITY,
        "predicate": {
            "race": Q006_RACE,
            "admission_type": Q006_ADMISSION_TYPE,
        },
        "dp_result": released_count,
    }


def release_noisy_encrypted_q006(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt an already-DP-noised Q006 BFV ciphertext.

    Expected input
    --------------
        Enc(global_count + integer_Laplace_noise)

    Differential privacy has already been applied under HE by the
    independent noise service.

    This function therefore performs only:

        1. trusted BFV decryption,
        2. structural validation,
        3. DP-safe non-negative post-processing.

    It MUST NOT sample or add any additional DP noise.

    Approved predicate
    ------------------
        race = ASIAN - KOREAN
        admission_type = URGENT

    Under add/remove-one-patient adjacency:

        sensitivity = 1

    The decrypted value is already a DP output and may legitimately
    be negative before post-processing.
    """

    epsilon = float(
        epsilon
    )

    if epsilon <= 0.0:
        raise ValueError(
            "epsilon must be greater than zero."
        )

    private_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_context.is_private():
        raise RuntimeError(
            "Q006 encrypted-noise release requires "
            "the private BFV context."
        )

    ciphertext = base64_to_bytes(
        ciphertext_base64
    )

    decrypted = decrypt_bfv(
        private_context,
        ciphertext,
    )

    if len(decrypted) != 1:
        raise ValueError(
            "Q006 already-noisy encrypted statistic must "
            "contain exactly one BFV slot."
        )

    raw_noisy_count = decrypted[0]

    if isinstance(
        raw_noisy_count,
        bool,
    ):
        raise ValueError(
            "Q006 already-noisy decrypted count must be "
            "an integer, not a boolean."
        )

    noisy_count = int(
        raw_noisy_count
    )

    if noisy_count != raw_noisy_count:
        raise ValueError(
            "Q006 already-noisy decrypted count must "
            "be an exact integer."
        )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT reject negative values here.
    #
    # Differential privacy has already been applied under HE, so a
    # negative noisy count is a valid intermediate DP output.
    #
    # Non-negative clipping is DP-safe post-processing.
    # -------------------------------------------------------------

    released_count = max(
        0,
        noisy_count,
    )

    return {
        "query_id": "Q006",
        "mechanism": "integer_laplace_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "sensitivity": Q006_SENSITIVITY,
        "predicate": {
            "race": Q006_RACE,
            "admission_type": Q006_ADMISSION_TYPE,
        },
        "dp_noise_applied_under_he": True,
        "postprocessing": (
            "nonnegative_integer_clipping"
        ),
        "dp_result": released_count,
    }


# =====================================================================
# Q007
# =====================================================================

def privatize_encrypted_q007(
    bounded_sum_ciphertext_base64: str,
    count_ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    """
    Decrypt the encrypted global Q007 rare-subgroup bounded-LOS sufficient
    statistics and apply the existing validated bounded-mean DP mechanism.

    Hybrid HE representation
    ------------------------
        bounded LOS sum -> CKKS
        subgroup count  -> BFV

    Approved predicate
    ------------------
        race = ASIAN - KOREAN
        admission_type = URGENT

    Approved bounds
    ---------------
        lower = 0 days
        upper = 30 days

    The requested epsilon is split equally between the bounded sum and
    subgroup count. Under add/remove-one-patient adjacency:

        bounded-sum sensitivity = 30
        count sensitivity       = 1

    Exact decrypted sufficient statistics remain inside this trusted
    release boundary and are never returned.
    """

    private_ckks_context = load_context(
        PRIVATE_CKKS_CONTEXT_PATH
    )

    private_bfv_context = load_context(
        PRIVATE_BFV_CONTEXT_PATH
    )

    if not private_ckks_context.is_private():
        raise RuntimeError(
            "Q007 DP release requires the private CKKS context."
        )

    if not private_bfv_context.is_private():
        raise RuntimeError(
            "Q007 DP release requires the private BFV context."
        )

    bounded_sum_ciphertext = base64_to_bytes(
        bounded_sum_ciphertext_base64
    )

    count_ciphertext = base64_to_bytes(
        count_ciphertext_base64
    )

    decrypted_sum = decrypt_ckks(
        private_ckks_context,
        bounded_sum_ciphertext,
    )

    if len(decrypted_sum) != 1:
        raise ValueError(
            "Q007 encrypted bounded sum must contain exactly "
            "one CKKS slot."
        )

    exact_sum = float(
        decrypted_sum[0]
    )

    decrypted_count = decrypt_bfv(
        private_bfv_context,
        count_ciphertext,
    )

    if len(decrypted_count) != 1:
        raise ValueError(
            "Q007 encrypted subgroup count must contain exactly "
            "one BFV slot."
        )

    raw_count = decrypted_count[0]

    if isinstance(
        raw_count,
        bool,
    ):
        raise ValueError(
            "Q007 decrypted count must be an integer, not a boolean."
        )

    exact_count = int(
        raw_count
    )

    if exact_count != raw_count:
        raise ValueError(
            "Q007 decrypted count must be an exact integer."
        )

    if exact_count <= 0:
        raise ValueError(
            "Q007 decrypted subgroup count must be greater than zero."
        )

    if not math.isfinite(
        exact_sum
    ):
        raise ValueError(
            "Q007 decrypted bounded sum is not finite."
        )

    minimum_possible_sum = (
        Q007_LOWER_BOUND
        * exact_count
    )

    maximum_possible_sum = (
        Q007_UPPER_BOUND
        * exact_count
    )

    if exact_sum < (
        minimum_possible_sum
        - Q007_CKKS_ABSOLUTE_TOLERANCE
    ):
        raise ValueError(
            "Q007 decrypted bounded sum is below the approved "
            "contribution bound."
        )

    if exact_sum > (
        maximum_possible_sum
        + Q007_CKKS_ABSOLUTE_TOLERANCE
    ):
        raise ValueError(
            "Q007 decrypted bounded sum exceeds the approved "
            "contribution bound."
        )

    # CKKS is approximate. After the tolerance checks above, clamp only
    # tiny numerical excursions back onto the valid theoretical interval.
    validated_sum = min(
        maximum_possible_sum,
        max(
            minimum_possible_sum,
            exact_sum,
        ),
    )

    dp_result = privatize_bounded_mean(
        exact_sum=float(
            validated_sum
        ),
        exact_count=exact_count,
        epsilon=epsilon,
        lower=Q007_LOWER_BOUND,
        upper=Q007_UPPER_BOUND,
        sum_budget_fraction=(
            Q007_SUM_BUDGET_FRACTION
        ),
    )

    required_fields = {
        "epsilon_sum",
        "epsilon_count",
        "sum_sensitivity",
        "count_sensitivity",
        "dp_mean",
    }

    missing_fields = (
        required_fields
        - set(
            dp_result.keys()
        )
    )

    if missing_fields:
        raise ValueError(
            "Q007 DP mechanism returned incomplete output: "
            f"{sorted(missing_fields)}"
        )

    released_mean = float(
        dp_result[
            "dp_mean"
        ]
    )

    if not math.isfinite(
        released_mean
    ):
        raise ValueError(
            "Q007 DP mechanism returned a non-finite bounded mean."
        )

    if not (
        Q007_LOWER_BOUND
        <= released_mean
        <= Q007_UPPER_BOUND
    ):
        raise ValueError(
            "Q007 DP mechanism returned a bounded mean outside "
            "the approved [0, 30] interval."
        )

    return {
        "query_id": "Q007",
        "mechanism": "laplace_sum_and_count",
        "epsilon": float(
            epsilon
        ),
        "delta": 0.0,
        "bounds": {
            "lower": Q007_LOWER_BOUND,
            "upper": Q007_UPPER_BOUND,
            "unit": "days",
        },
        "predicate": {
            "race": Q007_RACE,
            "admission_type": Q007_ADMISSION_TYPE,
        },
        "budget_allocation": {
            "epsilon_sum": float(
                dp_result[
                    "epsilon_sum"
                ]
            ),
            "epsilon_count": float(
                dp_result[
                    "epsilon_count"
                ]
            ),
        },
        "sensitivities": {
            "sum": float(
                dp_result[
                    "sum_sensitivity"
                ]
            ),
            "count": float(
                dp_result[
                    "count_sensitivity"
                ]
            ),
        },
        "dp_result": {
            "bounded_mean": released_mean,
            "unit": "days",
        },
    }



# =====================================================================
# Encrypted-noise release paths
# =====================================================================
# These functions receive ciphertexts to which DP noise has ALREADY been
# added by the independent public-key-only Noise Service. They must never
# sample or add further DP noise.


def release_noisy_encrypted_q001(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    private_context = load_context(PRIVATE_BFV_CONTEXT_PATH)
    if not private_context.is_private():
        raise RuntimeError("Q001 encrypted-noise release requires private BFV context.")

    decrypted = decrypt_bfv(private_context, base64_to_bytes(ciphertext_base64))
    if len(decrypted) != 1:
        raise ValueError("Q001 already-noisy ciphertext must contain one BFV slot.")

    raw = decrypted[0]
    if isinstance(raw, bool):
        raise ValueError("Q001 already-noisy count must be an integer.")
    noisy_count = int(raw)
    if noisy_count != raw:
        raise ValueError("Q001 already-noisy count must be an exact integer.")

    return {
        "query_id": "Q001",
        "mechanism": "integer_laplace_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "sensitivity": 1.0,
        "dp_noise_applied_under_he": True,
        "postprocessing": "nonnegative_integer_clipping",
        "dp_result": max(0, noisy_count),
    }


def release_noisy_encrypted_q002(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    private_context = load_context(PRIVATE_BFV_CONTEXT_PATH)
    if not private_context.is_private():
        raise RuntimeError("Q002 encrypted-noise release requires private BFV context.")

    decrypted = decrypt_bfv(private_context, base64_to_bytes(ciphertext_base64))
    expected_length = len(Q002_CATEGORIES)
    if len(decrypted) != expected_length:
        raise ValueError("Q002 already-noisy vector has an invalid length.")

    released = {}
    for category, raw in zip(Q002_CATEGORIES, decrypted):
        if isinstance(raw, bool):
            raise ValueError("Q002 already-noisy histogram values must be integers.")
        value = int(raw)
        if value != raw:
            raise ValueError("Q002 already-noisy histogram values must be exact integers.")
        released[category] = max(0, value)

    return {
        "query_id": "Q002",
        "mechanism": "integer_laplace_vector_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "sensitivity": 1.0,
        "vector_length": expected_length,
        "category_order": list(Q002_CATEGORIES),
        "dp_noise_applied_under_he": True,
        "postprocessing": "nonnegative_integer_clipping",
        "dp_result": released,
    }


def release_noisy_encrypted_q003(
    bounded_sum_ciphertext_base64: str,
    count_ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    private_ckks_context = load_context(PRIVATE_CKKS_CONTEXT_PATH)
    private_bfv_context = load_context(PRIVATE_BFV_CONTEXT_PATH)
    if not private_ckks_context.is_private() or not private_bfv_context.is_private():
        raise RuntimeError("Q003 encrypted-noise release requires private CKKS and BFV contexts.")

    decrypted_sum = decrypt_ckks(
        private_ckks_context,
        base64_to_bytes(bounded_sum_ciphertext_base64),
    )
    decrypted_count = decrypt_bfv(
        private_bfv_context,
        base64_to_bytes(count_ciphertext_base64),
    )
    if len(decrypted_sum) != 1 or len(decrypted_count) != 1:
        raise ValueError("Q003 already-noisy sufficient statistics must be scalar.")

    noisy_sum = float(decrypted_sum[0])
    raw_count = decrypted_count[0]
    if not math.isfinite(noisy_sum):
        raise ValueError("Q003 already-noisy bounded sum must be finite.")
    if isinstance(raw_count, bool):
        raise ValueError("Q003 already-noisy count must be an integer.")
    noisy_count = int(raw_count)
    if noisy_count != raw_count:
        raise ValueError("Q003 already-noisy count must be an exact integer.")

    safe_count = max(float(noisy_count), 1.0)
    released_mean = min(
        Q003_UPPER_BOUND,
        max(Q003_LOWER_BOUND, noisy_sum / safe_count),
    )
    epsilon_sum = epsilon * Q003_SUM_BUDGET_FRACTION
    epsilon_count = epsilon * (1.0 - Q003_SUM_BUDGET_FRACTION)

    return {
        "query_id": "Q003",
        "mechanism": "laplace_sum_and_count_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "bounds": {"lower": Q003_LOWER_BOUND, "upper": Q003_UPPER_BOUND, "unit": "days"},
        "budget_allocation": {"epsilon_sum": epsilon_sum, "epsilon_count": epsilon_count},
        "sensitivities": {"sum": 30.0, "count": 1.0},
        "dp_noise_applied_under_he": True,
        "postprocessing": "bounded_mean_from_noisy_sum_and_count",
        "dp_result": {"bounded_mean": float(released_mean), "unit": "days"},
    }


def release_noisy_encrypted_q004(
    ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    private_context = load_context(PRIVATE_BFV_CONTEXT_PATH)
    if not private_context.is_private():
        raise RuntimeError("Q004 encrypted-noise release requires private BFV context.")

    decrypted = decrypt_bfv(private_context, base64_to_bytes(ciphertext_base64))
    if len(decrypted) != Q004_VECTOR_LENGTH:
        raise ValueError("Q004 already-noisy contingency vector has an invalid length.")

    released_vector = []
    for raw in decrypted:
        if isinstance(raw, bool):
            raise ValueError("Q004 already-noisy cells must be integers.")
        value = int(raw)
        if value != raw:
            raise ValueError("Q004 already-noisy cells must be exact integers.")
        released_vector.append(max(0, value))

    table = {}
    index = 0
    for race in Q004_RACE_ORDER:
        table[race] = {}
        for admission_type in Q004_ADMISSION_TYPE_ORDER:
            table[race][admission_type] = released_vector[index]
            index += 1

    return {
        "query_id": "Q004",
        "mechanism": "integer_laplace_joint_histogram_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "sensitivity": 1.0,
        "cells": Q004_VECTOR_LENGTH,
        "dimensions": {
            "race_categories": len(Q004_RACE_ORDER),
            "admission_types": len(Q004_ADMISSION_TYPE_ORDER),
        },
        "race_order": list(Q004_RACE_ORDER),
        "admission_type_order": list(Q004_ADMISSION_TYPE_ORDER),
        "dp_noise_applied_under_he": True,
        "postprocessing": "nonnegative_integer_clipping",
        "dp_result": {"contingency_table": table},
    }


def release_noisy_encrypted_q005(
    bounded_sum_ciphertext_base64: str,
    count_ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    private_ckks_context = load_context(PRIVATE_CKKS_CONTEXT_PATH)
    private_bfv_context = load_context(PRIVATE_BFV_CONTEXT_PATH)
    if not private_ckks_context.is_private() or not private_bfv_context.is_private():
        raise RuntimeError("Q005 encrypted-noise release requires private CKKS and BFV contexts.")

    noisy_sums = [
        float(value)
        for value in decrypt_ckks(
            private_ckks_context,
            base64_to_bytes(bounded_sum_ciphertext_base64),
        )
    ]
    raw_counts = decrypt_bfv(
        private_bfv_context,
        base64_to_bytes(count_ciphertext_base64),
    )
    if len(noisy_sums) != Q005_VECTOR_LENGTH or len(raw_counts) != Q005_VECTOR_LENGTH:
        raise ValueError("Q005 already-noisy vectors have invalid lengths.")
    if any(not math.isfinite(value) for value in noisy_sums):
        raise ValueError("Q005 already-noisy bounded sums must be finite.")

    noisy_counts = []
    for raw in raw_counts:
        if isinstance(raw, bool):
            raise ValueError("Q005 already-noisy counts must be integers.")
        value = int(raw)
        if value != raw:
            raise ValueError("Q005 already-noisy counts must be exact integers.")
        noisy_counts.append(value)

    released_groups = {}
    for index, race in enumerate(Q005_RACE_ORDER):
        safe_count = max(float(noisy_counts[index]), 1.0)
        noisy_mean = noisy_sums[index] / safe_count
        released_groups[race] = {
            "bounded_mean": float(
                min(Q005_UPPER_BOUND, max(Q005_LOWER_BOUND, noisy_mean))
            )
        }

    epsilon_sum = epsilon * Q005_SUM_BUDGET_FRACTION
    epsilon_count = epsilon * (1.0 - Q005_SUM_BUDGET_FRACTION)

    return {
        "query_id": "Q005",
        "mechanism": "laplace_grouped_sum_and_count_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "bounds": {"lower": Q005_LOWER_BOUND, "upper": Q005_UPPER_BOUND, "unit": "days"},
        "groups": Q005_VECTOR_LENGTH,
        "race_order": list(Q005_RACE_ORDER),
        "budget_allocation": {"epsilon_sum": epsilon_sum, "epsilon_count": epsilon_count},
        "sensitivities": {"sum_vector": 30.0, "count_vector": 1.0},
        "dp_noise_applied_under_he": True,
        "postprocessing": "bounded_means_from_noisy_sum_and_count_vectors",
        "dp_result": {"bounded_mean_by_race": released_groups, "unit": "days"},
    }


def release_noisy_encrypted_q007(
    bounded_sum_ciphertext_base64: str,
    count_ciphertext_base64: str,
    *,
    epsilon: float,
) -> dict:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")

    private_ckks_context = load_context(PRIVATE_CKKS_CONTEXT_PATH)
    private_bfv_context = load_context(PRIVATE_BFV_CONTEXT_PATH)
    if not private_ckks_context.is_private() or not private_bfv_context.is_private():
        raise RuntimeError("Q007 encrypted-noise release requires private CKKS and BFV contexts.")

    decrypted_sum = decrypt_ckks(
        private_ckks_context,
        base64_to_bytes(bounded_sum_ciphertext_base64),
    )
    decrypted_count = decrypt_bfv(
        private_bfv_context,
        base64_to_bytes(count_ciphertext_base64),
    )
    if len(decrypted_sum) != 1 or len(decrypted_count) != 1:
        raise ValueError("Q007 already-noisy sufficient statistics must be scalar.")

    noisy_sum = float(decrypted_sum[0])
    raw_count = decrypted_count[0]
    if not math.isfinite(noisy_sum):
        raise ValueError("Q007 already-noisy bounded sum must be finite.")
    if isinstance(raw_count, bool):
        raise ValueError("Q007 already-noisy count must be an integer.")
    noisy_count = int(raw_count)
    if noisy_count != raw_count:
        raise ValueError("Q007 already-noisy count must be an exact integer.")

    safe_count = max(float(noisy_count), 1.0)
    released_mean = min(
        Q007_UPPER_BOUND,
        max(Q007_LOWER_BOUND, noisy_sum / safe_count),
    )
    epsilon_sum = epsilon * Q007_SUM_BUDGET_FRACTION
    epsilon_count = epsilon * (1.0 - Q007_SUM_BUDGET_FRACTION)

    return {
        "query_id": "Q007",
        "mechanism": "laplace_sum_and_count_under_he",
        "epsilon": epsilon,
        "delta": 0.0,
        "bounds": {"lower": Q007_LOWER_BOUND, "upper": Q007_UPPER_BOUND, "unit": "days"},
        "predicate": {"race": Q007_RACE, "admission_type": Q007_ADMISSION_TYPE},
        "budget_allocation": {"epsilon_sum": epsilon_sum, "epsilon_count": epsilon_count},
        "sensitivities": {"sum": 30.0, "count": 1.0},
        "dp_noise_applied_under_he": True,
        "postprocessing": "bounded_mean_from_noisy_sum_and_count",
        "dp_result": {"bounded_mean": float(released_mean), "unit": "days"},
    }
