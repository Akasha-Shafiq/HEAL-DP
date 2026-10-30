"""
Differential privacy mechanism for experimental Q008.

Q008 releases a two-cell histogram:

    [not_prolonged, prolonged]

Under add/remove-one-patient adjacency, each patient contributes
to exactly one cell, so the L1 sensitivity of the complete vector
is 1.

The implementation mirrors the existing histogram mechanisms in
coordinator/mechanisms.py.

Experimental only:
does not modify the production registry, accountant, or services.
"""

from __future__ import annotations

import opendp.prelude as dp


# Match the existing platform configuration.
dp.enable_features("contrib")


def privatize_prolonged_histogram(
    not_prolonged: int,
    prolonged: int,
    epsilon: float,
    sensitivity: float = 1.0,
) -> dict:
    """
    Apply one vector-valued Laplace mechanism to the two-cell
    prolonged-stay histogram.

    Parameters
    ----------
    not_prolonged:
        Exact number of patients with bounded LOS <= 14 days.

    prolonged:
        Exact number of patients with bounded LOS > 14 days.

    epsilon:
        Total epsilon for this complete two-cell release.

    sensitivity:
        L1 sensitivity. For Q008 this is 1 under
        add/remove-one-patient adjacency.

    Returns
    -------
    dict
        DP-safe released counts and derived prolonged-stay rate.
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

    if not_prolonged < 0 or prolonged < 0:
        raise ValueError(
            "Histogram counts must be non-negative."
        )

    exact_vector = [
        float(not_prolonged),
        float(prolonged),
    ]

    scale = sensitivity / epsilon

    mechanism = dp.m.make_laplace(
        dp.vector_domain(
            dp.atom_domain(
                T=float,
                nan=False,
            ),
            size=2,
        ),
        dp.l1_distance(
            T=float,
        ),
        scale=scale,
    )

    noisy_vector = mechanism(
        exact_vector
    )

    # ---------------------------------------------------------
    # Same DP-safe post-processing convention as the existing
    # histogram mechanisms:
    #
    # round + clamp to non-negative integers.
    # ---------------------------------------------------------

    released_not_prolonged = max(
        0,
        int(round(float(noisy_vector[0]))),
    )

    released_prolonged = max(
        0,
        int(round(float(noisy_vector[1]))),
    )

    released_total = (
        released_not_prolonged
        + released_prolonged
    )

    if released_total > 0:
        prolonged_rate = (
            released_prolonged
            / released_total
        )
    else:
        prolonged_rate = None

    return {
        "epsilon": epsilon,
        "sensitivity": sensitivity,
        "scale": scale,

        "not_prolonged": (
            released_not_prolonged
        ),

        "prolonged": (
            released_prolonged
        ),

        "total": released_total,

        "prolonged_rate": prolonged_rate,
    }