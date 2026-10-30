""" 
privacy budget accounting for the MIMIC-DP coorodinator 

The module wraps diffprivlib BudgetAccountant and stores comitted privacy spends in JSONL ledger
such that the budget states survives corrdinator restarts 
The intial aggregate query branch used pure epsilon DP
"""

"""
Persistent privacy-budget accounting for the MIMIC-DP coordinator.

The module wraps diffprivlib's BudgetAccountant and stores committed
privacy spends in a JSONL ledger so that budget state survives
coordinator restarts.

The initial aggregate-query branch uses pure epsilon-DP, so delta = 0.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from diffprivlib.accountant import BudgetAccountant

# ---------------------------------------------------------------------
##################  Configuration ########################  
# ---------------------------------------------------------------------

TOTAL_EPSILON_BUDGET = float(
    os.getenv("TOTAL_EPSILON_BUDGET", "10.0")
)

TOTAL_DELTA_BUDGET = float(
    os.getenv("TOTAL_DELTA_BUDGET", "0.0")
)


# ---------------------------------------------------------------------
# High-spend confirmation thresholds
# ---------------------------------------------------------------------

HIGH_SPEND_TOTAL_FRACTION = float(
    os.getenv(
        "HIGH_SPEND_TOTAL_FRACTION",
        "0.25",
    )
)

HIGH_SPEND_REMAINING_FRACTION = float(
    os.getenv(
        "HIGH_SPEND_REMAINING_FRACTION",
        "0.50",
    )
)


PRIVACY_LEDGER_PATH = Path(
    os.getenv(
        "PRIVACY_LEDGER_PATH",
        "results/privacy/privacy_ledger.jsonl",
    )
)

_LEDGER_LOCK = Lock()

# Active in-memory privacy-budget reservations.
# Reservations are used for admission control only. They prevent
# concurrent requests from independently claiming the same remaining
# privacy budget before an expensive distributed DP computation starts.
# Only successfully committed releases are written to the persistent
# privacy ledger.
_ACTIVE_RESERVATIONS: dict[str, dict] = {}


# ---------------------------------------------------------------------
################## Persistent ledger loading #########################
# ---------------------------------------------------------------------

def _load_spent_budget() -> list[tuple[float, float]]:
    """
    Load previously committed privacy spends from the ledger.
    """

    if not PRIVACY_LEDGER_PATH.exists():
        return []

    spends = []

    with PRIVACY_LEDGER_PATH.open(
        "r",
        encoding="utf-8",
    ) as ledger_file:

        for line in ledger_file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            spends.append(
                (
                    float(record["epsilon"]),
                    float(record["delta"]),
                )
            )

    return spends


def _build_accountant() -> BudgetAccountant:
    """
    Reconstruct the accountant from the persistent ledger.
    """

    return BudgetAccountant(
        epsilon=TOTAL_EPSILON_BUDGET,
        delta=TOTAL_DELTA_BUDGET,
        spent_budget=_load_spent_budget(),
    )


_accountant = _build_accountant()


# ---------------------------------------------------------------------
################## Budget operations #########################
# ---------------------------------------------------------------------

def _build_effective_accountant(
    *,
    exclude_request_id: str | None = None,
) -> BudgetAccountant:
    """
    Build a temporary accountant for admission control.

    The effective accountant contains all permanently committed spends
    from the persistent ledger plus all currently active reservations.
    A reservation can optionally be excluded when converting that same
    reservation into a permanent spend.
    """

    spent_budget = list(
        _load_spent_budget()
    )

    for request_id, reservation in _ACTIVE_RESERVATIONS.items():

        if (
            exclude_request_id is not None
            and request_id == exclude_request_id
        ):
            continue

        spent_budget.append(
            (
                float(reservation["epsilon"]),
                float(reservation["delta"]),
            )
        )

    return BudgetAccountant(
        epsilon=TOTAL_EPSILON_BUDGET,
        delta=TOTAL_DELTA_BUDGET,
        spent_budget=spent_budget,
    )


def check_budget(
    epsilon: float,
    delta: float = 0.0,
) -> bool:
    """
    Check whether a proposed privacy spend can be admitted.

    Active reservations are treated as temporarily unavailable budget.
    This operation does not consume or reserve privacy budget.
    """

    epsilon = float(epsilon)
    delta = float(delta)

    with _LEDGER_LOCK:

        effective_accountant = (
            _build_effective_accountant()
        )

        return effective_accountant.check(
            epsilon=epsilon,
            delta=delta,
        )


def reserve_budget(
    *,
    query_id: str,
    epsilon: float,
    delta: float = 0.0,
    request_id: str,
) -> dict:
    """
    Atomically reserve privacy budget for an in-flight request.

    The reservation blocks concurrent requests from claiming the same
    remaining budget, but it is not a permanent privacy spend and is not
    written to the persistent ledger.
    """

    epsilon = float(epsilon)
    delta = float(delta)

    with _LEDGER_LOCK:

        if request_id in _ACTIVE_RESERVATIONS:
            raise ValueError(
                "Privacy budget is already reserved for this request."
            )

        effective_accountant = (
            _build_effective_accountant()
        )

        # Raises BudgetError when committed spends plus active
        # reservations leave insufficient capacity.
        effective_accountant.check(
            epsilon=epsilon,
            delta=delta,
        )

        reservation = {
            "request_id": request_id,
            "query_id": query_id.upper(),
            "epsilon": epsilon,
            "delta": delta,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        _ACTIVE_RESERVATIONS[
            request_id
        ] = reservation

        return dict(reservation)


def release_budget_reservation(
    *,
    request_id: str,
) -> bool:
    """
    Release an active reservation without consuming privacy budget.

    This is used when computation fails before a DP result is returned.
    The operation is intentionally idempotent.
    """

    with _LEDGER_LOCK:

        reservation = (
            _ACTIVE_RESERVATIONS.pop(
                request_id,
                None,
            )
        )

        return reservation is not None


def commit_reserved_budget(
    *,
    request_id: str,
) -> dict:
    """
    Convert an active reservation into a permanent privacy spend.

    The committed spend is written exactly once to the persistent ledger.
    """

    with _LEDGER_LOCK:

        reservation = (
            _ACTIVE_RESERVATIONS.get(
                request_id
            )
        )

        if reservation is None:
            raise ValueError(
                "No active privacy-budget reservation exists "
                "for this request."
            )

        epsilon = float(
            reservation["epsilon"]
        )
        delta = float(
            reservation["delta"]
        )
        query_id = str(
            reservation["query_id"]
        )

        # Re-check against all committed spends and all OTHER active
        # reservations. The current request's own reservation is
        # excluded because it is being converted into a permanent spend.
        effective_accountant = (
            _build_effective_accountant(
                exclude_request_id=request_id,
            )
        )

        effective_accountant.check(
            epsilon=epsilon,
            delta=delta,
        )

        record = {
            "ledger_event_id": str(uuid4()),
            "request_id": request_id,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "query_id": query_id.upper(),
            "epsilon": epsilon,
            "delta": delta,
        }

        PRIVACY_LEDGER_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with PRIVACY_LEDGER_PATH.open(
            "a",
            encoding="utf-8",
        ) as ledger_file:
            ledger_file.write(
                json.dumps(record) + "\n"
            )

        _accountant.spend(
            epsilon=epsilon,
            delta=delta,
        )

        del _ACTIVE_RESERVATIONS[
            request_id
        ]

        return record


def commit_budget(
    *,
    query_id: str,
    epsilon: float,
    delta: float = 0.0,
    request_id: str | None = None,
) -> dict:
    """
    Persist and commit a privacy-budget spend.

    This legacy commit path remains available for the existing governed
    endpoints. Active reservations from reservation-enabled endpoints are
    treated as unavailable budget.
    """

    epsilon = float(epsilon)
    delta = float(delta)

    if request_id is None:
        request_id = str(uuid4())

    with _LEDGER_LOCK:

        # Re-check immediately before committing, including any active
        # reservations owned by other in-flight requests.
        effective_accountant = (
            _build_effective_accountant()
        )

        effective_accountant.check(
            epsilon=epsilon,
            delta=delta,
        )

        record = {
            "ledger_event_id": str(uuid4()),
            "request_id": request_id,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "query_id": query_id.upper(),
            "epsilon": epsilon,
            "delta": delta,
        }

        PRIVACY_LEDGER_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with PRIVACY_LEDGER_PATH.open(
            "a",
            encoding="utf-8",
        ) as ledger_file:
            ledger_file.write(
                json.dumps(record) + "\n"
            )

        _accountant.spend(
            epsilon=epsilon,
            delta=delta,
        )

    return record


def _normalize_budget_value(value: float) -> float:
    """
    Normalize tiny floating-point accounting residues for API display.

    This does not change the accountant's internal state or ledger values.
    Values extremely close to zero or an integer are presented cleanly.
    """

    value = float(value)

    if abs(value) < 1e-12:
        return 0.0

    nearest_integer = round(value)

    if abs(value - nearest_integer) < 1e-12:
        return float(nearest_integer)

    return value


def get_budget_status() -> dict:
    """
    Return the privacy-budget ceiling, amount spent,
    and amount remaining.
    """

    spent = _accountant.total()
    remaining = _accountant.remaining()

    spent_epsilon, spent_delta = spent
    remaining_epsilon, remaining_delta = remaining

    return {
        "ceiling": {
            "epsilon": TOTAL_EPSILON_BUDGET,
            "delta": TOTAL_DELTA_BUDGET,
        },
        "spent": {
            "epsilon": _normalize_budget_value(spent_epsilon),
            "delta": _normalize_budget_value(spent_delta),
        },
        "remaining": {
            "epsilon": _normalize_budget_value(remaining_epsilon),
            "delta": _normalize_budget_value(remaining_delta),
        },
    }
    

def get_budget_impact(
    epsilon: float,
) -> dict:
    """
    Preview the effect of a proposed epsilon spend.

    This function is advisory only:
        - it does not consume privacy budget,
        - it does not modify the ledger,
        - it does not execute a DP mechanism.

    A confirmation is requested when the proposed spend is
    either:

        1. at least 25% of the total epsilon budget, or
        2. at least 50% of the currently remaining budget.

    The thresholds are configurable through environment
    variables.
    """

    epsilon = float(epsilon)

    if epsilon < 0:
        raise ValueError(
            "epsilon must be non-negative."
        )

    budget = get_budget_status()

    total_epsilon = float(
        budget["ceiling"]["epsilon"]
    )

    spent_epsilon = float(
        budget["spent"]["epsilon"]
    )

    remaining_epsilon = float(
        budget["remaining"]["epsilon"]
    )

    # -------------------------------------------------------------
    # Percentage of original total budget
    # -------------------------------------------------------------

    if total_epsilon > 0:

        fraction_of_total = (
            epsilon
            / total_epsilon
        )

    else:

        fraction_of_total = float("inf")

    # -------------------------------------------------------------
    # Percentage of currently remaining budget
    # -------------------------------------------------------------

    if remaining_epsilon > 0:

        fraction_of_remaining = (
            epsilon
            / remaining_epsilon
        )

    else:

        fraction_of_remaining = float("inf")

    # -------------------------------------------------------------
    # Projected state if this request is executed
    # -------------------------------------------------------------

    projected_spent = _normalize_budget_value(
        spent_epsilon
        + epsilon
    )

    projected_remaining = _normalize_budget_value(
        remaining_epsilon
        - epsilon
    )

    # Avoid displaying tiny floating-point negatives such as
    # -8.881e-16 when the remaining budget should be zero.
    if abs(projected_remaining) < 1e-12:
        projected_remaining = 0.0

    # -------------------------------------------------------------
    # High-spend decision
    # -------------------------------------------------------------

    triggers = []

    if (
        fraction_of_total
        >= HIGH_SPEND_TOTAL_FRACTION
    ):

        triggers.append(
            "large_fraction_of_total_budget"
        )

    if (
        fraction_of_remaining
        >= HIGH_SPEND_REMAINING_FRACTION
    ):

        triggers.append(
            "large_fraction_of_remaining_budget"
        )

    confirmation_required = (
        len(triggers) > 0
    )

    return {
        "requested_epsilon": epsilon,

        "budget_before": {
            "total_epsilon": total_epsilon,
            "spent_epsilon": spent_epsilon,
            "remaining_epsilon": (
                remaining_epsilon
            ),
        },

        "impact": {
            "percent_of_total_budget": round(
                fraction_of_total * 100.0,
                2,
            ),
            "percent_of_remaining_budget": round(
                fraction_of_remaining * 100.0,
                2,
            ),
            "projected_spent_epsilon": (
                projected_spent
            ),
            "projected_remaining_epsilon": (
                projected_remaining
            ),
        },

        "thresholds": {
            "total_budget_percent": (
                HIGH_SPEND_TOTAL_FRACTION
                * 100.0
            ),
            "remaining_budget_percent": (
                HIGH_SPEND_REMAINING_FRACTION
                * 100.0
            ),
        },

        "confirmation_required": (
            confirmation_required
        ),

        "triggers": triggers,
    }