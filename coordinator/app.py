"""
FastAPI service for the MIMIC-DP coordinator.

The coordinator acts as the central orchestration layer. It:

    - exposes approved queries to the client,
    - validates query identifiers against the registry,
    - dispatches approved queries to hospital nodes,
    - aggregates local statistics,
    - returns the global result.

Differential privacy, privacy-budget accounting, audit logging, and
high-spend budget confirmation are implemented in the current stage.
"""

import os
from uuid import uuid4

from diffprivlib.utils import BudgetError
from fastapi import Depends, FastAPI, HTTPException

from coordinator.accountant import (
    check_budget,
    reserve_budget,
    commit_reserved_budget,
    release_budget_reservation,
    get_budget_impact,
    get_budget_status,
)

from coordinator.aggregator import (
    aggregate_q001,
    aggregate_q002,
    aggregate_q003,
    aggregate_q004,
    aggregate_q005,
    aggregate_q006,
    aggregate_q007,
)


from coordinator.audit import log_query_event

from coordinator.mechanisms import (
    privatize_bounded_mean,
    privatize_count,
    privatize_grouped_bounded_means,
    privatize_histogram,
    privatize_joint_histogram,
)

from coordinator.registry import get_query, list_queries

# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="MIMIC-DP Coordinator",
    description=(
        "Central orchestration service for distributed "
        "MIMIC-IV aggregate queries."
    ),
    version="0.1.0",
)

# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """
    Return coordinator service status.
    """

    return {
        "status": "ok",
        "service": "coordinator",
        "mode": "plain_baseline",
    }


# ---------------------------------------------------------------------
# Privacy-budget status
# ---------------------------------------------------------------------

@app.get("/privacy/budget")
def privacy_budget_status() -> dict:
    """
    Return the current platform privacy-budget state.

    This endpoint reports the configured privacy ceiling,
    cumulative committed spend, and remaining budget.
    """

    return get_budget_status()


# ---------------------------------------------------------------------
# Approved query registry
# ---------------------------------------------------------------------

@app.get("/queries")
def queries() -> dict:
    """
    Return the currently enabled approved queries.
    """

    approved_queries = list_queries()

    return {
        "count": len(approved_queries),
        "queries": approved_queries,
    }



# ---------------------------------------------------------------------
# High-spend privacy-budget safeguard
# ---------------------------------------------------------------------

def require_high_spend_confirmation(
    *,
    query_id: str,
    epsilon: float,
    delta: float,
    request_id: str,
    confirm_high_spend: bool,
) -> dict:
    """
    Evaluate the privacy-budget impact of a requested DP release.

    If the requested epsilon crosses the configured high-spend
    thresholds and explicit confirmation has not been provided,
    stop execution before any hospital aggregation or DP release.

    This function does not spend privacy budget.
    """

    budget_impact = get_budget_impact(
        epsilon=epsilon,
    )

    if (
        budget_impact["confirmation_required"]
        and not confirm_high_spend
    ):
        log_query_event(
            request_id=request_id,
            query_id=query_id,
            status="confirmation_required",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="high_privacy_budget_spend",
        )

        raise HTTPException(
            status_code=428,
            detail={
                "status": "confirmation_required",
                "message": (
                    "This request would consume a large "
                    "fraction of the available privacy budget. "
                    "Review the projected budget impact and "
                    "resubmit with confirm_high_spend=true "
                    "if you want to continue."
                ),
                "query_id": query_id,
                "requested_epsilon": epsilon,
                "budget_impact": budget_impact,
                "confirmation_required": True,
                "confirmation_parameter": (
                    "confirm_high_spend=true"
                ),
            },
        )

    return budget_impact


# ---------------------------------------------------------------------
# Experimental exact-baseline access control
# ---------------------------------------------------------------------

def require_exact_baseline_access() -> None:
    """Deny HTTP access to non-private results unless explicitly enabled.

    Exact hospital aggregates are for controlled experiments only.  This
    environment switch is not authentication and does not protect the
    hospital-node endpoints or any unrelated API/service.

    Leave disabled for a public-facing synthetic DP demonstration. Enable
    only in an isolated, access-controlled research environment.
    """
    enabled = os.getenv(
        "CDP_ENABLE_EXACT_BASELINES", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}

    if not enabled:
        raise HTTPException(
            status_code=403,
            detail=(
                "Exact non-private baseline endpoints are disabled. "
                "Use the approved differential-privacy endpoints."
            ),
        )


# ---------------------------------------------------------------------
# Q001 — Distributed unique-patient count
# ---------------------------------------------------------------------
@app.get(
    "/query/q001",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q001() -> dict:
    """
    Execute approved distributed query Q001.

    The coordinator validates the query against the approved-query
    registry, dispatches the local query to all participating
    hospitals, aggregates the responses, records an audit event,
    and returns the exact global result.

    This endpoint represents the non-private baseline.
    """

    request_id = str(uuid4())

    try:
        query_definition = get_query("Q001")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q001 is not an approved query.",
        ) from exc

    try:
        result = aggregate_q001()

    except Exception as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q001 failed.",
        ) from exc

    log_query_event(
        request_id=request_id,
        query_id="Q001",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "global_result": result["global_result"],
    }
    
    

 # ---------------------------------------------------------------------   
# Q001 — Differentially private unique-patient count
# ---------------------------------------------------------------------

@app.get("/query/q001/dp")
def execute_q001_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q001 using central differential privacy.

    The coordinator:

        1. validates Q001 against the approved-query registry,
        2. validates the requested epsilon,
        3. performs an advisory privacy-budget pre-check,
        4. previews the budget impact and obtains high-spend confirmation,
        5. reserves the requested privacy budget before dispatch,
        6. obtains the exact distributed aggregate internally,
        7. applies the OpenDP Laplace mechanism,
        8. commits the reservation before returning any DP result,
        9. records an audit event,
        10. returns only the DP result.

    The exact global statistic is never returned by this endpoint.
    """

    request_id = str(uuid4())
    delta = 0.0

    # ---------------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q001")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q001 is not an approved query.",
        ) from exc
        
    # ---------------------------------------------------------------------
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q001.",
                "allowed_epsilons": allowed_epsilons,
            },
        )
        
    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc

    # -------------------------------------------------------------
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q001",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q001",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc

   # --------------------------------------------------------------
    # 5. Distributed exact aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q001()
        exact_count = aggregate_result["global_result"]

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q001 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply OpenDP Laplace mechanism
    # -------------------------------------------------------------

    try:
        noisy_count = privatize_count(
            exact_count=exact_count,
            epsilon=epsilon,
            sensitivity=1.0,
        )

        # Safe DP post-processing:
        # counts should be non-negative integers.
        released_count = max(
            0,
            int(round(noisy_count)),
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit the privacy spend
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc

    # -------------------------------------------------------------
    # 8. Audit successful DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q001",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only the DP release
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace",
        "epsilon": epsilon,
        "delta": delta,
        "dp_result": released_count,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }    
    
    
    

# -------------------------------------------------------------
# Q002 — Distributed admission-type histogram
# ---------------------------------------------------------------------

@app.get(
    "/query/q002",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q002() -> dict:
    """
    Execute approved distributed query Q002.

    The coordinator validates the query against the approved-query
    registry, dispatches Q002 to all participating hospitals,
    aggregates the local histograms element-wise, records an audit
    event, and returns the exact global histogram.

    This endpoint represents the non-private baseline.
    """

    request_id = str(uuid4())

    try:
        query_definition = get_query("Q002")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q002 is not an approved query.",
        ) from exc

    try:
        result = aggregate_q002()

    except Exception as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q002 failed.",
        ) from exc

    log_query_event(
        request_id=request_id,
        query_id="Q002",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "global_result": result["global_result"],
    }


# -------------------------------------------------------------
# Q002 — Differentially private admission-type histogram
# ---------------------------------------------------------------------

@app.get("/query/q002/dp")
def execute_q002_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q002 using central differential privacy.

    The complete admission-type histogram is treated as one
    vector-valued query.

    Because each patient contributes to exactly one histogram bin,
    the L1 sensitivity of the full histogram is 1 under
    add/remove-one-patient adjacency.
    """

    request_id = str(uuid4())
    delta = 0.0

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q002")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q002 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q002.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc
        
        
        
        
    # -------------------------------------------------------------    
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q002",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q002",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc


    # -------------------------------------------------------------
    # 5. Distributed exact histogram aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q002()
        exact_histogram = aggregate_result["global_result"]

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q002 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply vector-valued OpenDP Laplace mechanism
    # -------------------------------------------------------------

    try:
        released_histogram = privatize_histogram(
            exact_histogram=exact_histogram,
            epsilon=epsilon,
            sensitivity=1.0,
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit privacy spend
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc

    # -------------------------------------------------------------
    # 8. Audit successful DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q002",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only the DP histogram
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace_vector",
        "epsilon": epsilon,
        "delta": delta,
        "dp_result": released_histogram,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }





# -------------------------------------------------------------
# Q003 — Distributed bounded mean length of stay
# ---------------------------------------------------------------------

@app.get(
    "/query/q003",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q003() -> dict:
    """
    Execute approved distributed query Q003.

    The coordinator validates the query, requests bounded LOS
    sufficient statistics from all participating hospitals,
    aggregates the bounded sums and counts, computes the exact
    global bounded mean, records an audit event, and returns the
    non-private baseline result.
    """

    request_id = str(uuid4())

    try:
        query_definition = get_query("Q003")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q003 is not an approved query.",
        ) from exc

    try:
        result = aggregate_q003()

    except Exception as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q003 failed.",
        ) from exc

    log_query_event(
        request_id=request_id,
        query_id="Q003",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "global_result": result["global_result"],
    }
    
    
# ---------------------------------------------------------------------
# Q003 — Differentially private bounded mean length of stay
# ---------------------------------------------------------------------

@app.get("/query/q003/dp")
def execute_q003_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q003 using central differential privacy.

    The requested epsilon is divided equally between:
        1. the bounded LOS sum, and
        2. the patient count.

    The DP mean is then computed from the two noisy sufficient
    statistics using post-processing.

    The exact sum, exact count, noisy sum, and noisy count are
    not returned to the client.
    """

    request_id = str(uuid4())
    delta = 0.0
    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q003")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q003 is not an approved query.",
        ) from exc
    # -------------------------------------------------------------
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q003.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc

    # -------------------------------------------------------------
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q003",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q003",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc


    # -------------------------------------------------------------
    # 5. Distributed exact sufficient-statistic aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q003()
        exact_result = aggregate_result["global_result"]

        exact_sum = exact_result["bounded_sum"]
        exact_count = exact_result["count"]
        lower = exact_result["lower"]
        upper = exact_result["upper"]

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q003 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply DP to bounded sum and count
    # -------------------------------------------------------------

    try:
        dp_result = privatize_bounded_mean(
            exact_sum=exact_sum,
            exact_count=exact_count,
            epsilon=epsilon,
            lower=lower,
            upper=upper,
            sum_budget_fraction=0.5,
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit total query privacy spend
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc
        
    # -------------------------------------------------------------
    # 8. Audit successful DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q003",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only DP-safe output
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace_sum_and_count",
        "epsilon": epsilon,
        "delta": delta,
        "budget_allocation": {
            "epsilon_sum": dp_result["epsilon_sum"],
            "epsilon_count": dp_result["epsilon_count"],
        },
        "dp_result": {
            "bounded_mean": dp_result["dp_mean"],
            "unit": "days",
        },
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }


# --------------------------------------------------------------------
# Q004 — Distributed race × admission-type contingency table
# ---------------------------------------------------------------------

@app.get(
    "/query/q004",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q004() -> dict:
    """
    Execute approved distributed query Q004.

    Each hospital computes a patient-level race × admission-type
    contingency table using the patient's earliest admission.

    The coordinator performs element-wise aggregation over the
    fixed 33 × 9 output domain and returns the exact non-private
    baseline result.

    This endpoint is intended for experimental validation only.
    """

    request_id = str(uuid4())

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q004")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q004 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Distributed aggregation
    # -------------------------------------------------------------

    try:
        result = aggregate_q004()

    except Exception as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q004 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 3. Audit successful baseline query
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q004",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    # -------------------------------------------------------------
    # 4. Return exact experimental baseline
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "global_result": result["global_result"],
    }


# ---------------------------------------------------------------------
# Q004 — Differentially private race × admission-type table
# ---------------------------------------------------------------------

@app.get("/query/q004/dp")
def execute_q004_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q004 using central differential privacy.

    Q004 is treated as one 297-cell vector-valued histogram over
    the fixed race × admission-type domain.

    Under add/remove-one-patient adjacency, each patient contributes
    to exactly one joint cell, so the complete histogram has
    L1 sensitivity 1.

    The requested epsilon applies to the complete table.
    """

    request_id = str(uuid4())
    delta = 0.0

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q004")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q004 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Validate requested epsilon
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q004.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc

    # -------------------------------------------------------------
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q004",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q004",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc


    # -------------------------------------------------------------
    # 5. Distributed exact aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q004()

        exact_table = (
            aggregate_result["global_result"][
                "contingency_table"
            ]
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q004 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply DP to the complete 297-cell histogram
    # -------------------------------------------------------------

    try:
        dp_result = privatize_joint_histogram(
            exact_table=exact_table,
            epsilon=epsilon,
            sensitivity=1.0,
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private Q004 release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit one total epsilon spend for Q004
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc

    # -------------------------------------------------------------
    # 8. Audit successful release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q004",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only DP-safe result
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace_joint_histogram",
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": dp_result["sensitivity"],
        "cells": dp_result["cells"],
        "dp_result": {
            "contingency_table": dp_result["dp_table"],
        },
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }
    
    
 
# ---------------------------------------------------------------------
# Q005 — Distributed bounded mean LOS by race
# ---------------------------------------------------------------------

@app.get(
    "/query/q005",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q005() -> dict:
    """
    Execute approved distributed query Q005.

    Each hospital computes bounded LOS sufficient statistics
    grouped by race using each patient's earliest admission.

    The coordinator aggregates the bounded sums and counts
    across hospitals and computes the exact bounded mean LOS
    for each race.

    This endpoint is intended for experimental validation only.
    """

    request_id = str(uuid4())

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q005")

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q005 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Distributed aggregation
    # -------------------------------------------------------------

    try:
        result = aggregate_q005()

    except Exception as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q005 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 3. Audit successful baseline query
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q005",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    # -------------------------------------------------------------
    # 4. Return exact experimental baseline
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "global_result": result["global_result"],
    }
    

# ---------------------------------------------------------------------
# Q005 — Differentially private bounded mean LOS by race
# ---------------------------------------------------------------------

@app.get("/query/q005/dp")
def execute_q005_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q005 using central differential privacy.

    Q005 releases bounded mean LOS for each race category.

    The 33 bounded sums are treated as one vector-valued query
    with L1 sensitivity 30, while the 33 counts are treated as
    one vector-valued query with L1 sensitivity 1.

    The requested epsilon is split equally between the grouped
    sum vector and grouped count vector.

    The released per-race means are computed only from noisy
    sufficient statistics via DP-safe post-processing.
    """

    request_id = str(uuid4())
    delta = 0.0

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q005")

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q005 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Validate requested epsilon
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q005.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc

    # -------------------------------------------------------------
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q005",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q005",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc


    # -------------------------------------------------------------
    # 5. Distributed exact aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q005()
        global_result = aggregate_result["global_result"]

        exact_groups = global_result["groups"]
        lower = global_result["lower"]
        upper = global_result["upper"]

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q005 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply DP to grouped sums and grouped counts
    # -------------------------------------------------------------

    try:
        dp_result = privatize_grouped_bounded_means(
            exact_groups=exact_groups,
            epsilon=epsilon,
            lower=lower,
            upper=upper,
            sum_budget_fraction=0.5,
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private Q005 release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit one total Q005 privacy spend
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc
    # -------------------------------------------------------------
    # 8. Audit successful DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q005",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only DP-safe results
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace_grouped_sum_and_count",
        "epsilon": epsilon,
        "delta": delta,
        "groups": dp_result["groups"],
        "budget_allocation": {
            "epsilon_sum": dp_result["epsilon_sum"],
            "epsilon_count": dp_result["epsilon_count"],
        },
        "sensitivities": {
            "sum_vector": dp_result["sum_sensitivity"],
            "count_vector": dp_result["count_sensitivity"],
        },
        "dp_result": {
            "bounded_mean_by_race": dp_result["dp_groups"],
            "unit": "days",
        },
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }
    
    
    
    
# ---------------------------------------------------------------------
# Q006 — Distributed rare subgroup count
# ---------------------------------------------------------------------

@app.get(
    "/query/q006",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q006() -> dict:
    """
    Execute approved distributed query Q006.

    Q006 counts patients whose earliest admission has:
        race = ASIAN - KOREAN
        admission_type = URGENT

    This is the exact non-private experimental baseline.
    """

    request_id = str(uuid4())

    # 1. Validate approved query
    try:
        query_definition = get_query("Q006")

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q006 is not an approved query.",
        ) from exc

    # 2. Execute distributed aggregation
    try:
        result = aggregate_q006()

    except Exception as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q006 failed.",
        ) from exc

    # 3. Audit successful baseline query
    log_query_event(
        request_id=request_id,
        query_id="Q006",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    # 4. Return exact baseline
    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "predicate": result["predicate"],
        "global_result": result["global_result"],
    }
    
    
# -------------------------------------------------------------
# Q006 — Central-DP rare subgroup count
# -------------------------------------------------------------

@app.get("/query/q006/dp")
def execute_q006_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q006 using central differential privacy.

    Q006 counts patients whose earliest admission has:
        race = ASIAN - KOREAN
        admission_type = URGENT

    Privacy model:
        privacy unit = subject_id
        adjacency = add/remove one patient
        sensitivity = 1

    The exact global count is never returned by this endpoint.
    """

    request_id = str(uuid4())
    delta = 0.0

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q006")

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q006 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q006.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc

    # -------------------------------------------------------------
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q006",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q006",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc


    # -------------------------------------------------------------
    # 5. Distributed exact aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q006()

        exact_count = aggregate_result["global_result"]["count"]

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q006 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply OpenDP Laplace mechanism
    # -------------------------------------------------------------

    try:
        noisy_count = privatize_count(
            exact_count=exact_count,
            epsilon=epsilon,
            sensitivity=1.0,
        )

        # Safe DP post-processing.
        # Released counts should be non-negative integers.
        released_count = max(
            0,
            int(round(noisy_count)),
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private Q006 release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit privacy spend
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc
        
    # -------------------------------------------------------------
    # 8. Audit successful DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q006",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only DP-safe result
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace",
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": 1.0,
        "predicate": query_definition["predicate"],
        "dp_result": {
            "count": released_count,
        },
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }
    
    
    
# ---------------------------------------------------------------------
# Q007 — Rare subgroup bounded mean LOS
# ---------------------------------------------------------------------

@app.get(
    "/query/q007",
    dependencies=[Depends(require_exact_baseline_access)],
    include_in_schema=False,
)
def execute_q007() -> dict:
    """
    Execute approved distributed query Q007.

    Q007 computes the bounded mean length of stay for patients
    whose earliest admission has:

        race = ASIAN - KOREAN
        admission_type = URGENT

    LOS is clipped to [0, 30] days.

    Each hospital returns a bounded sum and count. The coordinator
    aggregates these sufficient statistics and computes the correct
    weighted global mean.
    """

    request_id = str(uuid4())

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q007")

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=0,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q007 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Distributed aggregation
    # -------------------------------------------------------------

    try:
        result = aggregate_q007()

    except Exception as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="plain_baseline",
            participating_hospitals=5,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q007 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 3. Audit successful baseline query
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q007",
        status="success",
        mode="plain_baseline",
        participating_hospitals=5,
    )

    # -------------------------------------------------------------
    # 4. Return exact experimental baseline
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "plain_baseline",
        "predicate": query_definition["predicate"],
        "global_result": result["global_result"],
    }
    
    
# ---------------------------------------------------------------------
# Q007 — Differentially private rare-subgroup bounded mean LOS
# ---------------------------------------------------------------------

@app.get("/query/q007/dp")
def execute_q007_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute Q007 using central differential privacy.

    Q007 computes the bounded mean LOS for patients whose
    earliest admission satisfies:

        race = ASIAN - KOREAN
        admission_type = URGENT

    The requested epsilon is divided equally between:
        1. the bounded LOS sum, and
        2. the subgroup patient count.

    The final mean is computed by DP-safe post-processing.

    Exact and noisy sufficient statistics are not returned.
    """

    request_id = str(uuid4())
    delta = 0.0

    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:
        query_definition = get_query("Q007")

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q007 is not an approved query.",
        ) from exc

    # -------------------------------------------------------------
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Requested epsilon is not allowed for Q007.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # -------------------------------------------------------------
    # 3. Privacy-budget pre-check
    # -------------------------------------------------------------

    try:
        check_budget(
            epsilon=epsilon,
            delta=delta,
        )

    except BudgetError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining privacy budget.",
        ) from exc

    # -------------------------------------------------------------
    # 4. Privacy-budget impact preview / high-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q007",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )

    try:
        reserve_budget(
            query_id="Q007",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except BudgetError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="Insufficient unreserved privacy budget.",
        ) from exc


    # -------------------------------------------------------------
    # 5. Distributed exact sufficient-statistic aggregation
    # -------------------------------------------------------------

    try:
        aggregate_result = aggregate_q007()

        exact_result = aggregate_result["global_result"]

        exact_sum = exact_result["bounded_sum"]
        exact_count = exact_result["count"]
        lower = exact_result["lower"]
        upper = exact_result["upper"]

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Distributed execution of Q007 failed.",
        ) from exc

    # -------------------------------------------------------------
    # 6. Apply DP to bounded sum and count
    # -------------------------------------------------------------

    try:
        dp_result = privatize_bounded_mean(
            exact_sum=exact_sum,
            exact_count=exact_count,
            epsilon=epsilon,
            lower=lower,
            upper=upper,
            sum_budget_fraction=0.5,
        )

    except Exception as exc:
        # No DP result is released on this failure path.
        release_budget_reservation(request_id=request_id)

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="dp_mechanism_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="Differentially private Q007 release failed.",
        ) from exc

    # -------------------------------------------------------------
    # 7. Commit total query privacy spend
    # -------------------------------------------------------------

    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except Exception as exc:
        # An I/O failure may leave the ledger outcome uncertain.
        # Do not release the reservation or send the DP output.
        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=500,
            detail="DP release withheld: privacy ledger commitment failed.",
        ) from exc

    # -------------------------------------------------------------
    # 8. Audit successful DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q007",
        status="success",
        mode="central_dp",
        participating_hospitals=5,
        epsilon=epsilon,
        delta=delta,
    )

    # -------------------------------------------------------------
    # 9. Return only DP-safe result
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "central_dp",
        "mechanism": "laplace_bounded_sum_and_count",
        "epsilon": epsilon,
        "delta": delta,
        "predicate": query_definition["predicate"],
        "bounds": {
            "lower": lower,
            "upper": upper,
            "unit": "days",
        },
        "budget_allocation": {
            "epsilon_sum": dp_result["epsilon_sum"],
            "epsilon_count": dp_result["epsilon_count"],
        },
        "sensitivities": {
            "sum": dp_result["sum_sensitivity"],
            "count": dp_result["count_sensitivity"],
        },
        "dp_result": {
            "bounded_mean": dp_result["dp_mean"],
            "unit": "days",
        },
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }