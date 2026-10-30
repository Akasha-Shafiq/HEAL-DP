"""
Governed HE coordinator service for the MIMIC-DP prototype.

The coordinator:

    - validates approved queries,
    - validates the requested epsilon,
    - checks the HE-specific privacy budget,
    - applies the high-spend confirmation safeguard,
    - requests encrypted local statistics from hospital nodes,
    - homomorphically aggregates those ciphertexts,
    - sends only the encrypted global statistic to the trusted
      release service,
    - receives only the DP-protected result,
    - commits the privacy spend,
    - writes an HE-specific audit event,
    - returns only the DP result to the caller.

The coordinator does NOT:

    - access hospital datasets,
    - receive plaintext local statistics,
    - possess BFV or CKKS secret keys,
    - decrypt encrypted global aggregates.

The HE experiments use a separate privacy ledger and audit log from
the existing validated CDP platform.
"""

import json
import math
import os
from urllib.request import Request, urlopen
from uuid import uuid4


# ---------------------------------------------------------------------
# HE-specific governance paths
# ---------------------------------------------------------------------
#
# coordinator.accountant and coordinator.audit read their paths when
# their modules are imported. Therefore these environment variables
# must be set BEFORE those modules are imported.
#
# This keeps HE experiments separate from the existing CDP ledger
# and audit log.
# ---------------------------------------------------------------------

HE_PRIVACY_LEDGER_PATH = os.getenv(
    "HE_PRIVACY_LEDGER_PATH",
    "results/he/privacy/privacy_ledger.jsonl",
)

HE_AUDIT_LOG_PATH = os.getenv(
    "HE_AUDIT_LOG_PATH",
    "results/he/audit/query_audit.jsonl",
)

os.environ["PRIVACY_LEDGER_PATH"] = (
    HE_PRIVACY_LEDGER_PATH
)

os.environ["AUDIT_LOG_PATH"] = (
    HE_AUDIT_LOG_PATH
)


# ---------------------------------------------------------------------
# Imports that depend on the governance configuration above
# ---------------------------------------------------------------------

from diffprivlib.utils import BudgetError
from fastapi import FastAPI, HTTPException

from coordinator.accountant import (
    check_budget,
    commit_budget,
    commit_reserved_budget,
    get_budget_impact,
    get_budget_status,
    release_budget_reservation,
    reserve_budget,
)

from coordinator.audit import (
    log_query_event,
)

from coordinator.registry import (
    get_query,
)

from he_pipeline.coordinator_service.encrypted_aggregator import (
    Q002_CATEGORY_ORDER,
    aggregate_q001_ciphertexts,
    aggregate_q002_ciphertexts,
    aggregate_q003_ciphertexts,
    aggregate_q004_ciphertexts,
    aggregate_q005_ciphertexts,
    aggregate_q006_ciphertexts,
    aggregate_q007_ciphertexts,
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
)
# ---------------------------------------------------------------------
# Release-service configuration
# ---------------------------------------------------------------------

HE_RELEASE_Q001_DP_URL = os.getenv(
    "HE_RELEASE_Q001_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q001",
)


HE_RELEASE_Q002_DP_URL = os.getenv(
    "HE_RELEASE_Q002_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q002",
)


HE_RELEASE_Q003_DP_URL = os.getenv(
    "HE_RELEASE_Q003_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q003",
)


HE_RELEASE_Q004_DP_URL = os.getenv(
    "HE_RELEASE_Q004_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q004",
)


HE_RELEASE_Q005_DP_URL = os.getenv(
    "HE_RELEASE_Q005_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q005",
)


HE_RELEASE_Q006_DP_URL = os.getenv(
    "HE_RELEASE_Q006_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q006",
)


# Q006 encrypted-noise research path. The coordinator sends the exact
# encrypted global count to the independent public-key-only Noise Service,
# then forwards only the already-noisy ciphertext to the Release Service.
HE_NOISE_Q006_URL = os.getenv(
    "HE_NOISE_Q006_URL",
    "http://127.0.0.1:9400/internal/noise/q006",
)


HE_RELEASE_Q006_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q006_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q006",
)


HE_RELEASE_Q007_DP_URL = os.getenv(
    "HE_RELEASE_Q007_DP_URL",
    "http://127.0.0.1:9300/internal/dp/q007",
)

# ---------------------------------------------------------------------
# Trusted release-service authentication
# ---------------------------------------------------------------------
# Only the HE coordinator should possess this shared research-service token.
# The release service uses the same HE_RELEASE_SERVICE_TOKEN value.
HE_RELEASE_SERVICE_TOKEN = os.getenv("HE_RELEASE_SERVICE_TOKEN", "")


def _trusted_release_headers() -> dict[str, str]:
    if len(HE_RELEASE_SERVICE_TOKEN) < 32:
        raise RuntimeError(
            "Set HE_RELEASE_SERVICE_TOKEN to the same strong value "
            "in the HE coordinator and HE release-service processes."
        )
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HE_RELEASE_SERVICE_TOKEN}",
    }


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="MIMIC-DP HE Coordinator",
    description=(
        "Governed homomorphic-encryption coordinator with "
        "central differential privacy."
    ),
    version="0.7.0",
)


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """
    Return HE coordinator status.
    """

    return {
        "status": "ok",
        "service": "he_coordinator",
        "he_schemes": [
            "BFV",
            "CKKS",
        ],
        "supported_queries": [
            "Q001",
            "Q002",
            "Q003",
            "Q004",
            "Q005",
            "Q006",
            "Q007",
        ],
        "private_key_available": False,
        "central_dp_supported": True,
        "privacy_accounting_enabled": True,
        "encrypted_noise_supported_queries": [
            "Q001", "Q002", "Q003", "Q004",
            "Q005", "Q006", "Q007",
        ],
    }


# ---------------------------------------------------------------------
# HE privacy-budget status
# ---------------------------------------------------------------------

@app.get("/privacy/budget")
def privacy_budget_status() -> dict:
    """
    Return the HE experiment's privacy-budget state.

    This uses the HE-specific privacy ledger rather than the
    existing CDP ledger.
    """

    return get_budget_status()


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
    mode: str = "he_central_dp",
) -> dict:
    """
    Evaluate the privacy-budget impact of a requested HE+DP release.

    If the requested epsilon crosses the configured high-spend
    thresholds and confirmation has not been supplied, execution
    stops BEFORE encrypted hospital aggregation or DP release.

    This function does not consume privacy budget.
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
            mode=mode,
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
# Internal helper: request trusted Q001 DP release
# ---------------------------------------------------------------------

def request_q001_dp_release(
    *,
    ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send the encrypted global Q001 statistic to the trusted
    release service.

    The release service decrypts the aggregate internally and
    applies the existing OpenDP mechanism.

    Only the DP result is returned.
    """

    request_body = json.dumps(
        {
            "ciphertext": ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q001_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    # -------------------------------------------------------------
    # Validate the trusted release response
    # -------------------------------------------------------------

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "sensitivity",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Trusted release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    # The trusted release service must never send the exact
    # aggregate back to the coordinator.
    forbidden_fields = {
        "exact_count",
        "value",
        "global_result",
    }

    leaked_fields = (
        forbidden_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Trusted release response unexpectedly "
            "contains exact aggregate fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q001":
        raise ValueError(
            "Unexpected query returned by release service."
        )

    if payload["mechanism"] != "laplace":
        raise ValueError(
            "Unexpected Q001 DP mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Release-service epsilon does not match request."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Release service reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request trusted Q002 DP release
# ---------------------------------------------------------------------

def request_q002_dp_release(
    *,
    ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send the encrypted global Q002 histogram to the trusted release
    service.

    The release service decrypts the nine-element BFV vector
    internally and applies the existing OpenDP vector-valued
    histogram mechanism.

    Only the DP histogram is returned.
    """

    request_body = json.dumps(
        {
            "ciphertext": ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q002_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    # -------------------------------------------------------------
    # Validate trusted release response
    # -------------------------------------------------------------

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "sensitivity",
        "vector_length",
        "category_order",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Trusted Q002 release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    # Exact histogram information must never return to the
    # coordinator.
    forbidden_fields = {
        "exact_histogram",
        "exact_vector",
        "value",
        "global_result",
        "counts",
    }

    leaked_fields = (
        forbidden_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Trusted Q002 release unexpectedly contains exact "
            "aggregate fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q002":
        raise ValueError(
            "Unexpected query returned by Q002 release service."
        )

    if payload["mechanism"] != "laplace_vector":
        raise ValueError(
            "Unexpected Q002 DP mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q002 release-service epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q002 release must use delta = 0."
        )

    if float(payload["sensitivity"]) != 1.0:
        raise ValueError(
            "Q002 histogram sensitivity must equal 1."
        )

    if int(payload["vector_length"]) != len(
        Q002_CATEGORY_ORDER
    ):
        raise ValueError(
            "Q002 release returned an invalid vector length."
        )

    if tuple(
        payload["category_order"]
    ) != Q002_CATEGORY_ORDER:
        raise ValueError(
            "Q002 release returned an unexpected category order."
        )

    dp_result = payload["dp_result"]

    if not isinstance(dp_result, dict):
        raise ValueError(
            "Q002 DP result must be a histogram dictionary."
        )

    if set(dp_result.keys()) != set(
        Q002_CATEGORY_ORDER
    ):
        raise ValueError(
            "Q002 DP result does not contain exactly the "
            "approved histogram categories."
        )

    for category in Q002_CATEGORY_ORDER:

        released_value = dp_result[
            category
        ]

        if not isinstance(
            released_value,
            int,
        ):
            raise ValueError(
                "Q002 DP histogram values must be integers."
            )

        if released_value < 0:
            raise ValueError(
                "Q002 DP histogram values must be non-negative."
            )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q002 release service reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request trusted Q003 DP release
# ---------------------------------------------------------------------

def request_q003_dp_release(
    *,
    bounded_sum_ciphertext: str,
    count_ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send the encrypted global Q003 sufficient statistics to the
    trusted release service.

    The release service decrypts the CKKS bounded sum and BFV count
    internally, applies the existing bounded-mean OpenDP mechanism,
    and returns only the DP bounded mean plus public mechanism
    metadata.

    The release service must not commit privacy budget.
    """

    request_body = json.dumps(
        {
            "bounded_sum_ciphertext": bounded_sum_ciphertext,
            "count_ciphertext": count_ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q003_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "bounds",
        "budget_allocation",
        "sensitivities",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Trusted Q003 release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "exact_sum",
        "exact_count",
        "exact_mean",
        "noisy_sum",
        "noisy_count",
        "bounded_sum",
        "count",
        "mean",
        "value",
        "global_result",
    }

    leaked_fields = (
        forbidden_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Trusted Q003 release unexpectedly contains exact "
            "or intermediate statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q003":
        raise ValueError(
            "Unexpected query returned by Q003 release service."
        )

    if payload["mechanism"] != "laplace_sum_and_count":
        raise ValueError(
            "Unexpected Q003 DP mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q003 release-service epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q003 release must use delta = 0."
        )

    bounds = payload["bounds"]

    if not isinstance(bounds, dict):
        raise ValueError(
            "Q003 release bounds must be a dictionary."
        )

    if float(bounds.get("lower", float("nan"))) != 0.0:
        raise ValueError(
            "Q003 release returned an unexpected lower bound."
        )

    if float(bounds.get("upper", float("nan"))) != 30.0:
        raise ValueError(
            "Q003 release returned an unexpected upper bound."
        )

    if bounds.get("unit") != "days":
        raise ValueError(
            "Q003 release returned an unexpected unit."
        )

    budget_allocation = payload["budget_allocation"]

    if not isinstance(budget_allocation, dict):
        raise ValueError(
            "Q003 budget allocation must be a dictionary."
        )

    expected_half_epsilon = float(epsilon) * 0.5

    if float(budget_allocation.get("epsilon_sum", -1.0)) != expected_half_epsilon:
        raise ValueError(
            "Q003 release returned an unexpected epsilon_sum."
        )

    if float(budget_allocation.get("epsilon_count", -1.0)) != expected_half_epsilon:
        raise ValueError(
            "Q003 release returned an unexpected epsilon_count."
        )

    sensitivities = payload["sensitivities"]

    if not isinstance(sensitivities, dict):
        raise ValueError(
            "Q003 sensitivities must be a dictionary."
        )

    if float(sensitivities.get("sum", -1.0)) != 30.0:
        raise ValueError(
            "Q003 bounded-sum sensitivity must equal 30."
        )

    if float(sensitivities.get("count", -1.0)) != 1.0:
        raise ValueError(
            "Q003 count sensitivity must equal 1."
        )

    dp_result = payload["dp_result"]

    if not isinstance(dp_result, dict):
        raise ValueError(
            "Q003 DP result must be a dictionary."
        )

    if set(dp_result.keys()) != {
        "bounded_mean",
        "unit",
    }:
        raise ValueError(
            "Q003 DP result must contain only bounded_mean and unit."
        )

    released_mean = float(
        dp_result["bounded_mean"]
    )

    if not (0.0 <= released_mean <= 30.0):
        raise ValueError(
            "Q003 released bounded mean must lie in [0, 30]."
        )

    if dp_result["unit"] != "days":
        raise ValueError(
            "Q003 DP result returned an unexpected unit."
        )

    if payload["release_stage"] != "he_plus_central_dp":
        raise ValueError(
            "Q003 release returned an unexpected release stage."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q003 release service reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload



# ---------------------------------------------------------------------
# Internal helper: request trusted Q004 DP release
# ---------------------------------------------------------------------

def request_q004_dp_release(
    *,
    ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send the encrypted global Q004 297-cell BFV vector to the trusted
    release service.

    The release service decrypts only inside the trusted boundary,
    reconstructs the canonical 33 x 9 contingency table, applies the
    validated OpenDP joint-histogram mechanism, and returns only the
    DP contingency table plus public metadata.

    The release service must not commit privacy budget.
    """

    request_body = json.dumps(
        {
            "ciphertext": ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q004_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "sensitivity",
        "cells",
        "dimensions",
        "race_order",
        "admission_type_order",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = required_fields - set(payload.keys())

    if missing_fields:
        raise ValueError(
            "Trusted Q004 release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "table",
        "vector",
        "histogram",
        "counts",
        "count",
        "patient_count",
        "local_result",
        "global_result",
        "value",
        "exact_result",
        "exact_table",
        "exact_vector",
        "exact_contingency_table",
    }

    leaked_fields = forbidden_fields & set(payload.keys())

    if leaked_fields:
        raise ValueError(
            "Trusted Q004 release unexpectedly contains exact "
            "aggregate fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q004":
        raise ValueError(
            "Unexpected query returned by Q004 release service."
        )

    if payload["mechanism"] != "laplace_joint_histogram":
        raise ValueError(
            "Unexpected Q004 DP mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q004 release-service epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q004 release must use delta = 0."
        )

    if float(payload["sensitivity"]) != 1.0:
        raise ValueError(
            "Q004 joint-histogram sensitivity must equal 1."
        )

    if int(payload["cells"]) != Q004_VECTOR_LENGTH:
        raise ValueError(
            "Q004 release returned an invalid cell count."
        )

    dimensions = payload["dimensions"]

    if not isinstance(dimensions, dict):
        raise ValueError(
            "Q004 release dimensions must be a dictionary."
        )

    if int(dimensions.get("race_categories", -1)) != len(
        Q004_RACE_ORDER
    ):
        raise ValueError(
            "Q004 release returned an invalid race dimension."
        )

    if int(dimensions.get("admission_types", -1)) != len(
        Q004_ADMISSION_TYPE_ORDER
    ):
        raise ValueError(
            "Q004 release returned an invalid admission-type dimension."
        )

    if tuple(payload["race_order"]) != Q004_RACE_ORDER:
        raise ValueError(
            "Q004 release returned an unexpected race order."
        )

    if tuple(
        payload["admission_type_order"]
    ) != Q004_ADMISSION_TYPE_ORDER:
        raise ValueError(
            "Q004 release returned an unexpected admission-type order."
        )

    dp_result = payload["dp_result"]

    if not isinstance(dp_result, dict):
        raise ValueError(
            "Q004 DP result must be a dictionary."
        )

    if set(dp_result.keys()) != {"contingency_table"}:
        raise ValueError(
            "Q004 DP result must contain only contingency_table."
        )

    contingency_table = dp_result["contingency_table"]

    if not isinstance(contingency_table, dict):
        raise ValueError(
            "Q004 contingency table must be a dictionary."
        )

    if set(contingency_table.keys()) != set(Q004_RACE_ORDER):
        raise ValueError(
            "Q004 DP table does not contain exactly the approved races."
        )

    total_cells = 0

    for race in Q004_RACE_ORDER:
        row = contingency_table[race]

        if not isinstance(row, dict):
            raise ValueError(
                "Each Q004 race row must be a dictionary."
            )

        if set(row.keys()) != set(Q004_ADMISSION_TYPE_ORDER):
            raise ValueError(
                "Q004 DP table row does not contain exactly the "
                "approved admission types."
            )

        for admission_type in Q004_ADMISSION_TYPE_ORDER:
            released_value = row[admission_type]

            if (
                isinstance(released_value, bool)
                or not isinstance(released_value, int)
            ):
                raise ValueError(
                    "Q004 DP cell values must be integers."
                )

            if released_value < 0:
                raise ValueError(
                    "Q004 DP cell values must be non-negative."
                )

            total_cells += 1

    if total_cells != Q004_VECTOR_LENGTH:
        raise ValueError(
            "Q004 DP table has an invalid total number of cells."
        )

    if payload["release_stage"] != "he_plus_central_dp":
        raise ValueError(
            "Q004 release returned an unexpected release stage."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q004 release service reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request trusted Q005 DP release
# ---------------------------------------------------------------------

def request_q005_dp_release(
    *,
    bounded_sum_ciphertext: str,
    count_ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send the encrypted global Q005 grouped sufficient statistics to the
    trusted release service.

    The release service decrypts the canonical 33-element CKKS
    bounded-sum vector and 33-element BFV count vector only inside the
    trusted boundary, applies the existing grouped bounded-means OpenDP
    mechanism, and returns only DP-safe means plus public metadata.

    The release service must not commit privacy budget.
    """

    request_body = json.dumps(
        {
            "bounded_sum_ciphertext": bounded_sum_ciphertext,
            "count_ciphertext": count_ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q005_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=120,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    # -------------------------------------------------------------
    # Validate trusted release response.
    # -------------------------------------------------------------

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "bounds",
        "groups",
        "race_order",
        "budget_allocation",
        "sensitivities",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Trusted Q005 release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "exact_sums",
        "exact_counts",
        "exact_groups",
        "bounded_sums",
        "bounded_sum",
        "counts",
        "count",
        "means",
        "mean",
        "table",
        "vector",
        "histogram",
        "patient_count",
        "local_result",
        "global_result",
        "value",
        "exact_result",
        "exact_sum",
        "exact_count",
        "exact_mean",
        "noisy_sums",
        "noisy_counts",
    }

    leaked_fields = (
        forbidden_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Trusted Q005 release unexpectedly contains exact "
            "or intermediate statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q005":
        raise ValueError(
            "Unexpected query returned by Q005 release service."
        )

    if (
        payload["mechanism"]
        != "laplace_grouped_sum_and_count"
    ):
        raise ValueError(
            "Unexpected Q005 DP mechanism."
        )

    if float(
        payload["epsilon"]
    ) != float(
        epsilon
    ):
        raise ValueError(
            "Q005 release-service epsilon does not match request."
        )

    if float(
        payload["delta"]
    ) != 0.0:
        raise ValueError(
            "Q005 release must use delta = 0."
        )

    # -------------------------------------------------------------
    # Validate fixed bounds and race schema.
    # -------------------------------------------------------------

    bounds = payload[
        "bounds"
    ]

    if not isinstance(
        bounds,
        dict,
    ):
        raise ValueError(
            "Q005 release bounds must be a dictionary."
        )

    if float(
        bounds.get(
            "lower",
            float("nan"),
        )
    ) != Q005_LOWER_BOUND:
        raise ValueError(
            "Q005 release returned an unexpected lower bound."
        )

    if float(
        bounds.get(
            "upper",
            float("nan"),
        )
    ) != Q005_UPPER_BOUND:
        raise ValueError(
            "Q005 release returned an unexpected upper bound."
        )

    if bounds.get(
        "unit"
    ) != "days":
        raise ValueError(
            "Q005 release returned an unexpected unit."
        )

    if int(
        payload[
            "groups"
        ]
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 release returned an invalid number of groups."
        )

    if tuple(
        payload[
            "race_order"
        ]
    ) != Q005_RACE_ORDER:
        raise ValueError(
            "Q005 release returned an unexpected race order."
        )

    # -------------------------------------------------------------
    # Validate epsilon split and sensitivities.
    # -------------------------------------------------------------

    budget_allocation = payload[
        "budget_allocation"
    ]

    if not isinstance(
        budget_allocation,
        dict,
    ):
        raise ValueError(
            "Q005 budget allocation must be a dictionary."
        )

    expected_half_epsilon = (
        float(epsilon)
        * 0.5
    )

    if float(
        budget_allocation.get(
            "epsilon_sum",
            -1.0,
        )
    ) != expected_half_epsilon:
        raise ValueError(
            "Q005 release returned an unexpected epsilon_sum."
        )

    if float(
        budget_allocation.get(
            "epsilon_count",
            -1.0,
        )
    ) != expected_half_epsilon:
        raise ValueError(
            "Q005 release returned an unexpected epsilon_count."
        )

    sensitivities = payload[
        "sensitivities"
    ]

    if not isinstance(
        sensitivities,
        dict,
    ):
        raise ValueError(
            "Q005 sensitivities must be a dictionary."
        )

    if float(
        sensitivities.get(
            "sum_vector",
            -1.0,
        )
    ) != 30.0:
        raise ValueError(
            "Q005 sum-vector sensitivity must equal 30."
        )

    if float(
        sensitivities.get(
            "count_vector",
            -1.0,
        )
    ) != 1.0:
        raise ValueError(
            "Q005 count-vector sensitivity must equal 1."
        )

    # -------------------------------------------------------------
    # Validate DP-safe grouped-mean output.
    # -------------------------------------------------------------

    dp_result = payload[
        "dp_result"
    ]

    if not isinstance(
        dp_result,
        dict,
    ):
        raise ValueError(
            "Q005 DP result must be a dictionary."
        )

    if set(
        dp_result.keys()
    ) != {
        "bounded_mean_by_race",
        "unit",
    }:
        raise ValueError(
            "Q005 DP result must contain only "
            "bounded_mean_by_race and unit."
        )

    if dp_result[
        "unit"
    ] != "days":
        raise ValueError(
            "Q005 DP result returned an unexpected unit."
        )

    released_groups = dp_result[
        "bounded_mean_by_race"
    ]

    if not isinstance(
        released_groups,
        dict,
    ):
        raise ValueError(
            "Q005 grouped DP result must be a dictionary."
        )

    if set(
        released_groups.keys()
    ) != set(
        Q005_RACE_ORDER
    ):
        raise ValueError(
            "Q005 DP result does not contain exactly "
            "the approved race groups."
        )

    for race in Q005_RACE_ORDER:

        group_result = released_groups[
            race
        ]

        if not isinstance(
            group_result,
            dict,
        ):
            raise ValueError(
                "Each Q005 race result must be a dictionary."
            )

        if set(
            group_result.keys()
        ) != {
            "bounded_mean",
        }:
            raise ValueError(
                "Q005 race results must contain only bounded_mean."
            )

        released_mean = float(
            group_result[
                "bounded_mean"
            ]
        )

        if not math.isfinite(
            released_mean
        ):
            raise ValueError(
                "Q005 released bounded means must be finite."
            )

        if not (
            Q005_LOWER_BOUND
            <= released_mean
            <= Q005_UPPER_BOUND
        ):
            raise ValueError(
                "Q005 released bounded means must lie in [0, 30]."
            )

    if payload[
        "release_stage"
    ] != "he_plus_central_dp":
        raise ValueError(
            "Q005 release returned an unexpected release stage."
        )

    if payload[
        "exact_aggregate_exposed"
    ] is not False:
        raise ValueError(
            "Q005 release service reported exact aggregate exposure."
        )

    if payload[
        "budget_committed"
    ] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request trusted Q006 DP release
# ---------------------------------------------------------------------

def request_q006_dp_release(
    *,
    ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send the encrypted global Q006 rare-subgroup count to the trusted
    release service and accept only a DP-safe count response.
    """

    request_body = json.dumps(
        {
            "ciphertext": ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q006_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "sensitivity",
        "predicate",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Trusted Q006 release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "exact_count",
        "count",
        "value",
        "global_result",
        "noisy_count",
        "patient_count",
        "local_result",
        "exact_result",
    }

    leaked_fields = (
        forbidden_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Trusted Q006 release unexpectedly contains exact "
            "or intermediate statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q006":
        raise ValueError(
            "Unexpected query returned by Q006 release service."
        )

    if payload["mechanism"] != "laplace":
        raise ValueError(
            "Unexpected Q006 DP mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q006 release-service epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q006 release must use delta = 0."
        )

    if float(payload["sensitivity"]) != 1.0:
        raise ValueError(
            "Q006 count sensitivity must equal 1."
        )

    predicate = payload["predicate"]

    if not isinstance(predicate, dict):
        raise ValueError(
            "Q006 release predicate must be a dictionary."
        )

    if set(predicate.keys()) != {
        "race",
        "admission_type",
    }:
        raise ValueError(
            "Q006 release predicate contains unexpected fields."
        )

    if predicate["race"] != Q006_RACE:
        raise ValueError(
            "Q006 release returned an unexpected race predicate."
        )

    if predicate["admission_type"] != Q006_ADMISSION_TYPE:
        raise ValueError(
            "Q006 release returned an unexpected admission-type "
            "predicate."
        )

    released_count = payload["dp_result"]

    if (
        isinstance(released_count, bool)
        or not isinstance(released_count, int)
    ):
        raise ValueError(
            "Q006 DP result must be an integer count."
        )

    if released_count < 0:
        raise ValueError(
            "Q006 DP count must be non-negative."
        )

    if payload["release_stage"] != "he_plus_central_dp":
        raise ValueError(
            "Q006 release returned an unexpected release stage."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q006 release service reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request Q006 encrypted DP-noise addition
# ---------------------------------------------------------------------

def request_q006_encrypted_noise(
    *,
    ciphertext: str,
    ciphertext_bytes: int,
    epsilon: float,
) -> dict:
    """
    Send only the BFV-encrypted global Q006 count to the independent
    Noise Service. The service samples integer Laplace noise, encrypts it
    with the public BFV context, adds it homomorphically, and returns only
    the already-noisy ciphertext.

    The coordinator must never receive plaintext noise or a plaintext
    aggregate from this service.
    """

    request_body = json.dumps(
        {
            "query_id": "Q006",
            "ciphertext": ciphertext,
            "ciphertext_bytes": int(ciphertext_bytes),
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_NOISE_Q006_URL,
        data=request_body,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "sensitivity",
        "he_scheme",
        "encoding",
        "ciphertext",
        "ciphertext_bytes",
        "dp_noise_applied_under_he",
        "plaintext_noise_exposed",
        "exact_aggregate_exposed",
        "private_key_available",
    }

    missing_fields = required_fields - set(payload.keys())

    if missing_fields:
        raise ValueError(
            "Q006 Noise Service response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "noise",
        "plaintext_noise",
        "exact_count",
        "count",
        "value",
        "global_result",
        "noisy_count",
        "patient_count",
        "local_result",
        "exact_result",
    }

    leaked_fields = forbidden_fields & set(payload.keys())

    if leaked_fields:
        raise ValueError(
            "Q006 Noise Service unexpectedly returned plaintext "
            "or intermediate fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q006":
        raise ValueError(
            "Unexpected query returned by Q006 Noise Service."
        )

    if payload["mechanism"] != "integer_laplace":
        raise ValueError(
            "Unexpected Q006 encrypted-noise mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q006 Noise Service epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q006 encrypted-noise release must use delta = 0."
        )

    if float(payload["sensitivity"]) != 1.0:
        raise ValueError(
            "Q006 encrypted-noise sensitivity must equal 1."
        )

    if payload["he_scheme"] != "BFV":
        raise ValueError(
            "Q006 encrypted-noise service must return BFV ciphertext."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Q006 encrypted-noise ciphertext must use base64 encoding."
        )

    encrypted_noisy_count = payload["ciphertext"]

    if (
        not isinstance(encrypted_noisy_count, str)
        or not encrypted_noisy_count
    ):
        raise ValueError(
            "Q006 Noise Service returned an invalid ciphertext."
        )

    if int(payload["ciphertext_bytes"]) <= 0:
        raise ValueError(
            "Q006 Noise Service returned an invalid ciphertext size."
        )

    if payload["dp_noise_applied_under_he"] is not True:
        raise ValueError(
            "Q006 Noise Service did not confirm encrypted DP-noise addition."
        )

    if payload["plaintext_noise_exposed"] is not False:
        raise ValueError(
            "Q006 Noise Service reported plaintext-noise exposure."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q006 Noise Service reported exact-aggregate exposure."
        )

    if payload["private_key_available"] is not False:
        raise ValueError(
            "Q006 Noise Service must not possess a private HE key."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request Q006 already-noisy trusted release
# ---------------------------------------------------------------------

def request_q006_encrypted_noise_release(
    *,
    ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send only Enc(global_count + DP_noise) to the trusted Release Service.

    The release service decrypts an already-DP-noised count and performs
    only non-negative post-processing. It must not apply a second DP
    perturbation and must not commit privacy budget.
    """

    request_body = json.dumps(
        {
            "ciphertext": ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q006_ENCRYPTED_NOISE_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "sensitivity",
        "predicate",
        "dp_noise_applied_under_he",
        "postprocessing",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = required_fields - set(payload.keys())

    if missing_fields:
        raise ValueError(
            "Trusted Q006 encrypted-noise release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "noise",
        "plaintext_noise",
        "exact_count",
        "count",
        "value",
        "global_result",
        "noisy_count",
        "patient_count",
        "local_result",
        "exact_result",
    }

    leaked_fields = forbidden_fields & set(payload.keys())

    if leaked_fields:
        raise ValueError(
            "Trusted Q006 encrypted-noise release unexpectedly contains "
            "plaintext or intermediate statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q006":
        raise ValueError(
            "Unexpected query returned by Q006 encrypted-noise release."
        )

    if payload["mechanism"] != "integer_laplace_under_he":
        raise ValueError(
            "Unexpected Q006 encrypted-noise release mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q006 encrypted-noise release epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q006 encrypted-noise release must use delta = 0."
        )

    if float(payload["sensitivity"]) != 1.0:
        raise ValueError(
            "Q006 encrypted-noise count sensitivity must equal 1."
        )

    predicate = payload["predicate"]

    if not isinstance(predicate, dict):
        raise ValueError(
            "Q006 encrypted-noise release predicate must be a dictionary."
        )

    if set(predicate.keys()) != {
        "race",
        "admission_type",
    }:
        raise ValueError(
            "Q006 encrypted-noise release predicate contains "
            "unexpected fields."
        )

    if predicate["race"] != Q006_RACE:
        raise ValueError(
            "Q006 encrypted-noise release returned an unexpected race."
        )

    if predicate["admission_type"] != Q006_ADMISSION_TYPE:
        raise ValueError(
            "Q006 encrypted-noise release returned an unexpected "
            "admission type."
        )

    if payload["dp_noise_applied_under_he"] is not True:
        raise ValueError(
            "Q006 release did not confirm encrypted DP-noise application."
        )

    if payload["postprocessing"] != "nonnegative_integer_clipping":
        raise ValueError(
            "Unexpected Q006 encrypted-noise post-processing."
        )

    released_count = payload["dp_result"]

    if (
        isinstance(released_count, bool)
        or not isinstance(released_count, int)
    ):
        raise ValueError(
            "Q006 encrypted-noise DP result must be an integer count."
        )

    if released_count < 0:
        raise ValueError(
            "Q006 encrypted-noise DP count must be non-negative."
        )

    if (
        payload["release_stage"]
        != "he_encrypted_noise_plus_central_dp"
    ):
        raise ValueError(
            "Q006 encrypted-noise release returned an unexpected stage."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q006 encrypted-noise release reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Internal helper: request trusted Q007 DP release
# ---------------------------------------------------------------------

def request_q007_dp_release(
    *,
    bounded_sum_ciphertext: str,
    count_ciphertext: str,
    epsilon: float,
) -> dict:
    """
    Send encrypted global Q007 bounded-LOS sufficient statistics to the
    trusted release service and accept only a DP-safe bounded mean.
    """

    request_body = json.dumps(
        {
            "bounded_sum_ciphertext": bounded_sum_ciphertext,
            "count_ciphertext": count_ciphertext,
            "epsilon": float(epsilon),
        }
    ).encode("utf-8")

    request = Request(
        HE_RELEASE_Q007_DP_URL,
        data=request_body,
        headers=_trusted_release_headers(),
        method="POST",
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "mechanism",
        "epsilon",
        "delta",
        "bounds",
        "predicate",
        "budget_allocation",
        "sensitivities",
        "dp_result",
        "release_stage",
        "exact_aggregate_exposed",
        "budget_committed",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Trusted Q007 release response missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_fields = {
        "exact_sum",
        "exact_count",
        "exact_mean",
        "bounded_sum",
        "count",
        "mean",
        "noisy_sum",
        "noisy_count",
        "global_result",
        "value",
        "patient_count",
        "local_result",
        "exact_result",
    }

    leaked_fields = (
        forbidden_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Trusted Q007 release unexpectedly contains exact "
            "or intermediate statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q007":
        raise ValueError(
            "Unexpected query returned by Q007 release service."
        )

    if payload["mechanism"] != "laplace_sum_and_count":
        raise ValueError(
            "Unexpected Q007 DP mechanism."
        )

    if float(payload["epsilon"]) != float(epsilon):
        raise ValueError(
            "Q007 release-service epsilon does not match request."
        )

    if float(payload["delta"]) != 0.0:
        raise ValueError(
            "Q007 release must use delta = 0."
        )

    bounds = payload["bounds"]

    if not isinstance(bounds, dict):
        raise ValueError(
            "Q007 release bounds must be a dictionary."
        )

    if float(
        bounds.get(
            "lower",
            float("nan"),
        )
    ) != Q007_LOWER_BOUND:
        raise ValueError(
            "Q007 release returned an unexpected lower bound."
        )

    if float(
        bounds.get(
            "upper",
            float("nan"),
        )
    ) != Q007_UPPER_BOUND:
        raise ValueError(
            "Q007 release returned an unexpected upper bound."
        )

    if bounds.get("unit") != "days":
        raise ValueError(
            "Q007 release returned an unexpected unit."
        )

    predicate = payload["predicate"]

    if not isinstance(predicate, dict):
        raise ValueError(
            "Q007 release predicate must be a dictionary."
        )

    if set(predicate.keys()) != {
        "race",
        "admission_type",
    }:
        raise ValueError(
            "Q007 release predicate contains unexpected fields."
        )

    if predicate["race"] != Q007_RACE:
        raise ValueError(
            "Q007 release returned an unexpected race predicate."
        )

    if predicate["admission_type"] != Q007_ADMISSION_TYPE:
        raise ValueError(
            "Q007 release returned an unexpected admission-type "
            "predicate."
        )

    budget_allocation = payload["budget_allocation"]

    if not isinstance(budget_allocation, dict):
        raise ValueError(
            "Q007 budget allocation must be a dictionary."
        )

    expected_half_epsilon = (
        float(epsilon)
        * 0.5
    )

    if float(
        budget_allocation.get(
            "epsilon_sum",
            -1.0,
        )
    ) != expected_half_epsilon:
        raise ValueError(
            "Q007 release returned an unexpected epsilon_sum."
        )

    if float(
        budget_allocation.get(
            "epsilon_count",
            -1.0,
        )
    ) != expected_half_epsilon:
        raise ValueError(
            "Q007 release returned an unexpected epsilon_count."
        )

    sensitivities = payload["sensitivities"]

    if not isinstance(sensitivities, dict):
        raise ValueError(
            "Q007 sensitivities must be a dictionary."
        )

    if float(
        sensitivities.get(
            "sum",
            -1.0,
        )
    ) != 30.0:
        raise ValueError(
            "Q007 bounded-sum sensitivity must equal 30."
        )

    if float(
        sensitivities.get(
            "count",
            -1.0,
        )
    ) != 1.0:
        raise ValueError(
            "Q007 count sensitivity must equal 1."
        )

    dp_result = payload["dp_result"]

    if not isinstance(dp_result, dict):
        raise ValueError(
            "Q007 DP result must be a dictionary."
        )

    if set(dp_result.keys()) != {
        "bounded_mean",
        "unit",
    }:
        raise ValueError(
            "Q007 DP result must contain only bounded_mean and unit."
        )

    released_mean = float(
        dp_result["bounded_mean"]
    )

    if not math.isfinite(released_mean):
        raise ValueError(
            "Q007 released bounded mean must be finite."
        )

    if not (
        Q007_LOWER_BOUND
        <= released_mean
        <= Q007_UPPER_BOUND
    ):
        raise ValueError(
            "Q007 released bounded mean must lie in [0, 30]."
        )

    if dp_result["unit"] != "days":
        raise ValueError(
            "Q007 DP result returned an unexpected unit."
        )

    if payload["release_stage"] != "he_plus_central_dp":
        raise ValueError(
            "Q007 release returned an unexpected release stage."
        )

    if payload["exact_aggregate_exposed"] is not False:
        raise ValueError(
            "Q007 release service reported exact aggregate exposure."
        )

    if payload["budget_committed"] is not False:
        raise ValueError(
            "Release service must not commit privacy budget."
        )

    return payload


# ---------------------------------------------------------------------
# Q001 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q001")
def query_q001() -> dict:
    """
    Execute HE-only Q001 aggregation.

    This endpoint exists for HE correctness/research testing.
    It returns the encrypted global statistic and does NOT apply DP.
    """

    try:

        result = aggregate_q001_ciphertexts()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q001 aggregation failed.",
        ) from exc

    return result


# ---------------------------------------------------------------------
# Q001 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q001/dp")
def query_q001_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q001 using HE + central differential privacy.

    Workflow
    --------
    1. Validate Q001 against the existing approved-query registry.
    2. Validate the requested epsilon.
    3. Check the HE-specific remaining privacy budget.
    4. Apply the existing high-spend confirmation safeguard.
    5. Obtain the encrypted five-hospital global aggregate.
    6. Send only the encrypted aggregate to the trusted release
       service for decryption + OpenDP.
    7. Commit the HE privacy spend after successful DP generation.
    8. Record an HE-specific audit event.
    9. Return only the DP-protected result.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0


    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:

        query_definition = get_query(
            "Q001"
        )

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q001 is not an approved query.",
        ) from exc


    # -------------------------------------------------------------
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q001."
                ),
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 4. High-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q001",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q001",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc



    # -------------------------------------------------------------
    # 6. Encrypted distributed aggregation
    # -------------------------------------------------------------

    try:

        aggregate_result = (
            aggregate_q001_ciphertexts()
        )

        encrypted_global = aggregate_result[
            "ciphertext"
        ]

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q001 failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 7. Trusted decryption + OpenDP
    # -------------------------------------------------------------

    try:

        dp_release = request_q001_dp_release(
            ciphertext=encrypted_global,
            epsilon=epsilon,
        )

        released_count = int(
            dp_release["dp_result"]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted HE + DP Q001 release failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q001",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc

    # 9. Audit successful HE + DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q001",
        status="success",
        mode="he_central_dp",
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )


    # -------------------------------------------------------------
    # 10. Return only the DP release
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": "he_central_dp",
        "he_scheme": "BFV",
        "mechanism": dp_release[
            "mechanism"
        ],
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": dp_release[
            "sensitivity"
        ],
        "participating_hospitals": (
            participating_hospitals
        ),
        "dp_result": released_count,
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }


# ---------------------------------------------------------------------
# Q002 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q002")
def query_q002() -> dict:
    """
    Execute HE-only Q002 aggregation.

    Each hospital computes its validated patient-level admission-type
    histogram, converts it to the approved nine-element representation,
    and encrypts it with the shared BFV public context.

    The coordinator homomorphically adds those ciphertexts and returns
    only the encrypted global vector.

    No decryption or differential privacy occurs at this endpoint.
    """

    try:

        result = aggregate_q002_ciphertexts()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q002 aggregation failed.",
        ) from exc

    return result



# ---------------------------------------------------------------------
# Q002 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q002/dp")
def query_q002_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q002 using HE + central differential privacy.

    Workflow
    --------
    1. Validate Q002 against the approved-query registry.
    2. Validate requested epsilon.
    3. Check the HE-specific remaining privacy budget.
    4. Apply the high-spend confirmation safeguard.
    5. Obtain the encrypted five-hospital global histogram.
    6. Send only the encrypted aggregate to the trusted release
       service for decryption + OpenDP.
    7. Commit the HE privacy spend after successful DP generation.
    8. Record an HE-specific audit event.
    9. Return only the DP-protected histogram.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0


    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:

        query_definition = get_query(
            "Q002"
        )

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="he_central_dp",
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

    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q002."
                ),
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 4. High-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q002",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q002",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc



    # -------------------------------------------------------------
    # 6. Encrypted distributed aggregation
    # -------------------------------------------------------------

    try:

        aggregate_result = (
            aggregate_q002_ciphertexts()
        )

        encrypted_global = aggregate_result[
            "ciphertext"
        ]

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q002 failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 7. Trusted decryption + OpenDP
    # -------------------------------------------------------------

    try:

        dp_release = request_q002_dp_release(
            ciphertext=encrypted_global,
            epsilon=epsilon,
        )

        released_histogram = dp_release[
            "dp_result"
        ]

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted HE + DP Q002 release failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q002",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc

    # 9. Audit successful HE + DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q002",
        status="success",
        mode="he_central_dp",
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )


    # -------------------------------------------------------------
    # 10. Return only DP-safe output
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": "he_central_dp",
        "he_scheme": "BFV",
        "mechanism": dp_release[
            "mechanism"
        ],
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": dp_release[
            "sensitivity"
        ],
        "vector_length": dp_release[
            "vector_length"
        ],
        "category_order": dp_release[
            "category_order"
        ],
        "participating_hospitals": (
            participating_hospitals
        ),
        "dp_result": released_histogram,
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }

# ---------------------------------------------------------------------
# Q003 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q003")
def query_q003() -> dict:
    """
    Execute HE-only Q003 aggregation.

    Each hospital computes the validated Q003 sufficient statistics:

        bounded LOS sum -> CKKS
        patient count   -> BFV

    The coordinator homomorphically aggregates the five CKKS bounded-sum
    ciphertexts and the five BFV count ciphertexts independently.

    The coordinator uses public HE contexts only. No decryption, mean
    reconstruction, differential privacy, or privacy-budget spending
    occurs at this endpoint.

    Only encrypted global sufficient statistics and fixed public metadata
    are returned.
    """

    try:

        result = aggregate_q003_ciphertexts()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q003 aggregation failed.",
        ) from exc

    return result

# ---------------------------------------------------------------------
# Q003 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q003/dp")
def query_q003_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q003 using hybrid HE + central differential
    privacy.

    Workflow
    --------
    1. Validate Q003 against the approved-query registry.
    2. Validate the requested epsilon.
    3. Check the HE-specific remaining privacy budget.
    4. Apply the high-spend confirmation safeguard.
    5. Obtain the encrypted five-hospital global bounded sum and count.
    6. Send only those encrypted global sufficient statistics to the
       trusted release service for decryption + OpenDP.
    7. Commit the TOTAL requested epsilon once after successful DP
       generation.
    8. Record an HE-specific audit event.
    9. Return only the DP-protected bounded mean.

    Although the DP mechanism splits epsilon equally between the
    bounded sum and count internally, accounting commits the total
    query epsilon exactly once.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0


    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:

        query_definition = get_query(
            "Q003"
        )

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="he_central_dp",
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

    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q003."
                ),
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 4. High-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q003",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q003",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc



    # -------------------------------------------------------------
    # 6. Encrypted distributed aggregation
    # -------------------------------------------------------------

    try:

        aggregate_result = (
            aggregate_q003_ciphertexts()
        )

        encrypted_global_sum = aggregate_result[
            "bounded_sum_ciphertext"
        ]

        encrypted_global_count = aggregate_result[
            "count_ciphertext"
        ]

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q003 failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 7. Trusted decryption + OpenDP
    # -------------------------------------------------------------

    try:

        dp_release = request_q003_dp_release(
            bounded_sum_ciphertext=encrypted_global_sum,
            count_ciphertext=encrypted_global_count,
            epsilon=epsilon,
        )

        released_mean = float(
            dp_release["dp_result"][
                "bounded_mean"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted HE + DP Q003 release failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q003",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc

    # 9. Audit successful HE + DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q003",
        status="success",
        mode="he_central_dp",
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )


    # -------------------------------------------------------------
    # 10. Return only DP-safe output
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": "he_central_dp",
        "he_schemes": {
            "bounded_sum": "CKKS",
            "count": "BFV",
        },
        "mechanism": dp_release[
            "mechanism"
        ],
        "epsilon": epsilon,
        "delta": delta,
        "bounds": dp_release[
            "bounds"
        ],
        "budget_allocation": dp_release[
            "budget_allocation"
        ],
        "sensitivities": dp_release[
            "sensitivities"
        ],
        "participating_hospitals": (
            participating_hospitals
        ),
        "dp_result": {
            "bounded_mean": released_mean,
            "unit": "days",
        },
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }

# ---------------------------------------------------------------------
# Q004 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q004")
def query_q004() -> dict:
    """
    Execute HE-only Q004 aggregation.

    Each hospital returns only a BFV ciphertext for its canonical
    297-cell race x admission-type vector. The coordinator
    homomorphically adds the five ciphertexts and returns only the
    encrypted global vector plus fixed public metadata.

    No decryption, differential privacy, privacy-budget spending,
    or exact-table reconstruction occurs at this endpoint.
    """

    try:
        result = aggregate_q004_ciphertexts()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Encrypted Q004 aggregation failed.",
        ) from exc

    forbidden_fields = {
        "table",
        "vector",
        "histogram",
        "counts",
        "count",
        "patient_count",
        "local_result",
        "global_result",
        "value",
        "exact_result",
        "exact_table",
        "exact_vector",
    }

    leaked_fields = forbidden_fields & set(result.keys())

    if leaked_fields:
        raise HTTPException(
            status_code=500,
            detail=(
                "Encrypted Q004 aggregation unexpectedly returned "
                "plaintext fields."
            ),
        )

    return result

# ---------------------------------------------------------------------
# Q004 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q004/dp")
def query_q004_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q004 using BFV + central differential privacy.

    Workflow
    --------
    1. Validate Q004 against the approved-query registry.
    2. Validate the requested epsilon.
    3. Check the HE-specific remaining privacy budget.
    4. Apply the high-spend confirmation safeguard.
    5. Obtain the encrypted five-hospital global 297-cell vector.
    6. Send only that encrypted vector to the trusted release service.
    7. Apply the validated OpenDP joint-histogram mechanism there.
    8. Commit the requested epsilon exactly once after successful DP
       generation.
    9. Record an HE-specific audit event.
    10. Return only the DP 33 x 9 contingency table and public metadata.

    The coordinator never receives the exact decrypted contingency
    table and never possesses the BFV secret key.
    """

    request_id = str(uuid4())
    delta = 0.0

    # 1. Registry validation
    try:
        query_definition = get_query("Q004")

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q004 is not an approved query.",
        ) from exc

    # 2. Epsilon validation
    allowed_epsilons = query_definition["allowed_epsilons"]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q004."
                ),
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # 3. Privacy-budget pre-check
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail="Insufficient remaining HE privacy budget.",
        ) from exc

    # 4. High-spend safeguard
    budget_impact = require_high_spend_confirmation(
        query_id="Q004",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q004",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc


    # 6. Encrypted distributed aggregation
    try:
        aggregate_result = aggregate_q004_ciphertexts()
        encrypted_global = aggregate_result["ciphertext"]
        participating_hospitals = len(
            aggregate_result["participating_hospitals"]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution of Q004 failed."
            ),
        ) from exc

    # 7. Trusted decryption + OpenDP
    try:
        dp_release = request_q004_dp_release(
            ciphertext=encrypted_global,
            epsilon=epsilon,
        )

        released_table = dp_release[
            "dp_result"
        ]["contingency_table"]

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=participating_hospitals,
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail="Trusted HE + DP Q004 release failed.",
        ) from exc

    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q004",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc


    # 9. Audit successful HE + DP release
    log_query_event(
        request_id=request_id,
        query_id="Q004",
        status="success",
        mode="he_central_dp",
        participating_hospitals=participating_hospitals,
        epsilon=epsilon,
        delta=delta,
    )

    # 10. Return only DP-safe output
    return {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "he_central_dp",
        "he_scheme": "BFV",
        "mechanism": dp_release["mechanism"],
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": dp_release["sensitivity"],
        "cells": dp_release["cells"],
        "dimensions": dp_release["dimensions"],
        "race_order": dp_release["race_order"],
        "admission_type_order": dp_release[
            "admission_type_order"
        ],
        "participating_hospitals": participating_hospitals,
        "dp_result": {
            "contingency_table": released_table,
        },
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }

# ---------------------------------------------------------------------
# Q005 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q005")
def query_q005() -> dict:
    """
    Execute HE-only Q005 aggregation.

    Each hospital returns:

        33 bounded LOS sums -> CKKS ciphertext
        33 patient counts   -> BFV ciphertext

    The coordinator homomorphically aggregates both canonical vectors
    using public contexts only.

    No decryption, differential privacy, mean reconstruction, or
    privacy-budget spending occurs at this endpoint.
    """

    try:

        result = aggregate_q005_ciphertexts()

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q005 aggregation failed.",
        ) from exc

    forbidden_fields = {
        "bounded_sums",
        "bounded_sum",
        "counts",
        "count",
        "means",
        "mean",
        "groups",
        "table",
        "vector",
        "histogram",
        "patient_count",
        "local_result",
        "global_result",
        "value",
        "exact_result",
        "exact_sum",
        "exact_count",
        "exact_mean",
    }

    leaked_fields = (
        forbidden_fields
        & set(result.keys())
    )

    if leaked_fields:

        raise HTTPException(
            status_code=500,
            detail=(
                "Encrypted Q005 aggregation unexpectedly returned "
                "plaintext fields."
            ),
        )

    return result


# ---------------------------------------------------------------------
# Q005 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q005/dp")
def query_q005_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q005 using hybrid HE + central differential privacy.

    Workflow
    --------
    1. Validate Q005 against the approved-query registry.
    2. Validate the requested epsilon.
    3. Check the HE-specific remaining privacy budget.
    4. Apply the high-spend confirmation safeguard.
    5. Obtain encrypted five-hospital global Q005 sum/count vectors.
    6. Send only those encrypted vectors to the trusted release service.
    7. Apply the existing grouped bounded-means OpenDP mechanism there.
    8. Commit the TOTAL requested epsilon exactly once after successful
       DP generation.
    9. Record an HE-specific audit event.
    10. Return only DP bounded means by race and public metadata.

    The internal DP mechanism splits epsilon equally between the complete
    33-element sum vector and the complete 33-element count vector.
    Accounting commits the total query epsilon exactly once.

    The coordinator never possesses either HE secret key and never
    receives the exact decrypted grouped sufficient statistics.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0


    # -------------------------------------------------------------
    # 1. Registry validation
    # -------------------------------------------------------------

    try:

        query_definition = get_query(
            "Q005"
        )

    except KeyError as exc:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="he_central_dp",
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
    # 2. Epsilon validation
    # -------------------------------------------------------------

    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q005."
                ),
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 4. High-spend safeguard
    # -------------------------------------------------------------

    budget_impact = require_high_spend_confirmation(
        query_id="Q005",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q005",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc



    # -------------------------------------------------------------
    # 6. Encrypted distributed aggregation
    # -------------------------------------------------------------

    try:

        aggregate_result = (
            aggregate_q005_ciphertexts()
        )

        encrypted_global_sums = aggregate_result[
            "bounded_sum_ciphertext"
        ]

        encrypted_global_counts = aggregate_result[
            "count_ciphertext"
        ]

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q005 failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 7. Trusted decryption + OpenDP
    # -------------------------------------------------------------

    try:

        dp_release = request_q005_dp_release(
            bounded_sum_ciphertext=(
                encrypted_global_sums
            ),
            count_ciphertext=(
                encrypted_global_counts
            ),
            epsilon=epsilon,
        )

        released_groups = dp_release[
            "dp_result"
        ][
            "bounded_mean_by_race"
        ]

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )


        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted HE + DP Q005 release failed."
            ),
        ) from exc


    # -------------------------------------------------------------
    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q005",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc

    # 9. Audit successful HE + DP release
    # -------------------------------------------------------------

    log_query_event(
        request_id=request_id,
        query_id="Q005",
        status="success",
        mode="he_central_dp",
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )


    # -------------------------------------------------------------
    # 10. Return only DP-safe output
    # -------------------------------------------------------------

    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": "he_central_dp",

        "he_schemes": {
            "bounded_sum_vector": "CKKS",
            "count_vector": "BFV",
        },

        "mechanism": dp_release[
            "mechanism"
        ],

        "epsilon": epsilon,
        "delta": delta,

        "bounds": dp_release[
            "bounds"
        ],

        "groups": dp_release[
            "groups"
        ],

        "race_order": dp_release[
            "race_order"
        ],

        "budget_allocation": dp_release[
            "budget_allocation"
        ],

        "sensitivities": dp_release[
            "sensitivities"
        ],

        "participating_hospitals": (
            participating_hospitals
        ),

        "dp_result": {
            "bounded_mean_by_race": (
                released_groups
            ),
            "unit": "days",
        },

        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }

# ---------------------------------------------------------------------
# Q006 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q006")
def query_q006() -> dict:
    """
    Execute HE-only Q006 rare-subgroup count aggregation.

    Hospitals return only BFV-encrypted scalar counts. The coordinator
    homomorphically adds them using the public BFV context and returns
    only the encrypted global count plus fixed public metadata.
    """

    try:
        result = aggregate_q006_ciphertexts()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Encrypted Q006 aggregation failed.",
        ) from exc

    forbidden_fields = {
        "count",
        "value",
        "exact_count",
        "patient_count",
        "global_result",
        "local_result",
        "exact_result",
        "noisy_count",
    }

    leaked_fields = (
        forbidden_fields
        & set(result.keys())
    )

    if leaked_fields:
        raise HTTPException(
            status_code=500,
            detail=(
                "Encrypted Q006 aggregation unexpectedly returned "
                "plaintext fields."
            ),
        )

    return result


# ---------------------------------------------------------------------
# Q006 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q006/dp")
def query_q006_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q006 using BFV + central differential privacy.

    Validation order is intentionally identical to the existing governed
    HE endpoints: registry, epsilon allowlist, budget pre-check,
    high-spend confirmation, encrypted aggregation, trusted DP release,
    atomic reservation, encrypted aggregation, trusted DP release, one reserved epsilon commit, audit, DP-only response.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0

    # 1. Registry validation
    try:
        query_definition = get_query(
            "Q006"
        )

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q006 is not an approved query.",
        ) from exc

    # 2. Epsilon validation
    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q006."
                ),
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # 3. Privacy-budget pre-check
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc

    # 4. High-spend safeguard
    budget_impact = require_high_spend_confirmation(
        query_id="Q006",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q006",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc


    # 6. Encrypted distributed aggregation
    try:
        aggregate_result = (
            aggregate_q006_ciphertexts()
        )

        encrypted_global = aggregate_result[
            "ciphertext"
        ]

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q006 failed."
            ),
        ) from exc

    # 7. Trusted decryption + OpenDP
    try:
        dp_release = request_q006_dp_release(
            ciphertext=encrypted_global,
            epsilon=epsilon,
        )

        released_count = int(
            dp_release["dp_result"]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted HE + DP Q006 release failed."
            ),
        ) from exc

    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc


    # 9. Audit successful release
    log_query_event(
        request_id=request_id,
        query_id="Q006",
        status="success",
        mode="he_central_dp",
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )

    # 10. Return only DP-safe output
    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": "he_central_dp",
        "he_scheme": "BFV",
        "mechanism": dp_release[
            "mechanism"
        ],
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": dp_release[
            "sensitivity"
        ],
        "predicate": dp_release[
            "predicate"
        ],
        "participating_hospitals": (
            participating_hospitals
        ),
        "dp_result": released_count,
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }


# ---------------------------------------------------------------------
# Q006 governed encrypted-noise HE + DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q006/encrypted-noise/dp")
def query_q006_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q006 with DP noise added while the global count
    remains BFV-encrypted.

    Validation order
    ----------------
    1. Registry validation.
    2. Epsilon allowlist validation.
    3. HE privacy-budget pre-check.
    4. High-spend confirmation safeguard.
    5. Atomically reserve the requested privacy budget.
    6. Five-hospital encrypted aggregation.
    7. Independent encrypted DP-noise addition.
    8. Trusted decryption of only the already-noisy ciphertext and
       non-negative post-processing.
    9. Commit the reserved privacy spend exactly once.
    10. Audit the successful release.
    11. Return only the DP-safe result.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0
    mode = "he_encrypted_noise_dp"

    # 1. Registry validation
    try:
        query_definition = get_query(
            "Q006"
        )

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode=mode,
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q006 is not an approved query.",
        ) from exc

    # 2. Epsilon validation
    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode=mode,
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q006."
                ),
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # 3. Privacy-budget pre-check
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
            mode=mode,
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc

    # 4. High-spend safeguard
    budget_impact = require_high_spend_confirmation(
        query_id="Q006",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
        mode=mode,
    )

    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q006",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode=mode,
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc

    # 6. Encrypted distributed aggregation
    try:
        aggregate_result = (
            aggregate_q006_ciphertexts()
        )

        encrypted_global = aggregate_result[
            "ciphertext"
        ]

        encrypted_global_bytes = int(
            aggregate_result[
                "ciphertext_bytes"
            ]
        )

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode=mode,
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q006 failed."
            ),
        ) from exc

    # 7. Add DP noise while the global statistic remains encrypted
    try:
        noise_result = request_q006_encrypted_noise(
            ciphertext=encrypted_global,
            ciphertext_bytes=encrypted_global_bytes,
            epsilon=epsilon,
        )

        encrypted_noisy_global = noise_result[
            "ciphertext"
        ]

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode=mode,
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="encrypted_dp_noise_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted DP-noise addition for Q006 failed."
            ),
        ) from exc

    # 8. Decrypt only the already-noisy ciphertext and post-process
    try:
        dp_release = request_q006_encrypted_noise_release(
            ciphertext=encrypted_noisy_global,
            epsilon=epsilon,
        )

        released_count = int(
            dp_release["dp_result"]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="failed",
            mode=mode,
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_encrypted_noise_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted encrypted-noise Q006 release failed."
            ),
        ) from exc

    # 9. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # If commit fails, release any still-active reservation so the
        # request cannot strand budget in memory.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q006",
            status="rejected",
            mode=mode,
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE reserved privacy budget could not be committed."
            ),
        ) from exc

    # 10. Audit successful release
    log_query_event(
        request_id=request_id,
        query_id="Q006",
        status="success",
        mode=mode,
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )

    # 11. Return only the DP-safe output
    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": mode,
        "he_scheme": "BFV",
        "mechanism": dp_release[
            "mechanism"
        ],
        "epsilon": epsilon,
        "delta": delta,
        "sensitivity": dp_release[
            "sensitivity"
        ],
        "predicate": dp_release[
            "predicate"
        ],
        "dp_noise_applied_under_he": True,
        "noise_service_private_key_available": False,
        "release_stage": dp_release[
            "release_stage"
        ],
        "participating_hospitals": (
            participating_hospitals
        ),
        "dp_result": released_count,
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }


# ---------------------------------------------------------------------
# Q007 encrypted aggregation endpoint
# ---------------------------------------------------------------------

@app.get("/query/q007")
def query_q007() -> dict:
    """
    Execute HE-only Q007 rare-subgroup bounded-LOS aggregation.

    Hospitals return a CKKS-encrypted bounded LOS sum and BFV-encrypted
    subgroup count. The coordinator aggregates both using public contexts
    only and returns encrypted global sufficient statistics.
    """

    try:
        result = aggregate_q007_ciphertexts()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Encrypted Q007 aggregation failed.",
        ) from exc

    forbidden_fields = {
        "bounded_sum",
        "count",
        "mean",
        "value",
        "exact_sum",
        "exact_count",
        "exact_mean",
        "global_result",
        "local_result",
        "exact_result",
        "noisy_sum",
        "noisy_count",
        "patient_count",
    }

    leaked_fields = (
        forbidden_fields
        & set(result.keys())
    )

    if leaked_fields:
        raise HTTPException(
            status_code=500,
            detail=(
                "Encrypted Q007 aggregation unexpectedly returned "
                "plaintext fields."
            ),
        )

    return result


# ---------------------------------------------------------------------
# Q007 governed HE + Central DP endpoint
# ---------------------------------------------------------------------

@app.get("/query/q007/dp")
def query_q007_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    """
    Execute governed Q007 using hybrid HE + central differential privacy.

    The bounded LOS sum is aggregated with CKKS and the subgroup count
    with BFV. The trusted release service decrypts both and applies the
    existing bounded-mean DP mechanism with a 50/50 epsilon split.
    Accounting reserves the total requested epsilon before execution and commits it exactly once after a successful DP release.
    """

    request_id = str(
        uuid4()
    )

    delta = 0.0

    # 1. Registry validation
    try:
        query_definition = get_query(
            "Q007"
        )

    except KeyError as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="query_not_approved",
        )

        raise HTTPException(
            status_code=403,
            detail="Q007 is not an approved query.",
        ) from exc

    # 2. Epsilon validation
    allowed_epsilons = query_definition[
        "allowed_epsilons"
    ]

    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="epsilon_not_allowed",
        )

        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Requested epsilon is not allowed for Q007."
                ),
                "allowed_epsilons": allowed_epsilons,
            },
        )

    # 3. Privacy-budget pre-check
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
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_exceeded",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient remaining HE privacy budget."
            ),
        ) from exc

    # 4. High-spend safeguard
    budget_impact = require_high_spend_confirmation(
        query_id="Q007",
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
    )


    # 5. Atomically reserve privacy budget before computation
    try:
        reserve_budget(
            query_id="Q007",
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=0,
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_reservation_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be reserved."
            ),
        ) from exc


    # 6. Encrypted distributed aggregation
    try:
        aggregate_result = (
            aggregate_q007_ciphertexts()
        )

        encrypted_global_sum = aggregate_result[
            "bounded_sum_ciphertext"
        ]

        encrypted_global_count = aggregate_result[
            "count_ciphertext"
        ]

        participating_hospitals = len(
            aggregate_result[
                "participating_hospitals"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=5,
            epsilon=epsilon,
            delta=delta,
            error="encrypted_distributed_execution_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Encrypted distributed execution "
                "of Q007 failed."
            ),
        ) from exc

    # 7. Trusted decryption + OpenDP
    try:
        dp_release = request_q007_dp_release(
            bounded_sum_ciphertext=(
                encrypted_global_sum
            ),
            count_ciphertext=(
                encrypted_global_count
            ),
            epsilon=epsilon,
        )

        released_mean = float(
            dp_release[
                "dp_result"
            ][
                "bounded_mean"
            ]
        )

    except Exception as exc:
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="failed",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="trusted_dp_release_failed",
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Trusted HE + DP Q007 release failed."
            ),
        ) from exc

    # 8. Commit the reserved privacy spend exactly once
    try:
        commit_reserved_budget(
            request_id=request_id,
        )

    except (BudgetError, ValueError) as exc:
        # Do not strand an in-memory reservation if final commit fails.
        release_budget_reservation(
            request_id=request_id,
        )

        log_query_event(
            request_id=request_id,
            query_id="Q007",
            status="rejected",
            mode="he_central_dp",
            participating_hospitals=(
                participating_hospitals
            ),
            epsilon=epsilon,
            delta=delta,
            error="privacy_budget_commit_failed",
        )

        raise HTTPException(
            status_code=403,
            detail=(
                "HE privacy budget could not be committed."
            ),
        ) from exc


    # 9. Audit successful release
    log_query_event(
        request_id=request_id,
        query_id="Q007",
        status="success",
        mode="he_central_dp",
        participating_hospitals=(
            participating_hospitals
        ),
        epsilon=epsilon,
        delta=delta,
    )

    # 10. Return only DP-safe output
    return {
        "request_id": request_id,
        "query_id": query_definition[
            "query_id"
        ],
        "name": query_definition[
            "name"
        ],
        "privacy_unit": query_definition[
            "privacy_unit"
        ],
        "mode": "he_central_dp",
        "he_schemes": {
            "bounded_sum": "CKKS",
            "count": "BFV",
        },
        "mechanism": dp_release[
            "mechanism"
        ],
        "epsilon": epsilon,
        "delta": delta,
        "bounds": dp_release[
            "bounds"
        ],
        "predicate": dp_release[
            "predicate"
        ],
        "budget_allocation": dp_release[
            "budget_allocation"
        ],
        "sensitivities": dp_release[
            "sensitivities"
        ],
        "participating_hospitals": (
            participating_hospitals
        ),
        "dp_result": {
            "bounded_mean": released_mean,
            "unit": "days",
        },
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }



# =====================================================================
# Extended encrypted-noise-before-decryption research paths
# Q006 above remains the validated reference implementation.
# The routes below add equivalent research paths for Q001-Q005 and Q007
# without modifying the existing /dp baseline routes.
# =====================================================================

HE_NOISE_Q001_URL = os.getenv("HE_NOISE_Q001_URL", "http://127.0.0.1:9400/internal/noise/q001")
HE_NOISE_Q002_URL = os.getenv("HE_NOISE_Q002_URL", "http://127.0.0.1:9400/internal/noise/q002")
HE_NOISE_Q003_URL = os.getenv("HE_NOISE_Q003_URL", "http://127.0.0.1:9400/internal/noise/q003")
HE_NOISE_Q004_URL = os.getenv("HE_NOISE_Q004_URL", "http://127.0.0.1:9400/internal/noise/q004")
HE_NOISE_Q005_URL = os.getenv("HE_NOISE_Q005_URL", "http://127.0.0.1:9400/internal/noise/q005")
HE_NOISE_Q007_URL = os.getenv("HE_NOISE_Q007_URL", "http://127.0.0.1:9400/internal/noise/q007")

HE_RELEASE_Q001_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q001_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q001",
)
HE_RELEASE_Q002_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q002_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q002",
)
HE_RELEASE_Q003_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q003_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q003",
)
HE_RELEASE_Q004_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q004_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q004",
)
HE_RELEASE_Q005_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q005_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q005",
)
HE_RELEASE_Q007_ENCRYPTED_NOISE_URL = os.getenv(
    "HE_RELEASE_Q007_ENCRYPTED_NOISE_URL",
    "http://127.0.0.1:9300/internal/encrypted-noise/q007",
)

_ENCRYPTED_NOISE_URLS = {
    "Q001": HE_NOISE_Q001_URL,
    "Q002": HE_NOISE_Q002_URL,
    "Q003": HE_NOISE_Q003_URL,
    "Q004": HE_NOISE_Q004_URL,
    "Q005": HE_NOISE_Q005_URL,
    "Q007": HE_NOISE_Q007_URL,
}

_ENCRYPTED_NOISE_RELEASE_URLS = {
    "Q001": HE_RELEASE_Q001_ENCRYPTED_NOISE_URL,
    "Q002": HE_RELEASE_Q002_ENCRYPTED_NOISE_URL,
    "Q003": HE_RELEASE_Q003_ENCRYPTED_NOISE_URL,
    "Q004": HE_RELEASE_Q004_ENCRYPTED_NOISE_URL,
    "Q005": HE_RELEASE_Q005_ENCRYPTED_NOISE_URL,
    "Q007": HE_RELEASE_Q007_ENCRYPTED_NOISE_URL,
}

_ENCRYPTED_NOISE_AGGREGATORS = {
    "Q001": aggregate_q001_ciphertexts,
    "Q002": aggregate_q002_ciphertexts,
    "Q003": aggregate_q003_ciphertexts,
    "Q004": aggregate_q004_ciphertexts,
    "Q005": aggregate_q005_ciphertexts,
    "Q007": aggregate_q007_ciphertexts,
}

_HYBRID_ENCRYPTED_NOISE_QUERIES = {"Q003", "Q005", "Q007"}


def _post_internal_json(
    url: str, payload: dict, *, timeout: int = 120,
    trusted_release: bool = False,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers=(
            _trusted_release_headers()
            if trusted_release else {"Content-Type": "application/json"}
        ),
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Internal HE service returned a non-object response.")
    return result


def _request_extended_encrypted_noise(
    query_id: str,
    aggregate_result: dict,
    *,
    epsilon: float,
) -> dict:
    query_id = query_id.upper()
    if query_id not in _ENCRYPTED_NOISE_URLS:
        raise ValueError(f"Unsupported encrypted-noise query: {query_id}")

    if query_id in _HYBRID_ENCRYPTED_NOISE_QUERIES:
        payload = {
            "query_id": query_id,
            "bounded_sum_ciphertext": aggregate_result["bounded_sum_ciphertext"],
            "bounded_sum_ciphertext_bytes": int(aggregate_result["bounded_sum_ciphertext_bytes"]),
            "count_ciphertext": aggregate_result["count_ciphertext"],
            "count_ciphertext_bytes": int(aggregate_result["count_ciphertext_bytes"]),
            "epsilon": float(epsilon),
        }
    else:
        payload = {
            "query_id": query_id,
            "ciphertext": aggregate_result["ciphertext"],
            "ciphertext_bytes": int(aggregate_result["ciphertext_bytes"]),
            "epsilon": float(epsilon),
        }

    result = _post_internal_json(_ENCRYPTED_NOISE_URLS[query_id], payload)

    required = {
        "query_id", "epsilon", "delta", "encoding",
        "dp_noise_applied_under_he", "plaintext_noise_exposed",
        "exact_aggregate_exposed", "private_key_available",
    }
    if query_id in _HYBRID_ENCRYPTED_NOISE_QUERIES:
        required |= {
            "bounded_sum_ciphertext", "bounded_sum_ciphertext_bytes",
            "count_ciphertext", "count_ciphertext_bytes",
        }
    else:
        required |= {"ciphertext", "ciphertext_bytes"}

    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"Noise Service response missing fields: {sorted(missing)}")
    if result["query_id"] != query_id:
        raise ValueError("Noise Service returned an unexpected query_id.")
    if float(result["epsilon"]) != float(epsilon) or float(result["delta"]) != 0.0:
        raise ValueError("Noise Service returned unexpected privacy parameters.")
    if result["encoding"] != "base64":
        raise ValueError("Noise Service must use base64 ciphertext encoding.")
    if result["dp_noise_applied_under_he"] is not True:
        raise ValueError("Noise Service did not confirm encrypted DP-noise addition.")
    if result["plaintext_noise_exposed"] is not False:
        raise ValueError("Noise Service reported plaintext noise exposure.")
    if result["exact_aggregate_exposed"] is not False:
        raise ValueError("Noise Service reported exact aggregate exposure.")
    if result["private_key_available"] is not False:
        raise ValueError("Noise Service must not have a private key.")

    return result


def _request_extended_encrypted_noise_release(
    query_id: str,
    noise_result: dict,
    *,
    epsilon: float,
) -> dict:
    query_id = query_id.upper()
    if query_id not in _ENCRYPTED_NOISE_RELEASE_URLS:
        raise ValueError(f"Unsupported encrypted-noise release query: {query_id}")

    if query_id in _HYBRID_ENCRYPTED_NOISE_QUERIES:
        payload = {
            "bounded_sum_ciphertext": noise_result["bounded_sum_ciphertext"],
            "count_ciphertext": noise_result["count_ciphertext"],
            "epsilon": float(epsilon),
        }
    else:
        payload = {
            "ciphertext": noise_result["ciphertext"],
            "epsilon": float(epsilon),
        }

    result = _post_internal_json(
        _ENCRYPTED_NOISE_RELEASE_URLS[query_id],
        payload,
        trusted_release=True,
    )

    required = {
        "query_id", "mechanism", "epsilon", "delta", "dp_result",
        "dp_noise_applied_under_he", "release_stage",
        "exact_aggregate_exposed", "budget_committed",
    }
    missing = required - set(result.keys())
    if missing:
        raise ValueError(f"Encrypted-noise release missing fields: {sorted(missing)}")

    forbidden = {
        "exact_count", "exact_sum", "exact_mean", "exact_vector",
        "exact_table", "noise", "plaintext_noise", "noisy_count",
        "noisy_sum", "global_result", "local_result", "exact_result",
    }
    leaked = forbidden & set(result.keys())
    if leaked:
        raise ValueError(f"Encrypted-noise release exposed forbidden fields: {sorted(leaked)}")

    if result["query_id"] != query_id:
        raise ValueError("Encrypted-noise release returned an unexpected query_id.")
    if float(result["epsilon"]) != float(epsilon) or float(result["delta"]) != 0.0:
        raise ValueError("Encrypted-noise release returned unexpected privacy parameters.")
    if result["dp_noise_applied_under_he"] is not True:
        raise ValueError("Encrypted-noise release did not confirm under-HE DP noise.")
    if result["release_stage"] != "he_encrypted_noise_plus_central_dp":
        raise ValueError("Encrypted-noise release returned an unexpected release stage.")
    if result["exact_aggregate_exposed"] is not False:
        raise ValueError("Encrypted-noise release reported exact aggregate exposure.")
    if result["budget_committed"] is not False:
        raise ValueError("Release Service must not commit privacy budget.")

    expected_mechanisms = {
        "Q001": "integer_laplace_under_he",
        "Q002": "integer_laplace_vector_under_he",
        "Q003": "laplace_sum_and_count_under_he",
        "Q004": "integer_laplace_joint_histogram_under_he",
        "Q005": "laplace_grouped_sum_and_count_under_he",
        "Q007": "laplace_sum_and_count_under_he",
    }
    if result["mechanism"] != expected_mechanisms[query_id]:
        raise ValueError("Encrypted-noise release returned an unexpected mechanism.")

    dp_result = result["dp_result"]

    if query_id == "Q001":
        if float(result.get("sensitivity", -1.0)) != 1.0:
            raise ValueError("Q001 encrypted-noise sensitivity must equal 1.")
        if isinstance(dp_result, bool) or not isinstance(dp_result, int) or dp_result < 0:
            raise ValueError("Q001 encrypted-noise result must be a non-negative integer.")

    elif query_id == "Q002":
        if float(result.get("sensitivity", -1.0)) != 1.0:
            raise ValueError("Q002 encrypted-noise sensitivity must equal 1.")
        if int(result.get("vector_length", -1)) != len(Q002_CATEGORY_ORDER):
            raise ValueError("Q002 encrypted-noise vector length is invalid.")
        if tuple(result.get("category_order", [])) != Q002_CATEGORY_ORDER:
            raise ValueError("Q002 encrypted-noise category order is invalid.")
        if not isinstance(dp_result, dict) or set(dp_result) != set(Q002_CATEGORY_ORDER):
            raise ValueError("Q002 encrypted-noise histogram domain is invalid.")
        for value in dp_result.values():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Q002 encrypted-noise histogram values must be non-negative integers.")

    elif query_id == "Q003":
        bounds = result.get("bounds", {})
        if float(bounds.get("lower", float("nan"))) != 0.0 or float(bounds.get("upper", float("nan"))) != 30.0:
            raise ValueError("Q003 encrypted-noise bounds are invalid.")
        allocation = result.get("budget_allocation", {})
        expected_half = float(epsilon) * 0.5
        if float(allocation.get("epsilon_sum", -1.0)) != expected_half or float(allocation.get("epsilon_count", -1.0)) != expected_half:
            raise ValueError("Q003 encrypted-noise budget allocation is invalid.")
        sensitivities = result.get("sensitivities", {})
        if float(sensitivities.get("sum", -1.0)) != 30.0 or float(sensitivities.get("count", -1.0)) != 1.0:
            raise ValueError("Q003 encrypted-noise sensitivities are invalid.")
        if not isinstance(dp_result, dict) or set(dp_result) != {"bounded_mean", "unit"}:
            raise ValueError("Q003 encrypted-noise DP result is invalid.")
        mean = float(dp_result["bounded_mean"])
        if not math.isfinite(mean) or not (0.0 <= mean <= 30.0) or dp_result["unit"] != "days":
            raise ValueError("Q003 encrypted-noise bounded mean is invalid.")

    elif query_id == "Q004":
        if float(result.get("sensitivity", -1.0)) != 1.0 or int(result.get("cells", -1)) != Q004_VECTOR_LENGTH:
            raise ValueError("Q004 encrypted-noise public metadata is invalid.")
        if tuple(result.get("race_order", [])) != Q004_RACE_ORDER or tuple(result.get("admission_type_order", [])) != Q004_ADMISSION_TYPE_ORDER:
            raise ValueError("Q004 encrypted-noise domain ordering is invalid.")
        if not isinstance(dp_result, dict) or set(dp_result) != {"contingency_table"}:
            raise ValueError("Q004 encrypted-noise DP result is invalid.")
        table = dp_result["contingency_table"]
        if not isinstance(table, dict) or set(table) != set(Q004_RACE_ORDER):
            raise ValueError("Q004 encrypted-noise table race domain is invalid.")
        for race in Q004_RACE_ORDER:
            row = table[race]
            if not isinstance(row, dict) or set(row) != set(Q004_ADMISSION_TYPE_ORDER):
                raise ValueError("Q004 encrypted-noise table admission domain is invalid.")
            for value in row.values():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError("Q004 encrypted-noise cells must be non-negative integers.")

    elif query_id == "Q005":
        bounds = result.get("bounds", {})
        if float(bounds.get("lower", float("nan"))) != Q005_LOWER_BOUND or float(bounds.get("upper", float("nan"))) != Q005_UPPER_BOUND:
            raise ValueError("Q005 encrypted-noise bounds are invalid.")
        if int(result.get("groups", -1)) != Q005_VECTOR_LENGTH or tuple(result.get("race_order", [])) != Q005_RACE_ORDER:
            raise ValueError("Q005 encrypted-noise grouped domain is invalid.")
        allocation = result.get("budget_allocation", {})
        expected_half = float(epsilon) * 0.5
        if float(allocation.get("epsilon_sum", -1.0)) != expected_half or float(allocation.get("epsilon_count", -1.0)) != expected_half:
            raise ValueError("Q005 encrypted-noise budget allocation is invalid.")
        sensitivities = result.get("sensitivities", {})
        if float(sensitivities.get("sum_vector", -1.0)) != 30.0 or float(sensitivities.get("count_vector", -1.0)) != 1.0:
            raise ValueError("Q005 encrypted-noise sensitivities are invalid.")
        if not isinstance(dp_result, dict) or set(dp_result) != {"bounded_mean_by_race", "unit"} or dp_result["unit"] != "days":
            raise ValueError("Q005 encrypted-noise DP result is invalid.")
        groups = dp_result["bounded_mean_by_race"]
        if not isinstance(groups, dict) or set(groups) != set(Q005_RACE_ORDER):
            raise ValueError("Q005 encrypted-noise result race domain is invalid.")
        for group in groups.values():
            if not isinstance(group, dict) or set(group) != {"bounded_mean"}:
                raise ValueError("Q005 encrypted-noise group result is invalid.")
            mean = float(group["bounded_mean"])
            if not math.isfinite(mean) or not (Q005_LOWER_BOUND <= mean <= Q005_UPPER_BOUND):
                raise ValueError("Q005 encrypted-noise bounded mean is invalid.")

    elif query_id == "Q007":
        bounds = result.get("bounds", {})
        if float(bounds.get("lower", float("nan"))) != Q007_LOWER_BOUND or float(bounds.get("upper", float("nan"))) != Q007_UPPER_BOUND:
            raise ValueError("Q007 encrypted-noise bounds are invalid.")
        predicate = result.get("predicate", {})
        if predicate != {"race": Q007_RACE, "admission_type": Q007_ADMISSION_TYPE}:
            raise ValueError("Q007 encrypted-noise predicate is invalid.")
        allocation = result.get("budget_allocation", {})
        expected_half = float(epsilon) * 0.5
        if float(allocation.get("epsilon_sum", -1.0)) != expected_half or float(allocation.get("epsilon_count", -1.0)) != expected_half:
            raise ValueError("Q007 encrypted-noise budget allocation is invalid.")
        sensitivities = result.get("sensitivities", {})
        if float(sensitivities.get("sum", -1.0)) != 30.0 or float(sensitivities.get("count", -1.0)) != 1.0:
            raise ValueError("Q007 encrypted-noise sensitivities are invalid.")
        if not isinstance(dp_result, dict) or set(dp_result) != {"bounded_mean", "unit"}:
            raise ValueError("Q007 encrypted-noise DP result is invalid.")
        mean = float(dp_result["bounded_mean"])
        if not math.isfinite(mean) or not (Q007_LOWER_BOUND <= mean <= Q007_UPPER_BOUND) or dp_result["unit"] != "days":
            raise ValueError("Q007 encrypted-noise bounded mean is invalid.")

    return result


def _format_extended_encrypted_noise_response(
    query_id: str,
    query_definition: dict,
    request_id: str,
    epsilon: float,
    participating_hospitals: int,
    budget_impact: dict,
    dp_release: dict,
) -> dict:
    response = {
        "request_id": request_id,
        "query_id": query_definition["query_id"],
        "name": query_definition["name"],
        "privacy_unit": query_definition["privacy_unit"],
        "mode": "he_encrypted_noise_dp",
        "mechanism": dp_release["mechanism"],
        "epsilon": float(epsilon),
        "delta": 0.0,
        "dp_noise_applied_under_he": True,
        "noise_service_private_key_available": False,
        "release_stage": dp_release["release_stage"],
        "participating_hospitals": participating_hospitals,
        "dp_result": dp_release["dp_result"],
        "exact_aggregate_exposed": False,
        "budget_impact": budget_impact,
        "budget": get_budget_status(),
    }

    if query_id == "Q001":
        response.update({
            "he_scheme": "BFV",
            "sensitivity": dp_release["sensitivity"],
        })
    elif query_id == "Q002":
        response.update({
            "he_scheme": "BFV",
            "sensitivity": dp_release["sensitivity"],
            "vector_length": dp_release["vector_length"],
            "category_order": dp_release["category_order"],
        })
    elif query_id == "Q003":
        response.update({
            "he_schemes": {"bounded_sum": "CKKS", "count": "BFV"},
            "bounds": dp_release["bounds"],
            "budget_allocation": dp_release["budget_allocation"],
            "sensitivities": dp_release["sensitivities"],
        })
    elif query_id == "Q004":
        response.update({
            "he_scheme": "BFV",
            "sensitivity": dp_release["sensitivity"],
            "cells": dp_release["cells"],
            "dimensions": dp_release["dimensions"],
            "race_order": dp_release["race_order"],
            "admission_type_order": dp_release["admission_type_order"],
        })
    elif query_id == "Q005":
        response.update({
            "he_schemes": {"bounded_sum": "CKKS", "count": "BFV"},
            "bounds": dp_release["bounds"],
            "groups": dp_release["groups"],
            "race_order": dp_release["race_order"],
            "budget_allocation": dp_release["budget_allocation"],
            "sensitivities": dp_release["sensitivities"],
        })
    elif query_id == "Q007":
        response.update({
            "he_schemes": {"bounded_sum": "CKKS", "count": "BFV"},
            "bounds": dp_release["bounds"],
            "predicate": dp_release["predicate"],
            "budget_allocation": dp_release["budget_allocation"],
            "sensitivities": dp_release["sensitivities"],
        })

    return response


def _execute_extended_encrypted_noise_dp(
    query_id: str,
    *,
    epsilon: float,
    confirm_high_spend: bool,
) -> dict:
    query_id = query_id.upper()
    request_id = str(uuid4())
    delta = 0.0
    mode = "he_encrypted_noise_dp"

    try:
        query_definition = get_query(query_id)
    except KeyError as exc:
        log_query_event(
            request_id=request_id, query_id=query_id, status="rejected",
            mode=mode, participating_hospitals=0, epsilon=epsilon,
            delta=delta, error="query_not_approved",
        )
        raise HTTPException(status_code=403, detail=f"{query_id} is not an approved query.") from exc

    allowed_epsilons = query_definition["allowed_epsilons"]
    if epsilon not in allowed_epsilons:
        log_query_event(
            request_id=request_id, query_id=query_id, status="rejected",
            mode=mode, participating_hospitals=0, epsilon=epsilon,
            delta=delta, error="epsilon_not_allowed",
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Requested epsilon is not allowed for {query_id}.",
                "allowed_epsilons": allowed_epsilons,
            },
        )

    try:
        check_budget(epsilon=epsilon, delta=delta)
    except BudgetError as exc:
        log_query_event(
            request_id=request_id, query_id=query_id, status="rejected",
            mode=mode, participating_hospitals=0, epsilon=epsilon,
            delta=delta, error="privacy_budget_exceeded",
        )
        raise HTTPException(status_code=403, detail="Insufficient remaining HE privacy budget.") from exc

    budget_impact = require_high_spend_confirmation(
        query_id=query_id,
        epsilon=epsilon,
        delta=delta,
        request_id=request_id,
        confirm_high_spend=confirm_high_spend,
        mode=mode,
    )

    try:
        reserve_budget(
            query_id=query_id,
            epsilon=epsilon,
            delta=delta,
            request_id=request_id,
        )
    except (BudgetError, ValueError) as exc:
        log_query_event(
            request_id=request_id, query_id=query_id, status="rejected",
            mode=mode, participating_hospitals=0, epsilon=epsilon,
            delta=delta, error="privacy_budget_reservation_failed",
        )
        raise HTTPException(status_code=403, detail="HE privacy budget could not be reserved.") from exc

    try:
        aggregate_result = _ENCRYPTED_NOISE_AGGREGATORS[query_id]()
        participating_hospitals = len(aggregate_result["participating_hospitals"])
    except Exception as exc:
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id, query_id=query_id, status="failed",
            mode=mode, participating_hospitals=5, epsilon=epsilon,
            delta=delta, error="encrypted_distributed_execution_failed",
        )
        raise HTTPException(
            status_code=502,
            detail=f"Encrypted distributed execution of {query_id} failed.",
        ) from exc

    try:
        noise_result = _request_extended_encrypted_noise(
            query_id,
            aggregate_result,
            epsilon=epsilon,
        )
    except Exception as exc:
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id, query_id=query_id, status="failed",
            mode=mode, participating_hospitals=participating_hospitals,
            epsilon=epsilon, delta=delta, error="encrypted_dp_noise_failed",
        )
        raise HTTPException(
            status_code=502,
            detail=f"Encrypted DP-noise addition for {query_id} failed.",
        ) from exc

    try:
        dp_release = _request_extended_encrypted_noise_release(
            query_id,
            noise_result,
            epsilon=epsilon,
        )
    except Exception as exc:
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id, query_id=query_id, status="failed",
            mode=mode, participating_hospitals=participating_hospitals,
            epsilon=epsilon, delta=delta,
            error="trusted_encrypted_noise_release_failed",
        )
        raise HTTPException(
            status_code=502,
            detail=f"Trusted encrypted-noise {query_id} release failed.",
        ) from exc

    try:
        commit_reserved_budget(request_id=request_id)
    except (BudgetError, ValueError) as exc:
        release_budget_reservation(request_id=request_id)
        log_query_event(
            request_id=request_id, query_id=query_id, status="rejected",
            mode=mode, participating_hospitals=participating_hospitals,
            epsilon=epsilon, delta=delta, error="privacy_budget_commit_failed",
        )
        raise HTTPException(
            status_code=403,
            detail="HE reserved privacy budget could not be committed.",
        ) from exc

    log_query_event(
        request_id=request_id,
        query_id=query_id,
        status="success",
        mode=mode,
        participating_hospitals=participating_hospitals,
        epsilon=epsilon,
        delta=delta,
    )

    return _format_extended_encrypted_noise_response(
        query_id,
        query_definition,
        request_id,
        epsilon,
        participating_hospitals,
        budget_impact,
        dp_release,
    )


@app.get("/query/q001/encrypted-noise/dp")
def query_q001_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    return _execute_extended_encrypted_noise_dp(
        "Q001", epsilon=epsilon, confirm_high_spend=confirm_high_spend
    )


@app.get("/query/q002/encrypted-noise/dp")
def query_q002_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    return _execute_extended_encrypted_noise_dp(
        "Q002", epsilon=epsilon, confirm_high_spend=confirm_high_spend
    )


@app.get("/query/q003/encrypted-noise/dp")
def query_q003_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    return _execute_extended_encrypted_noise_dp(
        "Q003", epsilon=epsilon, confirm_high_spend=confirm_high_spend
    )


@app.get("/query/q004/encrypted-noise/dp")
def query_q004_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    return _execute_extended_encrypted_noise_dp(
        "Q004", epsilon=epsilon, confirm_high_spend=confirm_high_spend
    )


@app.get("/query/q005/encrypted-noise/dp")
def query_q005_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    return _execute_extended_encrypted_noise_dp(
        "Q005", epsilon=epsilon, confirm_high_spend=confirm_high_spend
    )


@app.get("/query/q007/encrypted-noise/dp")
def query_q007_encrypted_noise_dp(
    epsilon: float,
    confirm_high_spend: bool = False,
) -> dict:
    return _execute_extended_encrypted_noise_dp(
        "Q007", epsilon=epsilon, confirm_high_spend=confirm_high_spend
    )
