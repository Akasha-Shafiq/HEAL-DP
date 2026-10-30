"""
Trusted HE release/decryption service.

Purpose
-------
This service forms the trusted release boundary of the HE pipeline.

It supports:

Q001
----
1. HE correctness validation
   - obtains the encrypted global Q001 aggregate,
   - decrypts it using the private BFV context,
   - returns the exact result.

   This endpoint exists only for HE correctness testing.

2. HE + Central DP release
   - receives an encrypted global Q001 aggregate,
   - decrypts it only inside this trusted boundary,
   - applies the existing validated OpenDP Laplace mechanism,
   - returns only the differentially private result.

Q002
----
HE + Central DP release
   - receives an encrypted global nine-bin histogram,
   - decrypts it using the private BFV context,
   - applies the existing validated vector-valued OpenDP mechanism,
   - returns only the DP histogram.

Q003
----
HE + Central DP release
   - receives an encrypted CKKS bounded LOS sum,
   - receives an encrypted BFV patient count,
   - decrypts both only inside this trusted boundary,
   - applies the existing validated bounded-mean DP mechanism,
   - returns only the DP bounded mean.

Q004
----
HE + Central DP release
   - receives an encrypted global 297-cell contingency vector,
   - decrypts it using the private BFV context,
   - reconstructs the canonical 33 x 9 contingency table,
   - applies the existing validated OpenDP joint-histogram mechanism,
   - returns only the DP contingency table.

Q005
----
HE + Central DP release
   - receives a CKKS-encrypted 33-element bounded-sum vector,
   - receives a BFV-encrypted 33-element count vector,
   - decrypts both only inside this trusted boundary,
   - applies the existing validated grouped bounded-means mechanism,
   - returns only 33 DP bounded means by race.

Q006
----
HE + Central DP release
   - receives an encrypted global rare-subgroup count,
   - decrypts it using the private BFV context,
   - applies the existing validated OpenDP Laplace count mechanism,
   - returns only the DP count.

Q007
----
HE + Central DP release
   - receives an encrypted CKKS rare-subgroup bounded LOS sum,
   - receives an encrypted BFV rare-subgroup patient count,
   - decrypts both only inside this trusted boundary,
   - applies the existing validated bounded-mean DP mechanism,
   - returns only the DP bounded mean.

The service is the trusted component that has access to the
private HE contexts.

Privacy-budget validation, high-spend confirmation, budget commitment,
and audit logging remain coordinator responsibilities.
"""

import json
import os
from hmac import compare_digest
from pathlib import Path
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from he_pipeline.common.serialization import (
    base64_to_bytes,
)

from he_pipeline.crypto.interface import (
    decrypt_bfv,
)

from he_pipeline.key_service.key_manager import (
    load_context,
)

from he_pipeline.release_service.dp_bridge import (
    privatize_encrypted_q001,
    privatize_encrypted_q002,
    privatize_encrypted_q003,
    privatize_encrypted_q004,
    privatize_encrypted_q005,
    privatize_encrypted_q006,
    privatize_encrypted_q007,
    release_noisy_encrypted_q006,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

COORDINATOR_Q001_URL = os.getenv(
    "HE_COORDINATOR_Q001_URL",
    "http://127.0.0.1:9100/query/q001",
)


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


# ---------------------------------------------------------------------
# Internal caller authentication (research prototype boundary)
# ---------------------------------------------------------------------
# Configure the same independently generated secret in the HE coordinator
# and release-service processes. Never commit this value to source control.
# Missing or weak configuration fails closed at startup.
HE_RELEASE_SERVICE_TOKEN = os.getenv("HE_RELEASE_SERVICE_TOKEN", "")
if len(HE_RELEASE_SERVICE_TOKEN) < 32:
    raise RuntimeError(
        "Set HE_RELEASE_SERVICE_TOKEN to an unpredictable value of "
        "at least 32 characters before starting the HE release service."
    )

# HE correctness route is disabled even for authenticated callers unless
# the operator explicitly enables it for isolated research validation.
HE_ENABLE_EXACT_Q001_DECRYPTION = (
    os.getenv("HE_ENABLE_EXACT_Q001_DECRYPTION", "false").lower()
    == "true"
)


# ---------------------------------------------------------------------
# Validate PRIVATE contexts
# ---------------------------------------------------------------------

if not PRIVATE_BFV_CONTEXT_PATH.exists():
    raise RuntimeError(
        "BFV private context not found: "
        f"{PRIVATE_BFV_CONTEXT_PATH}"
    )


if not PRIVATE_CKKS_CONTEXT_PATH.exists():
    raise RuntimeError(
        "CKKS private context not found: "
        f"{PRIVATE_CKKS_CONTEXT_PATH}"
    )


# ---------------------------------------------------------------------
# Load PRIVATE BFV context
# ---------------------------------------------------------------------

private_bfv_context = load_context(
    PRIVATE_BFV_CONTEXT_PATH
)

if not private_bfv_context.is_private():
    raise RuntimeError(
        "Release service requires a private BFV context."
    )


# ---------------------------------------------------------------------
# Load PRIVATE CKKS context
#
# Loading it here validates the trusted release-service configuration
# at startup. Q003 decryption itself is performed inside dp_bridge.py.
# ---------------------------------------------------------------------

private_ckks_context = load_context(
    PRIVATE_CKKS_CONTEXT_PATH
)

if not private_ckks_context.is_private():
    raise RuntimeError(
        "Release service requires a private CKKS context."
    )


# ---------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------

class Q001DPRequest(BaseModel):
    """
    Internal encrypted Q001 + DP request.

    The coordinator sends only:

        - encrypted global Q001 count,
        - requested epsilon.

    The exact global count is never transmitted as plaintext.
    """

    ciphertext: str
    epsilon: float


class Q002DPRequest(BaseModel):
    """
    Internal encrypted Q002 + DP request.

    The coordinator sends only:

        - encrypted global nine-bin histogram,
        - requested epsilon.

    No exact histogram counts are transmitted as plaintext.
    """

    ciphertext: str
    epsilon: float


class Q003DPRequest(BaseModel):
    """
    Internal hybrid-HE Q003 + DP request.

    The coordinator sends only:

        - CKKS-encrypted global bounded LOS sum,
        - BFV-encrypted global patient count,
        - requested epsilon.

    Neither exact sufficient statistic is transmitted in plaintext.
    """

    bounded_sum_ciphertext: str
    count_ciphertext: str
    epsilon: float


class Q004DPRequest(BaseModel):
    """
    Internal encrypted Q004 + DP request.

    The coordinator sends only:

        - BFV-encrypted global canonical 297-cell vector,
        - requested epsilon.

    No exact contingency-table cell count is transmitted in plaintext.
    """

    ciphertext: str
    epsilon: float


class Q005DPRequest(BaseModel):
    """
    Internal hybrid-HE Q005 + DP request.

    The coordinator sends only:

        - CKKS-encrypted global 33-element bounded-sum vector,
        - BFV-encrypted global 33-element count vector,
        - requested epsilon.

    No exact grouped sufficient statistic is transmitted in plaintext.
    """

    bounded_sum_ciphertext: str
    count_ciphertext: str
    epsilon: float


class Q006DPRequest(BaseModel):
    """
    Internal encrypted Q006 + DP request.

    The coordinator sends only:

        - BFV-encrypted global rare-subgroup count,
        - requested epsilon.

    The exact rare-subgroup count is never transmitted as plaintext.
    """

    ciphertext: str
    epsilon: float


class Q007DPRequest(BaseModel):
    """
    Internal hybrid-HE Q007 + DP request.

    The coordinator sends only:

        - CKKS-encrypted global bounded LOS sum,
        - BFV-encrypted global subgroup patient count,
        - requested epsilon.

    Neither exact sufficient statistic is transmitted in plaintext.
    """

    bounded_sum_ciphertext: str
    count_ciphertext: str
    epsilon: float


# ---------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------

app = FastAPI(
    title="MIMIC-DP HE Trusted Release Service",
    description=(
        "Trusted HE decryption and central-DP release boundary "
        "for the MIMIC-DP prototype."
    ),
    version="0.7.0",
)


# ---------------------------------------------------------------------
# Protect all internal decryption and DP release routes.
# ---------------------------------------------------------------------
@app.middleware("http")
async def require_release_service_authorization(
    request: Request, call_next
):
    if request.url.path.startswith("/internal/"):
        # Preserve a non-disclosing default for the exact correctness path.
        if (
            request.url.path == "/internal/decrypt/q001"
            and not HE_ENABLE_EXACT_Q001_DECRYPTION
        ):
            return JSONResponse(
                status_code=404, content={"detail": "Not found"}
            )

        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {HE_RELEASE_SERVICE_TOKEN}"
        if not compare_digest(supplied, expected):
            return JSONResponse(
                status_code=403,
                content={"detail": "Release-service authorization failed."},
            )

    return await call_next(request)


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """
    Return release-service status without exposing secret material.
    """

    return {
        "status": "ok",
        "service": "he_release_service",
        "he_schemes": [
            "BFV",
            "CKKS",
        ],
        "private_contexts_loaded": True,
        "central_dp_supported": True,
        "supported_queries": [
            "Q001",
            "Q002",
            "Q003",
            "Q004",
            "Q005",
            "Q006",
            "Q007",
        ],
        "encrypted_noise_supported_queries": [
            "Q001", "Q002", "Q003", "Q004",
            "Q005", "Q006", "Q007",
        ],
    }


# =====================================================================
# Q001 exact HE correctness endpoint
# =====================================================================

@app.get("/internal/decrypt/q001", include_in_schema=False)
def decrypt_q001() -> dict:
    """
    Obtain the encrypted global Q001 statistic from the coordinator
    and decrypt it inside the trusted release boundary.

    IMPORTANT
    ---------
    This endpoint exists only for HE correctness validation.

    It returns the exact aggregate and therefore MUST NOT be treated
    as a differentially private user-facing release.
    """

    try:

        # ---------------------------------------------------------
        # 1. Request encrypted global statistic from coordinator
        # ---------------------------------------------------------

        with urlopen(
            COORDINATOR_Q001_URL,
            timeout=60,
        ) as response:

            payload = json.loads(
                response.read().decode("utf-8")
            )


        # ---------------------------------------------------------
        # 2. Validate coordinator response
        # ---------------------------------------------------------

        required_fields = {
            "query_id",
            "statistic",
            "he_scheme",
            "encoding",
            "participating_hospitals",
            "ciphertext",
            "ciphertext_bytes",
        }

        missing_fields = (
            required_fields
            - set(payload.keys())
        )

        if missing_fields:
            raise ValueError(
                "Coordinator response missing fields: "
                f"{sorted(missing_fields)}"
            )

        if "value" in payload:
            raise ValueError(
                "Coordinator unexpectedly returned plaintext."
            )

        if payload["query_id"] != "Q001":
            raise ValueError(
                "Unexpected query returned by coordinator."
            )

        if payload["he_scheme"] != "BFV":
            raise ValueError(
                "Q001 requires BFV."
            )

        if payload["encoding"] != "base64":
            raise ValueError(
                "Unsupported ciphertext encoding."
            )


        # ---------------------------------------------------------
        # 3. Decode ciphertext
        # ---------------------------------------------------------

        ciphertext = base64_to_bytes(
            payload["ciphertext"]
        )

        if (
            len(ciphertext)
            != payload["ciphertext_bytes"]
        ):
            raise ValueError(
                "Aggregate ciphertext size mismatch."
            )


        # ---------------------------------------------------------
        # 4. Trusted exact decryption
        # ---------------------------------------------------------

        decrypted = decrypt_bfv(
            private_bfv_context,
            ciphertext,
        )

        if len(decrypted) != 1:
            raise ValueError(
                "Q001 encrypted statistic must contain exactly "
                "one BFV slot."
            )

        global_count = int(
            decrypted[0]
        )


    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Trusted Q001 decryption failed.",
        ) from exc


    return {
        "query_id": "Q001",
        "statistic": "unique_patient_count",
        "he_scheme": "BFV",
        "participating_hospitals": payload[
            "participating_hospitals"
        ],
        "value": global_count,
        "release_stage": "he_correctness_only",
        "differential_privacy_applied": False,
    }


# =====================================================================
# Q001 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q001")
def release_q001_dp(
    request: Q001DPRequest,
) -> dict:
    """
    Apply central differential privacy to an encrypted global
    Q001 statistic.

    The exact global count is never returned.

    Privacy-budget accounting is deliberately NOT performed here.
    """

    try:

        result = privatize_encrypted_q001(
            request.ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q001 release failed.",
        ) from exc


    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }


# =====================================================================
# Q002 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q002")
def release_q002_dp(
    request: Q002DPRequest,
) -> dict:
    """
    Apply central differential privacy to an encrypted global
    Q002 admission-type histogram.

    The exact global histogram is never returned.

    Privacy-budget accounting is deliberately NOT performed here.
    """

    try:

        result = privatize_encrypted_q002(
            request.ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q002 release failed.",
        ) from exc


    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }


# =====================================================================
# Q003 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q003")
def release_q003_dp(
    request: Q003DPRequest,
) -> dict:
    """
    Apply central differential privacy to encrypted global Q003
    sufficient statistics.

    Q003 uses hybrid homomorphic encryption:

        bounded LOS sum -> CKKS
        patient count   -> BFV

    Workflow
    --------
    1. Receive only the two encrypted global sufficient statistics.
    2. Decrypt both inside the trusted release boundary.
    3. Apply the existing validated bounded-mean OpenDP mechanism.
    4. Split epsilon equally between bounded sum and patient count.
    5. Compute the released bounded mean through DP-safe
       post-processing.
    6. Return only DP-safe output and mechanism metadata.

    The endpoint does NOT return:

        - exact bounded sum,
        - exact patient count,
        - exact mean,
        - noisy bounded sum,
        - noisy patient count.

    Privacy-budget accounting is deliberately NOT performed here.

    The HE coordinator remains responsible for:

        - approved-query validation,
        - epsilon validation,
        - budget pre-check,
        - high-spend confirmation,
        - budget commit,
        - audit logging.
    """

    try:

        result = privatize_encrypted_q003(
            request.bounded_sum_ciphertext,
            request.count_ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q003 release failed.",
        ) from exc


    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }

# =====================================================================
# Q004 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q004")
def release_q004_dp(
    request: Q004DPRequest,
) -> dict:
    """
    Apply central differential privacy to the encrypted global Q004
    race x admission-type contingency table.

    Workflow
    --------
    1. Receive only the encrypted canonical 297-cell BFV vector.
    2. Decrypt it inside the trusted release boundary.
    3. Reconstruct the approved 33 x 9 contingency-table domain.
    4. Apply the existing validated OpenDP joint-histogram mechanism
       with L1 sensitivity 1.
    5. Return only the DP contingency table and public metadata.

    The endpoint does NOT return:

        - the exact decrypted 297-cell vector,
        - the exact contingency table,
        - exact cell counts,
        - the exact global patient total.

    Privacy-budget accounting is deliberately NOT performed here.

    The HE coordinator remains responsible for:

        - approved-query validation,
        - epsilon validation,
        - budget pre-check,
        - high-spend confirmation,
        - budget commit,
        - audit logging.
    """

    try:

        result = privatize_encrypted_q004(
            request.ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q004 release failed.",
        ) from exc


    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }

# =====================================================================
# Q005 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q005")
def release_q005_dp(
    request: Q005DPRequest,
) -> dict:
    """
    Apply central differential privacy to encrypted global Q005
    grouped bounded-LOS sufficient statistics.

    Q005 uses hybrid homomorphic encryption:

        33 bounded LOS sums -> CKKS
        33 patient counts   -> BFV

    Workflow
    --------
    1. Receive only the two encrypted global 33-element vectors.
    2. Decrypt both inside the trusted release boundary.
    3. Validate the fixed 33-race canonical representation.
    4. Apply the existing validated grouped bounded-means OpenDP
       mechanism.
    5. Split epsilon equally between the complete sum vector and the
       complete count vector.
    6. Return only DP bounded means by race and public mechanism
       metadata.

    Under add/remove-one-patient adjacency:

        sum-vector L1 sensitivity   = 30
        count-vector L1 sensitivity = 1

    The endpoint does NOT return:

        - exact bounded-sum vector,
        - exact count vector,
        - exact grouped sufficient statistics,
        - exact per-race means,
        - noisy sum vector,
        - noisy count vector.

    Privacy-budget accounting is deliberately NOT performed here.

    The HE coordinator remains responsible for:

        - approved-query validation,
        - epsilon validation,
        - budget pre-check,
        - high-spend confirmation,
        - budget commit,
        - audit logging.
    """

    try:

        result = privatize_encrypted_q005(
            request.bounded_sum_ciphertext,
            request.count_ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q005 release failed.",
        ) from exc


    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }

# =====================================================================
# Q006 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q006")
def release_q006_dp(
    request: Q006DPRequest,
) -> dict:
    """
    Apply central differential privacy to the encrypted global Q006
    rare-subgroup patient count.

    Workflow
    --------
    1. Receive only the encrypted global BFV count.
    2. Decrypt it inside the trusted release boundary.
    3. Apply the existing validated Laplace count mechanism with
       sensitivity 1.
    4. Return only the DP-safe non-negative count and public metadata.

    The endpoint does NOT return the exact decrypted subgroup count or
    the intermediate noisy real-valued count.

    Privacy-budget accounting is deliberately NOT performed here.
    """

    try:

        result = privatize_encrypted_q006(
            request.ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q006 release failed.",
        ) from exc

    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }


# =====================================================================
# Q006 encrypted-noise HE + DP endpoint
# =====================================================================

@app.post("/internal/encrypted-noise/q006")
def release_q006_encrypted_noise(
    request: Q006DPRequest,
) -> dict:
    """
    Release Q006 after differential-privacy noise has already been
    added while the global count remained encrypted.

    Expected workflow
    -----------------
        Enc(global count)
            ->
        independent noise service
            ->
        Enc(global count + integer Laplace noise)
            ->
        this trusted release service
            ->
        decrypt already-noisy count
            ->
        non-negative DP-safe post-processing
            ->
        released DP count

    IMPORTANT
    ---------
    This endpoint MUST NOT add any additional DP noise.

    Privacy-budget accounting remains a coordinator responsibility.
    """

    try:

        result = release_noisy_encrypted_q006(
            request.ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Encrypted-noise Q006 release failed."
            ),
        ) from exc

    return {
        **result,
        "release_stage": (
            "he_encrypted_noise_plus_central_dp"
        ),
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }



# =====================================================================
# Q007 HE + Central DP endpoint
# =====================================================================

@app.post("/internal/dp/q007")
def release_q007_dp(
    request: Q007DPRequest,
) -> dict:
    """
    Apply central differential privacy to encrypted global Q007
    rare-subgroup bounded-LOS sufficient statistics.

    Q007 uses hybrid homomorphic encryption:

        bounded LOS sum -> CKKS
        subgroup count  -> BFV

    Workflow
    --------
    1. Receive only the two encrypted global sufficient statistics.
    2. Decrypt both inside the trusted release boundary.
    3. Validate the fixed rare-subgroup predicate and [0, 30] LOS bounds.
    4. Apply the existing validated bounded-mean OpenDP mechanism.
    5. Split epsilon equally between bounded sum and patient count.
    6. Return only the DP bounded mean and public mechanism metadata.

    Under add/remove-one-patient adjacency:

        bounded-sum sensitivity = 30
        count sensitivity       = 1

    The endpoint does NOT return exact or noisy sufficient statistics.
    Privacy-budget accounting is deliberately NOT performed here.
    """

    try:

        result = privatize_encrypted_q007(
            request.bounded_sum_ciphertext,
            request.count_ciphertext,
            epsilon=request.epsilon,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="HE + DP Q007 release failed.",
        ) from exc

    return {
        **result,
        "release_stage": "he_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }




# =====================================================================
# Extended encrypted-noise release endpoints: Q001-Q005 and Q007
# =====================================================================

from he_pipeline.release_service.dp_bridge import (
    release_noisy_encrypted_q001,
    release_noisy_encrypted_q002,
    release_noisy_encrypted_q003,
    release_noisy_encrypted_q004,
    release_noisy_encrypted_q005,
    release_noisy_encrypted_q007,
)


def _encrypted_noise_release_response(result: dict) -> dict:
    return {
        **result,
        "release_stage": "he_encrypted_noise_plus_central_dp",
        "exact_aggregate_exposed": False,
        "budget_committed": False,
    }


@app.post("/internal/encrypted-noise/q001")
def release_q001_encrypted_noise(request: Q001DPRequest) -> dict:
    try:
        result = release_noisy_encrypted_q001(
            request.ciphertext,
            epsilon=request.epsilon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Encrypted-noise Q001 release failed.") from exc
    return _encrypted_noise_release_response(result)


@app.post("/internal/encrypted-noise/q002")
def release_q002_encrypted_noise(request: Q002DPRequest) -> dict:
    try:
        result = release_noisy_encrypted_q002(
            request.ciphertext,
            epsilon=request.epsilon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Encrypted-noise Q002 release failed.") from exc
    return _encrypted_noise_release_response(result)


@app.post("/internal/encrypted-noise/q003")
def release_q003_encrypted_noise(request: Q003DPRequest) -> dict:
    try:
        result = release_noisy_encrypted_q003(
            request.bounded_sum_ciphertext,
            request.count_ciphertext,
            epsilon=request.epsilon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Encrypted-noise Q003 release failed.") from exc
    return _encrypted_noise_release_response(result)


@app.post("/internal/encrypted-noise/q004")
def release_q004_encrypted_noise(request: Q004DPRequest) -> dict:
    try:
        result = release_noisy_encrypted_q004(
            request.ciphertext,
            epsilon=request.epsilon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Encrypted-noise Q004 release failed.") from exc
    return _encrypted_noise_release_response(result)


@app.post("/internal/encrypted-noise/q005")
def release_q005_encrypted_noise(request: Q005DPRequest) -> dict:
    try:
        result = release_noisy_encrypted_q005(
            request.bounded_sum_ciphertext,
            request.count_ciphertext,
            epsilon=request.epsilon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Encrypted-noise Q005 release failed.") from exc
    return _encrypted_noise_release_response(result)


@app.post("/internal/encrypted-noise/q007")
def release_q007_encrypted_noise(request: Q007DPRequest) -> dict:
    try:
        result = release_noisy_encrypted_q007(
            request.bounded_sum_ciphertext,
            request.count_ciphertext,
            epsilon=request.epsilon,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Encrypted-noise Q007 release failed.") from exc
    return _encrypted_noise_release_response(result)
