"""
Independent encrypted DP-noise service for the MIMIC-DP HE pipeline.

Scope
-----
Q001-Q007 encrypted-noise research paths.

The service receives an already-encrypted global Q006 count:

    Enc(global_count)

It then:

    1. samples integer Laplace noise with OpenDP,
    2. encrypts the noise using the PUBLIC BFV context,
    3. homomorphically adds encrypted noise,
    4. returns only:

           Enc(global_count + DP_noise)

The service does NOT:

    - possess the BFV secret key,
    - decrypt the global count,
    - reveal plaintext DP noise,
    - reveal the exact global count,
    - commit privacy budget,
    - perform high-spend confirmation.

Privacy accounting and governance remain coordinator responsibilities.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from he_pipeline.common.serialization import (
    base64_to_bytes,
    bytes_to_base64,
)

from he_pipeline.key_service.key_manager import (
    load_context,
)

from he_pipeline.noise_service.noise_generator import (
    Q006_SENSITIVITY,
    PUBLIC_BFV_CONTEXT_PATH,
    add_encrypted_q006_dp_noise,
)


# ---------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------

PUBLIC_BFV_CONTEXT_PATH = Path(
    PUBLIC_BFV_CONTEXT_PATH
)

if not PUBLIC_BFV_CONTEXT_PATH.exists():
    raise RuntimeError(
        "Noise-service public BFV context not found: "
        f"{PUBLIC_BFV_CONTEXT_PATH}"
    )

public_bfv_context = load_context(
    PUBLIC_BFV_CONTEXT_PATH
)

if not public_bfv_context.is_public():
    raise RuntimeError(
        "Noise service requires a public-only BFV context."
    )


# ---------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------

class Q006EncryptedNoiseRequest(BaseModel):
    """
    Request encrypted DP-noise addition for Q006.

    The input ciphertext already represents the homomorphically
    aggregated global Q006 count.

    No plaintext patient-derived statistic is accepted.
    """

    query_id: str
    ciphertext: str
    ciphertext_bytes: int
    epsilon: float


# ---------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------

app = FastAPI(
    title="MIMIC-DP HE Encrypted Noise Service",
    description=(
        "Independent public-key-only service for adding "
        "differential-privacy noise under homomorphic encryption."
    ),
    version="0.2.0",
)


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """
    Report service capabilities without exposing key material.
    """

    return {
        "status": "ok",
        "service": "he_noise_service",
        "supported_queries": [
            "Q001", "Q002", "Q003", "Q004",
            "Q005", "Q006", "Q007",
        ],
        "he_schemes": [
            "BFV",
            "CKKS",
        ],
        "public_context_loaded": True,
        "public_bfv_context_loaded": True,
        "public_ckks_context_loaded": True,
        "private_key_available": False,
        "encrypted_dp_noise_supported": True,
    }


# ---------------------------------------------------------------------
# Q006 encrypted-noise endpoint
# ---------------------------------------------------------------------

@app.post("/internal/noise/q006")
def add_q006_encrypted_noise(
    request: Q006EncryptedNoiseRequest,
) -> dict:
    """
    Add Q006 integer-Laplace DP noise under BFV encryption.

    Input
    -----
        Enc(global Q006 count)

    Output
    ------
        Enc(global Q006 count + DP noise)

    Plaintext noise is never returned.
    """

    try:

        # -------------------------------------------------------------
        # Validate public query metadata.
        # -------------------------------------------------------------

        if request.query_id != "Q006":
            raise ValueError(
                "Noise service received an unexpected query_id."
            )

        epsilon = float(
            request.epsilon
        )

        if epsilon <= 0.0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        if int(
            request.ciphertext_bytes
        ) <= 0:
            raise ValueError(
                "ciphertext_bytes must be greater than zero."
            )

        # -------------------------------------------------------------
        # Decode encrypted global count.
        # -------------------------------------------------------------

        encrypted_global_count = base64_to_bytes(
            request.ciphertext
        )

        if len(
            encrypted_global_count
        ) != int(
            request.ciphertext_bytes
        ):
            raise ValueError(
                "Q006 encrypted global-count ciphertext "
                "size does not match ciphertext_bytes."
            )

        # -------------------------------------------------------------
        # Add DP noise while the statistic remains encrypted.
        # -------------------------------------------------------------

        encrypted_noisy_count = (
            add_encrypted_q006_dp_noise(
                encrypted_global_count,
                epsilon=epsilon,
            )
        )

        # -------------------------------------------------------------
        # Return ciphertext only.
        # -------------------------------------------------------------

        return {
            "query_id": "Q006",
            "mechanism": "integer_laplace",
            "epsilon": epsilon,
            "delta": 0.0,
            "sensitivity": float(
                Q006_SENSITIVITY
            ),
            "he_scheme": "BFV",
            "encoding": "base64",
            "ciphertext": bytes_to_base64(
                encrypted_noisy_count
            ),
            "ciphertext_bytes": len(
                encrypted_noisy_count
            ),
            "dp_noise_applied_under_he": True,
            "plaintext_noise_exposed": False,
            "exact_aggregate_exposed": False,
            "private_key_available": False,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Q006 encrypted-noise generation failed."
            ),
        ) from exc


# =====================================================================
# Extended encrypted-noise support: Q001-Q005 and Q007
# =====================================================================

from he_pipeline.noise_service.noise_generator import (
    PUBLIC_CKKS_CONTEXT_PATH,
    Q001_SENSITIVITY,
    Q002_SENSITIVITY,
    Q002_VECTOR_LENGTH,
    Q003_SUM_SENSITIVITY,
    Q003_COUNT_SENSITIVITY,
    Q003_SUM_BUDGET_FRACTION,
    Q004_SENSITIVITY,
    Q004_VECTOR_LENGTH,
    Q005_SUM_SENSITIVITY,
    Q005_COUNT_SENSITIVITY,
    Q005_SUM_BUDGET_FRACTION,
    Q005_VECTOR_LENGTH,
    Q007_SUM_SENSITIVITY,
    Q007_COUNT_SENSITIVITY,
    Q007_SUM_BUDGET_FRACTION,
    add_encrypted_q001_dp_noise,
    add_encrypted_q002_dp_noise,
    add_encrypted_q003_dp_noise,
    add_encrypted_q004_dp_noise,
    add_encrypted_q005_dp_noise,
    add_encrypted_q007_dp_noise,
)

PUBLIC_CKKS_CONTEXT_PATH = Path(PUBLIC_CKKS_CONTEXT_PATH)

if not PUBLIC_CKKS_CONTEXT_PATH.exists():
    raise RuntimeError(
        "Noise-service public CKKS context not found: "
        f"{PUBLIC_CKKS_CONTEXT_PATH}"
    )

public_ckks_context = load_context(PUBLIC_CKKS_CONTEXT_PATH)

if not public_ckks_context.is_public():
    raise RuntimeError(
        "Noise service requires a public-only CKKS context."
    )


class BFVEncryptedNoiseRequest(BaseModel):
    query_id: str
    ciphertext: str
    ciphertext_bytes: int
    epsilon: float


class HybridEncryptedNoiseRequest(BaseModel):
    query_id: str
    bounded_sum_ciphertext: str
    bounded_sum_ciphertext_bytes: int
    count_ciphertext: str
    count_ciphertext_bytes: int
    epsilon: float


def _decode_checked_ciphertext(value: str, declared_bytes: int, label: str) -> bytes:
    if int(declared_bytes) <= 0:
        raise ValueError(f"{label}_bytes must be greater than zero.")
    decoded = base64_to_bytes(value)
    if len(decoded) != int(declared_bytes):
        raise ValueError(f"{label} size does not match declared byte length.")
    return decoded


def _common_noise_response(query_id: str, epsilon: float) -> dict:
    return {
        "query_id": query_id,
        "epsilon": float(epsilon),
        "delta": 0.0,
        "encoding": "base64",
        "dp_noise_applied_under_he": True,
        "plaintext_noise_exposed": False,
        "exact_aggregate_exposed": False,
        "private_key_available": False,
    }


@app.post("/internal/noise/q001")
def add_q001_encrypted_noise(request: BFVEncryptedNoiseRequest) -> dict:
    try:
        if request.query_id != "Q001":
            raise ValueError("Unexpected query_id for Q001 Noise Service route.")
        encrypted = _decode_checked_ciphertext(
            request.ciphertext, request.ciphertext_bytes, "ciphertext"
        )
        noisy = add_encrypted_q001_dp_noise(encrypted, epsilon=request.epsilon)
        return {
            **_common_noise_response("Q001", request.epsilon),
            "mechanism": "integer_laplace",
            "sensitivity": float(Q001_SENSITIVITY),
            "he_scheme": "BFV",
            "ciphertext": bytes_to_base64(noisy),
            "ciphertext_bytes": len(noisy),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Q001 encrypted-noise generation failed.") from exc


@app.post("/internal/noise/q002")
def add_q002_encrypted_noise(request: BFVEncryptedNoiseRequest) -> dict:
    try:
        if request.query_id != "Q002":
            raise ValueError("Unexpected query_id for Q002 Noise Service route.")
        encrypted = _decode_checked_ciphertext(
            request.ciphertext, request.ciphertext_bytes, "ciphertext"
        )
        noisy = add_encrypted_q002_dp_noise(encrypted, epsilon=request.epsilon)
        return {
            **_common_noise_response("Q002", request.epsilon),
            "mechanism": "integer_laplace_vector",
            "sensitivity": float(Q002_SENSITIVITY),
            "vector_length": Q002_VECTOR_LENGTH,
            "he_scheme": "BFV",
            "ciphertext": bytes_to_base64(noisy),
            "ciphertext_bytes": len(noisy),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Q002 encrypted-noise generation failed.") from exc


@app.post("/internal/noise/q003")
def add_q003_encrypted_noise(request: HybridEncryptedNoiseRequest) -> dict:
    try:
        if request.query_id != "Q003":
            raise ValueError("Unexpected query_id for Q003 Noise Service route.")
        encrypted_sum = _decode_checked_ciphertext(
            request.bounded_sum_ciphertext,
            request.bounded_sum_ciphertext_bytes,
            "bounded_sum_ciphertext",
        )
        encrypted_count = _decode_checked_ciphertext(
            request.count_ciphertext,
            request.count_ciphertext_bytes,
            "count_ciphertext",
        )
        noisy_sum, noisy_count = add_encrypted_q003_dp_noise(
            encrypted_sum,
            encrypted_count,
            epsilon=request.epsilon,
        )
        epsilon = float(request.epsilon)
        return {
            **_common_noise_response("Q003", epsilon),
            "mechanism": "laplace_sum_and_integer_count",
            "he_schemes": {"bounded_sum": "CKKS", "count": "BFV"},
            "budget_allocation": {
                "epsilon_sum": epsilon * Q003_SUM_BUDGET_FRACTION,
                "epsilon_count": epsilon * (1.0 - Q003_SUM_BUDGET_FRACTION),
            },
            "sensitivities": {
                "sum": float(Q003_SUM_SENSITIVITY),
                "count": float(Q003_COUNT_SENSITIVITY),
            },
            "bounded_sum_ciphertext": bytes_to_base64(noisy_sum),
            "bounded_sum_ciphertext_bytes": len(noisy_sum),
            "count_ciphertext": bytes_to_base64(noisy_count),
            "count_ciphertext_bytes": len(noisy_count),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Q003 encrypted-noise generation failed.") from exc


@app.post("/internal/noise/q004")
def add_q004_encrypted_noise(request: BFVEncryptedNoiseRequest) -> dict:
    try:
        if request.query_id != "Q004":
            raise ValueError("Unexpected query_id for Q004 Noise Service route.")
        encrypted = _decode_checked_ciphertext(
            request.ciphertext, request.ciphertext_bytes, "ciphertext"
        )
        noisy = add_encrypted_q004_dp_noise(encrypted, epsilon=request.epsilon)
        return {
            **_common_noise_response("Q004", request.epsilon),
            "mechanism": "integer_laplace_joint_histogram",
            "sensitivity": float(Q004_SENSITIVITY),
            "vector_length": Q004_VECTOR_LENGTH,
            "he_scheme": "BFV",
            "ciphertext": bytes_to_base64(noisy),
            "ciphertext_bytes": len(noisy),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Q004 encrypted-noise generation failed.") from exc


@app.post("/internal/noise/q005")
def add_q005_encrypted_noise(request: HybridEncryptedNoiseRequest) -> dict:
    try:
        if request.query_id != "Q005":
            raise ValueError("Unexpected query_id for Q005 Noise Service route.")
        encrypted_sums = _decode_checked_ciphertext(
            request.bounded_sum_ciphertext,
            request.bounded_sum_ciphertext_bytes,
            "bounded_sum_ciphertext",
        )
        encrypted_counts = _decode_checked_ciphertext(
            request.count_ciphertext,
            request.count_ciphertext_bytes,
            "count_ciphertext",
        )
        noisy_sums, noisy_counts = add_encrypted_q005_dp_noise(
            encrypted_sums,
            encrypted_counts,
            epsilon=request.epsilon,
        )
        epsilon = float(request.epsilon)
        return {
            **_common_noise_response("Q005", epsilon),
            "mechanism": "laplace_grouped_sum_and_integer_count",
            "he_schemes": {"bounded_sum": "CKKS", "count": "BFV"},
            "vector_length": Q005_VECTOR_LENGTH,
            "budget_allocation": {
                "epsilon_sum": epsilon * Q005_SUM_BUDGET_FRACTION,
                "epsilon_count": epsilon * (1.0 - Q005_SUM_BUDGET_FRACTION),
            },
            "sensitivities": {
                "sum_vector": float(Q005_SUM_SENSITIVITY),
                "count_vector": float(Q005_COUNT_SENSITIVITY),
            },
            "bounded_sum_ciphertext": bytes_to_base64(noisy_sums),
            "bounded_sum_ciphertext_bytes": len(noisy_sums),
            "count_ciphertext": bytes_to_base64(noisy_counts),
            "count_ciphertext_bytes": len(noisy_counts),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Q005 encrypted-noise generation failed.") from exc


@app.post("/internal/noise/q007")
def add_q007_encrypted_noise(request: HybridEncryptedNoiseRequest) -> dict:
    try:
        if request.query_id != "Q007":
            raise ValueError("Unexpected query_id for Q007 Noise Service route.")
        encrypted_sum = _decode_checked_ciphertext(
            request.bounded_sum_ciphertext,
            request.bounded_sum_ciphertext_bytes,
            "bounded_sum_ciphertext",
        )
        encrypted_count = _decode_checked_ciphertext(
            request.count_ciphertext,
            request.count_ciphertext_bytes,
            "count_ciphertext",
        )
        noisy_sum, noisy_count = add_encrypted_q007_dp_noise(
            encrypted_sum,
            encrypted_count,
            epsilon=request.epsilon,
        )
        epsilon = float(request.epsilon)
        return {
            **_common_noise_response("Q007", epsilon),
            "mechanism": "laplace_sum_and_integer_count",
            "he_schemes": {"bounded_sum": "CKKS", "count": "BFV"},
            "budget_allocation": {
                "epsilon_sum": epsilon * Q007_SUM_BUDGET_FRACTION,
                "epsilon_count": epsilon * (1.0 - Q007_SUM_BUDGET_FRACTION),
            },
            "sensitivities": {
                "sum": float(Q007_SUM_SENSITIVITY),
                "count": float(Q007_COUNT_SENSITIVITY),
            },
            "bounded_sum_ciphertext": bytes_to_base64(noisy_sum),
            "bounded_sum_ciphertext_bytes": len(noisy_sum),
            "count_ciphertext": bytes_to_base64(noisy_count),
            "count_ciphertext_bytes": len(noisy_count),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Q007 encrypted-noise generation failed.") from exc
