"""
Encrypted differential-privacy noise generation for the HE pipeline.

This module adds DP noise while approved global sufficient statistics remain
homomorphically encrypted. The Noise Service uses PUBLIC HE contexts only.
It never decrypts patient-derived statistics and never commits privacy budget.

Supported encrypted-noise representations
-----------------------------------------
Q001: BFV scalar count
Q002: BFV 9-element histogram
Q003: CKKS bounded sum + BFV count
Q004: BFV 297-element joint histogram
Q005: CKKS 33-element bounded-sum vector + BFV 33-element count vector
Q006: BFV scalar rare-subgroup count
Q007: CKKS rare-subgroup bounded sum + BFV subgroup count
"""

from pathlib import Path
import os
import opendp.prelude as dp

from he_pipeline.crypto.interface import (
    add_bfv_ciphertexts,
    add_ckks_ciphertexts,
    encrypt_bfv,
    encrypt_ckks,
)

from he_pipeline.key_service.key_manager import load_context


dp.enable_features("contrib")


# ---------------------------------------------------------------------
# Public HE contexts
# ---------------------------------------------------------------------


PUBLIC_BFV_CONTEXT_PATH = Path(
    os.getenv(
        "HE_PUBLIC_CONTEXT_PATH",
        ".he_runtime/public/bfv_public_context.bin",
    )
)

PUBLIC_CKKS_CONTEXT_PATH = Path(
    os.getenv(
        "HE_PUBLIC_CKKS_CONTEXT_PATH",
        ".he_runtime/public/ckks_public_context.bin",
    )
)


# ---------------------------------------------------------------------
# Approved query semantics
# ---------------------------------------------------------------------

Q001_SENSITIVITY = 1
Q002_SENSITIVITY = 1
Q002_VECTOR_LENGTH = 9

Q003_SUM_SENSITIVITY = 30.0
Q003_COUNT_SENSITIVITY = 1
Q003_SUM_BUDGET_FRACTION = 0.5

Q004_SENSITIVITY = 1
Q004_VECTOR_LENGTH = 297

Q005_SUM_SENSITIVITY = 30.0
Q005_COUNT_SENSITIVITY = 1
Q005_SUM_BUDGET_FRACTION = 0.5
Q005_VECTOR_LENGTH = 33

Q006_SENSITIVITY = 1

Q007_SUM_SENSITIVITY = 30.0
Q007_COUNT_SENSITIVITY = 1
Q007_SUM_BUDGET_FRACTION = 0.5


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _validate_epsilon(epsilon: float) -> float:
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be greater than zero.")
    return epsilon


def _validate_ciphertext(value: bytes, name: str) -> None:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be serialized bytes.")
    if not value:
        raise ValueError(f"{name} cannot be empty.")


def _load_public_bfv_context():
    context = load_context(PUBLIC_BFV_CONTEXT_PATH)
    if not context.is_public():
        raise RuntimeError(
            "Encrypted-noise generation requires a public-only BFV context."
        )
    return context


def _load_public_ckks_context():
    context = load_context(PUBLIC_CKKS_CONTEXT_PATH)
    if not context.is_public():
        raise RuntimeError(
            "Encrypted-noise generation requires a public-only CKKS context."
        )
    return context


def _sample_integer_laplace_scalar(*, epsilon: float, sensitivity: float) -> int:
    epsilon = _validate_epsilon(epsilon)
    sensitivity = float(sensitivity)
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be greater than zero.")

    mechanism = dp.m.make_laplace(
        dp.atom_domain(T=int),
        dp.absolute_distance(T=int),
        scale=(sensitivity / epsilon),
    )

    noise = mechanism(0)
    if isinstance(noise, bool) or not isinstance(noise, int):
        raise TypeError("OpenDP integer Laplace mechanism returned a non-integer value.")
    return int(noise)


def _sample_integer_laplace_vector(
    *, epsilon: float, sensitivity: float, length: int
) -> list[int]:
    epsilon = _validate_epsilon(epsilon)
    sensitivity = float(sensitivity)
    length = int(length)
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be greater than zero.")
    if length <= 0:
        raise ValueError("length must be greater than zero.")

    # Sampling one discrete-Laplace value per coordinate gives the product
    # mechanism corresponding to an L1-sensitive vector query.
    return [
        _sample_integer_laplace_scalar(
            epsilon=epsilon,
            sensitivity=sensitivity,
        )
        for _ in range(length)
    ]


def _sample_float_laplace_scalar(*, epsilon: float, sensitivity: float) -> float:
    epsilon = _validate_epsilon(epsilon)
    sensitivity = float(sensitivity)
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be greater than zero.")

    mechanism = dp.m.make_laplace(
        dp.atom_domain(T=float, nan=False),
        dp.absolute_distance(T=float),
        scale=(sensitivity / epsilon),
    )
    return float(mechanism(0.0))


def _sample_float_laplace_vector(
    *, epsilon: float, sensitivity: float, length: int
) -> list[float]:
    epsilon = _validate_epsilon(epsilon)
    sensitivity = float(sensitivity)
    length = int(length)
    if sensitivity <= 0.0:
        raise ValueError("sensitivity must be greater than zero.")
    if length <= 0:
        raise ValueError("length must be greater than zero.")

    mechanism = dp.m.make_laplace(
        dp.vector_domain(
            dp.atom_domain(T=float, nan=False),
            size=length,
        ),
        dp.l1_distance(T=float),
        scale=(sensitivity / epsilon),
    )
    values = mechanism([0.0] * length)
    return [float(value) for value in values]


def _add_bfv_noise(encrypted_value: bytes, noise: list[int]) -> bytes:
    _validate_ciphertext(encrypted_value, "encrypted_value")
    public_context = _load_public_bfv_context()
    encrypted_noise = encrypt_bfv(public_context, noise)
    return add_bfv_ciphertexts(
        public_context,
        [encrypted_value, encrypted_noise],
    )


def _add_ckks_noise(encrypted_value: bytes, noise: list[float]) -> bytes:
    _validate_ciphertext(encrypted_value, "encrypted_value")
    public_context = _load_public_ckks_context()
    encrypted_noise = encrypt_ckks(public_context, noise)
    return add_ckks_ciphertexts(
        public_context,
        [encrypted_value, encrypted_noise],
    )


# ---------------------------------------------------------------------
# Q001
# ---------------------------------------------------------------------

def add_encrypted_q001_dp_noise(
    encrypted_global_count: bytes,
    *,
    epsilon: float,
) -> bytes:
    noise = _sample_integer_laplace_scalar(
        epsilon=epsilon,
        sensitivity=Q001_SENSITIVITY,
    )
    return _add_bfv_noise(encrypted_global_count, [noise])


# ---------------------------------------------------------------------
# Q002
# ---------------------------------------------------------------------

def add_encrypted_q002_dp_noise(
    encrypted_global_histogram: bytes,
    *,
    epsilon: float,
) -> bytes:
    noise = _sample_integer_laplace_vector(
        epsilon=epsilon,
        sensitivity=Q002_SENSITIVITY,
        length=Q002_VECTOR_LENGTH,
    )
    return _add_bfv_noise(encrypted_global_histogram, noise)


# ---------------------------------------------------------------------
# Q003
# ---------------------------------------------------------------------

def add_encrypted_q003_dp_noise(
    encrypted_global_sum: bytes,
    encrypted_global_count: bytes,
    *,
    epsilon: float,
) -> tuple[bytes, bytes]:
    epsilon = _validate_epsilon(epsilon)
    epsilon_sum = epsilon * Q003_SUM_BUDGET_FRACTION
    epsilon_count = epsilon * (1.0 - Q003_SUM_BUDGET_FRACTION)

    sum_noise = _sample_float_laplace_scalar(
        epsilon=epsilon_sum,
        sensitivity=Q003_SUM_SENSITIVITY,
    )
    count_noise = _sample_integer_laplace_scalar(
        epsilon=epsilon_count,
        sensitivity=Q003_COUNT_SENSITIVITY,
    )

    return (
        _add_ckks_noise(encrypted_global_sum, [sum_noise]),
        _add_bfv_noise(encrypted_global_count, [count_noise]),
    )


# ---------------------------------------------------------------------
# Q004
# ---------------------------------------------------------------------

def add_encrypted_q004_dp_noise(
    encrypted_global_vector: bytes,
    *,
    epsilon: float,
) -> bytes:
    noise = _sample_integer_laplace_vector(
        epsilon=epsilon,
        sensitivity=Q004_SENSITIVITY,
        length=Q004_VECTOR_LENGTH,
    )
    return _add_bfv_noise(encrypted_global_vector, noise)


# ---------------------------------------------------------------------
# Q005
# ---------------------------------------------------------------------

def add_encrypted_q005_dp_noise(
    encrypted_global_sums: bytes,
    encrypted_global_counts: bytes,
    *,
    epsilon: float,
) -> tuple[bytes, bytes]:
    epsilon = _validate_epsilon(epsilon)
    epsilon_sum = epsilon * Q005_SUM_BUDGET_FRACTION
    epsilon_count = epsilon * (1.0 - Q005_SUM_BUDGET_FRACTION)

    sum_noise = _sample_float_laplace_vector(
        epsilon=epsilon_sum,
        sensitivity=Q005_SUM_SENSITIVITY,
        length=Q005_VECTOR_LENGTH,
    )
    count_noise = _sample_integer_laplace_vector(
        epsilon=epsilon_count,
        sensitivity=Q005_COUNT_SENSITIVITY,
        length=Q005_VECTOR_LENGTH,
    )

    return (
        _add_ckks_noise(encrypted_global_sums, sum_noise),
        _add_bfv_noise(encrypted_global_counts, count_noise),
    )


# ---------------------------------------------------------------------
# Q006 -- retained reference implementation
# ---------------------------------------------------------------------

def sample_q006_integer_laplace_noise(*, epsilon: float) -> int:
    return _sample_integer_laplace_scalar(
        epsilon=epsilon,
        sensitivity=Q006_SENSITIVITY,
    )


def add_encrypted_q006_dp_noise(
    encrypted_global_count: bytes,
    *,
    epsilon: float,
) -> bytes:
    noise = sample_q006_integer_laplace_noise(epsilon=epsilon)
    return _add_bfv_noise(encrypted_global_count, [noise])


# ---------------------------------------------------------------------
# Q007
# ---------------------------------------------------------------------

def add_encrypted_q007_dp_noise(
    encrypted_global_sum: bytes,
    encrypted_global_count: bytes,
    *,
    epsilon: float,
) -> tuple[bytes, bytes]:
    epsilon = _validate_epsilon(epsilon)
    epsilon_sum = epsilon * Q007_SUM_BUDGET_FRACTION
    epsilon_count = epsilon * (1.0 - Q007_SUM_BUDGET_FRACTION)

    sum_noise = _sample_float_laplace_scalar(
        epsilon=epsilon_sum,
        sensitivity=Q007_SUM_SENSITIVITY,
    )
    count_noise = _sample_integer_laplace_scalar(
        epsilon=epsilon_count,
        sensitivity=Q007_COUNT_SENSITIVITY,
    )

    return (
        _add_ckks_noise(encrypted_global_sum, [sum_noise]),
        _add_bfv_noise(encrypted_global_count, [count_noise]),
    )
