"""
TenSEAL context generation for the HE pipeline.

This module creates separate private and public contexts.

Private contexts:
    - contain the secret key,
    - must remain inside the trusted key/release boundary,
    - are required for decryption.

Public contexts:
    - contain no secret key,
    - may be distributed to hospital nodes and the HE coordinator,
    - support encryption and homomorphic evaluation.
"""

import tenseal as ts

from pathlib import Path

from he_pipeline.config.settings import (
    BFV_PLAIN_MODULUS,
    BFV_POLY_MODULUS_DEGREE,
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_GLOBAL_SCALE,
    CKKS_POLY_MODULUS_DEGREE,
)


def create_bfv_private_context() -> ts.Context:
    """
    Create a private BFV TenSEAL context.

    The context contains both public and secret key material.
    It must not be distributed to hospitals or the coordinator.
    """

    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=BFV_POLY_MODULUS_DEGREE,
        plain_modulus=BFV_PLAIN_MODULUS,
    )

    return context


def create_ckks_private_context() -> ts.Context:
    """
    Create a private CKKS TenSEAL context.

    The context contains both public and secret key material.
    """

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=CKKS_COEFF_MOD_BIT_SIZES,
    )

    context.global_scale = CKKS_GLOBAL_SCALE

    return context


def create_public_context(
    private_context: ts.Context,
) -> ts.Context:
    """
    Derive a public-only TenSEAL context from a private context.

    The serialized representation intentionally excludes the secret key.

    At the current stage we use ciphertext addition only, so Galois and
    relinearization keys are not required.
    """

    serialized_public_context = private_context.serialize(
        save_public_key=True,
        save_secret_key=False,
        save_galois_keys=False,
        save_relin_keys=False,
    )

    public_context = ts.context_from(
        serialized_public_context
    )

    return public_context




def save_context(
    context: ts.Context,
    output_path: str | Path,
    *,
    include_secret_key: bool,
) -> None:
    """
    Serialize a TenSEAL context to disk.

    Public contexts must be written with include_secret_key=False.
    Private contexts are written only inside the protected runtime
    directory with include_secret_key=True.
    """

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = context.serialize(
        save_public_key=True,
        save_secret_key=include_secret_key,
        save_galois_keys=False,
        save_relin_keys=False,
    )

    path.write_bytes(
        serialized
    )


def load_context(
    input_path: str | Path,
) -> ts.Context:
    """
    Load a serialized TenSEAL context from disk.
    """

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(
            f"TenSEAL context not found: {path}"
        )

    return ts.context_from(
        path.read_bytes()
    )


def generate_and_save_bfv_contexts(
    public_path: str | Path,
    private_path: str | Path,
) -> None:
    """
    Generate one BFV key pair and persist both views.

    The public context contains no secret key and may be distributed
    to hospital nodes and the coordinator.

    The private context contains the secret key and must remain inside
    the trusted release/key-service boundary.
    """

    private_context = create_bfv_private_context()

    public_context = create_public_context(
        private_context
    )

    save_context(
        public_context,
        public_path,
        include_secret_key=False,
    )

    save_context(
        private_context,
        private_path,
        include_secret_key=True,
    )
    
    
def generate_and_save_ckks_contexts(
    public_path: str | Path,
    private_path: str | Path,
) -> None:
    """
    Generate one CKKS key pair and persist both public and private
    views.

    The public context contains no secret key and may be distributed
    to hospital nodes and the coordinator.

    The private context contains the secret key and must remain inside
    the trusted release/key-service boundary.
    """

    private_context = create_ckks_private_context()

    public_context = create_public_context(
        private_context
    )

    save_context(
        public_context,
        public_path,
        include_secret_key=False,
    )

    save_context(
        private_context,
        private_path,
        include_secret_key=True,
    )