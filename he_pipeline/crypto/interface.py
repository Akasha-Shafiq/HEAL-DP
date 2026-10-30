"""
Reusable homomorphic-encryption interface for the MIMIC-DP HE pipeline.

The rest of the HE platform should use these functions rather than
calling TenSEAL directly.

Current scheme mapping:

    BFV:
        Q001, Q002, Q004, Q006
        exact integer-valued sufficient statistics

    CKKS:
        Q003, Q005, Q007
        approximate real-valued sufficient statistics

Ciphertexts are serialized as bytes so that the same interface can
later be used across HTTP service boundaries.
"""

from collections.abc import Sequence

import tenseal as ts



# BFV
# ---------------------------------------------------------------------

def encrypt_bfv(
    public_context: ts.Context,
    values: Sequence[int],
) -> bytes:
    """
    Encrypt an integer scalar/vector using a public BFV context.

    Returns a serialized ciphertext suitable for storage or transport.
    """

    encrypted = ts.bfv_vector(
        public_context,
        list(values),
    )

    return encrypted.serialize()


def add_bfv_ciphertexts(
    public_context: ts.Context,
    ciphertexts: Sequence[bytes],
) -> bytes:
    """
    Homomorphically add serialized BFV ciphertexts.

    No secret key is required.
    """

    if not ciphertexts:
        raise ValueError(
            "At least one BFV ciphertext is required."
        )

    encrypted_total = ts.bfv_vector_from(
        public_context,
        ciphertexts[0],
    )

    for ciphertext in ciphertexts[1:]:
        encrypted_value = ts.bfv_vector_from(
            public_context,
            ciphertext,
        )

        encrypted_total += encrypted_value

    return encrypted_total.serialize()


def decrypt_bfv(
    private_context: ts.Context,
    ciphertext: bytes,
) -> list[int]:
    """
    Decrypt a serialized BFV ciphertext using the private context.
    """

    encrypted = ts.bfv_vector_from(
        private_context,
        ciphertext,
    )

    return encrypted.decrypt()


# ---------------------------------------------------------------------
# CKKS
# ---------------------------------------------------------------------

def encrypt_ckks(
    public_context: ts.Context,
    values: Sequence[float],
) -> bytes:
    """
    Encrypt a real-valued scalar/vector using a public CKKS context.
    """

    encrypted = ts.ckks_vector(
        public_context,
        [float(value) for value in values],
    )

    return encrypted.serialize()


def add_ckks_ciphertexts(
    public_context: ts.Context,
    ciphertexts: Sequence[bytes],
) -> bytes:
    """
    Homomorphically add serialized CKKS ciphertexts.

    No secret key is required.
    """

    if not ciphertexts:
        raise ValueError(
            "At least one CKKS ciphertext is required."
        )

    encrypted_total = ts.ckks_vector_from(
        public_context,
        ciphertexts[0],
    )

    for ciphertext in ciphertexts[1:]:
        encrypted_value = ts.ckks_vector_from(
            public_context,
            ciphertext,
        )

        encrypted_total += encrypted_value

    return encrypted_total.serialize()


def decrypt_ckks(
    private_context: ts.Context,
    ciphertext: bytes,
) -> list[float]:
    """
    Decrypt a serialized CKKS ciphertext using the private context.
    """

    encrypted = ts.ckks_vector_from(
        private_context,
        ciphertext,
    )

    return encrypted.decrypt()