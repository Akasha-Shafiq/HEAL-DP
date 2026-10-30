"""
Tests for public/private TenSEAL context separation.

These tests model the intended HE trust boundary:

    public context
        -> encryption
        -> homomorphic aggregation

    private context
        -> final decryption

No MIMIC data or existing CDP services are involved.
"""

import pytest
import tenseal as ts

from he_pipeline.key_service.key_manager import (
    create_bfv_private_context,
    create_ckks_private_context,
    create_public_context,
)


def test_bfv_public_private_roundtrip():
    """
    BFV should support exact encrypted integer aggregation using only
    the public context, followed by decryption with the private context.
    """

    private_context = create_bfv_private_context()
    public_context = create_public_context(
        private_context
    )

    assert private_context.is_private()
    assert public_context.is_public()

    encrypted_a = ts.bfv_vector(
        public_context,
        [10],
    )

    encrypted_b = ts.bfv_vector(
        public_context,
        [20],
    )

    encrypted_sum = encrypted_a + encrypted_b

    serialized_result = encrypted_sum.serialize()

    decryptable_result = ts.bfv_vector_from(
        private_context,
        serialized_result,
    )

    result = decryptable_result.decrypt()

    assert result == [30]


def test_ckks_public_private_roundtrip():
    """
    CKKS should support approximate real-valued aggregation using the
    public context and decryption using the private context.
    """

    private_context = create_ckks_private_context()
    public_context = create_public_context(
        private_context
    )

    assert private_context.is_private()
    assert public_context.is_public()

    encrypted_a = ts.ckks_vector(
        public_context,
        [10.25],
    )

    encrypted_b = ts.ckks_vector(
        public_context,
        [20.50],
    )

    encrypted_sum = encrypted_a + encrypted_b

    serialized_result = encrypted_sum.serialize()

    decryptable_result = ts.ckks_vector_from(
        private_context,
        serialized_result,
    )

    result = decryptable_result.decrypt()[0]

    assert result == pytest.approx(
        30.75,
        abs=1e-6,
    )