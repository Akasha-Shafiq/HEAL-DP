"""
Tests for the reusable HE crypto interface.

These tests ensure that platform code can perform encrypted aggregation
without directly depending on TenSEAL implementation details.
"""

import pytest

from he_pipeline.crypto.interface import (
    add_bfv_ciphertexts,
    add_ckks_ciphertexts,
    decrypt_bfv,
    decrypt_ckks,
    encrypt_bfv,
    encrypt_ckks,
)

from he_pipeline.key_service.key_manager import (
    create_bfv_private_context,
    create_ckks_private_context,
    create_public_context,
)


def test_bfv_multi_party_aggregation():
    """
    Simulate several parties encrypting integer values independently.
    """

    private_context = create_bfv_private_context()
    public_context = create_public_context(
        private_context
    )

    values = [
        [10],
        [20],
        [30],
        [40],
        [50],
    ]

    ciphertexts = [
        encrypt_bfv(
            public_context,
            value,
        )
        for value in values
    ]

    encrypted_total = add_bfv_ciphertexts(
        public_context,
        ciphertexts,
    )

    result = decrypt_bfv(
        private_context,
        encrypted_total,
    )

    assert result == [150]


def test_ckks_multi_party_aggregation():
    """
    Simulate several parties encrypting real-valued statistics.
    """

    private_context = create_ckks_private_context()
    public_context = create_public_context(
        private_context
    )

    values = [
        [10.25],
        [20.50],
        [5.75],
    ]

    ciphertexts = [
        encrypt_ckks(
            public_context,
            value,
        )
        for value in values
    ]

    encrypted_total = add_ckks_ciphertexts(
        public_context,
        ciphertexts,
    )

    result = decrypt_ckks(
        private_context,
        encrypted_total,
    )[0]

    assert result == pytest.approx(
        36.50,
        abs=1e-6,
    )


def test_bfv_rejects_empty_aggregation():
    """
    Aggregation should fail clearly if no ciphertexts are supplied.
    """

    private_context = create_bfv_private_context()
    public_context = create_public_context(
        private_context
    )

    with pytest.raises(
        ValueError,
        match="At least one BFV ciphertext",
    ):
        add_bfv_ciphertexts(
            public_context,
            [],
        )