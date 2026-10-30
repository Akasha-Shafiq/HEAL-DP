"""
Tests for HTTP-safe ciphertext serialization.
"""

import pytest

from he_pipeline.common.serialization import (
    base64_to_bytes,
    bytes_to_base64,
)

from he_pipeline.crypto.interface import (
    decrypt_bfv,
    encrypt_bfv,
)

from he_pipeline.key_service.key_manager import (
    load_context,
)


PUBLIC_CONTEXT_PATH = (
    ".he_runtime/public/bfv_public_context.bin"
)

PRIVATE_CONTEXT_PATH = (
    ".he_runtime/private/bfv_private_context.bin"
)


def test_raw_bytes_base64_roundtrip():
    """
    Arbitrary bytes must survive Base64 encoding exactly.
    """

    original = b"MIMIC-DP-HE-test"

    encoded = bytes_to_base64(
        original
    )

    recovered = base64_to_bytes(
        encoded
    )

    assert recovered == original


def test_bfv_ciphertext_base64_roundtrip():
    """
    A real BFV ciphertext must survive conversion to a Base64
    transport string and back without altering its plaintext result.
    """

    public_context = load_context(
        PUBLIC_CONTEXT_PATH
    )

    private_context = load_context(
        PRIVATE_CONTEXT_PATH
    )

    ciphertext = encrypt_bfv(
        public_context,
        [44_690],
    )

    encoded = bytes_to_base64(
        ciphertext
    )

    recovered_ciphertext = base64_to_bytes(
        encoded
    )

    result = decrypt_bfv(
        private_context,
        recovered_ciphertext,
    )

    assert result == [44_690]


def test_invalid_base64_rejected():
    """
    Invalid transport data should fail clearly.
    """

    with pytest.raises(
        ValueError,
        match="Invalid Base64",
    ):
        base64_to_bytes(
            "this is not valid base64 !!!"
        )