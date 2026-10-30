"""
Serialization helpers for transporting HE ciphertexts between services.

TenSEAL ciphertexts are serialized as raw bytes. HTTP/JSON messages
require a text-safe representation, so ciphertext bytes are encoded
using Base64.

The transformation is:

    TenSEAL ciphertext
        ->
    serialized bytes
        ->
    Base64 string
        ->
    JSON / HTTP

and the reverse operation restores the original bytes exactly.
"""

import base64


def bytes_to_base64(data: bytes) -> str:
    """
    Convert serialized ciphertext bytes to a Base64 string.
    """

    if not isinstance(data, bytes):
        raise TypeError(
            "bytes_to_base64 expects a bytes object."
        )

    return base64.b64encode(
        data
    ).decode("ascii")


def base64_to_bytes(encoded: str) -> bytes:
    """
    Convert a Base64 string back to the original ciphertext bytes.
    """

    if not isinstance(encoded, str):
        raise TypeError(
            "base64_to_bytes expects a string."
        )

    try:
        return base64.b64decode(
            encoded.encode("ascii"),
            validate=True,
        )

    except Exception as exc:
        raise ValueError(
            "Invalid Base64 ciphertext encoding."
        ) from exc