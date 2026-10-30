"""
Configuration parameters for the MIMIC-DP homomorphic-encryption layer.

These parameters are kept separate from the existing CDP platform so that
HE experimentation does not alter the validated plaintext/DP implementation.
"""


# ---------------------------------------------------------------------
# BFV configuration
# Used initially for exact integer-valued sufficient statistics:
# Q001, Q002, Q004, Q006.
# ---------------------------------------------------------------------

BFV_POLY_MODULUS_DEGREE = 8192

BFV_PLAIN_MODULUS = 1032193


# ---------------------------------------------------------------------
# CKKS configuration
# Used later for real-valued sufficient statistics:
# Q003, Q005, Q007.
# ---------------------------------------------------------------------

CKKS_POLY_MODULUS_DEGREE = 8192

CKKS_COEFF_MOD_BIT_SIZES = [
    60,
    40,
    40,
    60,
]

CKKS_GLOBAL_SCALE = 2**40