"""
Generate persistent CKKS contexts for the MIMIC-DP HE prototype.

This script deliberately generates CKKS contexts only.

It does NOT regenerate the existing BFV key pair used by the
validated Q001/Q002 pipeline.
"""

from pathlib import Path
import os
from he_pipeline.key_service.key_manager import (
    generate_and_save_ckks_contexts,
    load_context,
)



# CKKS context paths
# Configurable for independent research environments.

PUBLIC_CKKS_CONTEXT = Path(
    os.getenv(
        "HE_PUBLIC_CKKS_CONTEXT_PATH",
        ".he_runtime/public/ckks_public_context.bin",
    )
)

PRIVATE_CKKS_CONTEXT = Path(
    os.getenv(
        "HE_PRIVATE_CKKS_CONTEXT_PATH",
        ".he_runtime/private/ckks_private_context.bin",
    )
)



def main() -> None:

    print("=" * 68)
    print("MIMIC-DP HE CKKS CONTEXT GENERATION")
    print("=" * 68)

    generate_and_save_ckks_contexts(
        PUBLIC_CKKS_CONTEXT,
        PRIVATE_CKKS_CONTEXT,
    )

    public_context = load_context(
        PUBLIC_CKKS_CONTEXT
    )

    private_context = load_context(
        PRIVATE_CKKS_CONTEXT
    )

    print(
        "\nPublic context:",
        PUBLIC_CKKS_CONTEXT,
    )

    print(
        "Public context is public:",
        public_context.is_public(),
    )

    print(
        "\nPrivate context:",
        PRIVATE_CKKS_CONTEXT,
    )

    print(
        "Private context is private:",
        private_context.is_private(),
    )

    print(
        "\nPublic context size:",
        f"{PUBLIC_CKKS_CONTEXT.stat().st_size:,} bytes",
    )

    print(
        "Private context size:",
        f"{PRIVATE_CKKS_CONTEXT.stat().st_size:,} bytes",
    )

    assert public_context.is_public()
    assert private_context.is_private()

    print(
        "\nStatus: PASS"
    )


if __name__ == "__main__":
    main()