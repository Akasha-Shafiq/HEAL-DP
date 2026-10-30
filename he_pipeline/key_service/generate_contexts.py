"""
Generate persistent HE contexts for the local distributed prototype.

The files are written under .he_runtime/, which must remain excluded
from version control.
"""

from pathlib import Path

import os
from he_pipeline.key_service.key_manager import (
    generate_and_save_bfv_contexts,
    load_context,
)



# BFV context paths
# Configurable for independent research environments.

PUBLIC_BFV_CONTEXT = Path(
    os.getenv(
        "HE_PUBLIC_CONTEXT_PATH",
        ".he_runtime/public/bfv_public_context.bin",
    )
)

PRIVATE_BFV_CONTEXT = Path(
    os.getenv(
        "HE_PRIVATE_CONTEXT_PATH",
        ".he_runtime/private/bfv_private_context.bin",
    )
)



def main() -> None:

    print("=" * 68)
    print("MIMIC-DP HE BFV CONTEXT GENERATION")
    print("=" * 68)

    generate_and_save_bfv_contexts(
        PUBLIC_BFV_CONTEXT,
        PRIVATE_BFV_CONTEXT,
    )

    public_context = load_context(
        PUBLIC_BFV_CONTEXT
    )

    private_context = load_context(
        PRIVATE_BFV_CONTEXT
    )

    print(
        "\nPublic context:",
        PUBLIC_BFV_CONTEXT,
    )

    print(
        "Public context is public:",
        public_context.is_public(),
    )

    print(
        "\nPrivate context:",
        PRIVATE_BFV_CONTEXT,
    )

    print(
        "Private context is private:",
        private_context.is_private(),
    )

    print(
        "\nPublic context size:",
        f"{PUBLIC_BFV_CONTEXT.stat().st_size:,} bytes",
    )

    print(
        "Private context size:",
        f"{PRIVATE_BFV_CONTEXT.stat().st_size:,} bytes",
    )

    assert public_context.is_public()
    assert private_context.is_private()

    print(
        "\nStatus: PASS"
    )


if __name__ == "__main__":
    main()