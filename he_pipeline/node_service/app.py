"""
HE-enabled hospital node service.

Each hospital:

    1. loads only its own local MIMIC partition,
    2. executes the existing validated query logic,
    3. converts the query result into the approved HE representation,
    4. encrypts sufficient statistics using public HE contexts,
    5. returns only encrypted statistics.

The hospital node does NOT:

    - load BFV or CKKS secret keys,
    - decrypt ciphertexts,
    - aggregate across hospitals,
    - apply differential privacy,
    - manage the global privacy budget.

Currently supported HE queries:

    Q001:
        unique-patient count
        scalar integer -> BFV

    Q002:
        admission-type histogram
        canonical 9-element integer vector -> BFV

    Q003:
        bounded mean LOS sufficient statistics
        bounded sum -> CKKS
        patient count -> BFV

    Q004:
        race x admission-type contingency table
        canonical 297-element integer vector -> BFV

    Q005:
        bounded mean LOS by race sufficient statistics
        canonical 33-element bounded-sum vector -> CKKS
        canonical 33-element count vector -> BFV

    Q006:
        rare-subgroup patient count
        scalar integer -> BFV

    Q007:
        rare-subgroup bounded mean LOS sufficient statistics
        bounded sum -> CKKS
        subgroup patient count -> BFV
"""

import math
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from he_pipeline.common.serialization import (
    bytes_to_base64,
)

from he_pipeline.crypto.interface import (
    encrypt_bfv,
    encrypt_ckks,
)

from he_pipeline.key_service.key_manager import (
    load_context,
)

from he_pipeline.node_service.query_adapter import (
    Q002_CATEGORY_ORDER,
    Q003_LOWER_BOUND,
    Q003_UPPER_BOUND,
    Q004_ADMISSION_TYPE_ORDER,
    Q004_RACE_ORDER,
    Q004_VECTOR_LENGTH,
    Q005_LOWER_BOUND,
    Q005_RACE_ORDER,
    Q005_UPPER_BOUND,
    Q005_VECTOR_LENGTH,
    Q006_ADMISSION_TYPE,
    Q006_RACE,
    Q007_ADMISSION_TYPE,
    Q007_LOWER_BOUND,
    Q007_RACE,
    Q007_UPPER_BOUND,
    execute_q001_from_dataframe,
    execute_q002_from_dataframe,
    execute_q003_from_dataframe,
    execute_q004_from_dataframe,
    execute_q005_from_dataframe,
    execute_q006_from_dataframe,
    execute_q007_from_dataframe,
)


# ---------------------------------------------------------------------
# Hospital configuration
# ---------------------------------------------------------------------

HOSPITAL_ID = os.getenv(
    "HOSPITAL_ID",
    "hospital_1",
)


DATA_FILE = Path(
    os.getenv(
        "DATA_FILE",
        "data/hospital_1/admissions.csv.gz",
    )
)


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
# Validate local resources
# ---------------------------------------------------------------------

if not DATA_FILE.exists():
    raise RuntimeError(
        f"Hospital dataset not found: {DATA_FILE}"
    )


if not PUBLIC_BFV_CONTEXT_PATH.exists():
    raise RuntimeError(
        "BFV public context not found: "
        f"{PUBLIC_BFV_CONTEXT_PATH}"
    )


if not PUBLIC_CKKS_CONTEXT_PATH.exists():
    raise RuntimeError(
        "CKKS public context not found: "
        f"{PUBLIC_CKKS_CONTEXT_PATH}"
    )


# ---------------------------------------------------------------------
# Load hospital data
# ---------------------------------------------------------------------

admissions_df = pd.read_csv(
    DATA_FILE
)


# ---------------------------------------------------------------------
# Load PUBLIC BFV context only
# ---------------------------------------------------------------------

public_bfv_context = load_context(
    PUBLIC_BFV_CONTEXT_PATH
)


if not public_bfv_context.is_public():
    raise RuntimeError(
        "Hospital node must load a public-only BFV context."
    )


# ---------------------------------------------------------------------
# Load PUBLIC CKKS context only
# ---------------------------------------------------------------------

public_ckks_context = load_context(
    PUBLIC_CKKS_CONTEXT_PATH
)


if not public_ckks_context.is_public():
    raise RuntimeError(
        "Hospital node must load a public-only CKKS context."
    )


# ---------------------------------------------------------------------
# FastAPI service
# ---------------------------------------------------------------------

app = FastAPI(
    title="MIMIC-DP HE Hospital Node",
    description=(
        "Hospital-side encrypted sufficient-statistic service "
        "for the MIMIC-DP HE prototype."
    ),
    version="0.7.0",
)


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """
    Return service status without releasing patient-derived
    statistics.
    """

    return {
        "status": "ok",
        "hospital": HOSPITAL_ID,
        "he_schemes": [
            "BFV",
            "CKKS",
        ],
        "public_context_only": True,
        "supported_queries": [
            "Q001",
            "Q002",
            "Q003",
            "Q004",
            "Q005",
            "Q006",
            "Q007",
        ],
    }


# =====================================================================
# Q001
# =====================================================================

@app.get("/query/q001")
def query_q001() -> dict:
    """
    Execute Q001 locally and return only the encrypted count.

    The plaintext local count exists only transiently inside this
    hospital process and is not included in the HTTP response.
    """

    try:

        local_count = execute_q001_from_dataframe(
            admissions_df
        )

        ciphertext = encrypt_bfv(
            public_bfv_context,
            [local_count],
        )

        ciphertext_base64 = bytes_to_base64(
            ciphertext
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q001 execution failed.",
        ) from exc

    return {
        "query_id": "Q001",
        "hospital": HOSPITAL_ID,
        "statistic": "unique_patient_count",
        "he_scheme": "BFV",
        "encoding": "base64",
        "ciphertext": ciphertext_base64,
        "ciphertext_bytes": len(
            ciphertext
        ),
    }


# =====================================================================
# Q002
# =====================================================================

@app.get("/query/q002")
def query_q002() -> dict:
    """
    Execute Q002 locally and return only the encrypted admission-type
    histogram.

    The validated Q002 implementation first assigns each patient to
    the admission type from the patient's earliest admission.

    The HE adapter represents that histogram as a canonical
    nine-element vector whose slot order is fixed by
    ``Q002_CATEGORY_ORDER``.

    The plaintext vector exists only transiently inside this hospital
    process and is never included in the HTTP response.
    """

    try:

        local_vector = execute_q002_from_dataframe(
            admissions_df
        )

        if len(local_vector) != len(
            Q002_CATEGORY_ORDER
        ):
            raise ValueError(
                "Q002 vector does not match the "
                "approved category domain."
            )

        ciphertext = encrypt_bfv(
            public_bfv_context,
            local_vector,
        )

        ciphertext_base64 = bytes_to_base64(
            ciphertext
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q002 execution failed.",
        ) from exc

    return {
        "query_id": "Q002",
        "hospital": HOSPITAL_ID,
        "statistic": "admission_type_histogram",
        "he_scheme": "BFV",
        "encoding": "base64",
        "vector_length": len(
            Q002_CATEGORY_ORDER
        ),
        "category_order": list(
            Q002_CATEGORY_ORDER
        ),
        "ciphertext": ciphertext_base64,
        "ciphertext_bytes": len(
            ciphertext
        ),
    }


# =====================================================================
# Q003
# =====================================================================

@app.get("/query/q003")
def query_q003() -> dict:
    """
    Execute Q003 locally and return only encrypted sufficient
    statistics.

    Q003 computes sufficient statistics for the bounded mean LOS:

        bounded LOS sum -> CKKS
        patient count   -> BFV

    Each patient contributes exactly once using the earliest
    admission, and LOS is clipped to the approved interval
    [0, 30] days.

    The plaintext bounded sum and count exist only transiently
    inside this hospital process and are never included in the
    HTTP response.
    """

    try:

        result = execute_q003_from_dataframe(
            admissions_df
        )

        local_sum = float(
            result["bounded_sum"]
        )

        local_count = int(
            result["count"]
        )

        returned_lower = float(
            result["lower"]
        )

        returned_upper = float(
            result["upper"]
        )


        # -------------------------------------------------------------
        # Validate approved Q003 bounds before encryption.
        # -------------------------------------------------------------

        if returned_lower != Q003_LOWER_BOUND:
            raise ValueError(
                "Q003 returned an unexpected lower bound."
            )

        if returned_upper != Q003_UPPER_BOUND:
            raise ValueError(
                "Q003 returned an unexpected upper bound."
            )

        if local_count < 0:
            raise ValueError(
                "Q003 patient count cannot be negative."
            )

        if local_sum < 0.0:
            raise ValueError(
                "Q003 bounded LOS sum cannot be negative."
            )


        # -------------------------------------------------------------
        # CKKS: real-valued bounded LOS sum
        # -------------------------------------------------------------

        encrypted_sum = encrypt_ckks(
            public_ckks_context,
            [local_sum],
        )


        # -------------------------------------------------------------
        # BFV: exact integer patient count
        # -------------------------------------------------------------

        encrypted_count = encrypt_bfv(
            public_bfv_context,
            [local_count],
        )


        encrypted_sum_base64 = bytes_to_base64(
            encrypted_sum
        )

        encrypted_count_base64 = bytes_to_base64(
            encrypted_count
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q003 execution failed.",
        ) from exc

    return {
        "query_id": "Q003",
        "hospital": HOSPITAL_ID,
        "statistic": "bounded_los_sum_and_count",
        "lower": Q003_LOWER_BOUND,
        "upper": Q003_UPPER_BOUND,
        "encoding": "base64",

        "bounded_sum_he_scheme": "CKKS",
        "bounded_sum_ciphertext": (
            encrypted_sum_base64
        ),
        "bounded_sum_ciphertext_bytes": (
            len(encrypted_sum)
        ),

        "count_he_scheme": "BFV",
        "count_ciphertext": (
            encrypted_count_base64
        ),
        "count_ciphertext_bytes": (
            len(encrypted_count)
        ),
    }


# =====================================================================
# Q004
# =====================================================================

@app.get("/query/q004")
def query_q004() -> dict:
    """
    Execute Q004 locally and return only the encrypted race x
    admission-type contingency table.

    Query semantics
    ---------------
    Each patient contributes exactly once using the patient's
    earliest admission.

    The validated Q004 implementation produces a complete:

        33 race categories
            x
        9 admission types
            =
        297-cell joint histogram

    The HE adapter flattens this table into a fixed race-major
    297-element integer vector.

    The complete vector is encrypted using BFV.

    No plaintext cell count, local table, patient total, or exact
    local statistic is included in the HTTP response.
    """

    try:

        # -------------------------------------------------------------
        # 1. Execute validated Q004 query through the HE adapter
        # -------------------------------------------------------------

        local_vector = execute_q004_from_dataframe(
            admissions_df
        )


        # -------------------------------------------------------------
        # 2. Validate canonical Q004 representation
        # -------------------------------------------------------------

        if len(local_vector) != Q004_VECTOR_LENGTH:
            raise ValueError(
                "Q004 vector does not match the approved "
                "297-cell domain."
            )

        if len(Q004_RACE_ORDER) != 33:
            raise ValueError(
                "Q004 race domain must contain exactly "
                "33 categories."
            )

        if len(Q004_ADMISSION_TYPE_ORDER) != 9:
            raise ValueError(
                "Q004 admission-type domain must contain "
                "exactly 9 categories."
            )

        if any(
            not isinstance(value, int)
            for value in local_vector
        ):
            raise ValueError(
                "Q004 vector must contain only integer counts."
            )

        if any(
            value < 0
            for value in local_vector
        ):
            raise ValueError(
                "Q004 vector cannot contain negative counts."
            )


        # -------------------------------------------------------------
        # 3. Encrypt complete 297-element vector using BFV
        # -------------------------------------------------------------

        ciphertext = encrypt_bfv(
            public_bfv_context,
            local_vector,
        )

        ciphertext_base64 = bytes_to_base64(
            ciphertext
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q004 execution failed.",
        ) from exc


    # -----------------------------------------------------------------
    # Return only ciphertext and fixed public-domain metadata.
    # -----------------------------------------------------------------

    return {
        "query_id": "Q004",
        "hospital": HOSPITAL_ID,
        "statistic": (
            "race_admission_type_contingency"
        ),
        "he_scheme": "BFV",
        "encoding": "base64",

        "dimensions": {
            "race_categories": len(
                Q004_RACE_ORDER
            ),
            "admission_types": len(
                Q004_ADMISSION_TYPE_ORDER
            ),
        },

        "vector_length": Q004_VECTOR_LENGTH,

        "race_order": list(
            Q004_RACE_ORDER
        ),

        "admission_type_order": list(
            Q004_ADMISSION_TYPE_ORDER
        ),

        "ciphertext": ciphertext_base64,

        "ciphertext_bytes": len(
            ciphertext
        ),
    }

# =====================================================================
# Q005
# =====================================================================

@app.get("/query/q005")
def query_q005() -> dict:
    """
    Execute Q005 locally and return only encrypted grouped bounded-LOS
    sufficient statistics.

    Query semantics
    ---------------
    Each patient contributes exactly once using the patient's earliest
    admission.

    The validated Q005 implementation groups patients by the fixed
    33-race domain and computes, for every race:

        bounded LOS sum
        patient count

    LOS is measured in days and clipped to the approved interval
    [0, 30].

    HE representation
    -----------------
    The adapter converts the grouped result into two canonical vectors
    using exactly the same race order:

        33 bounded LOS sums -> CKKS
        33 patient counts   -> BFV

    The plaintext vectors exist only transiently inside this hospital
    process and are never included in the HTTP response.
    """

    try:

        # -------------------------------------------------------------
        # 1. Execute validated Q005 query through the HE adapter
        # -------------------------------------------------------------

        result = execute_q005_from_dataframe(
            admissions_df
        )

        bounded_sums = result[
            "bounded_sums"
        ]

        counts = result[
            "counts"
        ]

        returned_lower = float(
            result[
                "lower"
            ]
        )

        returned_upper = float(
            result[
                "upper"
            ]
        )


        # -------------------------------------------------------------
        # 2. Validate canonical Q005 representation
        # -------------------------------------------------------------

        if len(
            Q005_RACE_ORDER
        ) != 33:
            raise ValueError(
                "Q005 race domain must contain exactly "
                "33 categories."
            )

        if Q005_VECTOR_LENGTH != 33:
            raise ValueError(
                "Q005 vector length must equal 33."
            )

        if len(
            bounded_sums
        ) != Q005_VECTOR_LENGTH:
            raise ValueError(
                "Q005 bounded-sum vector does not match "
                "the approved 33-race domain."
            )

        if len(
            counts
        ) != Q005_VECTOR_LENGTH:
            raise ValueError(
                "Q005 count vector does not match "
                "the approved 33-race domain."
            )

        if returned_lower != Q005_LOWER_BOUND:
            raise ValueError(
                "Q005 returned an unexpected lower bound."
            )

        if returned_upper != Q005_UPPER_BOUND:
            raise ValueError(
                "Q005 returned an unexpected upper bound."
            )

        if returned_lower >= returned_upper:
            raise ValueError(
                "Q005 requires lower < upper."
            )


        # -------------------------------------------------------------
        # 3. Validate bounded-sum and count slots before encryption
        # -------------------------------------------------------------

        for index in range(
            Q005_VECTOR_LENGTH
        ):

            bounded_sum = bounded_sums[
                index
            ]

            count = counts[
                index
            ]

            if isinstance(
                bounded_sum,
                bool,
            ):
                raise ValueError(
                    "Q005 bounded sums must be numeric, "
                    "not booleans."
                )

            bounded_sum = float(
                bounded_sum
            )

            if not math.isfinite(
                bounded_sum
            ):
                raise ValueError(
                    "Q005 bounded sums must be finite."
                )

            if isinstance(
                count,
                bool,
            ):
                raise ValueError(
                    "Q005 counts must be integers, "
                    "not booleans."
                )

            if not isinstance(
                count,
                int,
            ):
                raise ValueError(
                    "Q005 counts must be exact integers."
                )

            if count < 0:
                raise ValueError(
                    "Q005 counts cannot be negative."
                )

            minimum_sum = (
                float(count)
                * returned_lower
            )

            maximum_sum = (
                float(count)
                * returned_upper
            )

            tolerance = 1e-9

            if bounded_sum < (
                minimum_sum
                - tolerance
            ):
                raise ValueError(
                    "Q005 bounded sum is below the valid "
                    "count-scaled lower bound."
                )

            if bounded_sum > (
                maximum_sum
                + tolerance
            ):
                raise ValueError(
                    "Q005 bounded sum exceeds the valid "
                    "count-scaled upper bound."
                )


        # -------------------------------------------------------------
        # 4. CKKS: real-valued bounded-sum vector
        # -------------------------------------------------------------

        encrypted_sums = encrypt_ckks(
            public_ckks_context,
            bounded_sums,
        )


        # -------------------------------------------------------------
        # 5. BFV: exact integer count vector
        # -------------------------------------------------------------

        encrypted_counts = encrypt_bfv(
            public_bfv_context,
            counts,
        )


        # -------------------------------------------------------------
        # 6. Encode ciphertexts for HTTP transport
        # -------------------------------------------------------------

        encrypted_sums_base64 = bytes_to_base64(
            encrypted_sums
        )

        encrypted_counts_base64 = bytes_to_base64(
            encrypted_counts
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q005 execution failed.",
        ) from exc


    # -----------------------------------------------------------------
    # Return only ciphertexts and fixed public-domain metadata.
    # -----------------------------------------------------------------

    return {
        "query_id": "Q005",
        "hospital": HOSPITAL_ID,
        "statistic": (
            "grouped_bounded_los_sum_and_count"
        ),
        "lower": Q005_LOWER_BOUND,
        "upper": Q005_UPPER_BOUND,
        "encoding": "base64",

        "vector_length": Q005_VECTOR_LENGTH,

        "race_order": list(
            Q005_RACE_ORDER
        ),

        "bounded_sum_he_scheme": "CKKS",

        "bounded_sum_ciphertext": (
            encrypted_sums_base64
        ),

        "bounded_sum_ciphertext_bytes": (
            len(
                encrypted_sums
            )
        ),

        "count_he_scheme": "BFV",

        "count_ciphertext": (
            encrypted_counts_base64
        ),

        "count_ciphertext_bytes": (
            len(
                encrypted_counts
            )
        ),
    }


# =====================================================================
# Q006
# =====================================================================

@app.get("/query/q006")
def query_q006() -> dict:
    """
    Execute Q006 locally and return only the encrypted rare-subgroup
    patient count.

    Query semantics
    ---------------
    Each patient contributes at most once. The patient's earliest
    admission is selected first, and that admission must satisfy the
    fixed approved predicate:

        race = "ASIAN - KOREAN"
        admission_type = "URGENT"

    HE representation
    -----------------
    The exact local subgroup count is encrypted as one BFV slot.

    The plaintext count exists only transiently inside this hospital
    process and is never included in the HTTP response.
    """

    try:

        # -------------------------------------------------------------
        # 1. Execute validated Q006 query through the HE adapter.
        # -------------------------------------------------------------

        local_count = execute_q006_from_dataframe(
            admissions_df
        )


        # -------------------------------------------------------------
        # 2. Validate the approved scalar representation.
        # -------------------------------------------------------------

        if isinstance(
            local_count,
            bool,
        ):
            raise ValueError(
                "Q006 count must be an integer, not a boolean."
            )

        if not isinstance(
            local_count,
            int,
        ):
            raise ValueError(
                "Q006 count must be an exact integer."
            )

        if local_count < 0:
            raise ValueError(
                "Q006 count cannot be negative."
            )


        # -------------------------------------------------------------
        # 3. Encrypt the exact local count using BFV.
        # -------------------------------------------------------------

        ciphertext = encrypt_bfv(
            public_bfv_context,
            [local_count],
        )

        ciphertext_base64 = bytes_to_base64(
            ciphertext
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q006 execution failed.",
        ) from exc


    # -----------------------------------------------------------------
    # Return only ciphertext and fixed public query metadata.
    # -----------------------------------------------------------------

    return {
        "query_id": "Q006",
        "hospital": HOSPITAL_ID,
        "statistic": "rare_subgroup_count",
        "predicate": {
            "race": Q006_RACE,
            "admission_type": Q006_ADMISSION_TYPE,
        },
        "he_scheme": "BFV",
        "encoding": "base64",
        "ciphertext": ciphertext_base64,
        "ciphertext_bytes": len(
            ciphertext
        ),
    }


# =====================================================================
# Q007
# =====================================================================

@app.get("/query/q007")
def query_q007() -> dict:
    """
    Execute Q007 locally and return only encrypted rare-subgroup
    bounded-LOS sufficient statistics.

    Query semantics
    ---------------
    Each patient's earliest admission is selected first. Patients are
    then retained only when that admission satisfies the fixed approved
    rare-subgroup predicate:

        race = "ASIAN - KOREAN"
        admission_type = "URGENT"

    Each matching patient contributes one LOS value clipped to the
    approved interval [0, 30] days.

    HE representation
    -----------------

        bounded LOS sum        -> CKKS
        subgroup patient count -> BFV

    The plaintext bounded sum and count exist only transiently inside
    this hospital process and are never included in the HTTP response.
    """

    try:

        # -------------------------------------------------------------
        # 1. Execute validated Q007 query through the HE adapter.
        # -------------------------------------------------------------

        result = execute_q007_from_dataframe(
            admissions_df
        )

        local_sum = float(
            result[
                "bounded_sum"
            ]
        )

        local_count = result[
            "count"
        ]

        returned_lower = float(
            result[
                "lower"
            ]
        )

        returned_upper = float(
            result[
                "upper"
            ]
        )

        returned_predicate = result[
            "predicate"
        ]


        # -------------------------------------------------------------
        # 2. Validate the approved Q007 metadata and representation.
        # -------------------------------------------------------------

        if returned_lower != Q007_LOWER_BOUND:
            raise ValueError(
                "Q007 returned an unexpected lower bound."
            )

        if returned_upper != Q007_UPPER_BOUND:
            raise ValueError(
                "Q007 returned an unexpected upper bound."
            )

        if returned_lower >= returned_upper:
            raise ValueError(
                "Q007 requires lower < upper."
            )

        if not isinstance(
            returned_predicate,
            dict,
        ):
            raise ValueError(
                "Q007 predicate metadata must be a dictionary."
            )

        if set(
            returned_predicate.keys()
        ) != {
            "race",
            "admission_type",
        }:
            raise ValueError(
                "Q007 predicate metadata contains unexpected fields."
            )

        if returned_predicate[
            "race"
        ] != Q007_RACE:
            raise ValueError(
                "Q007 returned an unexpected race predicate."
            )

        if returned_predicate[
            "admission_type"
        ] != Q007_ADMISSION_TYPE:
            raise ValueError(
                "Q007 returned an unexpected admission-type predicate."
            )

        if isinstance(
            local_count,
            bool,
        ):
            raise ValueError(
                "Q007 count must be an integer, not a boolean."
            )

        if not isinstance(
            local_count,
            int,
        ):
            raise ValueError(
                "Q007 count must be an exact integer."
            )

        if local_count < 0:
            raise ValueError(
                "Q007 count cannot be negative."
            )

        if not math.isfinite(
            local_sum
        ):
            raise ValueError(
                "Q007 bounded sum must be finite."
            )

        minimum_sum = (
            float(local_count)
            * returned_lower
        )

        maximum_sum = (
            float(local_count)
            * returned_upper
        )

        tolerance = 1e-9

        if local_sum < (
            minimum_sum
            - tolerance
        ):
            raise ValueError(
                "Q007 bounded sum is below the valid "
                "count-scaled lower bound."
            )

        if local_sum > (
            maximum_sum
            + tolerance
        ):
            raise ValueError(
                "Q007 bounded sum exceeds the valid "
                "count-scaled upper bound."
            )

        if (
            local_count == 0
            and abs(local_sum) > tolerance
        ):
            raise ValueError(
                "Q007 zero-count result must have bounded_sum = 0."
            )


        # -------------------------------------------------------------
        # 3. CKKS: real-valued bounded LOS sum.
        # -------------------------------------------------------------

        encrypted_sum = encrypt_ckks(
            public_ckks_context,
            [local_sum],
        )


        # -------------------------------------------------------------
        # 4. BFV: exact integer subgroup patient count.
        # -------------------------------------------------------------

        encrypted_count = encrypt_bfv(
            public_bfv_context,
            [local_count],
        )


        # -------------------------------------------------------------
        # 5. Encode ciphertexts for HTTP transport.
        # -------------------------------------------------------------

        encrypted_sum_base64 = bytes_to_base64(
            encrypted_sum
        )

        encrypted_count_base64 = bytes_to_base64(
            encrypted_count
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Encrypted Q007 execution failed.",
        ) from exc


    # -----------------------------------------------------------------
    # Return only ciphertexts and fixed public query metadata.
    # -----------------------------------------------------------------

    return {
        "query_id": "Q007",
        "hospital": HOSPITAL_ID,
        "statistic": (
            "rare_subgroup_bounded_los_sum_and_count"
        ),
        "lower": Q007_LOWER_BOUND,
        "upper": Q007_UPPER_BOUND,
        "predicate": {
            "race": Q007_RACE,
            "admission_type": Q007_ADMISSION_TYPE,
        },
        "encoding": "base64",

        "bounded_sum_he_scheme": "CKKS",

        "bounded_sum_ciphertext": (
            encrypted_sum_base64
        ),

        "bounded_sum_ciphertext_bytes": len(
            encrypted_sum
        ),

        "count_he_scheme": "BFV",

        "count_ciphertext": (
            encrypted_count_base64
        ),

        "count_ciphertext_bytes": len(
            encrypted_count
        ),
    }
