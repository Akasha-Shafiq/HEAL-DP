"""
Encrypted aggregation logic for the MIMIC-DP HE coordinator.

The coordinator:

    1. requests encrypted sufficient statistics from hospital nodes,
    2. validates the response metadata,
    3. decodes Base64 transport strings back to ciphertext bytes,
    4. homomorphically aggregates ciphertexts using shared PUBLIC
       HE contexts,
    5. returns only encrypted global sufficient statistics.

The coordinator does NOT:

    - access hospital datasets,
    - receive plaintext local statistics,
    - load secret keys,
    - decrypt ciphertexts,
    - apply differential privacy at this stage.

Currently supported encrypted aggregation:

    Q001:
        unique-patient scalar count
        BFV

    Q002:
        canonical 9-element admission-type histogram
        BFV

    Q003:
        bounded LOS sufficient statistics
        bounded sum -> CKKS
        patient count -> BFV

    Q004:
        canonical 33 x 9 race x admission-type joint histogram
        297-element integer vector -> BFV

    Q005:
        grouped bounded-LOS sufficient statistics by race
        canonical 33-element bounded-sum vector -> CKKS
        canonical 33-element count vector -> BFV

    Q006:
        rare-subgroup patient count
        scalar integer -> BFV

    Q007:
        rare-subgroup bounded-mean sufficient statistics
        bounded LOS sum -> CKKS
        subgroup patient count -> BFV
"""

import json
import os
from collections.abc import Sequence
from urllib.request import urlopen

from he_pipeline.common.serialization import (
    base64_to_bytes,
    bytes_to_base64,
)

from he_pipeline.crypto.interface import (
    add_bfv_ciphertexts,
    add_ckks_ciphertexts,
)

from he_pipeline.key_service.key_manager import (
    load_context,
)

from node.queries import (
    Q004_ADMISSION_TYPES,
    Q004_RACE_CATEGORIES,
)

from he_pipeline.node_service.query_adapter import (
    Q006_ADMISSION_TYPE,
    Q006_RACE,
    Q007_ADMISSION_TYPE,
    Q007_LOWER_BOUND,
    Q007_RACE,
    Q007_UPPER_BOUND,
)


# ---------------------------------------------------------------------
# Hospital endpoints
# ---------------------------------------------------------------------
# Configure HE_HOSPITAL_1_URL ... HE_HOSPITAL_5_URL with base service URLs.
# Values are read at import/startup; restart the coordinator after changes.
# Local defaults retain the original five-node experimental configuration.

HE_HOSPITAL_BASE_URLS = [
    os.getenv(
        f"HE_HOSPITAL_{i}_URL",
        f"http://127.0.0.1:{8100 + i}",
    ).rstrip("/")
    for i in range(1, 6)
]

DEFAULT_HOSPITAL_Q001_URLS = [f"{base}/query/q001" for base in HE_HOSPITAL_BASE_URLS]
DEFAULT_HOSPITAL_Q002_URLS = [f"{base}/query/q002" for base in HE_HOSPITAL_BASE_URLS]
DEFAULT_HOSPITAL_Q003_URLS = [f"{base}/query/q003" for base in HE_HOSPITAL_BASE_URLS]
DEFAULT_HOSPITAL_Q004_URLS = [f"{base}/query/q004" for base in HE_HOSPITAL_BASE_URLS]
DEFAULT_HOSPITAL_Q005_URLS = [f"{base}/query/q005" for base in HE_HOSPITAL_BASE_URLS]
DEFAULT_HOSPITAL_Q006_URLS = [f"{base}/query/q006" for base in HE_HOSPITAL_BASE_URLS]
DEFAULT_HOSPITAL_Q007_URLS = [f"{base}/query/q007" for base in HE_HOSPITAL_BASE_URLS]


# ---------------------------------------------------------------------
# Public HE contexts
# ---------------------------------------------------------------------
# These must be public-only contexts. Private contexts are never loaded
# into the encrypted aggregator.

PUBLIC_BFV_CONTEXT_PATH = os.getenv(
    "HE_PUBLIC_CONTEXT_PATH",
    ".he_runtime/public/bfv_public_context.bin",
)

PUBLIC_CKKS_CONTEXT_PATH = os.getenv(
    "HE_PUBLIC_CKKS_CONTEXT_PATH",
    ".he_runtime/public/ckks_public_context.bin",
)


# ---------------------------------------------------------------------
# Approved Q002 vector domain
# ---------------------------------------------------------------------

Q002_CATEGORY_ORDER = (
    "AMBULATORY OBSERVATION",
    "DIRECT EMER.",
    "DIRECT OBSERVATION",
    "ELECTIVE",
    "EU OBSERVATION",
    "EW EMER.",
    "OBSERVATION ADMIT",
    "SURGICAL SAME DAY ADMISSION",
    "URGENT",
)


# ---------------------------------------------------------------------
# Approved Q003 LOS bounds
# ---------------------------------------------------------------------

Q003_LOWER_BOUND = 0.0
Q003_UPPER_BOUND = 30.0


# ---------------------------------------------------------------------
# Approved Q004 vector domain
# ---------------------------------------------------------------------

Q004_RACE_ORDER = tuple(
    Q004_RACE_CATEGORIES
)

Q004_ADMISSION_TYPE_ORDER = tuple(
    Q004_ADMISSION_TYPES
)

Q004_VECTOR_LENGTH = (
    len(Q004_RACE_ORDER)
    * len(Q004_ADMISSION_TYPE_ORDER)
)

if len(Q004_RACE_ORDER) != 33:
    raise RuntimeError(
        "Q004 canonical race domain must contain 33 categories."
    )

if len(Q004_ADMISSION_TYPE_ORDER) != 9:
    raise RuntimeError(
        "Q004 canonical admission-type domain must contain 9 categories."
    )

if Q004_VECTOR_LENGTH != 297:
    raise RuntimeError(
        "Q004 canonical HE vector must contain 297 slots."
    )


# ---------------------------------------------------------------------
# Approved Q005 grouped bounded-LOS domain
# ---------------------------------------------------------------------

Q005_RACE_ORDER = tuple(
    Q004_RACE_ORDER
)

Q005_VECTOR_LENGTH = len(
    Q005_RACE_ORDER
)

Q005_LOWER_BOUND = 0.0
Q005_UPPER_BOUND = 30.0

if len(Q005_RACE_ORDER) != 33:
    raise RuntimeError(
        "Q005 canonical race domain must contain 33 categories."
    )

if Q005_VECTOR_LENGTH != 33:
    raise RuntimeError(
        "Q005 canonical HE vectors must contain 33 slots."
    )


# =====================================================================
# Q001
# =====================================================================

def fetch_encrypted_q001(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request one encrypted Q001 statistic from a hospital node.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "he_scheme",
        "encoding",
        "ciphertext",
        "ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Hospital response is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    if "value" in payload:
        raise ValueError(
            "Hospital response unexpectedly contains "
            "a plaintext value."
        )

    if payload["query_id"] != "Q001":
        raise ValueError(
            "Unexpected query_id returned by hospital."
        )

    if payload["statistic"] != "unique_patient_count":
        raise ValueError(
            "Unexpected Q001 statistic type."
        )

    if payload["he_scheme"] != "BFV":
        raise ValueError(
            "Q001 requires BFV ciphertexts."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported ciphertext encoding."
        )

    return payload


def aggregate_q001_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q001_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q001 results.

    Returns an encrypted global statistic encoded as Base64.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    public_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    if not public_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    hospital_ciphertexts = []
    participating_hospitals = []
    hospital_ciphertext_sizes = {}

    seen_hospitals = set()

    for hospital_url in hospital_urls:

        payload = fetch_encrypted_q001(
            hospital_url
        )

        hospital_id = payload["hospital"]

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q001 aggregation: {hospital_id}"
            )

        seen_hospitals.add(
            hospital_id
        )

        ciphertext = base64_to_bytes(
            payload["ciphertext"]
        )

        if len(ciphertext) != payload["ciphertext_bytes"]:
            raise ValueError(
                "Ciphertext size mismatch for "
                f"{hospital_id}."
            )

        hospital_ciphertexts.append(
            ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = len(ciphertext)

    encrypted_global = add_bfv_ciphertexts(
        public_context,
        hospital_ciphertexts,
    )

    return {
        "query_id": "Q001",
        "statistic": "unique_patient_count",
        "he_scheme": "BFV",
        "encoding": "base64",
        "participating_hospitals": (
            participating_hospitals
        ),
        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),
        "ciphertext": bytes_to_base64(
            encrypted_global
        ),
        "ciphertext_bytes": len(
            encrypted_global
        ),
    }


# =====================================================================
# Q002
# =====================================================================

def fetch_encrypted_q002(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request one encrypted Q002 histogram from a hospital node.

    Only ciphertext and fixed query metadata are accepted.
    Plaintext histogram representations are explicitly rejected.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "he_scheme",
        "encoding",
        "vector_length",
        "category_order",
        "ciphertext",
        "ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q002 hospital response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_plaintext_fields = {
        "value",
        "vector",
        "histogram",
        "counts",
        "global_result",
    }

    leaked_fields = (
        forbidden_plaintext_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Q002 hospital response unexpectedly contains "
            "plaintext statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q002":
        raise ValueError(
            "Unexpected query_id returned by hospital "
            "for Q002 aggregation."
        )

    if (
        payload["statistic"]
        != "admission_type_histogram"
    ):
        raise ValueError(
            "Unexpected Q002 statistic type."
        )

    if payload["he_scheme"] != "BFV":
        raise ValueError(
            "Q002 requires BFV ciphertexts."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported Q002 ciphertext encoding."
        )

    if int(payload["vector_length"]) != 9:
        raise ValueError(
            "Q002 encrypted vector must contain exactly "
            "nine slots."
        )

    returned_category_order = tuple(
        payload["category_order"]
    )

    if returned_category_order != Q002_CATEGORY_ORDER:
        raise ValueError(
            "Q002 hospital category order does not match "
            "the approved canonical domain."
        )

    return payload


def aggregate_q002_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q002_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q002 vectors.

    The coordinator never decrypts the result. BFV addition acts
    element-wise over the canonical nine histogram slots.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    public_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    if not public_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    hospital_ciphertexts = []
    participating_hospitals = []
    hospital_ciphertext_sizes = {}

    seen_hospitals = set()

    for hospital_url in hospital_urls:

        payload = fetch_encrypted_q002(
            hospital_url
        )

        hospital_id = payload["hospital"]

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q002 aggregation: {hospital_id}"
            )

        seen_hospitals.add(
            hospital_id
        )

        ciphertext = base64_to_bytes(
            payload["ciphertext"]
        )

        if len(ciphertext) != payload["ciphertext_bytes"]:
            raise ValueError(
                "Q002 ciphertext size mismatch for "
                f"{hospital_id}."
            )

        hospital_ciphertexts.append(
            ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = len(ciphertext)

    encrypted_global = add_bfv_ciphertexts(
        public_context,
        hospital_ciphertexts,
    )

    return {
        "query_id": "Q002",
        "statistic": "admission_type_histogram",
        "he_scheme": "BFV",
        "encoding": "base64",
        "vector_length": len(
            Q002_CATEGORY_ORDER
        ),
        "category_order": list(
            Q002_CATEGORY_ORDER
        ),
        "participating_hospitals": (
            participating_hospitals
        ),
        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),
        "ciphertext": bytes_to_base64(
            encrypted_global
        ),
        "ciphertext_bytes": len(
            encrypted_global
        ),
    }


# =====================================================================
# Q003
# =====================================================================

def fetch_encrypted_q003(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request encrypted Q003 sufficient statistics from one hospital.

    Q003 uses a hybrid HE representation:

        bounded LOS sum -> CKKS
        patient count   -> BFV

    Only encrypted sufficient statistics and fixed public metadata
    are accepted.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "lower",
        "upper",
        "encoding",
        "bounded_sum_he_scheme",
        "bounded_sum_ciphertext",
        "bounded_sum_ciphertext_bytes",
        "count_he_scheme",
        "count_ciphertext",
        "count_ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q003 hospital response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    # -------------------------------------------------------------
    # Reject accidental plaintext sufficient-statistic leakage.
    # -------------------------------------------------------------

    forbidden_plaintext_fields = {
        "bounded_sum",
        "count",
        "mean",
        "value",
        "global_result",
        "exact_sum",
        "exact_count",
        "exact_mean",
    }

    leaked_fields = (
        forbidden_plaintext_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Q003 hospital response unexpectedly contains "
            "plaintext statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    # -------------------------------------------------------------
    # Validate query metadata.
    # -------------------------------------------------------------

    if payload["query_id"] != "Q003":
        raise ValueError(
            "Unexpected query_id returned by hospital "
            "for Q003 aggregation."
        )

    if (
        payload["statistic"]
        != "bounded_los_sum_and_count"
    ):
        raise ValueError(
            "Unexpected Q003 statistic type."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported Q003 ciphertext encoding."
        )

    if (
        payload["bounded_sum_he_scheme"]
        != "CKKS"
    ):
        raise ValueError(
            "Q003 bounded sum requires CKKS."
        )

    if payload["count_he_scheme"] != "BFV":
        raise ValueError(
            "Q003 patient count requires BFV."
        )

    # -------------------------------------------------------------
    # Validate approved contribution bounds.
    # -------------------------------------------------------------

    returned_lower = float(
        payload["lower"]
    )

    returned_upper = float(
        payload["upper"]
    )

    if returned_lower != Q003_LOWER_BOUND:
        raise ValueError(
            "Q003 hospital returned an unexpected "
            "lower LOS bound."
        )

    if returned_upper != Q003_UPPER_BOUND:
        raise ValueError(
            "Q003 hospital returned an unexpected "
            "upper LOS bound."
        )

    return payload


def aggregate_q003_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q003_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q003 sufficient
    statistics.

    The real-valued bounded LOS sums are aggregated using CKKS.

    The integer patient counts are aggregated independently using
    BFV.

    The coordinator uses public contexts only and never decrypts
    either global statistic.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    # -------------------------------------------------------------
    # Load public HE contexts only.
    # -------------------------------------------------------------

    public_bfv_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    public_ckks_context = load_context(
        PUBLIC_CKKS_CONTEXT_PATH
    )

    if not public_bfv_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    if not public_ckks_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only CKKS context."
        )

    # -------------------------------------------------------------
    # Collect encrypted hospital statistics.
    # -------------------------------------------------------------

    hospital_sum_ciphertexts = []
    hospital_count_ciphertexts = []

    participating_hospitals = []

    hospital_ciphertext_sizes = {}

    seen_hospitals = set()

    for hospital_url in hospital_urls:

        payload = fetch_encrypted_q003(
            hospital_url
        )

        hospital_id = payload["hospital"]

        # ---------------------------------------------------------
        # Prevent accidental double-counting if multiple URLs point
        # to the same hospital process.
        # ---------------------------------------------------------

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q003 aggregation: {hospital_id}"
            )

        seen_hospitals.add(
            hospital_id
        )

        # ---------------------------------------------------------
        # Decode CKKS bounded-sum ciphertext.
        # ---------------------------------------------------------

        sum_ciphertext = base64_to_bytes(
            payload["bounded_sum_ciphertext"]
        )

        expected_sum_size = int(
            payload[
                "bounded_sum_ciphertext_bytes"
            ]
        )

        if len(sum_ciphertext) != expected_sum_size:
            raise ValueError(
                "Q003 CKKS bounded-sum ciphertext size "
                f"mismatch for {hospital_id}."
            )

        # ---------------------------------------------------------
        # Decode BFV count ciphertext.
        # ---------------------------------------------------------

        count_ciphertext = base64_to_bytes(
            payload["count_ciphertext"]
        )

        expected_count_size = int(
            payload[
                "count_ciphertext_bytes"
            ]
        )

        if len(count_ciphertext) != expected_count_size:
            raise ValueError(
                "Q003 BFV count ciphertext size "
                f"mismatch for {hospital_id}."
            )

        hospital_sum_ciphertexts.append(
            sum_ciphertext
        )

        hospital_count_ciphertexts.append(
            count_ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = {
            "bounded_sum_ckks": len(
                sum_ciphertext
            ),
            "count_bfv": len(
                count_ciphertext
            ),
        }

    # -------------------------------------------------------------
    # Homomorphic aggregation.
    # -------------------------------------------------------------

    encrypted_global_sum = (
        add_ckks_ciphertexts(
            public_ckks_context,
            hospital_sum_ciphertexts,
        )
    )

    encrypted_global_count = (
        add_bfv_ciphertexts(
            public_bfv_context,
            hospital_count_ciphertexts,
        )
    )

    # -------------------------------------------------------------
    # Return encrypted global sufficient statistics only.
    # -------------------------------------------------------------

    return {
        "query_id": "Q003",
        "statistic": "bounded_los_sum_and_count",
        "lower": Q003_LOWER_BOUND,
        "upper": Q003_UPPER_BOUND,
        "encoding": "base64",

        "participating_hospitals": (
            participating_hospitals
        ),

        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),

        "bounded_sum_he_scheme": "CKKS",
        "bounded_sum_ciphertext": bytes_to_base64(
            encrypted_global_sum
        ),
        "bounded_sum_ciphertext_bytes": len(
            encrypted_global_sum
        ),

        "count_he_scheme": "BFV",
        "count_ciphertext": bytes_to_base64(
            encrypted_global_count
        ),
        "count_ciphertext_bytes": len(
            encrypted_global_count
        ),
    }

# =====================================================================
# Q004
# =====================================================================

def fetch_encrypted_q004(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request one encrypted Q004 joint-histogram vector from a hospital.

    Only ciphertext and fixed public-domain metadata are accepted.
    Patient-derived plaintext statistics are explicitly rejected.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "he_scheme",
        "encoding",
        "dimensions",
        "vector_length",
        "race_order",
        "admission_type_order",
        "ciphertext",
        "ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q004 hospital response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    # -------------------------------------------------------------
    # Reject accidental plaintext-statistic leakage.
    # -------------------------------------------------------------

    forbidden_plaintext_fields = {
        "table",
        "vector",
        "histogram",
        "counts",
        "count",
        "patient_count",
        "local_result",
        "global_result",
        "value",
        "exact_result",
        "exact_table",
        "exact_vector",
    }

    leaked_fields = (
        forbidden_plaintext_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Q004 hospital response unexpectedly contains "
            "plaintext statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    # -------------------------------------------------------------
    # Validate query and HE metadata.
    # -------------------------------------------------------------

    if payload["query_id"] != "Q004":
        raise ValueError(
            "Unexpected query_id returned by hospital "
            "for Q004 aggregation."
        )

    if (
        payload["statistic"]
        != "race_admission_type_contingency"
    ):
        raise ValueError(
            "Unexpected Q004 statistic type."
        )

    if payload["he_scheme"] != "BFV":
        raise ValueError(
            "Q004 requires BFV ciphertexts."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported Q004 ciphertext encoding."
        )

    if int(payload["vector_length"]) != Q004_VECTOR_LENGTH:
        raise ValueError(
            "Q004 encrypted vector must contain exactly "
            f"{Q004_VECTOR_LENGTH} slots."
        )

    # -------------------------------------------------------------
    # Validate fixed 33 x 9 public schema.
    # -------------------------------------------------------------

    dimensions = payload["dimensions"]

    if not isinstance(dimensions, dict):
        raise ValueError(
            "Q004 dimensions metadata must be an object."
        )

    if (
        int(dimensions.get("race_categories", -1))
        != len(Q004_RACE_ORDER)
    ):
        raise ValueError(
            "Q004 hospital returned an unexpected race dimension."
        )

    if (
        int(dimensions.get("admission_types", -1))
        != len(Q004_ADMISSION_TYPE_ORDER)
    ):
        raise ValueError(
            "Q004 hospital returned an unexpected "
            "admission-type dimension."
        )

    returned_race_order = tuple(
        payload["race_order"]
    )

    if returned_race_order != Q004_RACE_ORDER:
        raise ValueError(
            "Q004 hospital race order does not match "
            "the approved canonical domain."
        )

    returned_admission_type_order = tuple(
        payload["admission_type_order"]
    )

    if (
        returned_admission_type_order
        != Q004_ADMISSION_TYPE_ORDER
    ):
        raise ValueError(
            "Q004 hospital admission-type order does not match "
            "the approved canonical domain."
        )

    return payload


def aggregate_q004_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q004_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q004 vectors.

    Q004 is represented as one canonical 297-element BFV vector:

        33 race categories x 9 admission types.

    BFV addition acts element-wise over the fixed canonical slot order.

    The coordinator loads only the public BFV context and never
    decrypts the global result.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    public_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    if not public_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    hospital_ciphertexts = []
    participating_hospitals = []
    hospital_ciphertext_sizes = {}

    seen_hospitals = set()

    using_default_hospitals = (
        tuple(hospital_urls)
        == tuple(DEFAULT_HOSPITAL_Q004_URLS)
    )

    expected_default_hospitals = {
        f"hospital_{index}"
        for index in range(1, 6)
    }

    for index, hospital_url in enumerate(
        hospital_urls,
        start=1,
    ):

        payload = fetch_encrypted_q004(
            hospital_url
        )

        hospital_id = str(
            payload["hospital"]
        )

        # ---------------------------------------------------------
        # Prevent duplicate hospital contributions.
        # ---------------------------------------------------------

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q004 aggregation: {hospital_id}"
            )

        # For the normal five-node deployment, also bind each
        # configured endpoint position to its expected hospital ID.
        if using_default_hospitals:

            expected_hospital_id = (
                f"hospital_{index}"
            )

            if hospital_id != expected_hospital_id:
                raise ValueError(
                    "Q004 hospital identity does not match the "
                    "configured endpoint. "
                    f"Expected {expected_hospital_id}, "
                    f"received {hospital_id}."
                )

        seen_hospitals.add(
            hospital_id
        )

        # ---------------------------------------------------------
        # Decode and validate BFV ciphertext transport.
        # ---------------------------------------------------------

        ciphertext = base64_to_bytes(
            payload["ciphertext"]
        )

        expected_ciphertext_size = int(
            payload["ciphertext_bytes"]
        )

        if len(ciphertext) != expected_ciphertext_size:
            raise ValueError(
                "Q004 BFV ciphertext size mismatch for "
                f"{hospital_id}."
            )

        hospital_ciphertexts.append(
            ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = len(ciphertext)

    # -------------------------------------------------------------
    # Confirm the normal deployment contains exactly H1-H5.
    # -------------------------------------------------------------

    if (
        using_default_hospitals
        and seen_hospitals != expected_default_hospitals
    ):
        raise ValueError(
            "Q004 aggregation did not receive the expected "
            "five hospital identities."
        )

    # -------------------------------------------------------------
    # Homomorphic element-wise BFV aggregation.
    # -------------------------------------------------------------

    encrypted_global = add_bfv_ciphertexts(
        public_context,
        hospital_ciphertexts,
    )

    # -------------------------------------------------------------
    # Return encrypted global statistic only.
    # -------------------------------------------------------------

    return {
        "query_id": "Q004",
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

        "participating_hospitals": (
            participating_hospitals
        ),

        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),

        "ciphertext": bytes_to_base64(
            encrypted_global
        ),

        "ciphertext_bytes": len(
            encrypted_global
        ),
    }

# =====================================================================
# Q005
# =====================================================================

def fetch_encrypted_q005(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request encrypted Q005 sufficient statistics from one hospital.

    Q005 uses a hybrid HE representation over the fixed 33-race domain:

        bounded LOS sum vector -> CKKS
        patient count vector   -> BFV

    Only ciphertexts and fixed public metadata are accepted.
    Patient-derived plaintext sufficient statistics are rejected.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "lower",
        "upper",
        "encoding",
        "vector_length",
        "race_order",
        "bounded_sum_he_scheme",
        "bounded_sum_ciphertext",
        "bounded_sum_ciphertext_bytes",
        "count_he_scheme",
        "count_ciphertext",
        "count_ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q005 hospital response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    # -------------------------------------------------------------
    # Reject accidental plaintext sufficient-statistic leakage.
    # -------------------------------------------------------------

    forbidden_plaintext_fields = {
        "bounded_sums",
        "bounded_sum",
        "counts",
        "count",
        "means",
        "mean",
        "groups",
        "table",
        "vector",
        "histogram",
        "patient_count",
        "local_result",
        "global_result",
        "value",
        "exact_result",
        "exact_sum",
        "exact_count",
        "exact_mean",
    }

    leaked_fields = (
        forbidden_plaintext_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Q005 hospital response unexpectedly contains "
            "plaintext statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    # -------------------------------------------------------------
    # Validate query metadata.
    # -------------------------------------------------------------

    if payload["query_id"] != "Q005":
        raise ValueError(
            "Unexpected query_id returned by hospital "
            "for Q005 aggregation."
        )

    if (
        payload["statistic"]
        != "grouped_bounded_los_sum_and_count"
    ):
        raise ValueError(
            "Unexpected Q005 statistic type."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported Q005 ciphertext encoding."
        )

    if (
        payload["bounded_sum_he_scheme"]
        != "CKKS"
    ):
        raise ValueError(
            "Q005 bounded-sum vector requires CKKS."
        )

    if payload["count_he_scheme"] != "BFV":
        raise ValueError(
            "Q005 count vector requires BFV."
        )

    # -------------------------------------------------------------
    # Validate fixed public schema.
    # -------------------------------------------------------------

    if int(
        payload["vector_length"]
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 encrypted vectors must contain exactly "
            f"{Q005_VECTOR_LENGTH} slots."
        )

    returned_race_order = tuple(
        payload["race_order"]
    )

    if returned_race_order != Q005_RACE_ORDER:
        raise ValueError(
            "Q005 hospital race order does not match "
            "the approved canonical domain."
        )

    returned_lower = float(
        payload["lower"]
    )

    returned_upper = float(
        payload["upper"]
    )

    if returned_lower != Q005_LOWER_BOUND:
        raise ValueError(
            "Q005 hospital returned an unexpected "
            "lower LOS bound."
        )

    if returned_upper != Q005_UPPER_BOUND:
        raise ValueError(
            "Q005 hospital returned an unexpected "
            "upper LOS bound."
        )

    return payload


def aggregate_q005_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q005_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q005 sufficient
    statistics.

    Q005 is represented by two canonical 33-element vectors using the
    same fixed race order:

        bounded LOS sums -> CKKS
        patient counts   -> BFV

    The coordinator loads public HE contexts only and never decrypts
    either aggregate.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    # -------------------------------------------------------------
    # Load PUBLIC HE contexts only.
    # -------------------------------------------------------------

    public_bfv_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    public_ckks_context = load_context(
        PUBLIC_CKKS_CONTEXT_PATH
    )

    if not public_bfv_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    if not public_ckks_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only CKKS context."
        )

    # -------------------------------------------------------------
    # Collect encrypted hospital statistics.
    # -------------------------------------------------------------

    hospital_sum_ciphertexts = []
    hospital_count_ciphertexts = []

    participating_hospitals = []
    hospital_ciphertext_sizes = {}

    seen_hospitals = set()

    using_default_hospitals = (
        tuple(hospital_urls)
        == tuple(DEFAULT_HOSPITAL_Q005_URLS)
    )

    expected_default_hospitals = {
        f"hospital_{index}"
        for index in range(1, 6)
    }

    for index, hospital_url in enumerate(
        hospital_urls,
        start=1,
    ):

        payload = fetch_encrypted_q005(
            hospital_url
        )

        hospital_id = str(
            payload["hospital"]
        )

        # ---------------------------------------------------------
        # Prevent duplicate hospital contributions.
        # ---------------------------------------------------------

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q005 aggregation: {hospital_id}"
            )

        # For the normal five-node deployment, bind endpoint position
        # to the expected hospital identity.
        if using_default_hospitals:

            expected_hospital_id = (
                f"hospital_{index}"
            )

            if hospital_id != expected_hospital_id:
                raise ValueError(
                    "Q005 hospital identity does not match the "
                    "configured endpoint. "
                    f"Expected {expected_hospital_id}, "
                    f"received {hospital_id}."
                )

        seen_hospitals.add(
            hospital_id
        )

        # ---------------------------------------------------------
        # Decode and validate CKKS bounded-sum ciphertext transport.
        # ---------------------------------------------------------

        sum_ciphertext = base64_to_bytes(
            payload["bounded_sum_ciphertext"]
        )

        expected_sum_size = int(
            payload[
                "bounded_sum_ciphertext_bytes"
            ]
        )

        if len(
            sum_ciphertext
        ) != expected_sum_size:
            raise ValueError(
                "Q005 CKKS bounded-sum ciphertext size "
                f"mismatch for {hospital_id}."
            )

        # ---------------------------------------------------------
        # Decode and validate BFV count ciphertext transport.
        # ---------------------------------------------------------

        count_ciphertext = base64_to_bytes(
            payload["count_ciphertext"]
        )

        expected_count_size = int(
            payload[
                "count_ciphertext_bytes"
            ]
        )

        if len(
            count_ciphertext
        ) != expected_count_size:
            raise ValueError(
                "Q005 BFV count ciphertext size "
                f"mismatch for {hospital_id}."
            )

        hospital_sum_ciphertexts.append(
            sum_ciphertext
        )

        hospital_count_ciphertexts.append(
            count_ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = {
            "bounded_sum_ckks": len(
                sum_ciphertext
            ),
            "count_bfv": len(
                count_ciphertext
            ),
        }

    # -------------------------------------------------------------
    # Confirm the normal deployment contains exactly H1-H5.
    # -------------------------------------------------------------

    if (
        using_default_hospitals
        and seen_hospitals != expected_default_hospitals
    ):
        raise ValueError(
            "Q005 aggregation did not receive the expected "
            "five hospital identities."
        )

    # -------------------------------------------------------------
    # Homomorphic vector aggregation.
    # -------------------------------------------------------------

    encrypted_global_sums = (
        add_ckks_ciphertexts(
            public_ckks_context,
            hospital_sum_ciphertexts,
        )
    )

    encrypted_global_counts = (
        add_bfv_ciphertexts(
            public_bfv_context,
            hospital_count_ciphertexts,
        )
    )

    # -------------------------------------------------------------
    # Return encrypted global sufficient statistics only.
    # -------------------------------------------------------------

    return {
        "query_id": "Q005",
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

        "participating_hospitals": (
            participating_hospitals
        ),

        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),

        "bounded_sum_he_scheme": "CKKS",

        "bounded_sum_ciphertext": (
            bytes_to_base64(
                encrypted_global_sums
            )
        ),

        "bounded_sum_ciphertext_bytes": len(
            encrypted_global_sums
        ),

        "count_he_scheme": "BFV",

        "count_ciphertext": (
            bytes_to_base64(
                encrypted_global_counts
            )
        ),

        "count_ciphertext_bytes": len(
            encrypted_global_counts
        ),
    }

# =====================================================================
# Q006
# =====================================================================

def fetch_encrypted_q006(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request one encrypted Q006 rare-subgroup count from a hospital.

    Only ciphertext and fixed public predicate metadata are accepted.
    Patient-derived plaintext counts are explicitly rejected.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "predicate",
        "he_scheme",
        "encoding",
        "ciphertext",
        "ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q006 hospital response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_plaintext_fields = {
        "count",
        "value",
        "patient_count",
        "local_result",
        "global_result",
        "exact_result",
        "exact_count",
    }

    leaked_fields = (
        forbidden_plaintext_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Q006 hospital response unexpectedly contains "
            "plaintext statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q006":
        raise ValueError(
            "Unexpected query_id returned by hospital "
            "for Q006 aggregation."
        )

    if payload["statistic"] != "rare_subgroup_count":
        raise ValueError(
            "Unexpected Q006 statistic type."
        )

    predicate = payload["predicate"]

    if not isinstance(predicate, dict):
        raise ValueError(
            "Q006 predicate metadata must be an object."
        )

    if set(predicate.keys()) != {
        "race",
        "admission_type",
    }:
        raise ValueError(
            "Q006 predicate metadata contains unexpected fields."
        )

    if predicate["race"] != Q006_RACE:
        raise ValueError(
            "Q006 hospital returned an unexpected race predicate."
        )

    if predicate["admission_type"] != Q006_ADMISSION_TYPE:
        raise ValueError(
            "Q006 hospital returned an unexpected admission-type "
            "predicate."
        )

    if payload["he_scheme"] != "BFV":
        raise ValueError(
            "Q006 requires BFV ciphertexts."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported Q006 ciphertext encoding."
        )

    return payload


def aggregate_q006_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q006_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q006 scalar counts.

    The coordinator uses only the shared public BFV context and never
    decrypts the global rare-subgroup count.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    public_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    if not public_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    hospital_ciphertexts = []
    participating_hospitals = []
    hospital_ciphertext_sizes = {}
    seen_hospitals = set()

    using_default_hospitals = (
        tuple(hospital_urls)
        == tuple(DEFAULT_HOSPITAL_Q006_URLS)
    )

    expected_default_hospitals = {
        f"hospital_{index}"
        for index in range(1, 6)
    }

    for index, hospital_url in enumerate(
        hospital_urls,
        start=1,
    ):

        payload = fetch_encrypted_q006(
            hospital_url
        )

        hospital_id = str(
            payload["hospital"]
        )

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q006 aggregation: {hospital_id}"
            )

        if using_default_hospitals:

            expected_hospital_id = (
                f"hospital_{index}"
            )

            if hospital_id != expected_hospital_id:
                raise ValueError(
                    "Q006 hospital identity does not match the "
                    "configured endpoint. "
                    f"Expected {expected_hospital_id}, "
                    f"received {hospital_id}."
                )

        seen_hospitals.add(
            hospital_id
        )

        ciphertext = base64_to_bytes(
            payload["ciphertext"]
        )

        expected_ciphertext_size = int(
            payload["ciphertext_bytes"]
        )

        if len(ciphertext) != expected_ciphertext_size:
            raise ValueError(
                "Q006 BFV ciphertext size mismatch for "
                f"{hospital_id}."
            )

        hospital_ciphertexts.append(
            ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = len(ciphertext)

    if (
        using_default_hospitals
        and seen_hospitals != expected_default_hospitals
    ):
        raise ValueError(
            "Q006 aggregation did not receive the expected "
            "five hospital identities."
        )

    encrypted_global = add_bfv_ciphertexts(
        public_context,
        hospital_ciphertexts,
    )

    return {
        "query_id": "Q006",
        "statistic": "rare_subgroup_count",
        "predicate": {
            "race": Q006_RACE,
            "admission_type": Q006_ADMISSION_TYPE,
        },
        "he_scheme": "BFV",
        "encoding": "base64",
        "participating_hospitals": (
            participating_hospitals
        ),
        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),
        "ciphertext": bytes_to_base64(
            encrypted_global
        ),
        "ciphertext_bytes": len(
            encrypted_global
        ),
    }


# =====================================================================
# Q007
# =====================================================================

def fetch_encrypted_q007(
    hospital_url: str,
    *,
    timeout: int = 30,
) -> dict:
    """
    Request encrypted Q007 sufficient statistics from one hospital.

    Q007 uses a hybrid scalar representation:

        bounded LOS sum -> CKKS
        subgroup count  -> BFV

    Only ciphertexts and fixed public metadata are accepted.
    """

    with urlopen(
        hospital_url,
        timeout=timeout,
    ) as response:

        payload = json.loads(
            response.read().decode("utf-8")
        )

    required_fields = {
        "query_id",
        "hospital",
        "statistic",
        "lower",
        "upper",
        "predicate",
        "encoding",
        "bounded_sum_he_scheme",
        "bounded_sum_ciphertext",
        "bounded_sum_ciphertext_bytes",
        "count_he_scheme",
        "count_ciphertext",
        "count_ciphertext_bytes",
    }

    missing_fields = (
        required_fields
        - set(payload.keys())
    )

    if missing_fields:
        raise ValueError(
            "Q007 hospital response is missing fields: "
            f"{sorted(missing_fields)}"
        )

    forbidden_plaintext_fields = {
        "bounded_sum",
        "count",
        "mean",
        "value",
        "patient_count",
        "local_result",
        "global_result",
        "exact_result",
        "exact_sum",
        "exact_count",
        "exact_mean",
        "noisy_sum",
        "noisy_count",
    }

    leaked_fields = (
        forbidden_plaintext_fields
        & set(payload.keys())
    )

    if leaked_fields:
        raise ValueError(
            "Q007 hospital response unexpectedly contains "
            "plaintext statistic fields: "
            f"{sorted(leaked_fields)}"
        )

    if payload["query_id"] != "Q007":
        raise ValueError(
            "Unexpected query_id returned by hospital "
            "for Q007 aggregation."
        )

    if (
        payload["statistic"]
        != "rare_subgroup_bounded_los_sum_and_count"
    ):
        raise ValueError(
            "Unexpected Q007 statistic type."
        )

    if payload["encoding"] != "base64":
        raise ValueError(
            "Unsupported Q007 ciphertext encoding."
        )

    if payload["bounded_sum_he_scheme"] != "CKKS":
        raise ValueError(
            "Q007 bounded sum requires CKKS."
        )

    if payload["count_he_scheme"] != "BFV":
        raise ValueError(
            "Q007 subgroup count requires BFV."
        )

    returned_lower = float(
        payload["lower"]
    )

    returned_upper = float(
        payload["upper"]
    )

    if returned_lower != Q007_LOWER_BOUND:
        raise ValueError(
            "Q007 hospital returned an unexpected lower LOS bound."
        )

    if returned_upper != Q007_UPPER_BOUND:
        raise ValueError(
            "Q007 hospital returned an unexpected upper LOS bound."
        )

    predicate = payload["predicate"]

    if not isinstance(predicate, dict):
        raise ValueError(
            "Q007 predicate metadata must be an object."
        )

    if set(predicate.keys()) != {
        "race",
        "admission_type",
    }:
        raise ValueError(
            "Q007 predicate metadata contains unexpected fields."
        )

    if predicate["race"] != Q007_RACE:
        raise ValueError(
            "Q007 hospital returned an unexpected race predicate."
        )

    if predicate["admission_type"] != Q007_ADMISSION_TYPE:
        raise ValueError(
            "Q007 hospital returned an unexpected admission-type "
            "predicate."
        )

    return payload


def aggregate_q007_ciphertexts(
    hospital_urls: Sequence[str] = DEFAULT_HOSPITAL_Q007_URLS,
) -> dict:
    """
    Fetch and homomorphically aggregate encrypted Q007 sufficient
    statistics.

    The bounded LOS sums are aggregated with CKKS and subgroup counts
    are aggregated independently with BFV. The coordinator uses only
    public HE contexts and never decrypts either global statistic.
    """

    if not hospital_urls:
        raise ValueError(
            "At least one hospital URL is required."
        )

    public_bfv_context = load_context(
        PUBLIC_BFV_CONTEXT_PATH
    )

    public_ckks_context = load_context(
        PUBLIC_CKKS_CONTEXT_PATH
    )

    if not public_bfv_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only BFV context."
        )

    if not public_ckks_context.is_public():
        raise RuntimeError(
            "Coordinator must use a public-only CKKS context."
        )

    hospital_sum_ciphertexts = []
    hospital_count_ciphertexts = []
    participating_hospitals = []
    hospital_ciphertext_sizes = {}
    seen_hospitals = set()

    using_default_hospitals = (
        tuple(hospital_urls)
        == tuple(DEFAULT_HOSPITAL_Q007_URLS)
    )

    expected_default_hospitals = {
        f"hospital_{index}"
        for index in range(1, 6)
    }

    for index, hospital_url in enumerate(
        hospital_urls,
        start=1,
    ):

        payload = fetch_encrypted_q007(
            hospital_url
        )

        hospital_id = str(
            payload["hospital"]
        )

        if hospital_id in seen_hospitals:
            raise ValueError(
                "Duplicate hospital identity encountered during "
                f"Q007 aggregation: {hospital_id}"
            )

        if using_default_hospitals:

            expected_hospital_id = (
                f"hospital_{index}"
            )

            if hospital_id != expected_hospital_id:
                raise ValueError(
                    "Q007 hospital identity does not match the "
                    "configured endpoint. "
                    f"Expected {expected_hospital_id}, "
                    f"received {hospital_id}."
                )

        seen_hospitals.add(
            hospital_id
        )

        sum_ciphertext = base64_to_bytes(
            payload["bounded_sum_ciphertext"]
        )

        expected_sum_size = int(
            payload["bounded_sum_ciphertext_bytes"]
        )

        if len(sum_ciphertext) != expected_sum_size:
            raise ValueError(
                "Q007 CKKS bounded-sum ciphertext size mismatch for "
                f"{hospital_id}."
            )

        count_ciphertext = base64_to_bytes(
            payload["count_ciphertext"]
        )

        expected_count_size = int(
            payload["count_ciphertext_bytes"]
        )

        if len(count_ciphertext) != expected_count_size:
            raise ValueError(
                "Q007 BFV count ciphertext size mismatch for "
                f"{hospital_id}."
            )

        hospital_sum_ciphertexts.append(
            sum_ciphertext
        )

        hospital_count_ciphertexts.append(
            count_ciphertext
        )

        participating_hospitals.append(
            hospital_id
        )

        hospital_ciphertext_sizes[
            hospital_id
        ] = {
            "bounded_sum_ckks": len(
                sum_ciphertext
            ),
            "count_bfv": len(
                count_ciphertext
            ),
        }

    if (
        using_default_hospitals
        and seen_hospitals != expected_default_hospitals
    ):
        raise ValueError(
            "Q007 aggregation did not receive the expected "
            "five hospital identities."
        )

    encrypted_global_sum = add_ckks_ciphertexts(
        public_ckks_context,
        hospital_sum_ciphertexts,
    )

    encrypted_global_count = add_bfv_ciphertexts(
        public_bfv_context,
        hospital_count_ciphertexts,
    )

    return {
        "query_id": "Q007",
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
        "participating_hospitals": (
            participating_hospitals
        ),
        "hospital_ciphertext_bytes": (
            hospital_ciphertext_sizes
        ),
        "bounded_sum_he_scheme": "CKKS",
        "bounded_sum_ciphertext": bytes_to_base64(
            encrypted_global_sum
        ),
        "bounded_sum_ciphertext_bytes": len(
            encrypted_global_sum
        ),
        "count_he_scheme": "BFV",
        "count_ciphertext": bytes_to_base64(
            encrypted_global_count
        ),
        "count_ciphertext_bytes": len(
            encrypted_global_count
        ),
    }
