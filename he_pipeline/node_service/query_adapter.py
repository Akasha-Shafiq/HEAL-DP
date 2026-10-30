"""
Adapter between the validated hospital query implementation and
the homomorphic-encryption pipeline.

The adapter deliberately reuses query logic from ``node.queries``.
It does not redefine query semantics.

Q001
----
    local admissions.csv.gz
        ->
    node.queries.unique_patient_count(...)
        ->
    exact local integer statistic
        ->
    BFV encryption performed elsewhere

Q002
----
    local admissions.csv.gz
        ->
    node.queries.admission_type_histogram(...)
        ->
    canonical 9-element integer vector
        ->
    BFV encryption performed elsewhere

Q003
----
    local admissions.csv.gz
        ->
    node.queries.bounded_los_sum_and_count(...)
        ->
    bounded LOS sum + patient count
        ->
    CKKS sum / BFV count encryption performed elsewhere

Q004
----
    local admissions.csv.gz
        ->
    node.queries.race_admission_type_contingency(...)
        ->
    canonical 33 x 9 table
        ->
    canonical 297-element integer vector
        ->
    BFV encryption performed elsewhere

Q005
----
    local admissions.csv.gz
        ->
    node.queries.grouped_bounded_los_sum_and_count(...)
        ->
    canonical 33-element bounded-sum vector
    + canonical 33-element count vector
        ->
    CKKS sum-vector / BFV count-vector encryption performed elsewhere

Q006
----
    local admissions.csv.gz
        ->
    node.queries.rare_subgroup_count(...)
        ->
    exact rare-subgroup patient count
        ->
    BFV encryption performed elsewhere

Q007
----
    local admissions.csv.gz
        ->
    node.queries.rare_subgroup_bounded_los_sum_and_count(...)
        ->
    bounded LOS sum + rare-subgroup patient count
        ->
    CKKS sum / BFV count encryption performed elsewhere

No differential privacy is applied here.
"""

from pathlib import Path

import pandas as pd

from node.queries import (
    Q004_ADMISSION_TYPES,
    Q004_RACE_CATEGORIES,
    Q006_ADMISSION_TYPE as NODE_Q006_ADMISSION_TYPE,
    Q006_RACE as NODE_Q006_RACE,
    admission_type_histogram,
    bounded_los_sum_and_count,
    grouped_bounded_los_sum_and_count,
    race_admission_type_contingency,
    rare_subgroup_bounded_los_sum_and_count,
    rare_subgroup_count,
    unique_patient_count,
)


# =====================================================================
# Canonical domains
# =====================================================================


# ---------------------------------------------------------------------
# Q002
# ---------------------------------------------------------------------

Q002_CATEGORY_ORDER = tuple(
    Q004_ADMISSION_TYPES
)


# ---------------------------------------------------------------------
# Q004
# ---------------------------------------------------------------------
#
# Q004 is represented as one fixed 297-element vector:
#
#     33 race categories x 9 admission types
#
# Flattening is race-major:
#
#     race_0/admission_0
#     race_0/admission_1
#     ...
#     race_0/admission_8
#     race_1/admission_0
#     ...
#     race_32/admission_8
#
# This order must remain identical at:
#
#     hospital query adapter
#         ->
#     BFV encryption
#         ->
#     coordinator aggregation
#         ->
#     trusted decryption
#         ->
#     DP mechanism
#         ->
#     table reconstruction
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


# ---------------------------------------------------------------------
# Q005
# ---------------------------------------------------------------------
#
# Q005 uses the same fixed 33-race domain as Q004.
#
# Two canonical vectors are used:
#
#     bounded_sums[0:33] -> CKKS later
#     counts[0:33]       -> BFV later
#
# Both vectors use exactly the same race order. Keeping a shared
# canonical order is essential so that encrypted slot i always
# refers to the same race on every hospital, at the coordinator,
# and at the trusted release boundary.
# ---------------------------------------------------------------------

Q005_RACE_ORDER = tuple(
    Q004_RACE_ORDER
)

Q005_VECTOR_LENGTH = len(
    Q005_RACE_ORDER
)

Q005_LOWER_BOUND = 0.0
Q005_UPPER_BOUND = 30.0


# ---------------------------------------------------------------------
# Q006 / Q007 rare-subgroup predicate
# ---------------------------------------------------------------------
#
# Q006 and Q007 intentionally share one fixed, registry-approved
# predicate. Keeping these values canonical at the adapter boundary
# prevents hospital nodes from silently changing subgroup semantics.
# ---------------------------------------------------------------------

Q006_RACE = str(
    NODE_Q006_RACE
)

Q006_ADMISSION_TYPE = str(
    NODE_Q006_ADMISSION_TYPE
)

Q007_RACE = Q006_RACE
Q007_ADMISSION_TYPE = Q006_ADMISSION_TYPE

Q007_LOWER_BOUND = 0.0
Q007_UPPER_BOUND = 30.0


# =====================================================================
# Q001
# =====================================================================

def execute_q001_from_dataframe(
    dataframe: pd.DataFrame,
) -> int:
    """
    Execute the existing validated Q001 implementation.
    """

    return unique_patient_count(
        dataframe
    )


def execute_q001_from_file(
    data_file: str | Path,
) -> int:
    """
    Load a hospital admissions dataset and execute Q001 using
    the existing validated query implementation.
    """

    path = Path(data_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q001_from_dataframe(
        dataframe
    )


# =====================================================================
# Q002
# =====================================================================

def execute_q002_from_dataframe(
    dataframe: pd.DataFrame,
) -> list[int]:
    """
    Execute the existing validated Q002 implementation and convert
    its histogram dictionary into the canonical nine-element vector.

    The underlying Q002 semantics remain defined by
    ``node.queries.admission_type_histogram``:

        - privacy unit: subject_id,
        - one contribution per patient,
        - category selected from the earliest admission.

    Returns
    -------
    list[int]
        Nine integer counts in ``Q002_CATEGORY_ORDER``.
    """

    histogram = admission_type_histogram(
        dataframe
    )

    unexpected_categories = (
        set(histogram)
        - set(Q002_CATEGORY_ORDER)
    )

    if unexpected_categories:
        raise ValueError(
            "Q002 produced admission types outside the "
            "approved canonical domain: "
            f"{sorted(unexpected_categories)}"
        )

    return [
        int(
            histogram.get(
                category,
                0,
            )
        )
        for category in Q002_CATEGORY_ORDER
    ]


def execute_q002_from_file(
    data_file: str | Path,
) -> list[int]:
    """
    Load one hospital admissions dataset and execute Q002.

    The returned vector always has exactly nine slots in the
    canonical Q002 category order.
    """

    path = Path(data_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q002_from_dataframe(
        dataframe
    )


# =====================================================================
# Q003
# =====================================================================

Q003_LOWER_BOUND = 0.0
Q003_UPPER_BOUND = 30.0


def execute_q003_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    lower: float = Q003_LOWER_BOUND,
    upper: float = Q003_UPPER_BOUND,
) -> dict:
    """
    Execute the existing validated Q003 implementation.

    Q003 computes patient-level bounded length-of-stay sufficient
    statistics:

        - one contribution per patient,
        - earliest admission only,
        - LOS measured in days,
        - LOS clipped to [lower, upper],
        - return bounded sum and patient count.

    No mean, encryption, or differential privacy is computed here.
    """

    result = bounded_los_sum_and_count(
        dataframe,
        lower=lower,
        upper=upper,
    )

    required_fields = {
        "bounded_sum",
        "count",
        "lower",
        "upper",
    }

    missing_fields = (
        required_fields
        - set(result.keys())
    )

    if missing_fields:
        raise ValueError(
            "Validated Q003 implementation is missing fields: "
            f"{sorted(missing_fields)}"
        )

    bounded_sum = float(
        result["bounded_sum"]
    )

    count = int(
        result["count"]
    )

    returned_lower = float(
        result["lower"]
    )

    returned_upper = float(
        result["upper"]
    )

    if count < 0:
        raise ValueError(
            "Q003 patient count cannot be negative."
        )

    if bounded_sum < 0.0:
        raise ValueError(
            "Q003 bounded LOS sum cannot be negative."
        )

    if returned_lower != float(lower):
        raise ValueError(
            "Q003 returned an unexpected lower bound."
        )

    if returned_upper != float(upper):
        raise ValueError(
            "Q003 returned an unexpected upper bound."
        )

    return {
        "bounded_sum": bounded_sum,
        "count": count,
        "lower": returned_lower,
        "upper": returned_upper,
    }


def execute_q003_from_file(
    data_file: str | Path,
    *,
    lower: float = Q003_LOWER_BOUND,
    upper: float = Q003_UPPER_BOUND,
) -> dict:
    """
    Load one hospital admissions dataset and execute Q003 using
    the existing validated query implementation.
    """

    path = Path(
        data_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q003_from_dataframe(
        dataframe,
        lower=lower,
        upper=upper,
    )


# =====================================================================
# Q004
# =====================================================================

def flatten_q004_table(
    table: dict,
) -> list[int]:
    """
    Convert the validated Q004 nested contingency table into the
    canonical 297-element race-major integer vector.

    Expected input form:

        {
            race_1: {
                admission_type_1: count,
                ...
            },
            ...
        }

    Strict domain validation is performed so that ciphertext slot
    semantics can never silently change.
    """

    expected_races = set(
        Q004_RACE_ORDER
    )

    returned_races = set(
        table.keys()
    )

    if returned_races != expected_races:
        missing = (
            expected_races
            - returned_races
        )

        unexpected = (
            returned_races
            - expected_races
        )

        raise ValueError(
            "Q004 table does not match the canonical race domain. "
            f"Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    vector: list[int] = []

    for race in Q004_RACE_ORDER:

        race_row = table[race]

        if not isinstance(
            race_row,
            dict,
        ):
            raise ValueError(
                "Q004 race rows must be dictionaries."
            )

        expected_admission_types = set(
            Q004_ADMISSION_TYPE_ORDER
        )

        returned_admission_types = set(
            race_row.keys()
        )

        if (
            returned_admission_types
            != expected_admission_types
        ):
            missing = (
                expected_admission_types
                - returned_admission_types
            )

            unexpected = (
                returned_admission_types
                - expected_admission_types
            )

            raise ValueError(
                "Q004 table does not match the canonical "
                f"admission-type domain for race {race!r}. "
                f"Missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        for admission_type in (
            Q004_ADMISSION_TYPE_ORDER
        ):

            value = race_row[
                admission_type
            ]

            # Reject booleans explicitly because bool is a subclass
            # of int in Python.
            if isinstance(
                value,
                bool,
            ):
                raise ValueError(
                    "Q004 cell counts must be integers, "
                    "not booleans."
                )

            count = int(
                value
            )

            if count != value:
                raise ValueError(
                    "Q004 cell counts must be exact integers."
                )

            if count < 0:
                raise ValueError(
                    "Q004 cell counts cannot be negative."
                )

            vector.append(
                count
            )

    if len(vector) != Q004_VECTOR_LENGTH:
        raise RuntimeError(
            "Q004 canonical vector has an unexpected length."
        )

    return vector


def q004_vector_to_table(
    vector: list[int],
) -> dict:
    """
    Reconstruct the canonical nested Q004 table from a 297-element
    race-major vector.

    This inverse operation is useful later at the trusted release
    boundary after BFV decryption.
    """

    if len(vector) != Q004_VECTOR_LENGTH:
        raise ValueError(
            "Q004 vector has invalid length. "
            f"Expected {Q004_VECTOR_LENGTH}, "
            f"received {len(vector)}."
        )

    table = {}

    index = 0

    for race in Q004_RACE_ORDER:

        row = {}

        for admission_type in (
            Q004_ADMISSION_TYPE_ORDER
        ):

            value = vector[index]

            if isinstance(
                value,
                bool,
            ):
                raise ValueError(
                    "Q004 vector values must be integers, "
                    "not booleans."
                )

            count = int(
                value
            )

            if count != value:
                raise ValueError(
                    "Q004 vector values must be exact integers."
                )

            if count < 0:
                raise ValueError(
                    "Q004 vector values cannot be negative."
                )

            row[
                admission_type
            ] = count

            index += 1

        table[race] = row

    return table


def execute_q004_from_dataframe(
    dataframe: pd.DataFrame,
) -> list[int]:
    """
    Execute the existing validated Q004 implementation and flatten
    the resulting 33 x 9 contingency table into the canonical
    297-element integer vector.

    The underlying query semantics remain entirely defined by
    ``node.queries.race_admission_type_contingency``:

        - privacy unit: subject_id,
        - one contribution per patient,
        - earliest admission only,
        - one race x admission-type cell per patient,
        - fixed complete 33 x 9 domain.

    No encryption or differential privacy occurs in this adapter.
    """

    table = race_admission_type_contingency(
        dataframe
    )

    return flatten_q004_table(
        table
    )


def execute_q004_from_file(
    data_file: str | Path,
) -> list[int]:
    """
    Load one hospital admissions dataset and execute Q004.

    The returned vector always has exactly 297 integer slots in
    canonical race-major order.
    """

    path = Path(
        data_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q004_from_dataframe(
        dataframe
    )

# =====================================================================
# Q005
# =====================================================================

def _validate_q005_grouped_result(
    result: dict,
    *,
    lower: float,
    upper: float,
) -> dict:
    """
    Validate the output of the existing Q005 implementation without
    changing its semantics.

    Expected form:

        {
            "groups": {
                race: {
                    "bounded_sum": float,
                    "count": int,
                },
                ...
            },
            "lower": float,
            "upper": float,
        }

    The race domain must match the complete canonical 33-race domain.
    """

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            "Validated Q005 implementation must return a dictionary."
        )

    required_fields = {
        "groups",
        "lower",
        "upper",
    }

    missing_fields = (
        required_fields
        - set(result.keys())
    )

    if missing_fields:
        raise ValueError(
            "Validated Q005 implementation is missing fields: "
            f"{sorted(missing_fields)}"
        )

    groups = result[
        "groups"
    ]

    if not isinstance(
        groups,
        dict,
    ):
        raise ValueError(
            "Q005 groups must be a dictionary."
        )

    expected_races = set(
        Q005_RACE_ORDER
    )

    returned_races = set(
        groups.keys()
    )

    if returned_races != expected_races:

        missing = (
            expected_races
            - returned_races
        )

        unexpected = (
            returned_races
            - expected_races
        )

        raise ValueError(
            "Q005 groups do not match the canonical race domain. "
            f"Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    returned_lower = float(
        result["lower"]
    )

    returned_upper = float(
        result["upper"]
    )

    if returned_lower != float(lower):
        raise ValueError(
            "Q005 returned an unexpected lower bound."
        )

    if returned_upper != float(upper):
        raise ValueError(
            "Q005 returned an unexpected upper bound."
        )

    if returned_lower >= returned_upper:
        raise ValueError(
            "Q005 requires lower < upper."
        )

    validated_groups = {}

    for race in Q005_RACE_ORDER:

        group = groups[
            race
        ]

        if not isinstance(
            group,
            dict,
        ):
            raise ValueError(
                "Each Q005 race group must be a dictionary."
            )

        required_group_fields = {
            "bounded_sum",
            "count",
        }

        missing_group_fields = (
            required_group_fields
            - set(group.keys())
        )

        if missing_group_fields:
            raise ValueError(
                f"Q005 group {race!r} is missing fields: "
                f"{sorted(missing_group_fields)}"
            )

        raw_count = group[
            "count"
        ]

        # Reject booleans explicitly because bool is a subclass
        # of int in Python.
        if isinstance(
            raw_count,
            bool,
        ):
            raise ValueError(
                "Q005 group counts must be integers, not booleans."
            )

        count = int(
            raw_count
        )

        if count != raw_count:
            raise ValueError(
                "Q005 group counts must be exact integers."
            )

        if count < 0:
            raise ValueError(
                "Q005 group counts cannot be negative."
            )

        raw_sum = group[
            "bounded_sum"
        ]

        if isinstance(
            raw_sum,
            bool,
        ):
            raise ValueError(
                "Q005 bounded sums must be numeric, not booleans."
            )

        try:
            bounded_sum = float(
                raw_sum
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Q005 bounded sums must be numeric."
            ) from exc

        # Reject NaN and +/- infinity without adding an extra dependency.
        if not (
            float("-inf")
            < bounded_sum
            < float("inf")
        ):
            raise ValueError(
                "Q005 bounded sums must be finite."
            )

        # Because every contributing LOS value is clipped to
        # [lower, upper], the group sum must lie inside the
        # corresponding count-scaled interval.
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

        if (
            count == 0
            and abs(bounded_sum) > tolerance
        ):
            raise ValueError(
                "Q005 zero-count groups must have bounded_sum = 0."
            )

        validated_groups[
            race
        ] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

    return {
        "groups": validated_groups,
        "lower": returned_lower,
        "upper": returned_upper,
    }


def flatten_q005_groups(
    groups: dict,
) -> tuple[list[float], list[int]]:
    """
    Convert the validated Q005 grouped sufficient statistics into two
    canonical 33-element vectors using ``Q005_RACE_ORDER``.

    Returns
    -------
    tuple[list[float], list[int]]
        ``(bounded_sums, counts)`` where:

        - bounded_sums is intended for CKKS encryption later,
        - counts is intended for BFV encryption later.
    """

    if not isinstance(
        groups,
        dict,
    ):
        raise ValueError(
            "Q005 groups must be a dictionary."
        )

    expected_races = set(
        Q005_RACE_ORDER
    )

    returned_races = set(
        groups.keys()
    )

    if returned_races != expected_races:

        missing = (
            expected_races
            - returned_races
        )

        unexpected = (
            returned_races
            - expected_races
        )

        raise ValueError(
            "Q005 groups do not match the canonical race domain. "
            f"Missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    bounded_sums: list[float] = []
    counts: list[int] = []

    for race in Q005_RACE_ORDER:

        group = groups[
            race
        ]

        if not isinstance(
            group,
            dict,
        ):
            raise ValueError(
                "Each Q005 race group must be a dictionary."
            )

        required_fields = {
            "bounded_sum",
            "count",
        }

        missing_fields = (
            required_fields
            - set(group.keys())
        )

        if missing_fields:
            raise ValueError(
                f"Q005 group {race!r} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        raw_sum = group[
            "bounded_sum"
        ]

        if isinstance(
            raw_sum,
            bool,
        ):
            raise ValueError(
                "Q005 bounded sums must be numeric, not booleans."
            )

        bounded_sum = float(
            raw_sum
        )

        if not (
            float("-inf")
            < bounded_sum
            < float("inf")
        ):
            raise ValueError(
                "Q005 bounded sums must be finite."
            )

        raw_count = group[
            "count"
        ]

        if isinstance(
            raw_count,
            bool,
        ):
            raise ValueError(
                "Q005 counts must be integers, not booleans."
            )

        count = int(
            raw_count
        )

        if count != raw_count:
            raise ValueError(
                "Q005 counts must be exact integers."
            )

        if count < 0:
            raise ValueError(
                "Q005 counts cannot be negative."
            )

        bounded_sums.append(
            bounded_sum
        )

        counts.append(
            count
        )

    if len(
        bounded_sums
    ) != Q005_VECTOR_LENGTH:
        raise RuntimeError(
            "Q005 bounded-sum vector has an unexpected length."
        )

    if len(
        counts
    ) != Q005_VECTOR_LENGTH:
        raise RuntimeError(
            "Q005 count vector has an unexpected length."
        )

    return (
        bounded_sums,
        counts,
    )


def q005_vectors_to_groups(
    bounded_sums: list[float],
    counts: list[int],
) -> dict:
    """
    Reconstruct the canonical Q005 race-group mapping from two
    33-element vectors.

    This inverse mapping is intended for later use at the trusted
    release boundary after CKKS/BFV decryption.
    """

    if len(
        bounded_sums
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 bounded-sum vector has invalid length. "
            f"Expected {Q005_VECTOR_LENGTH}, "
            f"received {len(bounded_sums)}."
        )

    if len(
        counts
    ) != Q005_VECTOR_LENGTH:
        raise ValueError(
            "Q005 count vector has invalid length. "
            f"Expected {Q005_VECTOR_LENGTH}, "
            f"received {len(counts)}."
        )

    groups = {}

    for index, race in enumerate(
        Q005_RACE_ORDER
    ):

        raw_sum = bounded_sums[
            index
        ]

        if isinstance(
            raw_sum,
            bool,
        ):
            raise ValueError(
                "Q005 bounded-sum vector values must be numeric, "
                "not booleans."
            )

        bounded_sum = float(
            raw_sum
        )

        if not (
            float("-inf")
            < bounded_sum
            < float("inf")
        ):
            raise ValueError(
                "Q005 bounded-sum vector values must be finite."
            )

        raw_count = counts[
            index
        ]

        if isinstance(
            raw_count,
            bool,
        ):
            raise ValueError(
                "Q005 count-vector values must be integers, "
                "not booleans."
            )

        count = int(
            raw_count
        )

        if count != raw_count:
            raise ValueError(
                "Q005 count-vector values must be exact integers."
            )

        if count < 0:
            raise ValueError(
                "Q005 count-vector values cannot be negative."
            )

        groups[
            race
        ] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

    return groups


def execute_q005_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    lower: float = Q005_LOWER_BOUND,
    upper: float = Q005_UPPER_BOUND,
) -> dict:
    """
    Execute the existing validated Q005 implementation and convert its
    grouped sufficient statistics into canonical 33-element vectors.

    The underlying Q005 semantics remain entirely defined by
    ``node.queries.grouped_bounded_los_sum_and_count``:

        - privacy unit: subject_id,
        - one contribution per patient,
        - earliest admission only,
        - race selected from that earliest admission,
        - LOS measured in days,
        - LOS clipped to [lower, upper],
        - fixed complete 33-race domain,
        - each race returns bounded sum + patient count.

    No group mean, encryption, or differential privacy is computed
    in this adapter.

    Returns
    -------
    dict
        {
            "bounded_sums": list[float],   # length 33
            "counts": list[int],           # length 33
            "lower": float,
            "upper": float,
        }
    """

    result = grouped_bounded_los_sum_and_count(
        dataframe,
        lower=lower,
        upper=upper,
    )

    validated = _validate_q005_grouped_result(
        result,
        lower=lower,
        upper=upper,
    )

    bounded_sums, counts = flatten_q005_groups(
        validated[
            "groups"
        ]
    )

    return {
        "bounded_sums": bounded_sums,
        "counts": counts,
        "lower": validated[
            "lower"
        ],
        "upper": validated[
            "upper"
        ],
    }


def execute_q005_from_file(
    data_file: str | Path,
    *,
    lower: float = Q005_LOWER_BOUND,
    upper: float = Q005_UPPER_BOUND,
) -> dict:
    """
    Load one hospital admissions dataset and execute Q005.

    The returned bounded-sum and count vectors always contain exactly
    33 slots in canonical ``Q005_RACE_ORDER``.
    """

    path = Path(
        data_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q005_from_dataframe(
        dataframe,
        lower=lower,
        upper=upper,
    )


# =====================================================================
# Q006
# =====================================================================

def execute_q006_from_dataframe(
    dataframe: pd.DataFrame,
) -> int:
    """
    Execute the existing validated Q006 rare-subgroup count.

    Q006 semantics remain defined by
    ``node.queries.rare_subgroup_count``:

        - privacy unit: subject_id,
        - one contribution per patient,
        - earliest admission selected first,
        - fixed predicate:
              race = "ASIAN - KOREAN"
              admission_type = "URGENT",
        - return one exact non-negative integer count.

    No encryption or differential privacy is performed here.
    """

    raw_count = rare_subgroup_count(
        dataframe,
        race=Q006_RACE,
        admission_type=Q006_ADMISSION_TYPE,
    )

    # Reject booleans explicitly because bool is a subclass of int.
    if isinstance(
        raw_count,
        bool,
    ):
        raise ValueError(
            "Q006 count must be an integer, not a boolean."
        )

    count = int(
        raw_count
    )

    if count != raw_count:
        raise ValueError(
            "Q006 count must be an exact integer."
        )

    if count < 0:
        raise ValueError(
            "Q006 count cannot be negative."
        )

    return count


def execute_q006_from_file(
    data_file: str | Path,
) -> int:
    """
    Load one hospital admissions dataset and execute Q006.
    """

    path = Path(
        data_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q006_from_dataframe(
        dataframe
    )


# =====================================================================
# Q007
# =====================================================================

def _validate_q007_result(
    result: dict,
    *,
    lower: float,
    upper: float,
) -> dict:
    """
    Validate the output of the existing Q007 implementation without
    changing its statistical semantics.

    Expected form::

        {
            "bounded_sum": float,
            "count": int,
            "lower": float,
            "upper": float,
            "predicate": {
                "race": "ASIAN - KOREAN",
                "admission_type": "URGENT",
            },
        }

    CKKS encryption is performed later, so this adapter returns the exact
    local bounded sum and exact local subgroup count only to the hospital
    service process.
    """

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            "Validated Q007 implementation must return a dictionary."
        )

    required_fields = {
        "bounded_sum",
        "count",
        "lower",
        "upper",
        "predicate",
    }

    missing_fields = (
        required_fields
        - set(result.keys())
    )

    if missing_fields:
        raise ValueError(
            "Validated Q007 implementation is missing fields: "
            f"{sorted(missing_fields)}"
        )

    returned_lower = float(
        result["lower"]
    )

    returned_upper = float(
        result["upper"]
    )

    if returned_lower != float(lower):
        raise ValueError(
            "Q007 returned an unexpected lower bound."
        )

    if returned_upper != float(upper):
        raise ValueError(
            "Q007 returned an unexpected upper bound."
        )

    if returned_lower >= returned_upper:
        raise ValueError(
            "Q007 requires lower < upper."
        )

    predicate = result[
        "predicate"
    ]

    if not isinstance(
        predicate,
        dict,
    ):
        raise ValueError(
            "Q007 predicate metadata must be a dictionary."
        )

    if set(
        predicate.keys()
    ) != {
        "race",
        "admission_type",
    }:
        raise ValueError(
            "Q007 predicate metadata contains unexpected fields."
        )

    if predicate["race"] != Q007_RACE:
        raise ValueError(
            "Q007 returned an unexpected race predicate."
        )

    if predicate["admission_type"] != Q007_ADMISSION_TYPE:
        raise ValueError(
            "Q007 returned an unexpected admission-type predicate."
        )

    raw_count = result[
        "count"
    ]

    if isinstance(
        raw_count,
        bool,
    ):
        raise ValueError(
            "Q007 count must be an integer, not a boolean."
        )

    count = int(
        raw_count
    )

    if count != raw_count:
        raise ValueError(
            "Q007 count must be an exact integer."
        )

    if count < 0:
        raise ValueError(
            "Q007 count cannot be negative."
        )

    raw_sum = result[
        "bounded_sum"
    ]

    if isinstance(
        raw_sum,
        bool,
    ):
        raise ValueError(
            "Q007 bounded sum must be numeric, not a boolean."
        )

    try:
        bounded_sum = float(
            raw_sum
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Q007 bounded sum must be numeric."
        ) from exc

    if not (
        float("-inf")
        < bounded_sum
        < float("inf")
    ):
        raise ValueError(
            "Q007 bounded sum must be finite."
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
            "Q007 bounded sum is below the valid "
            "count-scaled lower bound."
        )

    if bounded_sum > (
        maximum_sum
        + tolerance
    ):
        raise ValueError(
            "Q007 bounded sum exceeds the valid "
            "count-scaled upper bound."
        )

    if (
        count == 0
        and abs(bounded_sum) > tolerance
    ):
        raise ValueError(
            "Q007 zero-count result must have bounded_sum = 0."
        )

    return {
        "bounded_sum": bounded_sum,
        "count": count,
        "lower": returned_lower,
        "upper": returned_upper,
        "predicate": {
            "race": Q007_RACE,
            "admission_type": Q007_ADMISSION_TYPE,
        },
    }


def execute_q007_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    lower: float = Q007_LOWER_BOUND,
    upper: float = Q007_UPPER_BOUND,
) -> dict:
    """
    Execute the existing validated Q007 rare-subgroup bounded-mean
    sufficient-statistic query.

    The underlying semantics remain entirely defined by
    ``node.queries.rare_subgroup_bounded_los_sum_and_count``:

        - privacy unit: subject_id,
        - earliest admission selected before subgroup filtering,
        - fixed subgroup predicate shared with Q006,
        - one LOS contribution per matching patient,
        - LOS clipped to [lower, upper],
        - return bounded sum + matching-patient count.

    No local mean, encryption, or differential privacy is computed here.
    """

    result = rare_subgroup_bounded_los_sum_and_count(
        dataframe,
        lower=lower,
        upper=upper,
        race=Q007_RACE,
        admission_type=Q007_ADMISSION_TYPE,
    )

    return _validate_q007_result(
        result,
        lower=lower,
        upper=upper,
    )


def execute_q007_from_file(
    data_file: str | Path,
    *,
    lower: float = Q007_LOWER_BOUND,
    upper: float = Q007_UPPER_BOUND,
) -> dict:
    """
    Load one hospital admissions dataset and execute Q007.
    """

    path = Path(
        data_file
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Hospital dataset not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    return execute_q007_from_dataframe(
        dataframe,
        lower=lower,
        upper=upper,
    )
