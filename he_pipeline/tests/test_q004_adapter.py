import pandas as pd
import pytest

from he_pipeline.node_service.query_adapter import (
    Q004_ADMISSION_TYPE_ORDER,
    Q004_RACE_ORDER,
    Q004_VECTOR_LENGTH,
    execute_q004_from_dataframe,
    flatten_q004_table,
    q004_vector_to_table,
)


def q004_index(
    race: str,
    admission_type: str,
) -> int:
    """
    Return the canonical flattened Q004 vector index.
    """

    race_index = (
        Q004_RACE_ORDER.index(
            race
        )
    )

    admission_index = (
        Q004_ADMISSION_TYPE_ORDER.index(
            admission_type
        )
    )

    return (
        race_index
        * len(Q004_ADMISSION_TYPE_ORDER)
        + admission_index
    )


def test_q004_domain_shape():
    """
    Q004 must always be represented as:

        33 races x 9 admission types = 297 cells.
    """

    assert len(
        Q004_RACE_ORDER
    ) == 33

    assert len(
        Q004_ADMISSION_TYPE_ORDER
    ) == 9

    assert (
        Q004_VECTOR_LENGTH
        == 297
    )


def test_q004_earliest_admission_and_canonical_order():
    """
    Each patient contributes exactly once using their earliest
    admission, and the flattened slot ordering must be canonical.
    """

    race_0 = Q004_RACE_ORDER[0]
    race_1 = Q004_RACE_ORDER[1]

    admission_0 = (
        Q004_ADMISSION_TYPE_ORDER[0]
    )

    admission_1 = (
        Q004_ADMISSION_TYPE_ORDER[1]
    )

    dataframe = pd.DataFrame(
        [
            # Patient 1 earliest admission.
            {
                "subject_id": 1,
                "admittime": "2025-01-01 10:00:00",
                "race": race_0,
                "admission_type": admission_0,
            },

            # Patient 1 later admission.
            # This MUST NOT contribute.
            {
                "subject_id": 1,
                "admittime": "2025-02-01 10:00:00",
                "race": race_1,
                "admission_type": admission_1,
            },

            # Patient 2.
            {
                "subject_id": 2,
                "admittime": "2025-01-05 10:00:00",
                "race": race_1,
                "admission_type": admission_1,
            },

            # Patient 3.
            {
                "subject_id": 3,
                "admittime": "2025-01-07 10:00:00",
                "race": race_0,
                "admission_type": admission_1,
            },
        ]
    )

    vector = execute_q004_from_dataframe(
        dataframe
    )

    assert len(vector) == 297

    assert all(
        isinstance(value, int)
        for value in vector
    )

    # Three unique patients.
    assert sum(vector) == 3

    assert vector[
        q004_index(
            race_0,
            admission_0,
        )
    ] == 1

    assert vector[
        q004_index(
            race_0,
            admission_1,
        )
    ] == 1

    assert vector[
        q004_index(
            race_1,
            admission_1,
        )
    ] == 1

    # The later admission of patient 1 must not produce a second
    # contribution.
    assert sum(vector) == 3


def test_q004_flatten_reconstruct_round_trip():
    """
    Flattening and reconstruction must preserve every cell exactly.
    """

    table = {
        race: {
            admission_type: 0
            for admission_type
            in Q004_ADMISSION_TYPE_ORDER
        }
        for race in Q004_RACE_ORDER
    }

    table[
        Q004_RACE_ORDER[0]
    ][
        Q004_ADMISSION_TYPE_ORDER[0]
    ] = 7

    table[
        Q004_RACE_ORDER[10]
    ][
        Q004_ADMISSION_TYPE_ORDER[5]
    ] = 11

    table[
        Q004_RACE_ORDER[-1]
    ][
        Q004_ADMISSION_TYPE_ORDER[-1]
    ] = 13

    vector = flatten_q004_table(
        table
    )

    reconstructed = q004_vector_to_table(
        vector
    )

    assert len(vector) == 297

    assert reconstructed == table


def test_q004_rejects_invalid_vector():
    """
    The reconstruction boundary must reject malformed HE output.
    """

    with pytest.raises(
        ValueError
    ):
        q004_vector_to_table(
            [0] * 296
        )

    invalid_vector = (
        [0] * 297
    )

    invalid_vector[100] = -1

    with pytest.raises(
        ValueError
    ):
        q004_vector_to_table(
            invalid_vector
        )