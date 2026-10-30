"""
Unit tests for the Q005 HE query adapter.

These tests validate only the adapter boundary:

    validated Q005 query semantics
        ->
    canonical 33-element bounded-sum vector
        +
    canonical 33-element count vector

No homomorphic encryption and no differential privacy are exercised here.
"""

import pandas as pd
import pytest

from he_pipeline.node_service.query_adapter import (
    Q005_LOWER_BOUND,
    Q005_RACE_ORDER,
    Q005_UPPER_BOUND,
    Q005_VECTOR_LENGTH,
    execute_q005_from_dataframe,
    flatten_q005_groups,
    q005_vectors_to_groups,
)


def test_q005_canonical_domain_shape() -> None:
    """
    Q005 must use one fixed 33-race domain for both vectors.
    """

    assert len(Q005_RACE_ORDER) == 33
    assert Q005_VECTOR_LENGTH == 33
    assert Q005_LOWER_BOUND == 0.0
    assert Q005_UPPER_BOUND == 30.0
    assert len(set(Q005_RACE_ORDER)) == 33


def test_q005_uses_earliest_admission_and_clips_los() -> None:
    """
    Each patient contributes exactly once using the earliest admission.

    The test also verifies that LOS values above 30 days are clipped.
    """

    race_0 = Q005_RACE_ORDER[0]
    race_1 = Q005_RACE_ORDER[1]

    dataframe = pd.DataFrame(
        {
            "subject_id": [
                1,
                1,
                2,
                3,
            ],
            "admittime": [
                "2026-01-01 00:00:00",
                "2026-02-01 00:00:00",
                "2026-01-03 00:00:00",
                "2026-01-04 00:00:00",
            ],
            "dischtime": [
                "2026-01-03 00:00:00",  # 2 days, earliest for patient 1
                "2026-02-11 00:00:00",  # later admission, must be ignored
                "2026-02-20 00:00:00",  # >30 days, clipped to 30
                "2026-01-08 00:00:00",  # 4 days
            ],
            "race": [
                race_0,
                race_1,
                race_1,
                race_0,
            ],
        }
    )

    result = execute_q005_from_dataframe(
        dataframe
    )

    bounded_sums = result["bounded_sums"]
    counts = result["counts"]

    assert len(bounded_sums) == Q005_VECTOR_LENGTH
    assert len(counts) == Q005_VECTOR_LENGTH

    race_0_index = Q005_RACE_ORDER.index(
        race_0
    )

    race_1_index = Q005_RACE_ORDER.index(
        race_1
    )

    assert counts[race_0_index] == 2
    assert bounded_sums[race_0_index] == pytest.approx(
        6.0
    )

    assert counts[race_1_index] == 1
    assert bounded_sums[race_1_index] == pytest.approx(
        30.0
    )

    assert sum(counts) == 3
    assert sum(bounded_sums) == pytest.approx(
        36.0
    )

    assert result["lower"] == Q005_LOWER_BOUND
    assert result["upper"] == Q005_UPPER_BOUND


def test_q005_flatten_and_inverse_mapping_roundtrip() -> None:
    """
    The two Q005 vectors must preserve the canonical race-slot mapping.
    """

    groups = {
        race: {
            "bounded_sum": 0.0,
            "count": 0,
        }
        for race in Q005_RACE_ORDER
    }

    first_race = Q005_RACE_ORDER[0]
    last_race = Q005_RACE_ORDER[-1]

    groups[first_race] = {
        "bounded_sum": 12.5,
        "count": 4,
    }

    groups[last_race] = {
        "bounded_sum": 30.0,
        "count": 1,
    }

    bounded_sums, counts = flatten_q005_groups(
        groups
    )

    assert len(bounded_sums) == Q005_VECTOR_LENGTH
    assert len(counts) == Q005_VECTOR_LENGTH

    assert bounded_sums[0] == pytest.approx(
        12.5
    )
    assert counts[0] == 4

    assert bounded_sums[-1] == pytest.approx(
        30.0
    )
    assert counts[-1] == 1

    reconstructed = q005_vectors_to_groups(
        bounded_sums,
        counts,
    )

    assert tuple(reconstructed.keys()) == Q005_RACE_ORDER

    assert reconstructed[first_race]["bounded_sum"] == pytest.approx(
        12.5
    )
    assert reconstructed[first_race]["count"] == 4

    assert reconstructed[last_race]["bounded_sum"] == pytest.approx(
        30.0
    )
    assert reconstructed[last_race]["count"] == 1


def test_q005_flatten_rejects_incomplete_race_domain() -> None:
    """
    Hospitals must not silently omit race slots.
    """

    groups = {
        race: {
            "bounded_sum": 0.0,
            "count": 0,
        }
        for race in Q005_RACE_ORDER
    }

    groups.pop(
        Q005_RACE_ORDER[-1]
    )

    with pytest.raises(
        ValueError,
        match="canonical race domain",
    ):
        flatten_q005_groups(
            groups
        )


def test_q005_inverse_rejects_invalid_vector_length() -> None:
    """
    Both decrypted vectors must contain exactly 33 slots.
    """

    with pytest.raises(
        ValueError,
        match="bounded-sum vector has invalid length",
    ):
        q005_vectors_to_groups(
            [0.0] * (Q005_VECTOR_LENGTH - 1),
            [0] * Q005_VECTOR_LENGTH,
        )

    with pytest.raises(
        ValueError,
        match="count vector has invalid length",
    ):
        q005_vectors_to_groups(
            [0.0] * Q005_VECTOR_LENGTH,
            [0] * (Q005_VECTOR_LENGTH - 1),
        )


def test_q005_rejects_negative_counts() -> None:
    """
    Count-vector slots must remain exact non-negative integers.
    """

    bounded_sums = [
        0.0
    ] * Q005_VECTOR_LENGTH

    counts = [
        0
    ] * Q005_VECTOR_LENGTH

    counts[0] = -1

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        q005_vectors_to_groups(
            bounded_sums,
            counts,
        )
