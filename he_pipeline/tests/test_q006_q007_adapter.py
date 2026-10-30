"""
Unit tests for the Q006 and Q007 HE query adapters.

These tests validate only the hospital adapter boundary:

Q006:
    validated rare-subgroup patient count
        ->
    exact non-negative integer for BFV encryption

Q007:
    validated rare-subgroup bounded LOS sufficient statistics
        ->
    bounded real sum for CKKS encryption
    + exact non-negative count for BFV encryption

No homomorphic encryption and no differential privacy are exercised here.
"""

import pandas as pd
import pytest

import he_pipeline.node_service.query_adapter as adapter
from he_pipeline.node_service.query_adapter import (
    Q006_ADMISSION_TYPE,
    Q006_RACE,
    Q007_ADMISSION_TYPE,
    Q007_LOWER_BOUND,
    Q007_RACE,
    Q007_UPPER_BOUND,
    execute_q006_from_dataframe,
    execute_q007_from_dataframe,
)


def test_q006_q007_share_fixed_predicate_and_bounds() -> None:
    """
    Q006 and Q007 must use the same registry-approved rare subgroup.
    """

    assert Q006_RACE == "ASIAN - KOREAN"
    assert Q006_ADMISSION_TYPE == "URGENT"

    assert Q007_RACE == Q006_RACE
    assert Q007_ADMISSION_TYPE == Q006_ADMISSION_TYPE

    assert Q007_LOWER_BOUND == 0.0
    assert Q007_UPPER_BOUND == 30.0


def test_q006_selects_earliest_admission_before_subgroup_filter() -> None:
    """
    A later matching admission must not make a patient eligible when that
    patient's earliest admission does not satisfy the Q006 predicate.
    """

    dataframe = pd.DataFrame(
        {
            "subject_id": [
                1,
                1,
                2,
                3,
                4,
            ],
            "admittime": [
                "2026-01-01 00:00:00",
                "2026-02-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-03 00:00:00",
                "2026-01-04 00:00:00",
            ],
            "race": [
                "WHITE",
                Q006_RACE,
                Q006_RACE,
                Q006_RACE,
                "ASIAN - CHINESE",
            ],
            "admission_type": [
                "URGENT",
                Q006_ADMISSION_TYPE,
                Q006_ADMISSION_TYPE,
                "ELECTIVE",
                Q006_ADMISSION_TYPE,
            ],
        }
    )

    count = execute_q006_from_dataframe(
        dataframe
    )

    # Only subject 2 has a matching earliest admission.
    assert count == 1


def test_q006_returns_zero_when_no_patient_matches() -> None:
    dataframe = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "admittime": [
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
            ],
            "race": ["WHITE", "ASIAN - CHINESE"],
            "admission_type": ["URGENT", "ELECTIVE"],
        }
    )

    assert execute_q006_from_dataframe(dataframe) == 0


def test_q006_rejects_non_integer_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The HE boundary must not accept an inexact count representation.
    """

    monkeypatch.setattr(
        adapter,
        "rare_subgroup_count",
        lambda *args, **kwargs: 1.5,
    )

    with pytest.raises(
        ValueError,
        match="exact integer",
    ):
        execute_q006_from_dataframe(
            pd.DataFrame()
        )


def test_q007_selects_earliest_admission_filters_subgroup_and_clips_los() -> None:
    """
    Q007 must select earliest admissions first, then apply the subgroup
    predicate, and clip matching LOS values to [0, 30] days.
    """

    dataframe = pd.DataFrame(
        {
            "subject_id": [
                1,
                1,
                2,
                3,
                4,
            ],
            "admittime": [
                "2026-01-01 00:00:00",
                "2026-02-01 00:00:00",
                "2026-01-02 00:00:00",
                "2026-01-03 00:00:00",
                "2026-01-04 00:00:00",
            ],
            "dischtime": [
                "2026-01-03 00:00:00",  # 2 days, but non-matching earliest
                "2026-02-11 00:00:00",  # later match: must be ignored
                "2026-02-16 00:00:00",  # 45 days -> clipped to 30
                "2026-01-08 00:00:00",  # 5 days and matching
                "2026-01-10 00:00:00",  # non-matching race
            ],
            "race": [
                "WHITE",
                Q007_RACE,
                Q007_RACE,
                Q007_RACE,
                "ASIAN - CHINESE",
            ],
            "admission_type": [
                "URGENT",
                Q007_ADMISSION_TYPE,
                Q007_ADMISSION_TYPE,
                Q007_ADMISSION_TYPE,
                Q007_ADMISSION_TYPE,
            ],
        }
    )

    result = execute_q007_from_dataframe(
        dataframe
    )

    assert result["count"] == 2
    assert result["bounded_sum"] == pytest.approx(35.0)
    assert result["lower"] == Q007_LOWER_BOUND
    assert result["upper"] == Q007_UPPER_BOUND
    assert result["predicate"] == {
        "race": Q007_RACE,
        "admission_type": Q007_ADMISSION_TYPE,
    }


def test_q007_zero_match_result_is_valid() -> None:
    dataframe = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "admittime": [
                "2026-01-01 00:00:00",
                "2026-01-02 00:00:00",
            ],
            "dischtime": [
                "2026-01-03 00:00:00",
                "2026-01-05 00:00:00",
            ],
            "race": ["WHITE", "ASIAN - CHINESE"],
            "admission_type": ["URGENT", "ELECTIVE"],
        }
    )

    result = execute_q007_from_dataframe(
        dataframe
    )

    assert result["count"] == 0
    assert result["bounded_sum"] == pytest.approx(0.0)


def test_q007_rejects_unexpected_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The adapter must reject any silent change to the approved subgroup.
    """

    monkeypatch.setattr(
        adapter,
        "rare_subgroup_bounded_los_sum_and_count",
        lambda *args, **kwargs: {
            "bounded_sum": 3.0,
            "count": 1,
            "lower": 0.0,
            "upper": 30.0,
            "predicate": {
                "race": "WHITE",
                "admission_type": "URGENT",
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="unexpected race predicate",
    ):
        execute_q007_from_dataframe(
            pd.DataFrame()
        )


def test_q007_rejects_sum_above_count_scaled_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    One matching patient can contribute at most 30 bounded LOS days.
    """

    monkeypatch.setattr(
        adapter,
        "rare_subgroup_bounded_los_sum_and_count",
        lambda *args, **kwargs: {
            "bounded_sum": 31.0,
            "count": 1,
            "lower": 0.0,
            "upper": 30.0,
            "predicate": {
                "race": Q007_RACE,
                "admission_type": Q007_ADMISSION_TYPE,
            },
        },
    )

    with pytest.raises(
        ValueError,
        match="count-scaled upper bound",
    ):
        execute_q007_from_dataframe(
            pd.DataFrame()
        )
