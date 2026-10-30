"""
Q008 experimental query:
Rare-cohort prolonged-stay rates.

For every fixed race × admission-type group, count patients whose
earliest admission has:

    prolonged_stay = bounded LOS > 14 days

Privacy/contribution unit:
    subject_id

Contribution rule:
    one contribution per patient, using the earliest admission.

This file is experimental only. It does not modify Q001-Q007.
"""

from __future__ import annotations

import pandas as pd

from node.queries import (
    Q004_RACE_CATEGORIES,
    Q004_ADMISSION_TYPES,
)


Q008_LOS_LOWER = 0.0
Q008_LOS_UPPER = 30.0
Q008_PROLONGED_THRESHOLD = 14.0


def prolonged_stay_histogram(
    dataframe: pd.DataFrame,
    threshold: float = Q008_PROLONGED_THRESHOLD,
    lower: float = Q008_LOS_LOWER,
    upper: float = Q008_LOS_UPPER,
) -> dict:
    """
    Compute Q008 exact patient-level prolonged-stay histogram.

    Output domain:

        33 race categories
        × 9 admission types
        × 2 prolonged-stay states

        = 594 fixed histogram cells.

    Each patient contributes to exactly one cell.
    """

    required_columns = {
        "subject_id",
        "admittime",
        "dischtime",
        "race",
        "admission_type",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Q008 requires columns: "
            "subject_id, admittime, dischtime, race, admission_type. "
            f"Missing: {sorted(missing_columns)}"
        )

    if lower >= upper:
        raise ValueError("Q008 requires lower < upper.")

    if not lower < threshold < upper:
        raise ValueError(
            "Q008 requires the prolonged-stay threshold "
            "to lie strictly inside the LOS bounds."
        )

    # ---------------------------------------------------------
    # Keep only required fields.
    # ---------------------------------------------------------

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "dischtime",
            "race",
            "admission_type",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Parse timestamps using the same convention as Q003/Q007.
    # ---------------------------------------------------------

    patient_data["admittime"] = pd.to_datetime(
        patient_data["admittime"],
        errors="coerce",
    )

    patient_data["dischtime"] = pd.to_datetime(
        patient_data["dischtime"],
        errors="coerce",
    )

    if (
        patient_data["admittime"].isna().any()
        or patient_data["dischtime"].isna().any()
    ):
        raise ValueError(
            "Q008 encountered invalid admission or discharge times."
        )

    # ---------------------------------------------------------
    # One contribution per patient:
    # retain the earliest admission.
    # ---------------------------------------------------------

    earliest_admission = (
        patient_data
        .sort_values(
            by=[
                "subject_id",
                "admittime",
            ]
        )
        .drop_duplicates(
            subset="subject_id",
            keep="first",
        )
        .copy()
    )

    # ---------------------------------------------------------
    # Validate categorical domain.
    # ---------------------------------------------------------

    if earliest_admission["race"].isna().any():
        raise ValueError(
            "Q008 encountered missing race values."
        )

    if earliest_admission["admission_type"].isna().any():
        raise ValueError(
            "Q008 encountered missing admission_type values."
        )

    observed_races = set(
        earliest_admission["race"].astype(str)
    )

    unexpected_races = (
        observed_races - set(Q004_RACE_CATEGORIES)
    )

    if unexpected_races:
        raise ValueError(
            "Q008 encountered race categories outside the "
            f"approved domain: {sorted(unexpected_races)}"
        )

    observed_admission_types = set(
        earliest_admission["admission_type"].astype(str)
    )

    unexpected_admission_types = (
        observed_admission_types
        - set(Q004_ADMISSION_TYPES)
    )

    if unexpected_admission_types:
        raise ValueError(
            "Q008 encountered admission types outside the "
            f"approved domain: {sorted(unexpected_admission_types)}"
        )

    # ---------------------------------------------------------
    # Calculate LOS exactly as in the existing query family.
    # ---------------------------------------------------------

    earliest_admission["los_days"] = (
        earliest_admission["dischtime"]
        - earliest_admission["admittime"]
    ).dt.total_seconds() / 86400.0

    # Use the same approved LOS domain [0, 30].
    earliest_admission["bounded_los_days"] = (
        earliest_admission["los_days"]
        .clip(
            lower=lower,
            upper=upper,
        )
    )

    # > 14 days = prolonged.
    earliest_admission["prolonged_stay"] = (
        earliest_admission["bounded_los_days"] > threshold
    )

    # ---------------------------------------------------------
    # Construct the complete fixed 33 × 9 × 2 histogram.
    # ---------------------------------------------------------

    grouped = (
        earliest_admission
        .groupby(
            [
                "race",
                "admission_type",
                "prolonged_stay",
            ]
        )
        .size()
    )

    histogram = {}
    rates = {}

    for race in Q004_RACE_CATEGORIES:

        histogram[race] = {}
        rates[race] = {}

        for admission_type in Q004_ADMISSION_TYPES:

            not_prolonged = int(
                grouped.get(
                    (
                        race,
                        admission_type,
                        False,
                    ),
                    0,
                )
            )

            prolonged = int(
                grouped.get(
                    (
                        race,
                        admission_type,
                        True,
                    ),
                    0,
                )
            )

            total = not_prolonged + prolonged

            if total > 0:
                prolonged_rate = prolonged / total
            else:
                prolonged_rate = None

            histogram[race][admission_type] = {
                "not_prolonged": not_prolonged,
                "prolonged": prolonged,
            }

            rates[race][admission_type] = {
                "total": total,
                "prolonged": prolonged,
                "prolonged_rate": prolonged_rate,
            }

    # ---------------------------------------------------------
    # Internal consistency checks.
    # ---------------------------------------------------------

    total_histogram_count = sum(
        cell["not_prolonged"] + cell["prolonged"]
        for race_data in histogram.values()
        for cell in race_data.values()
    )

    patient_count = int(len(earliest_admission))

    if total_histogram_count != patient_count:
        raise RuntimeError(
            "Q008 histogram total does not equal the "
            "number of contributing patients."
        )

    return {
        "query_id": "Q008",
        "description": (
            "Race × admission-type prolonged-stay histogram"
        ),
        "privacy_unit": "subject_id",
        "contribution_rule": "earliest_admission",
        "los_lower": float(lower),
        "los_upper": float(upper),
        "prolonged_threshold_days": float(threshold),
        "prolonged_definition": f"bounded_los_days > {threshold}",
        "race_categories": len(Q004_RACE_CATEGORIES),
        "admission_types": len(Q004_ADMISSION_TYPES),
        "histogram_cells": (
            len(Q004_RACE_CATEGORIES)
            * len(Q004_ADMISSION_TYPES)
            * 2
        ),
        "patient_count": patient_count,
        "histogram": histogram,
        "rates": rates,
    }