"""
Controlled synthetic cohorts for the Q008 disclosure experiment.

For each cohort size, two otherwise identical datasets are produced:

    SHORT case:
        target LOS = 7 days  -> not prolonged

    LONG case:
        target LOS = 20 days -> prolonged

All other patients are identical across the two datasets.

These datasets are experimental only and do not modify MIMIC data.
"""

from __future__ import annotations

import pandas as pd

from node.queries import (
    Q004_RACE_CATEGORIES,
    Q004_ADMISSION_TYPES,
)


# ---------------------------------------------------------------------
# Experimental constants
# ---------------------------------------------------------------------

DEFAULT_RACE = "ASIAN - KOREAN"
DEFAULT_ADMISSION_TYPE = "ELECTIVE"

SHORT_LOS_DAYS = 7.0
LONG_LOS_DAYS = 20.0

TARGET_SUBJECT_ID = 999_999_999

BASE_ADMISSION_TIME = pd.Timestamp(
    "2026-01-01 08:00:00"
)


def make_controlled_cohort(
    cohort_size: int,
    target_prolonged: bool,
    known_prolonged_fraction: float = 0.20,
    race: str = DEFAULT_RACE,
    admission_type: str = DEFAULT_ADMISSION_TYPE,
) -> pd.DataFrame:
    """
    Construct one controlled synthetic Q008 cohort.

    Parameters
    ----------
    cohort_size:
        Total number of patients, including the target.

    target_prolonged:
        False -> target LOS = 7 days
        True  -> target LOS = 20 days

    known_prolonged_fraction:
        Approximate fraction of the OTHER patients whose LOS
        is prolonged.

    Returns
    -------
    pandas.DataFrame

        Columns:
            subject_id
            admittime
            dischtime
            race
            admission_type
    """

    if cohort_size < 2:
        raise ValueError(
            "cohort_size must be at least 2."
        )

    if not 0.0 <= known_prolonged_fraction <= 1.0:
        raise ValueError(
            "known_prolonged_fraction must be between 0 and 1."
        )

    if race not in Q004_RACE_CATEGORIES:
        raise ValueError(
            f"Race is outside the approved domain: {race}"
        )

    if admission_type not in Q004_ADMISSION_TYPES:
        raise ValueError(
            "Admission type is outside the approved domain: "
            f"{admission_type}"
        )

    number_of_other_patients = cohort_size - 1

    # Approx. 20% prolonged among the patients whose status
    # is assumed known to the attacker.
    known_prolonged_count = round(
        number_of_other_patients
        * known_prolonged_fraction
    )

    # Make sure our very small cohorts contain at least one
    # prolonged background patient.
    if (
        known_prolonged_fraction > 0
        and known_prolonged_count == 0
    ):
        known_prolonged_count = 1

    known_prolonged_count = min(
        known_prolonged_count,
        number_of_other_patients,
    )

    rows = []

    # -----------------------------------------------------------------
    # Create all non-target patients.
    # Their data is identical in SHORT and LONG datasets.
    # -----------------------------------------------------------------

    for index in range(number_of_other_patients):

        subject_id = 800_000_000 + index

        admittime = (
            BASE_ADMISSION_TIME
            + pd.Timedelta(days=index)
        )

        if index < known_prolonged_count:
            los_days = LONG_LOS_DAYS
        else:
            los_days = SHORT_LOS_DAYS

        dischtime = (
            admittime
            + pd.Timedelta(days=los_days)
        )

        rows.append(
            {
                "subject_id": subject_id,
                "admittime": admittime,
                "dischtime": dischtime,
                "race": race,
                "admission_type": admission_type,
            }
        )

    # -----------------------------------------------------------------
    # Controlled target patient.
    # Only this patient's LOS differs between the two datasets.
    # -----------------------------------------------------------------

    target_admittime = (
        BASE_ADMISSION_TIME
        + pd.Timedelta(days=number_of_other_patients)
    )

    target_los = (
        LONG_LOS_DAYS
        if target_prolonged
        else SHORT_LOS_DAYS
    )

    target_dischtime = (
        target_admittime
        + pd.Timedelta(days=target_los)
    )

    rows.append(
        {
            "subject_id": TARGET_SUBJECT_ID,
            "admittime": target_admittime,
            "dischtime": target_dischtime,
            "race": race,
            "admission_type": admission_type,
        }
    )

    dataframe = pd.DataFrame(rows)

    return dataframe


def make_attribute_inference_pair(
    cohort_size: int,
    known_prolonged_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return the controlled SHORT and LONG datasets.

    Dataset S:
        target LOS = 7 days.

    Dataset P:
        target LOS = 20 days.

    Everything except the target's discharge time is identical.
    """

    short_dataset = make_controlled_cohort(
        cohort_size=cohort_size,
        target_prolonged=False,
        known_prolonged_fraction=known_prolonged_fraction,
    )

    long_dataset = make_controlled_cohort(
        cohort_size=cohort_size,
        target_prolonged=True,
        known_prolonged_fraction=known_prolonged_fraction,
    )

    return short_dataset, long_dataset