"""
    Approved local query implementtions for hospital nodes 
    
    Each function is this modules operates only on the local hospital dataset and returns an aggregate statistic. 
    
    This module does not perform 
    Differential privacy,
    privacy budget accounting , 
    inter-hospital communication,   
    global aggregation,
    
    That is managed by the coordinator
     
"""

import pandas as pd

Q004_RACE_CATEGORIES = (
    "AMERICAN INDIAN/ALASKA NATIVE",
    "ASIAN",
    "ASIAN - ASIAN INDIAN",
    "ASIAN - CHINESE",
    "ASIAN - KOREAN",
    "ASIAN - SOUTH EAST ASIAN",
    "BLACK/AFRICAN",
    "BLACK/AFRICAN AMERICAN",
    "BLACK/CAPE VERDEAN",
    "BLACK/CARIBBEAN ISLAND",
    "HISPANIC OR LATINO",
    "HISPANIC/LATINO - CENTRAL AMERICAN",
    "HISPANIC/LATINO - COLUMBIAN",
    "HISPANIC/LATINO - CUBAN",
    "HISPANIC/LATINO - DOMINICAN",
    "HISPANIC/LATINO - GUATEMALAN",
    "HISPANIC/LATINO - HONDURAN",
    "HISPANIC/LATINO - MEXICAN",
    "HISPANIC/LATINO - PUERTO RICAN",
    "HISPANIC/LATINO - SALVADORAN",
    "MULTIPLE RACE/ETHNICITY",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER",
    "OTHER",
    "PATIENT DECLINED TO ANSWER",
    "PORTUGUESE",
    "SOUTH AMERICAN",
    "UNABLE TO OBTAIN",
    "UNKNOWN",
    "WHITE",
    "WHITE - BRAZILIAN",
    "WHITE - EASTERN EUROPEAN",
    "WHITE - OTHER EUROPEAN",
    "WHITE - RUSSIAN",
)


Q004_ADMISSION_TYPES = (
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

Q006_RACE = "ASIAN - KOREAN"
Q006_ADMISSION_TYPE = "URGENT"




def unique_patient_count(
    dataframe: pd.DataFrame,
) -> int:
    """
    Q001 — Unique patient count.

    Count the number of distinct patients represented in the
    local hospital admissions table.

    Privacy unit
    ------------
    Patient, identified by ``subject_id``.

    Contribution rule
    -----------------
    Each patient contributes once to the count.

    Returns
    -------
    int
        Exact number of unique patients in the local hospital.
    """

    if "subject_id" not in dataframe.columns:
        raise ValueError(
            "Q001 requires the 'subject_id' column."
        )

    return int(
        dataframe["subject_id"].nunique()
    )
    
def admission_type_histogram(dataframe) -> dict[str, int]:
    """
    Compute the patient-level admission-type histogram for Q002.

    Each patient contributes to exactly one category. The category is
    determined from the patient's earliest hospital admission.

    Parameters
    ----------
    dataframe:
        Hospital admissions dataframe containing at least
        subject_id, admittime, and admission_type.

    Returns
    -------
    dict[str, int]
        Mapping from admission type to number of unique patients
        assigned to that category.
    """

    required_columns = {
        "subject_id",
        "admittime",
        "admission_type",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Q002 requires columns: "
            "subject_id, admittime, admission_type."
        )

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "admission_type",
        ]
    ].copy()

    patient_data["admittime"] = pd.to_datetime(
        patient_data["admittime"],
        errors="coerce",
    )

    if patient_data["admittime"].isna().any():
        raise ValueError(
            "Q002 encountered invalid admittime values."
        )

    # Sort chronologically so the first row for each patient
    # represents that patient's earliest admission.
    patient_data = patient_data.sort_values(
        by=[
            "subject_id",
            "admittime",
        ]
    )

    earliest_admission = patient_data.drop_duplicates(
        subset="subject_id",
        keep="first",
    )

    if earliest_admission["admission_type"].isna().any():
        raise ValueError(
            "Q002 encountered missing admission_type values."
        )

    histogram = (
        earliest_admission["admission_type"]
        .value_counts()
        .sort_index()
    )

    # Ensure a fixed public output domain for the vector DP mechanism.
    # In particular, synthetic or small hospitals may have no patients
    # for some approved admission-type categories.
    unexpected_categories = (
        set(histogram.index.astype(str)) - set(Q004_ADMISSION_TYPES)
    )
    if unexpected_categories:
        raise ValueError(
            "Q002 encountered admission types outside the approved domain: "
            f"{sorted(unexpected_categories)}"
        )

    return {
        category: int(histogram.get(category, 0))
        for category in Q004_ADMISSION_TYPES
    }
    
    
def bounded_los_sum_and_count(
    dataframe,
    lower: float = 0.0,
    upper: float = 30.0,
) -> dict:
    """
    Compute local sufficient statistics for Q003.

    Each patient contributes one length-of-stay value from the
    patient's earliest admission. LOS is measured in days and
    clipped to the interval [lower, upper].

    The hospital returns a bounded sum and patient count rather
    than a local mean. The coordinator can therefore construct
    the correct global mean from distributed sufficient statistics.

    Parameters
    ----------
    dataframe:
        Hospital admissions dataframe containing subject_id,
        admittime, and dischtime.

    lower:
        Lower LOS bound in days.

    upper:
        Upper LOS bound in days.

    Returns
    -------
    dict
        Bounded LOS sum and contributing patient count.
    """

    required_columns = {
        "subject_id",
        "admittime",
        "dischtime",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Q003 requires columns: "
            "subject_id, admittime, dischtime."
        )

    if lower >= upper:
        raise ValueError(
            "Q003 requires lower < upper."
        )

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "dischtime",
        ]
    ].copy()

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
            "Q003 encountered invalid admission or discharge times."
        )

    # Select each patient's earliest admission.
    patient_data = patient_data.sort_values(
        by=[
            "subject_id",
            "admittime",
        ]
    )

    earliest_admission = patient_data.drop_duplicates(
        subset="subject_id",
        keep="first",
    ).copy()

    # Length of stay expressed in days.
    earliest_admission["los_days"] = (
        earliest_admission["dischtime"]
        - earliest_admission["admittime"]
    ).dt.total_seconds() / 86400.0

    # Enforce the approved contribution bounds.
    earliest_admission["bounded_los_days"] = (
        earliest_admission["los_days"]
        .clip(
            lower=lower,
            upper=upper,
        )
    )

    bounded_sum = float(
        earliest_admission["bounded_los_days"].sum()
    )

    patient_count = int(
        len(earliest_admission)
    )

    return {
        "bounded_sum": bounded_sum,
        "count": patient_count,
        "lower": float(lower),
        "upper": float(upper),
    }
    
#-------------------------------------------------------------    
#-------------------------------------------------------------    

def race_admission_type_contingency(dataframe) -> dict:
    """
    Compute the patient-level race × admission-type contingency
    table for Q004.

    Each patient contributes to exactly one joint cell determined
    by the patient's earliest hospital admission.

    The output domain is fixed to the approved Cartesian product:

        33 race categories × 9 admission types = 297 cells.

    Cells not represented in a particular hospital are returned
    explicitly with count zero.

    Parameters
    ----------
    dataframe:
        Hospital admissions dataframe containing subject_id,
        admittime, race, and admission_type.

    Returns
    -------
    dict
        Nested mapping:
            race -> admission_type -> patient count
    """

    required_columns = {
        "subject_id",
        "admittime",
        "race",
        "admission_type",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Q004 requires columns: "
            "subject_id, admittime, race, admission_type."
        )

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "race",
            "admission_type",
        ]
    ].copy()

    patient_data["admittime"] = pd.to_datetime(
        patient_data["admittime"],
        errors="coerce",
    )

    if patient_data["admittime"].isna().any():
        raise ValueError(
            "Q004 encountered invalid admittime values."
        )

    patient_data = patient_data.sort_values(
        by=[
            "subject_id",
            "admittime",
        ]
    )

    earliest_admission = patient_data.drop_duplicates(
        subset="subject_id",
        keep="first",
    )

    if earliest_admission["race"].isna().any():
        raise ValueError(
            "Q004 encountered missing race values."
        )

    if earliest_admission["admission_type"].isna().any():
        raise ValueError(
            "Q004 encountered missing admission_type values."
        )

    observed_races = set(
        earliest_admission["race"].astype(str)
    )

    unexpected_races = (
        observed_races - set(Q004_RACE_CATEGORIES)
    )

    if unexpected_races:
        raise ValueError(
            "Q004 encountered race categories outside the "
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
            "Q004 encountered admission types outside the "
            f"approved domain: {sorted(unexpected_admission_types)}"
        )

    table = pd.crosstab(
        earliest_admission["race"],
        earliest_admission["admission_type"],
    )

    # Expand to the complete approved 33 × 9 domain.
    table = table.reindex(
        index=Q004_RACE_CATEGORIES,
        columns=Q004_ADMISSION_TYPES,
        fill_value=0,
    )

    return {
        race: {
            admission_type: int(
                table.loc[race, admission_type]
            )
            for admission_type in Q004_ADMISSION_TYPES
        }
        for race in Q004_RACE_CATEGORIES
    }
    
   #------------------------------------------------------------- 
   ########################bounded LOS###########################
   #-------------------------------------------------------------
def grouped_bounded_los_sum_and_count(
    dataframe,
    lower: float = 0.0,
    upper: float = 30.0,
) -> dict:
    """
    Compute patient-level bounded LOS sum and count grouped by race
    for Q005.

    Each patient contributes exactly once, using the patient's
    earliest hospital admission.

    LOS is measured in days and clipped to [lower, upper].

    The output uses the same fixed 33-category race domain as Q004.
    Race categories with no patients at a hospital are returned
    explicitly with bounded_sum=0 and count=0.
    """

    lower = float(lower)
    upper = float(upper)

    if lower >= upper:
        raise ValueError(
            "lower must be smaller than upper."
        )

    required_columns = {
        "subject_id",
        "admittime",
        "dischtime",
        "race",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Q005 requires columns: "
            "subject_id, admittime, dischtime, race."
        )

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "dischtime",
            "race",
        ]
    ].copy()

    patient_data["admittime"] = pd.to_datetime(
        patient_data["admittime"],
        errors="coerce",
    )

    patient_data["dischtime"] = pd.to_datetime(
        patient_data["dischtime"],
        errors="coerce",
    )

    if patient_data["admittime"].isna().any():
        raise ValueError(
            "Q005 encountered invalid admittime values."
        )

    if patient_data["dischtime"].isna().any():
        raise ValueError(
            "Q005 encountered invalid dischtime values."
        )

    patient_data = patient_data.sort_values(
        by=[
            "subject_id",
            "admittime",
        ]
    )

    earliest_admission = patient_data.drop_duplicates(
        subset="subject_id",
        keep="first",
    ).copy()

    if earliest_admission["race"].isna().any():
        raise ValueError(
            "Q005 encountered missing race values."
        )

    observed_races = set(
        earliest_admission["race"].astype(str)
    )

    unexpected_races = (
        observed_races
        - set(Q004_RACE_CATEGORIES)
    )

    if unexpected_races:
        raise ValueError(
            "Q005 encountered race categories outside the "
            f"approved domain: {sorted(unexpected_races)}"
        )

    #-------------------------------------------------------------
    # Calculate LOS in days
    # -------------------------------------------------------------

    earliest_admission["los_days"] = (
        (
            earliest_admission["dischtime"]
            - earliest_admission["admittime"]
        )
        .dt.total_seconds()
        / 86400.0
    )

    #-------------------------------------------------------------
    # Apply approved LOS bounds
    # -------------------------------------------------------------

    earliest_admission["bounded_los"] = (
        earliest_admission["los_days"].clip(
            lower=lower,
            upper=upper,
        )
    )
    # -------------------------------------------------------------
    # Group sufficient statistics by race
    # -------------------------------------------------------------

    grouped = (
        earliest_admission
        .groupby("race")["bounded_los"]
        .agg(["sum", "count"])
    )

    result = {}

    for race in Q004_RACE_CATEGORIES:

        if race in grouped.index:

            bounded_sum = float(
                grouped.loc[race, "sum"]
            )

            count = int(
                grouped.loc[race, "count"]
            )

        else:

            bounded_sum = 0.0
            count = 0

        result[race] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

    return {
        "groups": result,
        "lower": lower,
        "upper": upper,
    }
    
    
#-------------------------------------------------------------
##########################Q006################################
#-------------------------------------------------------------
def rare_subgroup_count(
    dataframe,
    race: str = Q006_RACE,
    admission_type: str = Q006_ADMISSION_TYPE,
) -> int:
    """
    Count patients whose earliest admission matches the
    approved Q006 rare-subgroup predicate.

    Q006 predicate:
        race = ASIAN - KOREAN
        admission_type = URGENT

    Each patient contributes at most one count because only
    the earliest admission per subject_id is considered.

    Parameters
    ----------
    dataframe:
        Hospital admissions dataframe.

    race:
        Approved race category.

    admission_type:
        Approved admission-type category.

    Returns
    -------
    int
        Number of patients satisfying the Q006 predicate.
    """

    required_columns = {
        "subject_id",
        "admittime",
        "race",
        "admission_type",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Q006 requires columns: "
            "subject_id, admittime, race, admission_type."
        )

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "race",
            "admission_type",
        ]
    ].copy()

    patient_data["admittime"] = pd.to_datetime(
        patient_data["admittime"],
        errors="coerce",
    )

    if patient_data["admittime"].isna().any():
        raise ValueError(
            "Q006 encountered invalid admittime values."
        )

    # One contribution per patient:
    # retain only the earliest admission.
    earliest_admission = (
        patient_data
        .sort_values(
            [
                "subject_id",
                "admittime",
            ]
        )
        .drop_duplicates(
            subset="subject_id",
            keep="first",
        )
    )

    if earliest_admission["race"].isna().any():
        raise ValueError(
            "Q006 encountered missing race values."
        )

    if earliest_admission["admission_type"].isna().any():
        raise ValueError(
            "Q006 encountered missing admission_type values."
        )

    matches = (
        (earliest_admission["race"] == race)
        &
        (
            earliest_admission["admission_type"]
            == admission_type
        )
    )

    return int(matches.sum())

#-------------------------------------------------------------
#############################Q007############################
#-------------------------------------------------------------


def rare_subgroup_bounded_los_sum_and_count(
    dataframe,
    lower: float = 0.0,
    upper: float = 30.0,
    race: str = Q006_RACE,
    admission_type: str = Q006_ADMISSION_TYPE,
) -> dict:
    """
    Compute local sufficient statistics for Q007.

    Q007 computes the bounded mean LOS for patients whose
    earliest admission satisfies the approved rare-subgroup
    predicate:

        race = ASIAN - KOREAN
        admission_type = URGENT

    Each matching patient contributes one LOS value from the
    patient's earliest admission. LOS is measured in days and
    clipped to [lower, upper].

    The hospital returns bounded sum and count rather than a
    local mean so that the coordinator can construct the correct
    weighted global mean.

    Returns
    -------
    dict
        Bounded LOS sum, subgroup patient count, bounds,
        and approved predicate.
    """

    required_columns = {
        "subject_id",
        "admittime",
        "dischtime",
        "race",
        "admission_type",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Q007 requires columns: "
            "subject_id, admittime, dischtime, "
            "race, admission_type."
        )

    if lower >= upper:
        raise ValueError(
            "Q007 requires lower < upper."
        )

    patient_data = dataframe[
        [
            "subject_id",
            "admittime",
            "dischtime",
            "race",
            "admission_type",
        ]
    ].copy()

    # Convert timestamps using the same logic as Q003.
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
            "Q007 encountered invalid admission or discharge times."
        )

    #-------------------------------------------------------------
    # Select each patient's earliest admission FIRST.
    # ---------------------------------------------------------

    patient_data = patient_data.sort_values(
        by=[
            "subject_id",
            "admittime",
        ]
    )

    earliest_admission = patient_data.drop_duplicates(
        subset="subject_id",
        keep="first",
    ).copy()

    #-------------------------------------------------------------
    # Apply the fixed Q007 subgroup predicate only after
    # selecting the earliest admission.
    # ---------------------------------------------------------

    subgroup = earliest_admission[
        (earliest_admission["race"] == race)
        &
        (
            earliest_admission["admission_type"]
            == admission_type
        )
    ].copy()

    # Length of stay expressed in days.
    subgroup["los_days"] = (
        subgroup["dischtime"]
        - subgroup["admittime"]
    ).dt.total_seconds() / 86400.0

    # Enforce approved contribution bounds.
    subgroup["bounded_los_days"] = (
        subgroup["los_days"]
        .clip(
            lower=lower,
            upper=upper,
        )
    )

    bounded_sum = float(
        subgroup["bounded_los_days"].sum()
    )

    patient_count = int(
        len(subgroup)
    )

    return {
        "bounded_sum": bounded_sum,
        "count": patient_count,
        "lower": float(lower),
        "upper": float(upper),
        "predicate": {
            "race": race,
            "admission_type": admission_type,
        },
    }