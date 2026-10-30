""" 
API service for the MIMIC Differential Privacy hospital node 

Each hospital node has access only to its own local data partititon 
and executes approved local  aggrgate queries

The node does not:
    - aggregate statistics across hospitals,
    - manage the global privacy budget,
    - apply central differential privacy, or
    - maintain the global audit log.

"""


import os 

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from node.queries import (
    admission_type_histogram,
    bounded_los_sum_and_count,
    grouped_bounded_los_sum_and_count,
    race_admission_type_contingency,
    rare_subgroup_count,
    unique_patient_count,
    rare_subgroup_bounded_los_sum_and_count,
)

#-------------------------------------------------------------
################## Node configuration ########################
#-------------------------------------------------------------

HOSPITAL_ID = os.getenv("HOSPITAL_ID", "hospital_1")

DATA_FILE = Path(
    os.getenv(
        "DATA_FILE",
        "data/hospital_1/admissions.csv.gz",
    )
)

#-------------------------------------------------------------
##################Validate and load local data###############
#-------------------------------------------------------------


if not DATA_FILE.exists():
    raise RuntimeError(
        f"Hospital dataset not found: {DATA_FILE}"
    )
    
admissions_df = pd.read_csv(DATA_FILE)

#-------------------------------------------------------------
##################Fast API application ###############
#-------------------------------------------------------------

app = FastAPI(
    title="MIMIC-DP Hospital Node",
    description= (
        "Local aggregate-query serivice for a simulated"
        "MIMIC-IV hospital "
    ),
    version="0.1.0",
)
#-------------------------------------------------------------
##################Fast API application ###############
#-------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """ 
    Return serivce staus without exposing patient-derived statistics 
    """
    
    return{
        "status": "ok",
        "hospital": HOSPITAL_ID,
    }
#-------------------------------------------------------------    
##################Q001- Unique patient count###############
#-------------------------------------------------------------

@app.get("/query/q001")
def query_q001() -> dict:
    """
    Execute approved local query Q001.

    Returns the exact local unique-patient count for the current
    non-private distributed baseline.
    """

    try:
        value = unique_patient_count(
            admissions_df
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q001 execution failed.",
        ) from exc

    return {
        "query_id": "Q001",
        "hospital": HOSPITAL_ID,
        "statistic": "unique_patient_count",
        "value": value,
    }
    
@app.get("/query/q002")
def query_q002() -> dict:
    """
    Execute local Q002 admission-type histogram.
    """

    try:
        histogram = admission_type_histogram(
            admissions_df
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q002 execution failed.",
        ) from exc

    return {
        "query_id": "Q002",
        "hospital": HOSPITAL_ID,
        "statistic": "admission_type_histogram",
        "value": histogram,
    }
    
@app.get("/query/q003")
def query_q003() -> dict:
    """
    Execute local Q003 bounded-LOS sufficient statistics.
    """

    try:
        statistics = bounded_los_sum_and_count(
            admissions_df,
            lower=0.0,
            upper=30.0,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q003 execution failed.",
        ) from exc

    return {
        "query_id": "Q003",
        "hospital": HOSPITAL_ID,
        "statistic": "bounded_los_sum_and_count",
        "value": statistics,
    }
    
    
@app.get("/query/q004")
def query_q004() -> dict:
    """
    Execute local Q004 race × admission-type contingency table.
    """

    try:
        contingency_table = race_admission_type_contingency(
            admissions_df
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q004 execution failed.",
        ) from exc

    return {
        "query_id": "Q004",
        "hospital": HOSPITAL_ID,
        "statistic": "race_admission_type_contingency",
        "value": contingency_table,
    }
    
    
@app.get("/query/q005")
def query_q005() -> dict:
    """
    Execute local Q005 bounded mean LOS by race sufficient statistics.
    """

    try:
        statistics = grouped_bounded_los_sum_and_count(
            admissions_df,
            lower=0.0,
            upper=30.0,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q005 execution failed.",
        ) from exc

    return {
        "query_id": "Q005",
        "hospital": HOSPITAL_ID,
        "statistic": "grouped_bounded_los_sum_and_count",
        "value": statistics,
    }
    
    
    
    
@app.get("/query/q006")
def query_q006() -> dict:
    """
    Execute local Q006 rare subgroup count.

    Counts patients whose earliest admission has:
        race = ASIAN - KOREAN
        admission_type = URGENT
    """

    try:
        count = rare_subgroup_count(
            admissions_df
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q006 execution failed.",
        ) from exc

    return {
        "query_id": "Q006",
        "hospital": HOSPITAL_ID,
        "statistic": "rare_subgroup_count",
        "predicate": {
            "race": "ASIAN - KOREAN",
            "admission_type": "URGENT",
        },
        "value": count,
    }
#-------------------------------------------------------------    
#############Q007 rare subgroup los sum and count#############
#-------------------------------------------------------------

@app.get("/query/q007")
def query_q007() -> dict:
    """
    Execute local Q007 rare-subgroup bounded-LOS
    sufficient statistics.
    """

    try:
        statistics = rare_subgroup_bounded_los_sum_and_count(
            admissions_df,
            lower=0.0,
            upper=30.0,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Q007 execution failed.",
        ) from exc

    return {
        "query_id": "Q007",
        "hospital": HOSPITAL_ID,
        "statistic": "rare_subgroup_bounded_los_sum_and_count",
        "value": statistics,
    }