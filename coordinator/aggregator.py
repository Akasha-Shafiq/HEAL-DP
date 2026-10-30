""" 
Distributed aggregation for the MIMIC-DP coordinator 

This module contacts the participating hospital nodes,
collects approved local statistics, and combines them into a global exact aggregate

Dp is not applied here
"""

from collections import defaultdict
import os
import httpx

# -------------------------------------------------------------
# Hospital node configuration
# -------------------------------------------------------------
# Node addresses can be provided through environment variables.

HOSPITAL_NODES = {
    f"hospital_{i}": os.getenv(
        f"HOSPITAL_{i}_URL",
        f"http://127.0.0.1:{8000 + i}",
    )
    for i in range(1, 6)
}


def aggregate_q001() -> dict:
    """
    Execute Q001 across all hospital nodes and aggregate the
    local unique-patient counts.

    Returns
    -------
    dict
        Local hospital results and the global exact count.
    """

    local_results = {}

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q001",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()

        local_results[hospital] = int(
            payload["value"]
        )

    global_count = sum(
        local_results.values()
    )

    return {
        "query_id": "Q001",
        "local_results": local_results,
        "global_result": global_count,
    }
    
def aggregate_q002() -> dict:
    """
    Execute Q002 across all hospital nodes and aggregate the
    local admission-type histograms element-wise.

    Returns
    -------
    dict
        Local hospital histograms and the exact global histogram.
    """

    local_results = {}
    global_histogram = defaultdict(int)

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q002",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()

        histogram = payload["value"]

        local_results[hospital] = histogram

        for category, count in histogram.items():
            global_histogram[category] += int(count)

    # Sort categories to give deterministic output.
    global_histogram = dict(
        sorted(global_histogram.items())
    )

    return {
        "query_id": "Q002",
        "local_results": local_results,
        "global_result": global_histogram,
    }
    
def aggregate_q003() -> dict:
    """
    Execute Q003 across all hospital nodes and aggregate the
    bounded LOS sufficient statistics.

    Each hospital returns:
        - bounded_sum
        - count

    The coordinator sums these values and computes the exact
    global bounded mean.

    Returns
    -------
    dict
        Local sufficient statistics and global Q003 statistics.
    """

    local_results = {}

    global_bounded_sum = 0.0
    global_count = 0

    expected_lower = 0.0
    expected_upper = 30.0

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q003",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()
        statistics = payload["value"]

        # Verify that every hospital used the approved bounds.
        if (
            float(statistics["lower"]) != expected_lower
            or float(statistics["upper"]) != expected_upper
        ):
            raise ValueError(
                f"{hospital} returned unexpected Q003 bounds."
            )

        bounded_sum = float(
            statistics["bounded_sum"]
        )

        count = int(
            statistics["count"]
        )

        local_results[hospital] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

        global_bounded_sum += bounded_sum
        global_count += count

    if global_count <= 0:
        raise ValueError(
            "Q003 global patient count must be positive."
        )

    global_mean = (
        global_bounded_sum / global_count
    )

    return {
        "query_id": "Q003",
        "local_results": local_results,
        "global_result": {
            "bounded_sum": global_bounded_sum,
            "count": global_count,
            "bounded_mean": global_mean,
            "lower": expected_lower,
            "upper": expected_upper,
        },
    }
    
    

def aggregate_q004() -> dict:
    """
    Execute Q004 across all hospital nodes and aggregate the
    patient-level race × admission-type contingency tables.

    Each hospital returns the same fixed 33 × 9 joint domain.
    The coordinator performs element-wise summation of all cells.

    Returns
    -------
    dict
        Local contingency tables and the exact global
        contingency table.
    """

    local_results = {}
    global_table = None

    expected_races = None
    expected_admission_types = None

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q004",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()
        table = payload["value"]

        # --------------------------------------------------------
        # Validate and establish the fixed output domain
        # ---------------------------------------------------------

        races = tuple(table.keys())

        if not races:
            raise ValueError(
                f"{hospital} returned an empty Q004 race domain."
            )

        first_race = races[0]

        admission_types = tuple(
            table[first_race].keys()
        )

        if expected_races is None:
            expected_races = races
            expected_admission_types = admission_types

            global_table = {
                race: {
                    admission_type: 0
                    for admission_type
                    in expected_admission_types
                }
                for race in expected_races
            }

        else:
            if races != expected_races:
                raise ValueError(
                    f"{hospital} returned a Q004 race domain "
                    "that does not match the approved domain."
                )

            if admission_types != expected_admission_types:
                raise ValueError(
                    f"{hospital} returned a Q004 admission-type "
                    "domain that does not match the approved domain."
                )

        # ---------------------------------------------------------
        # Validate every row and aggregate element-wise
        # ---------------------------------------------------------

        for race in expected_races:

            row_admission_types = tuple(
                table[race].keys()
            )

            if row_admission_types != expected_admission_types:
                raise ValueError(
                    f"{hospital} returned an inconsistent "
                    f"Q004 domain for race '{race}'."
                )

            for admission_type in expected_admission_types:

                count = int(
                    table[race][admission_type]
                )

                if count < 0:
                    raise ValueError(
                        f"{hospital} returned a negative Q004 count."
                    )

                global_table[race][admission_type] += count

        local_results[hospital] = table

    global_total = sum(
        sum(row.values())
        for row in global_table.values()
    )

    return {
        "query_id": "Q004",
        "local_results": local_results,
        "global_result": {
            "contingency_table": global_table,
            "total": global_total,
            "race_categories": len(expected_races),
            "admission_type_categories": len(
                expected_admission_types
            ),
            "cells": (
                len(expected_races)
                * len(expected_admission_types)
            ),
        },
    }
    
# ---------------------------------------------------------------------   
# ---------------------------------------------------------------------
def aggregate_q005() -> dict:
    """
    Execute Q005 across all hospital nodes and aggregate
    patient-level bounded LOS sufficient statistics by race.

    Each hospital returns, for every fixed race category:
        - bounded_sum
        - count

    The coordinator sums these sufficient statistics
    element-wise and computes the exact bounded mean LOS
    for each race.

    Returns
    -------
    dict
        Local grouped statistics and exact global grouped
        statistics.
    """

    local_results = {}

    global_groups = None
    expected_races = None

    expected_lower = 0.0
    expected_upper = 30.0

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q005",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()
        value = payload["value"]

        groups = value["groups"]
        lower = float(value["lower"])
        upper = float(value["upper"])

        # ---------------------------------------------------------
        # Verify that all hospitals used the approved LOS bounds
        # ---------------------------------------------------------

        if (
            lower != expected_lower
            or upper != expected_upper
        ):
            raise ValueError(
                f"{hospital} returned unexpected Q005 bounds."
            )

        races = tuple(groups.keys())

        if not races:
            raise ValueError(
                f"{hospital} returned an empty Q005 race domain."
            )

        # ---------------------------------------------------------
        # Establish and validate fixed race domain
        # ---------------------------------------------------------

        if expected_races is None:

            expected_races = races

            global_groups = {
                race: {
                    "bounded_sum": 0.0,
                    "count": 0,
                }
                for race in expected_races
            }

        elif races != expected_races:

            raise ValueError(
                f"{hospital} returned a Q005 race domain "
                "that does not match the approved domain."
            )

        # ---------------------------------------------------------
        # Aggregate sum and count for each race
        # ---------------------------------------------------------

        for race in expected_races:

            bounded_sum = float(
                groups[race]["bounded_sum"]
            )

            count = int(
                groups[race]["count"]
            )

            if count < 0:
                raise ValueError(
                    f"{hospital} returned a negative Q005 count."
                )

            global_groups[race]["bounded_sum"] += (
                bounded_sum
            )

            global_groups[race]["count"] += count

        local_results[hospital] = groups

    # -------------------------------------------------------------
    # Compute exact global bounded mean for each race
    # -------------------------------------------------------------

    for race in expected_races:

        bounded_sum = global_groups[race]["bounded_sum"]
        count = global_groups[race]["count"]

        if count > 0:
            bounded_mean = bounded_sum / count
        else:
            bounded_mean = None

        global_groups[race]["bounded_mean"] = (
            bounded_mean
        )

    # -------------------------------------------------------------
    # Global consistency statistics
    # -------------------------------------------------------------

    global_total_count = sum(
        statistics["count"]
        for statistics in global_groups.values()
    )

    global_total_sum = sum(
        statistics["bounded_sum"]
        for statistics in global_groups.values()
    )

    return {
        "query_id": "Q005",
        "local_results": local_results,
        "global_result": {
            "groups": global_groups,
            "race_categories": len(expected_races),
            "total_count": global_total_count,
            "total_bounded_sum": global_total_sum,
            "lower": expected_lower,
            "upper": expected_upper,
        },
    }
    
# -------------------------------------------------------------
# -------------------------------------------------------------
def aggregate_q006() -> dict:
    """
    Execute Q006 across all hospital nodes and aggregate
    the rare subgroup patient counts.

    Q006 counts patients whose earliest admission has:
        race = ASIAN - KOREAN
        admission_type = URGENT

    Each hospital returns one local integer count.
    The coordinator sums the five local counts.

    Returns
    -------
    dict
        Local hospital counts and exact global Q006 count.
    """

    local_results = {}
    global_count = 0

    expected_predicate = {
        "race": "ASIAN - KOREAN",
        "admission_type": "URGENT",
    }

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q006",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()

        # -----------------------------------------------------
        # Validate query identity
        # -----------------------------------------------------

        if payload["query_id"] != "Q006":
            raise ValueError(
                f"{hospital} returned an unexpected query ID."
            )

        # -----------------------------------------------------
        # Verify all hospitals executed the same predicate
        # -----------------------------------------------------

        predicate = payload["predicate"]

        if predicate != expected_predicate:
            raise ValueError(
                f"{hospital} returned an unexpected Q006 predicate."
            )

        # -----------------------------------------------------
        # Validate and aggregate local count
        # -----------------------------------------------------

        count = int(
            payload["value"]
        )

        if count < 0:
            raise ValueError(
                f"{hospital} returned a negative Q006 count."
            )

        local_results[hospital] = count

        global_count += count

    return {
        "query_id": "Q006",
        "predicate": expected_predicate,
        "local_results": local_results,
        "global_result": {
            "count": global_count,
        },
    }    
    
    
    
# -------------------------------------------------------------    
########################q007###############################
# -------------------------------------------------------------
def aggregate_q007() -> dict:
    """
    Execute Q007 across all hospital nodes and aggregate the
    rare-subgroup bounded LOS sufficient statistics.

    Each hospital returns:
        - bounded_sum
        - count
        - lower
        - upper
        - predicate

    The coordinator sums the bounded sums and counts and then
    computes the correct weighted global bounded mean.

    Q007 predicate:
        race = ASIAN - KOREAN
        admission_type = URGENT
    """

    local_results = {}

    global_bounded_sum = 0.0
    global_count = 0

    expected_lower = 0.0
    expected_upper = 30.0

    expected_predicate = {
        "race": "ASIAN - KOREAN",
        "admission_type": "URGENT",
    }

    for hospital, base_url in HOSPITAL_NODES.items():

        response = httpx.get(
            f"{base_url}/query/q007",
            timeout=10.0,
        )

        response.raise_for_status()

        payload = response.json()

        # -----------------------------------------------------
        # Verify query identity
        # -----------------------------------------------------

        if payload["query_id"] != "Q007":
            raise ValueError(
                f"{hospital} returned an unexpected query ID for Q007."
            )

        statistics = payload["value"]

        # -----------------------------------------------------
        # Verify approved LOS bounds
        # -----------------------------------------------------

        if (
            float(statistics["lower"]) != expected_lower
            or float(statistics["upper"]) != expected_upper
        ):
            raise ValueError(
                f"{hospital} returned unexpected Q007 bounds."
            )

        # -----------------------------------------------------
        # Verify approved rare-subgroup predicate
        # -----------------------------------------------------

        if statistics["predicate"] != expected_predicate:
            raise ValueError(
                f"{hospital} returned an unexpected Q007 predicate."
            )

        # -----------------------------------------------------
        # Read local sufficient statistics
        # -----------------------------------------------------

        bounded_sum = float(
            statistics["bounded_sum"]
        )

        count = int(
            statistics["count"]
        )

        if count < 0:
            raise ValueError(
                f"{hospital} returned a negative Q007 count."
            )

        local_results[hospital] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

        # -----------------------------------------------------
        # Aggregate sufficient statistics
        # -----------------------------------------------------

        global_bounded_sum += bounded_sum
        global_count += count

    # ---------------------------------------------------------
    # Ensure the global subgroup is non-empty
    # ---------------------------------------------------------

    if global_count <= 0:
        raise ValueError(
            "Q007 global subgroup count must be positive."
        )

    # ---------------------------------------------------------
    # Correct weighted global mean
    # ---------------------------------------------------------

    global_mean = (
        global_bounded_sum / global_count
    )

    return {
        "query_id": "Q007",
        "predicate": expected_predicate,
        "local_results": local_results,
        "global_result": {
            "bounded_sum": global_bounded_sum,
            "count": global_count,
            "bounded_mean": global_mean,
            "lower": expected_lower,
            "upper": expected_upper,
        },
    }