"""
Run the experimental Q008 query across the five existing
patient-disjoint hospital datasets.

This script:

1. Loads each hospital dataset independently.
2. Executes exact Q008 locally.
3. Validates Q008 against existing Q001/Q004 logic.
4. Aggregates the five local histograms.
5. Computes exact prolonged-stay rates.
6. Saves JSON and CSV results.

No DP, HE, services, registry, or privacy ledger are modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from node.queries import (
    Q004_RACE_CATEGORIES,
    Q004_ADMISSION_TYPES,
    unique_patient_count,
    race_admission_type_contingency,
)

from experiments_disclosure.disclosure_evaluation.q008_prolonged_stay.query import (
    prolonged_stay_histogram,
)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

RESULTS_DIR = (
    REPO_ROOT
    / "experiments_disclosure"
    / "disclosure_evaluation"
    / "q008_prolonged_stay"
    / "results"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HOSPITALS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5",
]


# ---------------------------------------------------------------------
# Empty global histogram
# ---------------------------------------------------------------------

global_histogram = {
    race: {
        admission_type: {
            "not_prolonged": 0,
            "prolonged": 0,
        }
        for admission_type in Q004_ADMISSION_TYPES
    }
    for race in Q004_RACE_CATEGORIES
}


hospital_summaries = []

total_unique_patients = 0


# ---------------------------------------------------------------------
# Process each hospital independently
# ---------------------------------------------------------------------

for hospital_id in HOSPITALS:

    data_file = (
        REPO_ROOT
        / "data"
        / hospital_id
        / "admissions.csv.gz"
    )

    print()
    print("=" * 70)
    print(f"Processing {hospital_id}")
    print(f"Data: {data_file}")
    print("=" * 70)

    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_file}"
        )

    dataframe = pd.read_csv(data_file)

    # -------------------------------------------------------------
    # Existing Q001 logic
    # -------------------------------------------------------------

    q001_count = unique_patient_count(dataframe)

    # -------------------------------------------------------------
    # Existing Q004 logic
    # -------------------------------------------------------------

    q004 = race_admission_type_contingency(dataframe)

    # -------------------------------------------------------------
    # Experimental Q008
    # -------------------------------------------------------------

    q008 = prolonged_stay_histogram(dataframe)

    q008_count = q008["patient_count"]

    # -------------------------------------------------------------
    # Validation 1:
    # Q001 unique patients must equal Q008 contributing patients.
    # -------------------------------------------------------------

    if q001_count != q008_count:
        raise RuntimeError(
            f"{hospital_id}: Q001/Q008 patient-count mismatch: "
            f"Q001={q001_count}, Q008={q008_count}"
        )

    # -------------------------------------------------------------
    # Validation 2:
    #
    # For every race × admission-type group:
    #
    # Q004(r,a)
    # =
    # Q008(r,a,not_prolonged)
    # +
    # Q008(r,a,prolonged)
    # -------------------------------------------------------------

    for race in Q004_RACE_CATEGORIES:

        for admission_type in Q004_ADMISSION_TYPES:

            q004_count = int(
                q004[race][admission_type]
            )

            q008_not = int(
                q008["histogram"][race][admission_type][
                    "not_prolonged"
                ]
            )

            q008_yes = int(
                q008["histogram"][race][admission_type][
                    "prolonged"
                ]
            )

            q008_cell_total = q008_not + q008_yes

            if q004_count != q008_cell_total:
                raise RuntimeError(
                    f"{hospital_id}: Q004/Q008 mismatch for "
                    f"{race} / {admission_type}: "
                    f"Q004={q004_count}, "
                    f"Q008={q008_cell_total}"
                )

            # Add local Q008 counts to global histogram.

            global_histogram[race][admission_type][
                "not_prolonged"
            ] += q008_not

            global_histogram[race][admission_type][
                "prolonged"
            ] += q008_yes

    # -------------------------------------------------------------
    # Hospital-level summary
    # -------------------------------------------------------------

    local_prolonged = sum(
        cell["prolonged"]
        for race_data in q008["histogram"].values()
        for cell in race_data.values()
    )

    local_not_prolonged = sum(
        cell["not_prolonged"]
        for race_data in q008["histogram"].values()
        for cell in race_data.values()
    )

    hospital_summaries.append(
        {
            "hospital_id": hospital_id,
            "patient_count": q008_count,
            "prolonged_count": local_prolonged,
            "not_prolonged_count": local_not_prolonged,
            "prolonged_rate": (
                local_prolonged / q008_count
                if q008_count > 0
                else None
            ),
        }
    )

    total_unique_patients += q008_count

    print(f"Patients:        {q008_count}")
    print(f"Prolonged:       {local_prolonged}")
    print(f"Not prolonged:   {local_not_prolonged}")
    print("Q001 validation: PASS")
    print("Q004 validation: PASS")


# ---------------------------------------------------------------------
# Global rates
# ---------------------------------------------------------------------

global_rates = {}

csv_rows = []

global_prolonged = 0
global_not_prolonged = 0


for race in Q004_RACE_CATEGORIES:

    global_rates[race] = {}

    for admission_type in Q004_ADMISSION_TYPES:

        not_prolonged = int(
            global_histogram[race][admission_type][
                "not_prolonged"
            ]
        )

        prolonged = int(
            global_histogram[race][admission_type][
                "prolonged"
            ]
        )

        total = not_prolonged + prolonged

        rate = (
            prolonged / total
            if total > 0
            else None
        )

        global_rates[race][admission_type] = {
            "total": total,
            "prolonged": prolonged,
            "not_prolonged": not_prolonged,
            "prolonged_rate": rate,
        }

        global_prolonged += prolonged
        global_not_prolonged += not_prolonged

        csv_rows.append(
            {
                "race": race,
                "admission_type": admission_type,
                "not_prolonged": not_prolonged,
                "prolonged": prolonged,
                "total": total,
                "prolonged_rate": rate,
            }
        )


# ---------------------------------------------------------------------
# Final validation
# ---------------------------------------------------------------------

global_histogram_total = (
    global_prolonged + global_not_prolonged
)

if global_histogram_total != total_unique_patients:
    raise RuntimeError(
        "Global Q008 histogram does not equal total patient count: "
        f"histogram={global_histogram_total}, "
        f"patients={total_unique_patients}"
    )


# ---------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------

output = {
    "query_id": "Q008",
    "description": (
        "Race × admission-type prolonged-stay rate"
    ),
    "experimental": True,
    "privacy_unit": "subject_id",
    "contribution_rule": "earliest_admission",
    "los_bounds_days": [0.0, 30.0],
    "prolonged_definition": "bounded_los_days > 14.0",
    "race_categories": len(Q004_RACE_CATEGORIES),
    "admission_types": len(Q004_ADMISSION_TYPES),
    "histogram_cells": (
        len(Q004_RACE_CATEGORIES)
        * len(Q004_ADMISSION_TYPES)
        * 2
    ),
    "hospital_count": len(HOSPITALS),
    "patient_count": total_unique_patients,
    "prolonged_count": global_prolonged,
    "not_prolonged_count": global_not_prolonged,
    "overall_prolonged_rate": (
        global_prolonged / total_unique_patients
        if total_unique_patients > 0
        else None
    ),
    "hospital_summaries": hospital_summaries,
    "histogram": global_histogram,
    "rates": global_rates,
}

json_path = RESULTS_DIR / "q008_exact_results.json"

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(
        output,
        file,
        indent=2,
    )


# ---------------------------------------------------------------------
# Save flattened CSV
# ---------------------------------------------------------------------

csv_path = RESULTS_DIR / "q008_exact_rates.csv"

pd.DataFrame(csv_rows).to_csv(
    csv_path,
    index=False,
)


# ---------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------

nonempty_groups = sum(
    1
    for row in csv_rows
    if row["total"] > 0
)

print()
print("=" * 70)
print("Q008 EXACT GLOBAL RESULT")
print("=" * 70)

print(f"Hospitals:             {len(HOSPITALS)}")
print(f"Patients:              {total_unique_patients}")
print(f"Histogram cells:       {len(Q004_RACE_CATEGORIES) * len(Q004_ADMISSION_TYPES) * 2}")
print(f"Race×admission groups: {len(Q004_RACE_CATEGORIES) * len(Q004_ADMISSION_TYPES)}")
print(f"Non-empty groups:      {nonempty_groups}")
print(f"Prolonged patients:    {global_prolonged}")
print(f"Not prolonged:         {global_not_prolonged}")

if total_unique_patients > 0:
    print(
        f"Overall prolonged rate: "
        f"{100 * global_prolonged / total_unique_patients:.4f}%"
    )

print()
print("VALIDATION")
print("Q001 patient-count consistency: PASS")
print("Q004 cell-refinement consistency: PASS")
print("Global histogram consistency: PASS")

print()
print(f"JSON: {json_path}")
print(f"CSV:  {csv_path}")