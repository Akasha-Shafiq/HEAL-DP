from pathlib import Path
import math

import pandas as pd

from node.queries import (
    rare_subgroup_count,
    rare_subgroup_bounded_los_sum_and_count,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENARIO_ROOT = (
    PROJECT_ROOT
    / "data"
    / "experiments"
    / "heterogeneity"
)

HOSPITAL_IDS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
    "hospital_5",
]

SCENARIOS = [
    "h5_retention_100",
    "h5_retention_75",
    "h5_retention_50",
    "h5_retention_25",
    "h5_retention_0",
]

EXPECTED_H5_COUNTS = {
    "h5_retention_100": 14,
    "h5_retention_75": 11,
    "h5_retention_50": 7,
    "h5_retention_25": 4,
    "h5_retention_0": 0,
}

EXPECTED_GLOBAL_PATIENTS = 223_452
EXPECTED_GLOBAL_ADMISSIONS = 546_028

EXPECTED_Q006_COUNT = 38

EXPECTED_Q007_COUNT = 38
EXPECTED_Q007_SUM = 174.42708333333331
EXPECTED_Q007_MEAN = 4.590186403508771

FLOAT_TOLERANCE = 1e-9



# Helpers
# ---------------------------------------------------------------------

def load_scenario(
    scenario_name: str,
) -> dict[str, pd.DataFrame]:

    scenario_data = {}

    for hospital_id in HOSPITAL_IDS:

        path = (
            SCENARIO_ROOT
            / scenario_name
            / hospital_id
            / "admissions.csv.gz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing scenario file: {path}"
            )

        scenario_data[hospital_id] = (
            pd.read_csv(path)
        )

    return scenario_data


def verify_patient_disjointness(
    hospital_data: dict[str, pd.DataFrame],
) -> None:

    owners = {}

    for hospital_id, df in hospital_data.items():

        subject_ids = (
            df["subject_id"]
            .astype(int)
            .unique()
        )

        for subject_id in subject_ids:

            if subject_id in owners:
                raise AssertionError(
                    f"Patient {subject_id} appears in "
                    f"{owners[subject_id]} and "
                    f"{hospital_id}."
                )

            owners[subject_id] = hospital_id


def validate_scenario(
    scenario_name: str,
) -> dict:

    print()
    print("=" * 72)
    print(f"VALIDATING: {scenario_name}")
    print("=" * 72)

    hospital_data = load_scenario(
        scenario_name
    )

    
    # 1. Global dataset size
    # ---------------------------------------------------------

    total_admissions = sum(
        len(df)
        for df in hospital_data.values()
    )

    global_subject_ids = set()

    for df in hospital_data.values():

        global_subject_ids.update(
            df["subject_id"]
            .astype(int)
            .unique()
        )

    total_patients = len(
        global_subject_ids
    )

    assert (
        total_admissions
        == EXPECTED_GLOBAL_ADMISSIONS
    ), (
        f"{scenario_name}: expected "
        f"{EXPECTED_GLOBAL_ADMISSIONS} admissions, "
        f"got {total_admissions}."
    )

    assert (
        total_patients
        == EXPECTED_GLOBAL_PATIENTS
    ), (
        f"{scenario_name}: expected "
        f"{EXPECTED_GLOBAL_PATIENTS} patients, "
        f"got {total_patients}."
    )

    
    # 2. Patient disjointness
    # ---------------------------------------------------------

    verify_patient_disjointness(
        hospital_data
    )
    
    # 3. Q006 — rare subgroup count
    # ---------------------------------------------------------

    q006_local = {}

    for hospital_id in HOSPITAL_IDS:

        q006_local[hospital_id] = (
            rare_subgroup_count(
                hospital_data[hospital_id]
            )
        )

    q006_global = sum(
        q006_local.values()
    )

    assert (
        q006_global
        == EXPECTED_Q006_COUNT
    ), (
        f"{scenario_name}: Q006 global count "
        f"expected {EXPECTED_Q006_COUNT}, "
        f"got {q006_global}."
    )

    expected_h5 = EXPECTED_H5_COUNTS[
        scenario_name
    ]

    assert (
        q006_local["hospital_5"]
        == expected_h5
    ), (
        f"{scenario_name}: H5 Q006 expected "
        f"{expected_h5}, got "
        f"{q006_local['hospital_5']}."
    )

    
    # 4. Q007 — rare subgroup bounded mean LOS
    # ---------------------------------------------------------

    q007_local = {}

    global_bounded_sum = 0.0
    global_count = 0

    for hospital_id in HOSPITAL_IDS:

        result = (
            rare_subgroup_bounded_los_sum_and_count(
                hospital_data[hospital_id],
                lower=0.0,
                upper=30.0,
            )
        )

        q007_local[hospital_id] = {
            "bounded_sum": result["bounded_sum"],
            "count": result["count"],
        }

        global_bounded_sum += (
            result["bounded_sum"]
        )

        global_count += (
            result["count"]
        )

    assert (
        global_count
        == EXPECTED_Q007_COUNT
    ), (
        f"{scenario_name}: Q007 count expected "
        f"{EXPECTED_Q007_COUNT}, "
        f"got {global_count}."
    )

    if not math.isclose(
        global_bounded_sum,
        EXPECTED_Q007_SUM,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise AssertionError(
            f"{scenario_name}: Q007 sum expected "
            f"{EXPECTED_Q007_SUM}, "
            f"got {global_bounded_sum}."
        )

    global_mean = (
        global_bounded_sum
        / global_count
    )

    if not math.isclose(
        global_mean,
        EXPECTED_Q007_MEAN,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise AssertionError(
            f"{scenario_name}: Q007 mean expected "
            f"{EXPECTED_Q007_MEAN}, "
            f"got {global_mean}."
        )


    # 5. Print local distribution
    # ---------------------------------------------------------

    print("Q006 local target counts:")

    for hospital_id in HOSPITAL_IDS:

        print(
            f"  {hospital_id}: "
            f"{q006_local[hospital_id]}"
        )

    print()
    print("Q007 local sufficient statistics:")

    for hospital_id in HOSPITAL_IDS:

        local = q007_local[hospital_id]

        local_mean = (
            local["bounded_sum"]
            / local["count"]
            if local["count"] > 0
            else None
        )

        print(
            f"  {hospital_id}: "
            f"count={local['count']}, "
            f"sum={local['bounded_sum']}, "
            f"mean={local_mean}"
        )

    print()
    print(
        f"Global patients: {total_patients:,}"
    )

    print(
        f"Global admissions: {total_admissions:,}"
    )

    print(
        f"Global Q006 count: {q006_global}"
    )

    print(
        f"Global Q007 count: {global_count}"
    )

    print(
        f"Global Q007 sum: {global_bounded_sum}"
    )

    print(
        f"Global Q007 mean: {global_mean}"
    )

    print(
        "Patient disjointness: PASSED"
    )

    print(
        "Global invariance: PASSED"
    )

    return {
        "scenario": scenario_name,
        "h5_target_count": (
            q006_local["hospital_5"]
        ),
        "q006_global_count": q006_global,
        "q007_global_count": global_count,
        "q007_global_sum": global_bounded_sum,
        "q007_global_mean": global_mean,
        "global_patients": total_patients,
        "global_admissions": total_admissions,
        "patient_disjoint": True,
        "invariance_passed": True,
    }



# Main
# ---------------------------------------------------------------------

def main() -> None:

    summaries = []

    for scenario_name in SCENARIOS:

        summaries.append(
            validate_scenario(
                scenario_name
            )
        )

    summary_df = pd.DataFrame(
        summaries
    )

    output_path = (
        SCENARIO_ROOT
        / "validation_summary.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("ALL HETEROGENEITY SCENARIOS VALIDATED")
    print("=" * 72)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Validation summary saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()