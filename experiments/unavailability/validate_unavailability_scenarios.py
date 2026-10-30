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
    / "unavailability"
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

EXPECTED = {
    "h5_retention_100": {
        "h5_target_count": 14,
        "global_target_count": 38,
        "global_patients": 223452,
        "global_admissions": 546028,
    },
    "h5_retention_75": {
        "h5_target_count": 11,
        "global_target_count": 35,
        "global_patients": 223449,
        "global_admissions": 546025,
    },
    "h5_retention_50": {
        "h5_target_count": 7,
        "global_target_count": 31,
        "global_patients": 223445,
        "global_admissions": 546011,
    },
    "h5_retention_25": {
        "h5_target_count": 4,
        "global_target_count": 28,
        "global_patients": 223442,
        "global_admissions": 546006,
    },
    "h5_retention_0": {
        "h5_target_count": 0,
        "global_target_count": 24,
        "global_patients": 223438,
        "global_admissions": 545999,
    },
}

FLOAT_TOLERANCE = 1e-9


# ---------------------------------------------------------------------
# Load scenario
# ---------------------------------------------------------------------

def load_scenario(
    scenario_name: str,
) -> dict[str, pd.DataFrame]:

    data = {}

    for hospital_id in HOSPITAL_IDS:

        path = (
            SCENARIO_ROOT
            / scenario_name
            / hospital_id
            / "admissions.csv.gz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

        data[hospital_id] = pd.read_csv(path)

    return data


# ---------------------------------------------------------------------
# Patient-disjointness
# ---------------------------------------------------------------------

def verify_patient_disjointness(
    hospital_data: dict[str, pd.DataFrame],
) -> None:

    owners = {}

    for hospital_id, df in hospital_data.items():

        for subject_id in (
            df["subject_id"]
            .astype(int)
            .unique()
        ):

            if subject_id in owners:
                raise AssertionError(
                    f"Patient {subject_id} appears in "
                    f"{owners[subject_id]} and "
                    f"{hospital_id}."
                )

            owners[subject_id] = hospital_id


# ---------------------------------------------------------------------
# Validate one scenario
# ---------------------------------------------------------------------

def validate_scenario(
    scenario_name: str,
    baseline_q007_local: dict | None,
) -> tuple[dict, dict]:

    print()
    print("=" * 72)
    print(f"VALIDATING: {scenario_name}")
    print("=" * 72)

    hospital_data = load_scenario(
        scenario_name
    )

    expected = EXPECTED[
        scenario_name
    ]

    # -------------------------------------------------------------
    # 1. Global population
    # -------------------------------------------------------------

    global_admissions = sum(
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

    global_patients = len(
        global_subject_ids
    )

    assert (
        global_patients
        == expected["global_patients"]
    ), (
        f"{scenario_name}: patient count mismatch. "
        f"Expected {expected['global_patients']}, "
        f"got {global_patients}."
    )

    assert (
        global_admissions
        == expected["global_admissions"]
    ), (
        f"{scenario_name}: admission count mismatch. "
        f"Expected {expected['global_admissions']}, "
        f"got {global_admissions}."
    )

    verify_patient_disjointness(
        hospital_data
    )

    # -------------------------------------------------------------
    # 2. Q006 exact count
    # -------------------------------------------------------------

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
        == expected["global_target_count"]
    ), (
        f"{scenario_name}: expected global Q006 "
        f"{expected['global_target_count']}, "
        f"got {q006_global}."
    )

    assert (
        q006_local["hospital_5"]
        == expected["h5_target_count"]
    ), (
        f"{scenario_name}: expected H5 Q006 "
        f"{expected['h5_target_count']}, "
        f"got {q006_local['hospital_5']}."
    )

    
    # 3. Q007 exact bounded sum/count
    # -------------------------------------------------------------

    q007_local = {}

    global_sum = 0.0
    global_count = 0

    for hospital_id in HOSPITAL_IDS:

        result = (
            rare_subgroup_bounded_los_sum_and_count(
                hospital_data[hospital_id],
                lower=0.0,
                upper=30.0,
            )
        )

        bounded_sum = float(
            result["bounded_sum"]
        )

        count = int(
            result["count"]
        )

        q007_local[hospital_id] = {
            "bounded_sum": bounded_sum,
            "count": count,
        }

        global_sum += bounded_sum
        global_count += count

    # Q006 and Q007 use the same subgroup predicate,
    # so their patient counts must agree.
    assert (
        global_count
        == q006_global
    ), (
        f"{scenario_name}: Q007 count "
        f"{global_count} does not match "
        f"Q006 count {q006_global}."
    )

    if global_count <= 0:
        raise AssertionError(
            f"{scenario_name}: Q007 global "
            "count must remain positive."
        )

    global_mean = (
        global_sum
        / global_count
    )

    
    # 4. H1-H4 must remain unchanged.
    #
    # Only H5 loses data in this experiment.
    # -------------------------------------------------------------

    if baseline_q007_local is not None:

        for hospital_id in [
            "hospital_1",
            "hospital_2",
            "hospital_3",
            "hospital_4",
        ]:

            current = q007_local[
                hospital_id
            ]

            baseline = baseline_q007_local[
                hospital_id
            ]

            assert (
                current["count"]
                == baseline["count"]
            ), (
                f"{scenario_name}: "
                f"{hospital_id} Q007 count changed."
            )

            if not math.isclose(
                current["bounded_sum"],
                baseline["bounded_sum"],
                rel_tol=0.0,
                abs_tol=FLOAT_TOLERANCE,
            ):
                raise AssertionError(
                    f"{scenario_name}: "
                    f"{hospital_id} Q007 sum changed."
                )


    # Print
    # -------------------------------------------------------------

    print("Q006 local counts:")

    for hospital_id in HOSPITAL_IDS:

        print(
            f"  {hospital_id}: "
            f"{q006_local[hospital_id]}"
        )

    print()
    print("Q007 local sufficient statistics:")

    for hospital_id in HOSPITAL_IDS:

        local = q007_local[
            hospital_id
        ]

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
        f"Global patients: {global_patients:,}"
    )

    print(
        f"Global admissions: {global_admissions:,}"
    )

    print(
        f"Global Q006 count: {q006_global}"
    )

    print(
        f"Global Q007 count: {global_count}"
    )

    print(
        f"Global Q007 sum: {global_sum}"
    )

    print(
        f"Global Q007 mean: {global_mean}"
    )

    print(
        "Patient disjointness: PASSED"
    )

    print(
        "Unchanged H1-H4 check: PASSED"
    )

    return (
        {
            "scenario": scenario_name,
            "h5_target_count": (
                q006_local["hospital_5"]
            ),
            "q006_global_count": (
                q006_global
            ),
            "q007_global_count": (
                global_count
            ),
            "q007_global_sum": (
                global_sum
            ),
            "q007_global_mean": (
                global_mean
            ),
            "global_patients": (
                global_patients
            ),
            "global_admissions": (
                global_admissions
            ),
        },
        q007_local,
    )



# Main
# ---------------------------------------------------------------------

def main() -> None:

    summaries = []

    baseline_q007_local = None

    for scenario_name in SCENARIOS:

        summary, q007_local = (
            validate_scenario(
                scenario_name,
                baseline_q007_local,
            )
        )

        summaries.append(
            summary
        )

        # The 100% scenario defines the unchanged
        # H1-H4 baseline.
        if scenario_name == "h5_retention_100":
            baseline_q007_local = q007_local

    summary_df = pd.DataFrame(
        summaries
    )

    # 
    # Add changes relative to full availability.
    # -------------------------------------------------------------

    baseline = summary_df.iloc[0]

    summary_df[
        "q006_count_change"
    ] = (
        summary_df["q006_global_count"]
        - baseline["q006_global_count"]
    )

    summary_df[
        "q007_sum_change"
    ] = (
        summary_df["q007_global_sum"]
        - baseline["q007_global_sum"]
    )

    summary_df[
        "q007_mean_change"
    ] = (
        summary_df["q007_global_mean"]
        - baseline["q007_global_mean"]
    )

    summary_df[
        "q007_mean_percent_change"
    ] = (
        (
            summary_df["q007_global_mean"]
            - baseline["q007_global_mean"]
        )
        /
        baseline["q007_global_mean"]
        * 100.0
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
    print(
        "ALL UNAVAILABILITY SCENARIOS VALIDATED"
    )
    print("=" * 72)

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Validation summary saved to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()