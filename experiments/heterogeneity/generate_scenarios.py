from pathlib import Path

import pandas as pd



# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATA_DIR = PROJECT_ROOT / "data"

OUTPUT_ROOT = (
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

DESTINATION_HOSPITALS = [
    "hospital_1",
    "hospital_2",
    "hospital_3",
    "hospital_4",
]

TARGET_SOURCE_HOSPITAL = "hospital_5"

TARGET_RACE = "ASIAN - KOREAN"
TARGET_ADMISSION_TYPE = "URGENT"

RETENTION_LEVELS = {
    "h5_retention_100": 1.00,
    "h5_retention_75": 0.75,
    "h5_retention_50": 0.50,
    "h5_retention_25": 0.25,
    "h5_retention_0": 0.00,
}



# Helpers
# ---------------------------------------------------------------------

def load_hospital_data() -> dict[str, pd.DataFrame]:
    """
    Load the original five hospital admissions datasets.

    These source files are read only. The script never overwrites them.
    """

    hospital_data = {}

    for hospital_id in HOSPITAL_IDS:

        path = (
            SOURCE_DATA_DIR
            / hospital_id
            / "admissions.csv.gz"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing source file: {path}"
            )

        df = pd.read_csv(path)

        hospital_data[hospital_id] = df

        print(
            f"Loaded {hospital_id}: "
            f"{len(df):,} admissions, "
            f"{df['subject_id'].nunique():,} patients"
        )

    return hospital_data


def get_earliest_admission(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one earliest-admission row per patient.
    """

    required_columns = {
        "subject_id",
        "admittime",
    }

    missing = required_columns.difference(
        dataframe.columns
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = dataframe.copy()

    df["admittime"] = pd.to_datetime(
        df["admittime"],
        errors="coerce",
    )

    if df["admittime"].isna().any():
        raise ValueError(
            "Invalid admittime values encountered."
        )

    return (
        df
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
        .copy()
    )


def identify_target_subjects(
    hospital_5_df: pd.DataFrame,
) -> list[int]:
    """
    Identify H5 patients whose earliest admission satisfies:

        race = ASIAN - KOREAN
        admission_type = URGENT
    """

    earliest = get_earliest_admission(
        hospital_5_df
    )

    required_columns = {
        "race",
        "admission_type",
    }

    missing = required_columns.difference(
        earliest.columns
    )

    if missing:
        raise ValueError(
            f"Missing target columns: {sorted(missing)}"
        )

    target = earliest[
        (earliest["race"] == TARGET_RACE)
        &
        (
            earliest["admission_type"]
            == TARGET_ADMISSION_TYPE
        )
    ]

    subject_ids = sorted(
        target["subject_id"]
        .astype(int)
        .tolist()
    )

    return subject_ids


def calculate_retained_count(
    total: int,
    retention_fraction: float,
) -> int:
    """
    Convert a retention fraction to an integer number of patients.

    Uses conventional half-up rounding:
        floor(x + 0.5)

    For 14 target patients:
        100% -> 14
         75% -> 11
         50% -> 7
         25% -> 4
          0% -> 0
    """

    return int(
        total * retention_fraction + 0.5
    )


def build_destination_map(
    target_subject_ids: list[int],
) -> dict[int, str]:
    """
    Assign every target patient a fixed experimental destination.

    The assignment is deterministic and independent of scenario.

    Patient 1 -> H1
    Patient 2 -> H2
    Patient 3 -> H3
    Patient 4 -> H4
    Patient 5 -> H1
    ...

    This means that when a patient is moved in more than one
    scenario, that patient always goes to the same hospital.
    """

    destination_map = {}

    for index, subject_id in enumerate(
        target_subject_ids
    ):
        destination = DESTINATION_HOSPITALS[
            index % len(DESTINATION_HOSPITALS)
        ]

        destination_map[subject_id] = destination

    return destination_map


def verify_patient_disjointness(
    scenario_data: dict[str, pd.DataFrame],
) -> None:
    """
    Verify that no subject_id exists in more than one hospital.
    """

    seen = {}

    for hospital_id, df in scenario_data.items():

        subject_ids = set(
            df["subject_id"]
            .astype(int)
            .unique()
        )

        for subject_id in subject_ids:

            if subject_id in seen:
                raise AssertionError(
                    f"Patient {subject_id} appears in both "
                    f"{seen[subject_id]} and {hospital_id}."
                )

            seen[subject_id] = hospital_id


def count_target_patients(
    dataframe: pd.DataFrame,
) -> int:
    """
    Count Q006 target patients using earliest admission.
    """

    earliest = get_earliest_admission(
        dataframe
    )

    mask = (
        (earliest["race"] == TARGET_RACE)
        &
        (
            earliest["admission_type"]
            == TARGET_ADMISSION_TYPE
        )
    )

    return int(mask.sum())



# Scenario generation
# ---------------------------------------------------------------------

def generate_scenario(
    original_data: dict[str, pd.DataFrame],
    target_subject_ids: list[int],
    destination_map: dict[int, str],
    scenario_name: str,
    retention_fraction: float,
    baseline_total_rows: int,
    baseline_total_patients: int,
) -> dict:
    """
    Generate one experimental patient-distribution scenario.
    """

    print()
    print("=" * 70)
    print(f"Generating scenario: {scenario_name}")
    print("=" * 70)

    total_target = len(target_subject_ids)

    retained_count = calculate_retained_count(
        total_target,
        retention_fraction,
    )

    # Keep the first N target subjects at H5.
    retained_subjects = set(
        target_subject_ids[:retained_count]
    )

    # Remaining target subjects are moved out of H5.
    moved_subjects = set(
        target_subject_ids[retained_count:]
    )

    print(
        f"Target patients total: {total_target}"
    )
    print(
        f"Retained at H5: {len(retained_subjects)}"
    )
    print(
        f"Moved from H5: {len(moved_subjects)}"
    )

    # Deep-enough copies for dataframe modification.
    scenario_data = {
        hospital_id: df.copy()
        for hospital_id, df
        in original_data.items()
    }

    h5_original = original_data[
        TARGET_SOURCE_HOSPITAL
    ]

    
    # Extract ALL admission rows belonging to moved patients.
    # -------------------------------------------------------------

    moved_rows = h5_original[
        h5_original["subject_id"]
        .astype(int)
        .isin(moved_subjects)
    ].copy()

    # Remove all rows for those patients from H5.
    scenario_data[TARGET_SOURCE_HOSPITAL] = (
        h5_original[
            ~h5_original["subject_id"]
            .astype(int)
            .isin(moved_subjects)
        ]
        .copy()
    )

    
    # Move complete patient histories to H1-H4.
    # -------------------------------------------------------------

    movement_records = []

    for subject_id in sorted(moved_subjects):

        destination = destination_map[
            subject_id
        ]

        patient_rows = moved_rows[
            moved_rows["subject_id"].astype(int)
            == subject_id
        ].copy()

        if patient_rows.empty:
            raise AssertionError(
                f"No rows found for moved patient "
                f"{subject_id}."
            )

        scenario_data[destination] = pd.concat(
            [
                scenario_data[destination],
                patient_rows,
            ],
            ignore_index=True,
        )

        movement_records.append(
            {
                "subject_id": subject_id,
                "source_hospital": (
                    TARGET_SOURCE_HOSPITAL
                ),
                "destination_hospital": destination,
                "admission_rows_moved": len(
                    patient_rows
                ),
            }
        )

    
    # Safety checks
    # -------------------------------------------------------------

    scenario_total_rows = sum(
        len(df)
        for df in scenario_data.values()
    )

    scenario_subject_ids = set()

    for df in scenario_data.values():
        scenario_subject_ids.update(
            df["subject_id"]
            .astype(int)
            .unique()
        )

    scenario_total_patients = len(
        scenario_subject_ids
    )

    if scenario_total_rows != baseline_total_rows:
        raise AssertionError(
            f"{scenario_name}: admission-row count changed. "
            f"Expected {baseline_total_rows}, "
            f"got {scenario_total_rows}."
        )

    if (
        scenario_total_patients
        != baseline_total_patients
    ):
        raise AssertionError(
            f"{scenario_name}: global patient count changed. "
            f"Expected {baseline_total_patients}, "
            f"got {scenario_total_patients}."
        )

    verify_patient_disjointness(
        scenario_data
    )

    
    # Check target distribution
    # -------------------------------------------------------------

    target_counts = {}

    for hospital_id in HOSPITAL_IDS:

        target_counts[hospital_id] = (
            count_target_patients(
                scenario_data[hospital_id]
            )
        )

    global_target_count = sum(
        target_counts.values()
    )

    if global_target_count != 38:
        raise AssertionError(
            f"{scenario_name}: global target count "
            f"changed from 38 to "
            f"{global_target_count}."
        )

    if (
        target_counts["hospital_5"]
        != retained_count
    ):
        raise AssertionError(
            f"{scenario_name}: H5 target count "
            f"expected {retained_count}, got "
            f"{target_counts['hospital_5']}."
        )

    
    # Save scenario
    # -------------------------------------------------------------

    scenario_dir = (
        OUTPUT_ROOT
        / scenario_name
    )

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for hospital_id, df in scenario_data.items():

        hospital_dir = (
            scenario_dir
            / hospital_id
        )

        hospital_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            hospital_dir
            / "admissions.csv.gz"
        )

        df.to_csv(
            output_file,
            index=False,
            compression="gzip",
        )

    movement_df = pd.DataFrame(
        movement_records,
        columns=[
            "subject_id",
            "source_hospital",
            "destination_hospital",
            "admission_rows_moved",
        ],
    )

    movement_df.to_csv(
        scenario_dir
        / "movement_manifest.csv",
        index=False,
    )

    
    # Scenario summary
    # -------------------------------------------------------------

    summary = {
        "scenario": scenario_name,
        "retention_fraction": (
            retention_fraction
        ),
        "target_total_global": (
            global_target_count
        ),
        "target_h5": (
            target_counts["hospital_5"]
        ),
        "target_h1": (
            target_counts["hospital_1"]
        ),
        "target_h2": (
            target_counts["hospital_2"]
        ),
        "target_h3": (
            target_counts["hospital_3"]
        ),
        "target_h4": (
            target_counts["hospital_4"]
        ),
        "moved_patients": len(
            moved_subjects
        ),
        "global_patients": (
            scenario_total_patients
        ),
        "global_admissions": (
            scenario_total_rows
        ),
    }

    print("Target distribution:")

    for hospital_id in HOSPITAL_IDS:
        print(
            f"  {hospital_id}: "
            f"{target_counts[hospital_id]}"
        )

    print(
        f"  GLOBAL: {global_target_count}"
    )

    print(
        f"Global patients preserved: "
        f"{scenario_total_patients:,}"
    )

    print(
        f"Global admissions preserved: "
        f"{scenario_total_rows:,}"
    )

    print(
        "Patient-disjointness check: PASSED"
    )

    return summary



# Main
# ---------------------------------------------------------------------

def main() -> None:

    print(
        "Loading original hospital datasets..."
    )

    original_data = load_hospital_data()

    baseline_total_rows = sum(
        len(df)
        for df in original_data.values()
    )

    baseline_subject_ids = set()

    for df in original_data.values():
        baseline_subject_ids.update(
            df["subject_id"]
            .astype(int)
            .unique()
        )

    baseline_total_patients = len(
        baseline_subject_ids
    )

    # Verify original partition is already patient-disjoint.
    verify_patient_disjointness(
        original_data
    )

    print()
    print("Original distributed dataset:")
    print(
        f"  patients: "
        f"{baseline_total_patients:,}"
    )
    print(
        f"  admissions: "
        f"{baseline_total_rows:,}"
    )
    print(
        "  patient-disjointness: PASSED"
    )

    
    # Identify H5 target patients
    # -------------------------------------------------------------

    target_subject_ids = (
        identify_target_subjects(
            original_data[
                TARGET_SOURCE_HOSPITAL
            ]
        )
    )

    print()
    print(
        "H5 target subgroup:"
    )

    print(
        f"  race = {TARGET_RACE}"
    )

    print(
        "  admission_type = "
        f"{TARGET_ADMISSION_TYPE}"
    )

    print(
        f"  target patients in H5 = "
        f"{len(target_subject_ids)}"
    )

    if len(target_subject_ids) != 14:
        raise AssertionError(
            "Expected 14 Q006/Q007 target "
            "patients in H5."
        )

    
    # Give every target patient a permanent destination
    # -------------------------------------------------------------

    destination_map = build_destination_map(
        target_subject_ids
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_manifest = pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "destination_hospital": (
                    destination_map[
                        subject_id
                    ]
                ),
            }
            for subject_id
            in target_subject_ids
        ]
    )

    destination_manifest.to_csv(
        OUTPUT_ROOT
        / "target_destination_map.csv",
        index=False,
    )

    
    # Generate all scenarios
    # -------------------------------------------------------------

    summaries = []

    for (
        scenario_name,
        retention_fraction,
    ) in RETENTION_LEVELS.items():

        summary = generate_scenario(
            original_data=original_data,
            target_subject_ids=(
                target_subject_ids
            ),
            destination_map=(
                destination_map
            ),
            scenario_name=scenario_name,
            retention_fraction=(
                retention_fraction
            ),
            baseline_total_rows=(
                baseline_total_rows
            ),
            baseline_total_patients=(
                baseline_total_patients
            ),
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_ROOT
        / "scenario_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("ALL SCENARIOS GENERATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"Scenario summary saved to:"
    )
    print(summary_path)

    print()
    print(
        "Original hospital datasets were "
        "not modified."
    )


if __name__ == "__main__":
    main()