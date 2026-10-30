from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
)

OUTPUT_ROOT = (
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

TARGET_HOSPITAL = "hospital_5"

TARGET_RACE = "ASIAN - KOREAN"
TARGET_ADMISSION_TYPE = "URGENT"

RETENTION_LEVELS = {
    "h5_retention_100": 1.00,
    "h5_retention_75": 0.75,
    "h5_retention_50": 0.50,
    "h5_retention_25": 0.25,
    "h5_retention_0": 0.00,
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_hospital_data() -> dict[str, pd.DataFrame]:
    """
    Load the original five hospital datasets.

    Source files are read-only. Nothing under data/hospital_*
    is modified by this experiment.
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
    Return the earliest admission for each patient.
    """

    required = {
        "subject_id",
        "admittime",
    }

    missing = required.difference(
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

    required = {
        "race",
        "admission_type",
    }

    missing = required.difference(
        earliest.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    target = earliest[
        (earliest["race"] == TARGET_RACE)
        &
        (
            earliest["admission_type"]
            == TARGET_ADMISSION_TYPE
        )
    ]

    return sorted(
        target["subject_id"]
        .astype(int)
        .tolist()
    )


def calculate_retained_count(
    total: int,
    retention_fraction: float,
) -> int:
    """
    Conventional half-up rounding.

    With 14 target patients:

        100% -> 14
         75% -> 11
         50% -> 7
         25% -> 4
          0% -> 0
    """

    return int(
        total * retention_fraction + 0.5
    )


def verify_patient_disjointness(
    hospital_data: dict[str, pd.DataFrame],
) -> None:
    """
    Ensure no subject_id belongs to more than one hospital.
    """

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


def count_target_patients(
    dataframe: pd.DataFrame,
) -> int:
    """
    Apply the Q006 target definition locally.
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

    return int(
        mask.sum()
    )


# ---------------------------------------------------------------------
# Generate one unavailability scenario
# ---------------------------------------------------------------------

def generate_scenario(
    original_data: dict[str, pd.DataFrame],
    target_subject_ids: list[int],
    scenario_name: str,
    retention_fraction: float,
    baseline_patients: int,
    baseline_admissions: int,
) -> dict:

    print()
    print("=" * 72)
    print(
        f"Generating unavailability scenario: "
        f"{scenario_name}"
    )
    print("=" * 72)

    total_target = len(
        target_subject_ids
    )

    retained_count = calculate_retained_count(
        total_target,
        retention_fraction,
    )

    retained_subjects = set(
        target_subject_ids[:retained_count]
    )

    unavailable_subjects = set(
        target_subject_ids[retained_count:]
    )

    print(
        f"Original H5 target patients: "
        f"{total_target}"
    )

    print(
        f"Remaining available at H5: "
        f"{len(retained_subjects)}"
    )

    print(
        f"Made unavailable: "
        f"{len(unavailable_subjects)}"
    )

    # -------------------------------------------------------------
    # Start from independent dataframe copies.
    # -------------------------------------------------------------

    scenario_data = {
        hospital_id: df.copy()
        for hospital_id, df
        in original_data.items()
    }

    h5_original = original_data[
        TARGET_HOSPITAL
    ]

    # -------------------------------------------------------------
    # Find ALL admission rows belonging to unavailable patients.
    # -------------------------------------------------------------

    unavailable_rows = h5_original[
        h5_original["subject_id"]
        .astype(int)
        .isin(unavailable_subjects)
    ].copy()

    unavailable_admission_count = len(
        unavailable_rows
    )

    # -------------------------------------------------------------
    # Remove the complete patient histories from H5.
    #
    # CRITICAL DIFFERENCE FROM THE HETEROGENEITY EXPERIMENT:
    # these rows are NOT added to another hospital.
    # -------------------------------------------------------------

    scenario_data[TARGET_HOSPITAL] = (
        h5_original[
            ~h5_original["subject_id"]
            .astype(int)
            .isin(unavailable_subjects)
        ]
        .copy()
    )

    # -------------------------------------------------------------
    # Validate total population reduction.
    # -------------------------------------------------------------

    total_admissions = sum(
        len(df)
        for df in scenario_data.values()
    )

    global_subject_ids = set()

    for df in scenario_data.values():

        global_subject_ids.update(
            df["subject_id"]
            .astype(int)
            .unique()
        )

    total_patients = len(
        global_subject_ids
    )

    expected_patients = (
        baseline_patients
        - len(unavailable_subjects)
    )

    expected_admissions = (
        baseline_admissions
        - unavailable_admission_count
    )

    if total_patients != expected_patients:
        raise AssertionError(
            f"{scenario_name}: expected "
            f"{expected_patients} patients, "
            f"got {total_patients}."
        )

    if total_admissions != expected_admissions:
        raise AssertionError(
            f"{scenario_name}: expected "
            f"{expected_admissions} admissions, "
            f"got {total_admissions}."
        )

    
    # Verify removed patients are truly absent globally.
    # -------------------------------------------------------------

    remaining_global_ids = set()

    for df in scenario_data.values():

        remaining_global_ids.update(
            df["subject_id"]
            .astype(int)
            .unique()
        )

    accidentally_remaining = (
        unavailable_subjects
        .intersection(
            remaining_global_ids
        )
    )

    if accidentally_remaining:
        raise AssertionError(
            "Unavailable patients still exist "
            f"in experimental dataset: "
            f"{sorted(accidentally_remaining)}"
        )

    # -------------------------------------------------------------
    # Patient-disjointness must still hold.
    # -------------------------------------------------------------

    verify_patient_disjointness(
        scenario_data
    )

    # -------------------------------------------------------------
    # Q006 target counts.
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

    expected_global_target_count = (
        38
        - len(unavailable_subjects)
    )

    if (
        global_target_count
        != expected_global_target_count
    ):
        raise AssertionError(
            f"{scenario_name}: expected global "
            f"Q006 count "
            f"{expected_global_target_count}, "
            f"got {global_target_count}."
        )

    if (
        target_counts["hospital_5"]
        != retained_count
    ):
        raise AssertionError(
            f"{scenario_name}: expected H5 "
            f"target count {retained_count}, "
            f"got "
            f"{target_counts['hospital_5']}."
        )

    
    # Save scenario.
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

        df.to_csv(
            hospital_dir
            / "admissions.csv.gz",
            index=False,
            compression="gzip",
        )


    # Save unavailable-patient manifest.
    # -------------------------------------------------------------

    manifest_rows = []

    for subject_id in sorted(
        unavailable_subjects
    ):

        patient_rows = unavailable_rows[
            unavailable_rows["subject_id"]
            .astype(int)
            == subject_id
        ]

        manifest_rows.append(
            {
                "subject_id": subject_id,
                "source_hospital": (
                    TARGET_HOSPITAL
                ),
                "status": "unavailable",
                "admission_rows_removed": len(
                    patient_rows
                ),
            }
        )

    manifest_df = pd.DataFrame(
        manifest_rows,
        columns=[
            "subject_id",
            "source_hospital",
            "status",
            "admission_rows_removed",
        ],
    )

    manifest_df.to_csv(
        scenario_dir
        / "unavailable_patient_manifest.csv",
        index=False,
    )

    
    # Print scenario results.
    # -------------------------------------------------------------

    print()
    print("Target distribution:")

    for hospital_id in HOSPITAL_IDS:

        print(
            f"  {hospital_id}: "
            f"{target_counts[hospital_id]}"
        )

    print(
        f"  GLOBAL: {global_target_count}"
    )

    print()
    print(
        f"Global patients: "
        f"{total_patients:,}"
    )

    print(
        f"Patients unavailable: "
        f"{len(unavailable_subjects)}"
    )

    print(
        f"Global admissions: "
        f"{total_admissions:,}"
    )

    print(
        f"Admissions removed: "
        f"{unavailable_admission_count:,}"
    )

    print(
        "Patient disjointness: PASSED"
    )

    print(
        "Unavailable-patient absence: PASSED"
    )

    return {
        "scenario": scenario_name,
        "retention_fraction": (
            retention_fraction
        ),
        "h5_target_count": (
            target_counts["hospital_5"]
        ),
        "global_target_count": (
            global_target_count
        ),
        "unavailable_target_patients": (
            len(unavailable_subjects)
        ),
        "admission_rows_removed": (
            unavailable_admission_count
        ),
        "global_patients": (
            total_patients
        ),
        "global_admissions": (
            total_admissions
        ),
    }



# Main
# ---------------------------------------------------------------------

def main() -> None:

    print(
        "Loading original hospital datasets..."
    )

    original_data = load_hospital_data()

    
    # Original global baseline.
    # -------------------------------------------------------------

    baseline_admissions = sum(
        len(df)
        for df in original_data.values()
    )

    baseline_ids = set()

    for df in original_data.values():

        baseline_ids.update(
            df["subject_id"]
            .astype(int)
            .unique()
        )

    baseline_patients = len(
        baseline_ids
    )

    verify_patient_disjointness(
        original_data
    )

    print()
    print("Original distributed dataset:")

    print(
        f"  patients: "
        f"{baseline_patients:,}"
    )

    print(
        f"  admissions: "
        f"{baseline_admissions:,}"
    )

    print(
        "  patient-disjointness: PASSED"
    )

    
    # Identify the same 14 H5 target patients used previously.
    # -------------------------------------------------------------

    target_subject_ids = (
        identify_target_subjects(
            original_data[
                TARGET_HOSPITAL
            ]
        )
    )

    print()
    print("H5 target subgroup:")

    print(
        f"  race = {TARGET_RACE}"
    )

    print(
        f"  admission_type = "
        f"{TARGET_ADMISSION_TYPE}"
    )

    print(
        f"  target patients = "
        f"{len(target_subject_ids)}"
    )

    if len(target_subject_ids) != 14:
        raise AssertionError(
            "Expected 14 target patients in H5."
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    
    # Save complete original target list.
    # -------------------------------------------------------------

    pd.DataFrame(
        {
            "subject_id": target_subject_ids,
        }
    ).to_csv(
        OUTPUT_ROOT
        / "original_h5_target_subjects.csv",
        index=False,
    )

    
    # Generate scenarios.
    # -------------------------------------------------------------

    summaries = []

    for (
        scenario_name,
        retention_fraction,
    ) in RETENTION_LEVELS.items():

        result = generate_scenario(
            original_data=original_data,
            target_subject_ids=(
                target_subject_ids
            ),
            scenario_name=scenario_name,
            retention_fraction=(
                retention_fraction
            ),
            baseline_patients=(
                baseline_patients
            ),
            baseline_admissions=(
                baseline_admissions
            ),
        )

        summaries.append(
            result
        )

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
    print("=" * 72)
    print(
        "ALL UNAVAILABILITY SCENARIOS "
        "GENERATED SUCCESSFULLY"
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
        "Scenario summary saved to:"
    )

    print(
        summary_path
    )

    print()
    print(
        "Original hospital datasets were "
        "not modified."
    )


if __name__ == "__main__":
    main()