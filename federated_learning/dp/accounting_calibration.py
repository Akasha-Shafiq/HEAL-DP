"""
Patient-level DP accounting calibration for F2.

This script performs NO model training and accesses NO patient features.

It calibrates the Gaussian noise multiplier required to achieve
a target patient-level (epsilon, delta) budget under the planned
30-round federated DP-SGD experiment.

Privacy unit:
    subject_id (patient)

Adjacency:
    add/remove one patient

Sampling:
    Poisson patient sampling

Accountant:
    Opacus RDP accountant

The same accounting grid applies to:
    Strategy A:
        one stay/example per patient

    Strategy B:
        up to five stays internally aggregated into one patient
        gradient before clipping/noising
"""

from __future__ import annotations

import math

import pandas as pd

from opacus.accountants import create_accountant
from opacus.accountants.utils import get_noise_multiplier

from federated_learning.config import PROJECT_ROOT


# =====================================================================
# Primary accounting design
# =====================================================================

ACCOUNTANT = "rdp"

DELTA = 1e-5

NUM_ROUNDS = 30

LOCAL_EPOCHS_PER_ROUND = 1

TARGET_EPSILONS = [
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
]

NOMINAL_PATIENT_BATCH_SIZES = [
    64,
    128,
    256,
]


# Frozen patient counts from the hospital-first training split.
HOSPITAL_PATIENTS = {
    "H1": 8690,
    "H2": 8690,
    "H3": 8690,
    "H4": 2172,
    "H5": 15208,
}


OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "federated_learning"
    / "f2_dp"
    / "accounting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Helper: independently recompute epsilon from calibrated sigma
# =====================================================================

def compute_epsilon(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """
    Re-run the accountant forward to verify the epsilon obtained
    from the calibrated noise multiplier.
    """

    accountant = create_accountant(
        mechanism=ACCOUNTANT
    )

    for _ in range(steps):

        accountant.step(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )

    return float(
        accountant.get_epsilon(
            delta=delta
        )
    )


# =====================================================================
# Build accounting grid
# =====================================================================

rows = []


for batch_size in NOMINAL_PATIENT_BATCH_SIZES:

    for hospital, n_patients in HOSPITAL_PATIENTS.items():

        # This mirrors the logical length of a standard PyTorch
        # DataLoader before Opacus converts it to Poisson sampling.
        steps_per_local_epoch = math.ceil(
            n_patients
            / batch_size
        )

        # Opacus DPDataLoader semantics:
        # sample_rate = 1 / len(data_loader)
        sample_rate = (
            1.0
            / steps_per_local_epoch
        )

        expected_patient_batch_size = (
            n_patients
            * sample_rate
        )

        total_steps = (
            NUM_ROUNDS
            * LOCAL_EPOCHS_PER_ROUND
            * steps_per_local_epoch
        )


        for target_epsilon in TARGET_EPSILONS:

            sigma = get_noise_multiplier(
                target_epsilon=target_epsilon,
                target_delta=DELTA,
                sample_rate=sample_rate,
                steps=total_steps,
                accountant=ACCOUNTANT,
                epsilon_tolerance=0.001,
            )


            achieved_epsilon = compute_epsilon(
                noise_multiplier=sigma,
                sample_rate=sample_rate,
                steps=total_steps,
                delta=DELTA,
            )


            if achieved_epsilon > (
                target_epsilon
                + 0.01
            ):

                raise RuntimeError(
                    "Calibrated epsilon exceeded "
                    f"target unexpectedly: "
                    f"hospital={hospital}, "
                    f"batch={batch_size}, "
                    f"target={target_epsilon}, "
                    f"achieved={achieved_epsilon}"
                )


            rows.append(
                {
                    "hospital":
                        hospital,

                    "training_patients":
                        n_patients,

                    "nominal_patient_batch_size":
                        batch_size,

                    "steps_per_local_epoch":
                        steps_per_local_epoch,

                    "expected_patient_batch_size":
                        expected_patient_batch_size,

                    "sample_rate":
                        sample_rate,

                    "num_rounds":
                        NUM_ROUNDS,

                    "local_epochs_per_round":
                        LOCAL_EPOCHS_PER_ROUND,

                    "total_dp_steps":
                        total_steps,

                    "delta":
                        DELTA,

                    "accountant":
                        ACCOUNTANT,

                    "target_epsilon":
                        target_epsilon,

                    "noise_multiplier":
                        sigma,

                    "achieved_epsilon":
                        achieved_epsilon,

                    "epsilon_difference":
                        achieved_epsilon
                        - target_epsilon,
                }
            )


accounting = pd.DataFrame(
    rows
)


# =====================================================================
# Basic validations
# =====================================================================

assert len(
    accounting
) == (
    len(
        HOSPITAL_PATIENTS
    )
    * len(
        NOMINAL_PATIENT_BATCH_SIZES
    )
    * len(
        TARGET_EPSILONS
    )
)


# H1/H2/H3 have the same number of patients, so they must have
# identical privacy-accounting parameters for each configuration.

for batch_size in NOMINAL_PATIENT_BATCH_SIZES:

    for epsilon in TARGET_EPSILONS:

        subset = accounting.loc[
            (
                accounting[
                    "nominal_patient_batch_size"
                ]
                == batch_size
            )
            &
            (
                accounting[
                    "target_epsilon"
                ]
                == epsilon
            )
            &
            (
                accounting[
                    "hospital"
                ].isin(
                    [
                        "H1",
                        "H2",
                        "H3",
                    ]
                )
            )
        ]


        assert (
            subset[
                "noise_multiplier"
            ].nunique()
            == 1
        )


# =====================================================================
# Save complete accounting grid
# =====================================================================

full_path = (
    OUTPUT_DIR
    / "patient_dp_accounting_grid_30rounds.csv"
)


accounting.to_csv(
    full_path,
    index=False,
)


# =====================================================================
# Sampling-design table
# =====================================================================

sampling_design = (
    accounting[
        [
            "hospital",
            "training_patients",
            "nominal_patient_batch_size",
            "steps_per_local_epoch",
            "expected_patient_batch_size",
            "sample_rate",
            "total_dp_steps",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "nominal_patient_batch_size",
            "hospital",
        ]
    )
)


sampling_design.to_csv(
    OUTPUT_DIR
    / "patient_sampling_design_30rounds.csv",
    index=False,
)


# =====================================================================
# Compact sigma comparison
# =====================================================================

# H1 represents H1/H2/H3 because their patient counts are identical.
compact = (
    accounting.loc[
        accounting[
            "hospital"
        ].isin(
            [
                "H1",
                "H4",
                "H5",
            ]
        ),
        [
            "hospital",
            "nominal_patient_batch_size",
            "target_epsilon",
            "noise_multiplier",
        ],
    ]
    .pivot(
        index=[
            "nominal_patient_batch_size",
            "target_epsilon",
        ],
        columns="hospital",
        values="noise_multiplier",
    )
    .reset_index()
    .rename(
        columns={
            "H1":
                "sigma_H1_H2_H3",

            "H4":
                "sigma_H4",

            "H5":
                "sigma_H5",
        }
    )
)


compact.to_csv(
    OUTPUT_DIR
    / "noise_multiplier_summary_30rounds.csv",
    index=False,
)


# =====================================================================
# Console output
# =====================================================================

print("=" * 80)
print("PATIENT-LEVEL DP ACCOUNTING CALIBRATION")
print("=" * 80)

print()

print(
    f"Accountant: {ACCOUNTANT.upper()}"
)

print(
    f"Delta:      {DELTA}"
)

print(
    f"Rounds:     {NUM_ROUNDS}"
)

print(
    "Privacy unit: patient (subject_id)"
)

print(
    "Adjacency:    add/remove one patient"
)


print()
print("=" * 80)
print("PATIENT SAMPLING DESIGN")
print("=" * 80)

print(
    sampling_design.round(
        {
            "expected_patient_batch_size":
                3,

            "sample_rate":
                6,
        }
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("REQUIRED NOISE MULTIPLIER")
print("=" * 80)

print(
    compact.round(
        6
    ).to_string(
        index=False
    )
)


print()
print("=" * 80)
print("CALIBRATION VALIDATION")
print("=" * 80)

validation_summary = (
    accounting.groupby(
        "nominal_patient_batch_size"
    )
    .agg(
        maximum_target_epsilon=(
            "target_epsilon",
            "max",
        ),

        maximum_absolute_epsilon_error=(
            "epsilon_difference",
            lambda x: float(
                x.abs().max()
            ),
        ),
    )
    .reset_index()
)


print(
    validation_summary.round(
        6
    ).to_string(
        index=False
    )
)


print()
print(
    "Full accounting grid saved to:"
)

print(
    f"  {full_path}"
)

print()
print(
    "No model was trained and no DP budget was consumed."
)