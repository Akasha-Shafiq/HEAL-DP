"""
Patient-grouped Poisson-sampling utilities.

This is a NON-PRIVATE control experiment.

Supported training representations
----------------------------------

Strategy A:
    - optimization unit = patient
    - exactly one retained ICU stay per patient
    - patients are sampled using Poisson sampling
    - no clipping
    - no Gaussian noise

Strategy B (K=5):
    - optimization unit = patient
    - each patient has 1..5 retained ICU stays
    - patient loss = mean loss across that patient's retained stays
    - patients are sampled using Poisson sampling
    - no clipping
    - no Gaussian noise

These controls match the sampling/optimization structure planned
for A-DP and B-DP before privacy-specific clipping and Gaussian
noise are introduced.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from torch import nn

from opacus.utils.uniform_sampler import (
    UniformWithReplacementSampler,
)

from federated_learning.federated.representation.task import (
    load_strategy_training_partition,
)


# =====================================================================
# Patient-grouped training data
# =====================================================================

_CACHE = {}


def load_patient_grouped_training_data(
    context,
):

    (
        dataframe,
        X,
        y,
        feature_columns,
        hospital,
        partition_id,
        strategy,
    ) = load_strategy_training_partition(
        context
    )


    # -------------------------------------------------------------
    # This matched control supports the two frozen patient-level
    # representations only.
    # -------------------------------------------------------------

    if strategy not in {
        "strategy_a",
        "strategy_b_k5",
    }:

        raise RuntimeError(
            f"Unsupported patient-control strategy: {strategy}"
        )


    cache_key = (
        hospital,
        strategy,
    )


    if cache_key in _CACHE:

        return _CACHE[
            cache_key
        ]


    dataframe = dataframe.reset_index(
        drop=True
    )


    # -------------------------------------------------------------
    # Build patient -> row-index mapping.
    #
    # Strategy A:
    #     one row for every patient.
    #
    # Strategy B:
    #     one to five rows for every patient.
    # -------------------------------------------------------------

    grouped = dataframe.groupby(
        "subject_id",
        sort=False,
    ).indices


    patient_ids = np.asarray(
        list(
            grouped.keys()
        )
    )


    patient_rows = [
        np.asarray(
            grouped[
                patient_id
            ],
            dtype=np.int64,
        )
        for patient_id
        in patient_ids
    ]


    n_patients = len(
        patient_ids
    )


    n_stays = len(
        dataframe
    )


    contribution_counts = np.asarray(
        [
            len(
                rows
            )
            for rows
            in patient_rows
        ],
        dtype=np.int64,
    )


    # -------------------------------------------------------------
    # General patient-mapping validation.
    # -------------------------------------------------------------

    if n_patients == 0:

        raise RuntimeError(
            "No patients were found in the training partition."
        )


    if contribution_counts.min() < 1:

        raise RuntimeError(
            "A patient has zero retained stays."
        )


    # -------------------------------------------------------------
    # Representation-specific contribution validation.
    # -------------------------------------------------------------

    if strategy == "strategy_a":

        if (
            contribution_counts.min() != 1
            or contribution_counts.max() != 1
        ):

            raise RuntimeError(
                "Strategy A must contain exactly "
                "one retained stay per patient."
            )


        if n_stays != n_patients:

            raise RuntimeError(
                "Strategy A validation failed: "
                "number of stays must equal number of patients."
            )


    elif strategy == "strategy_b_k5":

        if contribution_counts.max() > 5:

            raise RuntimeError(
                "Strategy B contribution bound violated: "
                "a patient has more than five retained stays."
            )


    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )


    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    ).reshape(
        -1,
        1,
    )


    result = {
        "dataframe":
            dataframe,

        "X":
            X_tensor,

        "y":
            y_tensor,

        "feature_columns":
            feature_columns,

        "hospital":
            hospital,

        "partition_id":
            partition_id,

        "strategy":
            strategy,

        "patient_ids":
            patient_ids,

        "patient_rows":
            patient_rows,

        "n_patients":
            n_patients,

        "n_stays":
            n_stays,

        "min_stays_per_patient":
            int(
                contribution_counts.min()
            ),

        "max_stays_per_patient":
            int(
                contribution_counts.max()
            ),
    }


    _CACHE[
        cache_key
    ] = result


    return result


# =====================================================================
# Patient-grouped Poisson training
# =====================================================================

def train_patient_grouped_poisson(
    *,
    model,
    context,
    learning_rate,
    nominal_patient_batch_size,
    server_round,
    local_epochs,
    device,
):

    # -------------------------------------------------------------
    # The matched non-private controls use exactly one local epoch
    # per Flower round, matching the planned DP experiments.
    # -------------------------------------------------------------

    if local_epochs != 1:

        raise ValueError(
            "This matched control is currently fixed "
            "to one local epoch per Flower round."
        )


    info = load_patient_grouped_training_data(
        context
    )


    X = info[
        "X"
    ].to(
        device
    )


    y = info[
        "y"
    ].to(
        device
    )


    patient_rows = info[
        "patient_rows"
    ]


    n_patients = info[
        "n_patients"
    ]


    partition_id = info[
        "partition_id"
    ]


    strategy = info[
        "strategy"
    ]


    # -------------------------------------------------------------
    # Sampling design.
    #
    # This must match the accounting calibration:
    #
    #     steps = ceil(N / nominal_batch_size)
    #
    #     q = 1 / steps
    #
    # Under Poisson sampling, the realized minibatch size varies
    # around N*q rather than being exactly the nominal batch size.
    # -------------------------------------------------------------

    steps_per_local_epoch = math.ceil(
        n_patients
        / nominal_patient_batch_size
    )


    sample_rate = (
        1.0
        / steps_per_local_epoch
    )


    expected_patient_batch_size = (
        n_patients
        * sample_rate
    )


    # -------------------------------------------------------------
    # Deterministic experimental seed.
    #
    # This is intentionally deterministic because this is a
    # NON-PRIVATE experimental control.
    #
    # The future DP implementation will treat randomness separately.
    # -------------------------------------------------------------

    generator = torch.Generator()


    generator.manual_seed(
        42
        + partition_id * 10_000
        + server_round
    )


    sampler = UniformWithReplacementSampler(
        num_samples=n_patients,
        sample_rate=sample_rate,
        steps=steps_per_local_epoch,
        generator=generator,
    )


    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )


    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )


    model.train()


    step_losses = []

    sampled_patient_counts = []

    sampled_stay_counts = []

    empty_batches = 0


    # -------------------------------------------------------------
    # Each sampler output contains PATIENT indices.
    #
    # A:
    #     every selected patient expands to exactly one stay.
    #
    # B:
    #     every selected patient expands to 1..5 stays.
    # -------------------------------------------------------------

    for sampled_patient_indices in sampler:

        sampled_patient_indices = list(
            sampled_patient_indices
        )


        m = len(
            sampled_patient_indices
        )


        sampled_patient_counts.append(
            m
        )


        if m == 0:

            # An empty Poisson sample is mathematically possible,
            # although extremely unlikely at our expected batch
            # sizes.
            empty_batches += 1
            continue


        # ---------------------------------------------------------
        # Expand sampled patients into retained stay rows.
        #
        # group_ids identifies which sampled patient generated
        # each stay row.
        # ---------------------------------------------------------

        row_blocks = []

        group_blocks = []


        for group_index, patient_index in enumerate(
            sampled_patient_indices
        ):

            rows = patient_rows[
                patient_index
            ]


            row_blocks.append(
                rows
            )


            group_blocks.append(
                np.full(
                    len(
                        rows
                    ),
                    group_index,
                    dtype=np.int64,
                )
            )


        selected_rows = np.concatenate(
            row_blocks
        )


        group_ids = np.concatenate(
            group_blocks
        )


        sampled_stay_counts.append(
            len(
                selected_rows
            )
        )


        row_tensor = torch.tensor(
            selected_rows,
            dtype=torch.long,
            device=device,
        )


        group_tensor = torch.tensor(
            group_ids,
            dtype=torch.long,
            device=device,
        )


        xb = X[
            row_tensor
        ]


        yb = y[
            row_tensor
        ]


        # ---------------------------------------------------------
        # Stay-level losses.
        # ---------------------------------------------------------

        logits = model(
            xb
        )


        stay_losses = criterion(
            logits,
            yb,
        ).reshape(
            -1
        )


        # ---------------------------------------------------------
        # Convert stay-level losses into exactly one loss per
        # sampled patient.
        #
        # Strategy A:
        #     patient_mean_loss = the single retained stay loss.
        #
        # Strategy B:
        #     patient_mean_loss = mean loss across the patient's
        #     1..5 retained stays.
        # ---------------------------------------------------------

        patient_loss_sums = torch.zeros(
            m,
            dtype=stay_losses.dtype,
            device=device,
        )


        patient_loss_sums.scatter_add_(
            0,
            group_tensor,
            stay_losses,
        )


        patient_stay_counts = torch.bincount(
            group_tensor,
            minlength=m,
        ).to(
            dtype=stay_losses.dtype
        )


        if torch.any(
            patient_stay_counts <= 0
        ):

            raise RuntimeError(
                "A sampled patient produced zero stay rows."
            )


        patient_mean_losses = (
            patient_loss_sums
            / patient_stay_counts
        )


        # ---------------------------------------------------------
        # Every sampled patient receives equal weight in the
        # optimization objective regardless of stay count.
        # ---------------------------------------------------------

        loss = patient_mean_losses.mean()


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()


        step_losses.append(
            float(
                loss.item()
            )
        )


    # =================================================================
    # Training summary
    # =================================================================

    mean_loss = (
        float(
            np.mean(
                step_losses
            )
        )
        if step_losses
        else float(
            "nan"
        )
    )


    realized_mean_patient_batch = float(
        np.mean(
            sampled_patient_counts
        )
    )


    realized_min_patient_batch = int(
        np.min(
            sampled_patient_counts
        )
    )


    realized_max_patient_batch = int(
        np.max(
            sampled_patient_counts
        )
    )


    realized_mean_stays = (
        float(
            np.mean(
                sampled_stay_counts
            )
        )
        if sampled_stay_counts
        else 0.0
    )


    return {
        "train_loss":
            mean_loss,

        "strategy":
            strategy,

        "hospital":
            info[
                "hospital"
            ],

        "n_patients":
            n_patients,

        "n_stays":
            info[
                "n_stays"
            ],

        "min_stays_per_patient":
            info[
                "min_stays_per_patient"
            ],

        "max_stays_per_patient":
            info[
                "max_stays_per_patient"
            ],

        "steps":
            steps_per_local_epoch,

        "sample_rate":
            sample_rate,

        "expected_patient_batch_size":
            expected_patient_batch_size,

        "realized_mean_patient_batch":
            realized_mean_patient_batch,

        "realized_min_patient_batch":
            realized_min_patient_batch,

        "realized_max_patient_batch":
            realized_max_patient_batch,

        "realized_mean_stays_per_step":
            realized_mean_stays,

        "empty_batches":
            empty_batches,
    }