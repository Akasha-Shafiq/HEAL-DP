"""
Patient-level differentially private training.

This module implements the DP training mechanism used for the
patient-level federated-learning experiments.

Privacy unit
------------
subject_id (one patient).

Supported representations
-------------------------
Strategy A:
    - exactly one retained stay per patient.

Strategy B (K=5):
    - one to five retained stays per patient.
    - the patient objective is the mean loss across that patient's
      retained stays.

For every Poisson-sampled patient minibatch:

    1. construct one full model gradient per sampled patient;
    2. clip that patient gradient once to L2 norm C;
    3. sum the clipped patient gradients;
    4. add Gaussian noise N(0, (sigma * C)^2 I) to the clipped sum;
    5. divide by the expected Poisson batch size N*q;
    6. apply the resulting gradient with SGD.

The Gaussian mechanism is therefore applied to a patient-gradient
sum whose add/remove sensitivity is bounded by C.

IMPORTANT
---------
Poisson sampling and Gaussian noise use fresh private randomness.
Do not replace these seeds with public deterministic experiment
seeds in runs that are claimed to satisfy the calibrated DP bound.
"""

from __future__ import annotations

import math
import secrets

import numpy as np
import torch

from opacus.utils.uniform_sampler import (
    UniformWithReplacementSampler,
)

from federated_learning.federated.patient_control.task import (
    load_patient_grouped_training_data,
)


# =====================================================================
# Patient-level DP training
# =====================================================================

def train_patient_dp_poisson(
    *,
    model,
    context,
    learning_rate,
    nominal_patient_batch_size,
    clipping_norm,
    noise_multiplier,
    server_round,
    local_epochs,
    device,
):
    # -----------------------------------------------------------------
    # Configuration validation
    # -----------------------------------------------------------------

    if local_epochs != 1:
        raise ValueError(
            "Patient-level DP training is fixed to one "
            "local epoch per Flower round."
        )

    if clipping_norm <= 0:
        raise ValueError(
            f"clipping_norm must be > 0, got {clipping_norm}"
        )

    if noise_multiplier <= 0:
        raise ValueError(
            "noise_multiplier must be > 0 for a DP run, "
            f"got {noise_multiplier}"
        )

    if nominal_patient_batch_size <= 0:
        raise ValueError(
            "nominal_patient_batch_size must be > 0, "
            f"got {nominal_patient_batch_size}"
        )

    # -----------------------------------------------------------------
    # Load the frozen patient-grouped representation.
    # -----------------------------------------------------------------

    info = load_patient_grouped_training_data(
        context
    )

    X = info["X"].to(device)

    y = (
        info["y"]
        .to(device)
        .reshape(-1)
    )

    patient_rows = info["patient_rows"]
    n_patients = info["n_patients"]
    partition_id = info["partition_id"]
    strategy = info["strategy"]

    # -----------------------------------------------------------------
    # Validate the frozen logistic-regression model.
    # -----------------------------------------------------------------

    if not hasattr(model, "linear"):
        raise RuntimeError(
            "Expected LogisticRegressionModel to expose "
            "a 'linear' layer."
        )

    weight = model.linear.weight
    bias = model.linear.bias

    if weight.ndim != 2:
        raise RuntimeError(
            "Unexpected logistic-regression weight shape."
        )

    if weight.shape[0] != 1:
        raise RuntimeError(
            "Expected binary logistic regression with "
            "one output logit."
        )

    if bias is None:
        raise RuntimeError(
            "Expected logistic-regression bias parameter."
        )

    n_features = weight.shape[1]

    if X.shape[1] != n_features:
        raise RuntimeError(
            "Feature dimension does not match model dimension: "
            f"X={X.shape[1]}, model={n_features}"
        )

    # =================================================================
    # Poisson sampling design
    # =================================================================
    #
    # This must match the accounting calibration:
    #
    #     steps = ceil(N / nominal_batch_size)
    #     q     = 1 / steps
    #
    # Opacus-style Poisson DP-SGD scales the noisy clipped sum by
    # the EXPECTED batch size N*q, not by the realized sampled count.
    # =================================================================

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

    if expected_patient_batch_size <= 0:
        raise RuntimeError(
            "Expected patient batch size must be positive."
        )

    # =================================================================
    # Private randomness
    # =================================================================
    #
    # Privacy amplification assumes random Poisson sampling.
    # The Gaussian mechanism also requires fresh random noise.
    #
    # Independent PyTorch generators are seeded from Python's
    # system-backed secrets source. Seeds are intentionally not
    # returned or logged.
    # =================================================================

    sampling_generator = torch.Generator()

    sampling_generator.manual_seed(
        secrets.randbits(63)
    )

    noise_generator = torch.Generator(
        device=device
    )

    noise_generator.manual_seed(
        secrets.randbits(63)
    )

    sampler = UniformWithReplacementSampler(
        num_samples=n_patients,
        sample_rate=sample_rate,
        steps=steps_per_local_epoch,
        generator=sampling_generator,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    # =================================================================
    # DP mechanism constants
    # =================================================================

    noise_std_sum = float(
        noise_multiplier
        * clipping_norm
    )

    noise_std_normalized = float(
        noise_std_sum
        / expected_patient_batch_size
    )

    # =================================================================
    # Diagnostics
    # =================================================================

    sampled_patient_counts = []
    sampled_stay_counts = []
    empty_batches = 0

    step_mean_gradient_norms = []
    step_max_gradient_norms = []
    step_clip_fractions = []
    step_mean_clip_factors = []

    # =================================================================
    # Training
    # =================================================================

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

        # -------------------------------------------------------------
        # Empty Poisson minibatch.
        #
        # The planned DP step is retained. Its clipped gradient sum is
        # zero and Gaussian noise is still added before normalization.
        # -------------------------------------------------------------

        if m == 0:

            empty_batches += 1

            sampled_stay_counts.append(
                0
            )

            with torch.no_grad():

                clipped_weight_gradient_sum = torch.zeros(
                    n_features,
                    dtype=weight.dtype,
                    device=device,
                )

                clipped_bias_gradient_sum = torch.zeros(
                    (),
                    dtype=bias.dtype,
                    device=device,
                )

        else:

            # ---------------------------------------------------------
            # Expand sampled patients to retained stay rows.
            #
            # Strategy A:
            #     one row per sampled patient.
            #
            # Strategy B:
            #     1..5 rows per sampled patient.
            #
            # group_ids maps every retained stay back to its patient.
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
                        len(rows),
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
                len(selected_rows)
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

            # =========================================================
            # Construct exactly one gradient per sampled patient.
            # =========================================================

            with torch.no_grad():

                logits = model(
                    xb
                ).reshape(-1)

                probabilities = torch.sigmoid(
                    logits
                )

                residuals = (
                    probabilities
                    - yb
                )

                patient_stay_counts = torch.bincount(
                    group_tensor,
                    minlength=m,
                ).to(
                    dtype=xb.dtype
                )

                if torch.any(
                    patient_stay_counts <= 0
                ):
                    raise RuntimeError(
                        "A sampled patient produced zero stay rows."
                    )

                # -----------------------------------------------------
                # Weight component of each patient gradient.
                #
                # Stay gradient:
                #     (sigmoid(z) - y) * x
                #
                # Strategy B averages these stay gradients within
                # patient BEFORE patient-level clipping.
                # -----------------------------------------------------

                stay_weight_gradients = (
                    residuals.unsqueeze(1)
                    * xb
                )

                patient_weight_gradient_sums = torch.zeros(
                    (
                        m,
                        n_features,
                    ),
                    dtype=xb.dtype,
                    device=device,
                )

                patient_weight_gradient_sums.index_add_(
                    0,
                    group_tensor,
                    stay_weight_gradients,
                )

                patient_weight_gradients = (
                    patient_weight_gradient_sums
                    / patient_stay_counts.unsqueeze(1)
                )

                # -----------------------------------------------------
                # Bias component of each patient gradient.
                # -----------------------------------------------------

                patient_bias_gradient_sums = torch.zeros(
                    m,
                    dtype=xb.dtype,
                    device=device,
                )

                patient_bias_gradient_sums.index_add_(
                    0,
                    group_tensor,
                    residuals,
                )

                patient_bias_gradients = (
                    patient_bias_gradient_sums
                    / patient_stay_counts
                )

                # =====================================================
                # Full patient-gradient L2 norm.
                #
                # Includes BOTH weight and bias, so the patient's full
                # model-gradient contribution is clipped exactly once.
                # =====================================================

                patient_gradient_norms = torch.sqrt(
                    torch.sum(
                        patient_weight_gradients
                        * patient_weight_gradients,
                        dim=1,
                    )
                    +
                    (
                        patient_bias_gradients
                        * patient_bias_gradients
                    )
                )

                # =====================================================
                # Patient-level L2 clipping.
                # =====================================================

                clip_factors = torch.clamp(
                    clipping_norm
                    / patient_gradient_norms.clamp_min(
                        1e-12
                    ),
                    max=1.0,
                )

                clipped_weight_gradients = (
                    patient_weight_gradients
                    * clip_factors.unsqueeze(1)
                )

                clipped_bias_gradients = (
                    patient_bias_gradients
                    * clip_factors
                )

                # =====================================================
                # SUM clipped patient gradients BEFORE Gaussian noise.
                #
                # Under add/remove-one-patient adjacency, this sum has
                # L2 sensitivity bounded by C.
                # =====================================================

                clipped_weight_gradient_sum = (
                    clipped_weight_gradients.sum(
                        dim=0
                    )
                )

                clipped_bias_gradient_sum = (
                    clipped_bias_gradients.sum()
                )

                # =====================================================
                # Pre-noise diagnostics.
                # =====================================================

                clipped_mask = (
                    patient_gradient_norms
                    > clipping_norm
                )

                step_mean_gradient_norms.append(
                    float(
                        patient_gradient_norms.mean().item()
                    )
                )

                step_max_gradient_norms.append(
                    float(
                        patient_gradient_norms.max().item()
                    )
                )

                step_clip_fractions.append(
                    float(
                        clipped_mask.float().mean().item()
                    )
                )

                step_mean_clip_factors.append(
                    float(
                        clip_factors.mean().item()
                    )
                )

        # =============================================================
        # Gaussian mechanism on clipped patient-gradient SUM.
        #
        #     noise ~ N(0, (sigma*C)^2 I)
        #
        # Division by N*q occurs AFTER noise and is DP post-processing.
        # =============================================================

        with torch.no_grad():

            weight_noise = (
                torch.randn(
                    n_features,
                    dtype=weight.dtype,
                    device=device,
                    generator=noise_generator,
                )
                * noise_std_sum
            )

            bias_noise = (
                torch.randn(
                    (),
                    dtype=bias.dtype,
                    device=device,
                    generator=noise_generator,
                )
                * noise_std_sum
            )

            noisy_weight_gradient = (
                (
                    clipped_weight_gradient_sum
                    + weight_noise
                )
                / expected_patient_batch_size
            )

            noisy_bias_gradient = (
                (
                    clipped_bias_gradient_sum
                    + bias_noise
                )
                / expected_patient_batch_size
            )

        # =============================================================
        # Apply the DP gradient using PyTorch SGD.
        # =============================================================

        optimizer.zero_grad(
            set_to_none=True
        )

        weight.grad = (
            noisy_weight_gradient
            .reshape_as(weight)
            .clone()
        )

        bias.grad = (
            noisy_bias_gradient
            .reshape_as(bias)
            .clone()
        )

        optimizer.step()

    # =================================================================
    # Summary diagnostics
    # =================================================================

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

    realized_mean_stays = float(
        np.mean(
            sampled_stay_counts
        )
    )

    mean_gradient_norm = (
        float(
            np.mean(
                step_mean_gradient_norms
            )
        )
        if step_mean_gradient_norms
        else float("nan")
    )

    max_gradient_norm = (
        float(
            np.max(
                step_max_gradient_norms
            )
        )
        if step_max_gradient_norms
        else float("nan")
    )

    mean_clip_fraction = (
        float(
            np.mean(
                step_clip_fractions
            )
        )
        if step_clip_fractions
        else float("nan")
    )

    mean_clip_factor = (
        float(
            np.mean(
                step_mean_clip_factors
            )
        )
        if step_mean_clip_factors
        else float("nan")
    )

    return {
        "strategy":
            strategy,

        "hospital":
            info["hospital"],

        "partition_id":
            partition_id,

        "server_round":
            int(server_round),

        "n_patients":
            n_patients,

        "n_stays":
            info["n_stays"],

        "min_stays_per_patient":
            info["min_stays_per_patient"],

        "max_stays_per_patient":
            info["max_stays_per_patient"],

        "clipping_norm":
            float(clipping_norm),

        "noise_multiplier":
            float(noise_multiplier),

        "noise_std_sum":
            noise_std_sum,

        "noise_std_normalized":
            noise_std_normalized,

        "steps":
            steps_per_local_epoch,

        "sample_rate":
            sample_rate,

        "expected_patient_batch_size":
            float(
                expected_patient_batch_size
            ),

        "realized_mean_patient_batch":
            realized_mean_patient_batch,

        "realized_min_patient_batch":
            realized_min_patient_batch,

        "realized_max_patient_batch":
            realized_max_patient_batch,

        "realized_mean_stays_per_step":
            realized_mean_stays,

        "mean_gradient_norm":
            mean_gradient_norm,

        "max_gradient_norm":
            max_gradient_norm,

        "mean_clip_fraction":
            mean_clip_fraction,

        "mean_clip_factor":
            mean_clip_factor,

        "empty_batches":
            empty_batches,
    }