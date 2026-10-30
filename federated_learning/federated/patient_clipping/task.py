"""
Patient-level gradient-clipping control.

This module implements the CLIPPING-ONLY control experiment.

There is:
    - patient-level Poisson sampling
    - one gradient contribution per sampled patient
    - L2 clipping of each patient gradient
    - NO Gaussian noise

Therefore this module DOES NOT provide differential privacy.

Supported representations
-------------------------

Strategy A:
    - exactly one retained stay per patient

Strategy B (K=5):
    - one to five retained stays per patient
    - patient objective is the mean loss across the patient's
      retained stays

The Poisson sampling design and deterministic sampling seeds are
kept identical to the matched non-private patient controls so that
the effect of clipping can be isolated experimentally.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from opacus.utils.uniform_sampler import (
    UniformWithReplacementSampler,
)

from federated_learning.federated.patient_control.task import (
    load_patient_grouped_training_data,
)


# =====================================================================
# Patient-level clipped training
# =====================================================================

def train_patient_clipped_poisson(
    *,
    model,
    context,
    learning_rate,
    nominal_patient_batch_size,
    clipping_norm,
    server_round,
    local_epochs,
    device,
):

    # -----------------------------------------------------------------
    # Experimental configuration validation
    # -----------------------------------------------------------------

    if local_epochs != 1:

        raise ValueError(
            "The clipping control is fixed to one "
            "local epoch per Flower round."
        )


    if clipping_norm <= 0:

        raise ValueError(
            f"clipping_norm must be > 0, got {clipping_norm}"
        )


    # -----------------------------------------------------------------
    # Load exactly the same patient-grouped representation used by
    # the matched non-private controls.
    # -----------------------------------------------------------------

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
    ).reshape(
        -1
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


    # -----------------------------------------------------------------
    # Validate the model structure.
    #
    # The frozen model is:
    #
    #     nn.Linear(351, 1)
    #
    # exposed as model.linear.
    #
    # We use the exact analytical logistic-regression gradients so
    # that all patients in a sampled minibatch can be processed
    # efficiently before patient-level clipping.
    # -----------------------------------------------------------------

    if not hasattr(
        model,
        "linear",
    ):

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
    # Sampling design
    # =================================================================
    #
    # This is intentionally identical to the accounting calibration
    # and matched non-private controls.
    #
    #     steps = ceil(N / nominal_batch_size)
    #
    #     q = 1 / steps
    #
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


    # =================================================================
    # Deterministic sampling seed
    # =================================================================
    #
    # IMPORTANT:
    #
    # This is a NON-PRIVATE clipping-control experiment.
    #
    # The seed is intentionally identical to the patient-control
    # experiment so that, for a given hospital and round, the same
    # Poisson patient samples are generated.
    #
    # The future DP experiment will treat randomness separately.
    # =================================================================

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


    model.train()


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


        if m == 0:

            empty_batches += 1
            continue


        # -------------------------------------------------------------
        # Expand sampled patients to their retained stay rows.
        #
        # Strategy A:
        #     one row per sampled patient.
        #
        # Strategy B:
        #     1..5 rows per sampled patient.
        #
        # group_ids maps every stay back to its sampled patient.
        # -------------------------------------------------------------

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


        # =============================================================
        # Logistic-regression residual
        # =============================================================
        #
        # For BCE with logits:
        #
        #     d loss / d logit = sigmoid(logit) - y
        #
        # =============================================================

        with torch.no_grad():

            logits = model(
                xb
            ).reshape(
                -1
            )


            probabilities = torch.sigmoid(
                logits
            )


            residuals = (
                probabilities
                - yb
            )


            # ---------------------------------------------------------
            # One contribution count per sampled patient.
            # ---------------------------------------------------------

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


            # =========================================================
            # Patient gradient: weight
            # =========================================================
            #
            # Stay gradient:
            #
            #     (sigmoid(z) - y) * x
            #
            # Then mean across all retained stays belonging to each
            # sampled patient.
            # =========================================================

            stay_weight_gradients = (
                residuals.unsqueeze(
                    1
                )
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
                / patient_stay_counts.unsqueeze(
                    1
                )
            )


            # =========================================================
            # Patient gradient: bias
            # =========================================================

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


            # =========================================================
            # Full patient-gradient L2 norm
            # =========================================================
            #
            # Includes BOTH:
            #
            #     linear.weight
            #     linear.bias
            #
            # so each patient's complete model-gradient contribution
            # is clipped once.
            # =========================================================

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


            # =========================================================
            # L2 clipping
            # =========================================================
            #
            #     factor_i =
            #
            #         min(
            #             1,
            #             C / ||g_i||_2
            #         )
            #
            # =========================================================

            clip_factors = torch.clamp(
                clipping_norm
                / patient_gradient_norms.clamp_min(
                    1e-12
                ),
                max=1.0,
            )


            clipped_weight_gradients = (
                patient_weight_gradients
                * clip_factors.unsqueeze(
                    1
                )
            )


            clipped_bias_gradients = (
                patient_bias_gradients
                * clip_factors
            )


            # =========================================================
            # Average the clipped PATIENT gradients.
            # =========================================================

            mean_clipped_weight_gradient = (
                clipped_weight_gradients.mean(
                    dim=0
                )
            )


            mean_clipped_bias_gradient = (
                clipped_bias_gradients.mean()
            )


            # =========================================================
            # Diagnostics
            # =========================================================

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
        # Apply the already-computed clipped gradient using PyTorch SGD.
        # =============================================================

        optimizer.zero_grad(
            set_to_none=True
        )


        weight.grad = (
            mean_clipped_weight_gradient
            .reshape_as(
                weight
            )
            .clone()
        )


        bias.grad = (
            mean_clipped_bias_gradient
            .reshape_as(
                bias
            )
            .clone()
        )


        optimizer.step()


    # =================================================================
    # Summary
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


    realized_mean_stays = (
        float(
            np.mean(
                sampled_stay_counts
            )
        )
        if sampled_stay_counts
        else 0.0
    )


    mean_gradient_norm = (
        float(
            np.mean(
                step_mean_gradient_norms
            )
        )
        if step_mean_gradient_norms
        else float(
            "nan"
        )
    )


    max_gradient_norm = (
        float(
            np.max(
                step_max_gradient_norms
            )
        )
        if step_max_gradient_norms
        else float(
            "nan"
        )
    )


    mean_clip_fraction = (
        float(
            np.mean(
                step_clip_fractions
            )
        )
        if step_clip_fractions
        else float(
            "nan"
        )
    )


    mean_clip_factor = (
        float(
            np.mean(
                step_mean_clip_factors
            )
        )
        if step_mean_clip_factors
        else float(
            "nan"
        )
    )


    return {
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

        "clipping_norm":
            float(
                clipping_norm
            ),

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