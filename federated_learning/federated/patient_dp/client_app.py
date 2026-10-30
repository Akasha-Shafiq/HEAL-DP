from __future__ import annotations

import math

import torch

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import (
    ClientApp,
)

from federated_learning.models import (
    LogisticRegressionModel,
)

from federated_learning.federated.patient_control.task import (
    load_patient_grouped_training_data,
)

from federated_learning.federated.patient_dp.task import (
    train_patient_dp_poisson,
)


app = ClientApp()


# =====================================================================
# Frozen privacy-accounting configuration
# =====================================================================
#
# These noise multipliers were calibrated previously using:
#
#     accountant       = RDP
#     privacy unit     = subject_id
#     adjacency        = add/remove one patient
#     delta            = 1e-5
#     server rounds    = 30
#     local epochs     = 1
#     patient batch    = 256
#     Poisson q        = 1 / ceil(N / 256)
#
# H1, H2, H3 have identical patient counts and therefore identical
# sampling rates and calibrated noise multipliers.
#
# IMPORTANT:
# Do not change these values independently of the frozen accounting
# table.
# =====================================================================

ACCOUNTING_DELTA = 1e-5

ACCOUNTING_BATCH_SIZE = 256

ACCOUNTING_LOCAL_EPOCHS = 1

ACCOUNTING_ROUNDS = 30


NOISE_MULTIPLIERS = {

    0.5: {
        "H1": 7.304688,
        "H2": 7.304688,
        "H3": 7.304688,
        "H4": 14.140625,
        "H5": 5.517578,
    },

    1.0: {
        "H1": 3.925781,
        "H2": 3.925781,
        "H3": 3.925781,
        "H4": 7.529297,
        "H5": 2.993164,
    },

    2.0: {
        "H1": 2.193604,
        "H2": 2.193604,
        "H3": 2.193604,
        "H4": 4.088135,
        "H5": 1.713257,
    },

    4.0: {
        "H1": 1.333618,
        "H2": 1.333618,
        "H3": 1.333618,
        "H4": 2.315674,
        "H5": 1.094208,
    },

    8.0: {
        "H1": 0.915375,
        "H2": 0.915375,
        "H3": 0.915375,
        "H4": 1.414642,
        "H5": 0.794258,
    },
}


SUPPORTED_EPSILONS = tuple(
    sorted(
        NOISE_MULTIPLIERS.keys()
    )
)


# =====================================================================
# Frozen patient counts covered by the published calibration
# =====================================================================
# These are the training PATIENT counts, not admission/stay counts.
# Reusing a previously calibrated sigma with a different cohort could
# invalidate the stated patient-level privacy accounting.

CALIBRATED_TRAIN_PATIENT_COUNTS = {
    "H1": 8690,
    "H2": 8690,
    "H3": 8690,
    "H4": 2172,
    "H5": 15208,
}


def require_calibrated_patient_count(
    *,
    hospital: str,
    n_patients: int,
) -> None:
    """Fail closed when training population differs from DP calibration."""
    expected = CALIBRATED_TRAIN_PATIENT_COUNTS.get(hospital)
    if expected is None:
        raise ValueError(
            f"No patient-level DP calibration for hospital {hospital!r}."
        )
    if n_patients != expected:
        raise ValueError(
            f"Hospital {hospital} has {n_patients} training patients, "
            f"but its fixed noise multiplier was calibrated for {expected}. "
            "Do not reuse the published DP epsilon claim. "
            "Recompute and validate accounting for this cohort and "
            "training/sampling configuration before training."
        )


# =====================================================================
# Calibrated noise lookup
# =====================================================================

def get_noise_multiplier(
    *,
    target_epsilon: float,
    hospital: str,
) -> float:

    if target_epsilon not in NOISE_MULTIPLIERS:

        raise ValueError(
            "Unsupported target epsilon. "
            f"Received {target_epsilon}; "
            f"supported values are "
            f"{SUPPORTED_EPSILONS}."
        )

    hospital_table = NOISE_MULTIPLIERS[
        target_epsilon
    ]

    if hospital not in hospital_table:

        raise ValueError(
            f"No calibrated noise multiplier "
            f"for hospital {hospital}."
        )

    return float(
        hospital_table[
            hospital
        ]
    )


# =====================================================================
# Client training
# =====================================================================

@app.train()
def train(
    msg: Message,
    context: Context,
) -> Message:

    # =================================================================
    # Training configuration received from ServerApp
    # =================================================================

    config = msg.content[
        "config"
    ]

    server_round = int(
        config[
            "server-round"
        ]
    )

    learning_rate = float(
        config[
            "learning-rate"
        ]
    )

    local_epochs = int(
        config[
            "local-epochs"
        ]
    )

    batch_size = int(
        config[
            "batch-size"
        ]
    )

    clipping_norm = float(
        config[
            "clipping-norm"
        ]
    )

    target_epsilon = float(
        config[
            "target-epsilon"
        ]
    )

    delta = float(
        config[
            "delta"
        ]
    )

    # =================================================================
    # Accounting-contract validation
    # =================================================================
    #
    # The sigma table above is valid only for the frozen accounting
    # configuration. Fail loudly rather than accidentally claiming a
    # privacy guarantee for a different training configuration.
    # =================================================================

    if batch_size != ACCOUNTING_BATCH_SIZE:

        raise ValueError(
            "The calibrated patient-DP noise multipliers "
            f"require batch-size={ACCOUNTING_BATCH_SIZE}, "
            f"but received {batch_size}."
        )

    if local_epochs != ACCOUNTING_LOCAL_EPOCHS:

        raise ValueError(
            "The calibrated patient-DP noise multipliers "
            f"require local-epochs={ACCOUNTING_LOCAL_EPOCHS}, "
            f"but received {local_epochs}."
        )

    if not math.isclose(
        delta,
        ACCOUNTING_DELTA,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):

        raise ValueError(
            "The calibrated patient-DP noise multipliers "
            f"require delta={ACCOUNTING_DELTA}, "
            f"but received {delta}."
        )

    if clipping_norm <= 0:

        raise ValueError(
            f"clipping-norm must be > 0, "
            f"got {clipping_norm}"
        )

    # =================================================================
    # Load patient metadata before training.
    # =================================================================
    #
    # This uses the same cached loader used by the DP task, so we do not
    # rely on an assumed Flower partition-id -> hospital mapping.
    # =================================================================

    patient_info = (
        load_patient_grouped_training_data(
            context
        )
    )

    hospital = str(
        patient_info[
            "hospital"
        ]
    )

    strategy_name = str(
        patient_info[
            "strategy"
        ]
    )

    # Verify that the observed training cohort matches the frozen
    # patient counts used for the precomputed hospital noise multipliers.
    require_calibrated_patient_count(
        hospital=hospital,
        n_patients=int(patient_info["n_patients"]),
    )

    # =================================================================
    # Select hospital-specific calibrated sigma.
    # =================================================================

    noise_multiplier = (
        get_noise_multiplier(
            target_epsilon=target_epsilon,
            hospital=hospital,
        )
    )

    # =================================================================
    # Device
    # =================================================================

    device = torch.device(
        "cpu"
    )

    # =================================================================
    # Frozen logistic-regression model
    # =================================================================

    model = LogisticRegressionModel(
        n_features=351
    )

    model.load_state_dict(
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )

    model.to(
        device
    )

    # =================================================================
    # Patient-level DP training
    # =================================================================

    result = train_patient_dp_poisson(
        model=model,
        context=context,
        learning_rate=learning_rate,
        nominal_patient_batch_size=batch_size,
        clipping_norm=clipping_norm,
        noise_multiplier=noise_multiplier,
        server_round=server_round,
        local_epochs=local_epochs,
        device=device,
    )

    # =================================================================
    # Consistency checks
    # =================================================================

    if result["hospital"] != hospital:

        raise RuntimeError(
            "Hospital changed between metadata loading "
            "and DP training."
        )

    if result["strategy"] != strategy_name:

        raise RuntimeError(
            "Training strategy changed between metadata "
            "loading and DP training."
        )

    # =================================================================
    # Human-readable strategy label
    # =================================================================

    if strategy_name == "strategy_a":

        strategy_label = "A"

    elif strategy_name == "strategy_b_k5":

        strategy_label = "B5"

    else:

        raise RuntimeError(
            "Unexpected strategy returned by "
            f"patient-DP training: {strategy_name}"
        )

    # =================================================================
    # Console diagnostics
    # =================================================================

    print(
        f"[Round {server_round:02d}] "
        f"PATIENT-DP {strategy_label} | "
        f"{hospital}: "
        f"{result['n_patients']:,} patients / "
        f"{result['n_stays']:,} stays | "
        f"eps_target={target_epsilon:.1f} | "
        f"delta={delta:.1e} | "
        f"C={result['clipping_norm']:.4f} | "
        f"sigma={result['noise_multiplier']:.6f} | "
        f"noise_std_sum="
        f"{result['noise_std_sum']:.6f} | "
        f"noise_std_avg="
        f"{result['noise_std_normalized']:.6f} | "
        f"q={result['sample_rate']:.6f} | "
        f"steps={result['steps']} | "
        f"expected_batch="
        f"{result['expected_patient_batch_size']:.2f} | "
        f"mean_batch="
        f"{result['realized_mean_patient_batch']:.2f} | "
        f"mean_grad_norm="
        f"{result['mean_gradient_norm']:.4f} | "
        f"max_grad_norm="
        f"{result['max_gradient_norm']:.4f} | "
        f"clip_fraction="
        f"{result['mean_clip_fraction']:.4f} | "
        f"mean_clip_factor="
        f"{result['mean_clip_factor']:.4f}"
    )

    # =================================================================
    # Return updated model and DP diagnostics
    # =================================================================
    #
    # IMPORTANT:
    #
    # num-examples remains PATIENT count.
    #
    # Therefore Flower FedAvg weights hospitals by the number of
    # training patients, not the number of retained stays.
    # =================================================================

    return Message(
        content=RecordDict(
            {
                "arrays":
                    ArrayRecord(
                        model.state_dict()
                    ),

                "metrics":
                    MetricRecord(
                        {
                            "num-examples":
                                int(
                                    result[
                                        "n_patients"
                                    ]
                                ),

                            "target_epsilon":
                                float(
                                    target_epsilon
                                ),

                            "delta":
                                float(
                                    delta
                                ),

                            "clipping_norm":
                                float(
                                    result[
                                        "clipping_norm"
                                    ]
                                ),

                            "noise_multiplier":
                                float(
                                    result[
                                        "noise_multiplier"
                                    ]
                                ),

                            "noise_std_sum":
                                float(
                                    result[
                                        "noise_std_sum"
                                    ]
                                ),

                            "noise_std_normalized":
                                float(
                                    result[
                                        "noise_std_normalized"
                                    ]
                                ),

                            "sample_rate":
                                float(
                                    result[
                                        "sample_rate"
                                    ]
                                ),

                            "expected_patient_batch":
                                float(
                                    result[
                                        "expected_patient_batch_size"
                                    ]
                                ),

                            "mean_patient_batch":
                                float(
                                    result[
                                        "realized_mean_patient_batch"
                                    ]
                                ),

                            "mean_gradient_norm":
                                float(
                                    result[
                                        "mean_gradient_norm"
                                    ]
                                ),

                            "max_gradient_norm":
                                float(
                                    result[
                                        "max_gradient_norm"
                                    ]
                                ),

                            "clip_fraction":
                                float(
                                    result[
                                        "mean_clip_fraction"
                                    ]
                                ),

                            "mean_clip_factor":
                                float(
                                    result[
                                        "mean_clip_factor"
                                    ]
                                ),

                            "empty_batches":
                                int(
                                    result[
                                        "empty_batches"
                                    ]
                                ),
                        }
                    ),
            }
        ),
        reply_to=msg,
    )