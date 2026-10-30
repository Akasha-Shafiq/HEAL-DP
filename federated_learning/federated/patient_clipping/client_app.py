from __future__ import annotations

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

from federated_learning.federated.patient_clipping.task import (
    train_patient_clipped_poisson,
)


app = ClientApp()


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
    # Patient-level clipping-only training
    # =================================================================

    result = train_patient_clipped_poisson(
        model=model,
        context=context,
        learning_rate=learning_rate,
        nominal_patient_batch_size=batch_size,
        clipping_norm=clipping_norm,
        server_round=server_round,
        local_epochs=local_epochs,
        device=device,
    )


    # =================================================================
    # Human-readable strategy label
    # =================================================================

    hospital = result[
        "hospital"
    ]


    strategy_name = result[
        "strategy"
    ]


    if strategy_name == "strategy_a":

        strategy_label = "A"


    elif strategy_name == "strategy_b_k5":

        strategy_label = "B5"


    else:

        raise RuntimeError(
            f"Unexpected strategy returned by clipping training: "
            f"{strategy_name}"
        )


    # =================================================================
    # Console diagnostics
    # =================================================================

    print(
        f"[Round {server_round:02d}] "
        f"CLIP-ONLY {strategy_label} | "
        f"{hospital}: "
        f"{result['n_patients']:,} patients / "
        f"{result['n_stays']:,} stays | "
        f"C={result['clipping_norm']:.4f} | "
        f"q={result['sample_rate']:.6f} | "
        f"steps={result['steps']} | "
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
    # Return updated model and diagnostics
    # =================================================================
    #
    # "num-examples" remains the PATIENT count.
    #
    # Therefore Flower FedAvg continues to weight hospitals by the
    # number of training patients rather than by retained stay count.
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

                            "clipping_norm":
                                float(
                                    result[
                                        "clipping_norm"
                                    ]
                                ),

                            "sample_rate":
                                float(
                                    result[
                                        "sample_rate"
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