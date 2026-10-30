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

from federated_learning.federated.patient_control.task import (
    train_patient_grouped_poisson,
)


app = ClientApp()


@app.train()
def train(
    msg: Message,
    context: Context,
) -> Message:

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


    device = torch.device(
        "cpu"
    )


    # The feature dimension remains frozen at 351.
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


    result = train_patient_grouped_poisson(
        model=model,
        context=context,
        learning_rate=learning_rate,
        nominal_patient_batch_size=batch_size,
        server_round=server_round,
        local_epochs=local_epochs,
        device=device,
    )


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
            f"Unexpected strategy returned by training: "
            f"{strategy_name}"
        )


    print(
        f"[Round {server_round:02d}] "
        f"PATIENT-POISSON {strategy_label} | "
        f"{hospital}: "
        f"{result['n_patients']:,} patients / "
        f"{result['n_stays']:,} stays | "
        f"q={result['sample_rate']:.6f} | "
        f"steps={result['steps']} | "
        f"mean_batch={result['realized_mean_patient_batch']:.2f} | "
        f"batch_range="
        f"{result['realized_min_patient_batch']}-"
        f"{result['realized_max_patient_batch']} | "
        f"loss={result['train_loss']:.6f}"
    )


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
                            "train_loss":
                                float(
                                    result[
                                        "train_loss"
                                    ]
                                ),

                            # FedAvg is weighted by PATIENT COUNT,
                            # because the optimization/privacy unit
                            # is the patient.
                            "num-examples":
                                int(
                                    result[
                                        "n_patients"
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