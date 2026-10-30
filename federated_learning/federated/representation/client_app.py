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

from federated_learning.federated.task import (
    train_local_model,
)

from federated_learning.federated.representation.task import (
    make_strategy_train_loader,
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


    (
        train_loader,
        hospital,
        n_examples,
        n_patients,
        n_features,
        strategy,
    ) = make_strategy_train_loader(
        context=context,
        batch_size=batch_size,
        server_round=server_round,
    )


    device = torch.device(
        "cpu"
    )


    model = LogisticRegressionModel(
        n_features=n_features
    )


    model.load_state_dict(
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )


    model.to(
        device
    )


    mean_loss = train_local_model(
        model=model,
        train_loader=train_loader,
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        device=device,
    )


    print(
        f"[Round {server_round:02d}] "
        f"{strategy} | "
        f"{hospital}: "
        f"{n_examples:,} examples / "
        f"{n_patients:,} patients, "
        f"loss={mean_loss:.6f}"
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
                                    mean_loss
                                ),

                            # Standard FedAvg weighting for this
                            # non-private representation experiment.
                            "num-examples":
                                int(
                                    n_examples
                                ),
                        }
                    ),
            }
        ),
        reply_to=msg,
    )