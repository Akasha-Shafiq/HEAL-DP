"""
F1 Flower ClientApp.

Each simulated SuperNode corresponds to one frozen hospital:

    partition 0 -> H1
    partition 1 -> H2
    partition 2 -> H3
    partition 3 -> H4
    partition 4 -> H5

Each hospital performs one local SGD epoch per federated round.
"""

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
    make_train_loader,
    train_local_model,
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
        n_features,
    ) = make_train_loader(
        context=context,
        batch_size=batch_size,
        server_round=server_round,
    )


    # CPU keeps the first F1 experiment simple and matches
    # the CPU F0 execution on this machine.

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
        f"{hospital}: "
        f"{n_examples:,} examples, "
        f"loss={mean_loss:.6f}"
    )


    arrays = ArrayRecord(
        model.state_dict()
    )


    metrics = MetricRecord(
        {
            "train_loss":
                float(
                    mean_loss
                ),

            # Flower FedAvg uses this key as the
            # parameter-aggregation weight.
            "num-examples":
                int(
                    n_examples
                ),
        }
    )


    content = RecordDict(
        {
            "arrays":
                arrays,

            "metrics":
                metrics,
        }
    )


    return Message(
        content=content,
        reply_to=msg,
    )