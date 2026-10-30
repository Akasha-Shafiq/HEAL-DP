"""
F1 Flower ServerApp.

Baseline:
    - 5 hospitals
    - all 5 participate every round
    - 30 FedAvg rounds
    - 1 local epoch per round
    - sample-size weighted FedAvg
    - no differential privacy
    - no client dropout
    - server-side global held-out evaluation after each round
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
)

from flwr.serverapp import (
    Grid,
    ServerApp,
)

from flwr.serverapp.strategy import (
    FedAvg,
)

from federated_learning.models import (
    LogisticRegressionModel,
)

from federated_learning.evaluation import (
    binary_metrics,
    evaluate_groups,
)

from federated_learning.federated.task import (
    load_global_test,
    predict_model,
    set_seed,
)


app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:

    # ================================================================
    # Run configuration
    # ================================================================

    project_root = Path(
        str(
            context.run_config[
                "project-root"
            ]
        )
    )


    num_rounds = int(
        context.run_config[
            "num-server-rounds"
        ]
    )

    local_epochs = int(
        context.run_config[
            "local-epochs"
        ]
    )

    batch_size = int(
        context.run_config[
            "batch-size"
        ]
    )

    learning_rate = float(
        context.run_config[
            "learning-rate"
        ]
    )

    seed = int(
        context.run_config[
            "seed"
        ]
    )


    set_seed(
        seed
    )


    device = torch.device(
        "cpu"
    )


    # ================================================================
    # Results directory
    # ================================================================

    base_results_dir = (
        project_root
        / "results"
        / "federated_learning"
        / "f1_federated"
    )


    # Keep smoke tests separate from the final 30-round experiment.

    if num_rounds == 30:

        results_dir = (
            base_results_dir
        )

    else:

        results_dir = (
            base_results_dir
            / f"smoke_{num_rounds}_rounds"
        )


    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ================================================================
    # Frozen global test population
    # ================================================================

    (
        test_data,
        X_test,
        y_test,
        feature_columns,
    ) = load_global_test(
        project_root
    )


    if len(
        test_data
    ) != 14979:

        raise RuntimeError(
            "Unexpected F1 test-row count."
        )


    if test_data[
        "subject_id"
    ].nunique() != 10864:

        raise RuntimeError(
            "Unexpected F1 test-patient count."
        )


    # ================================================================
    # Initial global model
    # ================================================================
    #
    # The seed is fixed before construction so F1 starts from the
    # same reproducible initialization scheme used by F0.
    #

    global_model = (
        LogisticRegressionModel(
            n_features=len(
                feature_columns
            )
        )
    )


    initial_arrays = (
        ArrayRecord(
            global_model.state_dict()
        )
    )


    # ================================================================
    # Server-side evaluation after every round
    # ================================================================

    round_history = []


    def global_evaluate(
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:

        model = LogisticRegressionModel(
            n_features=len(
                feature_columns
            )
        )


        model.load_state_dict(
            arrays.to_torch_state_dict()
        )


        model.to(
            device
        )


        probabilities = predict_model(
            model=model,
            X=X_test,
            device=device,
        )


        predictions = (
            probabilities
            > 0.5
        ).astype(
            int
        )


        metrics = binary_metrics(
            y_true=y_test,
            probabilities=probabilities,
            predictions=predictions,
        )


        row = {
            "round":
                int(
                    server_round
                ),
            **metrics,
        }


        round_history.append(
            row
        )


        print(
            f"[Global round {server_round:02d}] "
            f"ROC-AUC={metrics['roc_auc']:.5f} "
            f"PR-AUC={metrics['pr_auc']:.5f} "
            f"Accuracy={metrics['accuracy']:.5f} "
            f"Recall={metrics['long_recall']:.5f}"
        )


        return MetricRecord(
            {
                "num-examples":
                    int(
                        len(
                            y_test
                        )
                    ),

                "roc_auc":
                    float(
                        metrics[
                            "roc_auc"
                        ]
                    ),

                "pr_auc":
                    float(
                        metrics[
                            "pr_auc"
                        ]
                    ),

                "accuracy":
                    float(
                        metrics[
                            "accuracy"
                        ]
                    ),

                "long_recall":
                    float(
                        metrics[
                            "long_recall"
                        ]
                    ),

                "long_f1":
                    float(
                        metrics[
                            "long_f1"
                        ]
                    ),

                "FPR":
                    float(
                        metrics[
                            "FPR"
                        ]
                    ),
            }
        )


    # ================================================================
    # FedAvg
    # ================================================================

    strategy = FedAvg(
        fraction_train=1.0,

        # We evaluate centrally on the fixed union of all local
        # held-out sets after every round.
        fraction_evaluate=0.0,

        min_train_nodes=5,
        min_available_nodes=5,

        weighted_by_key=(
            "num-examples"
        ),
    )


    train_config = ConfigRecord(
        {
            "learning-rate":
                learning_rate,

            "local-epochs":
                local_epochs,

            "batch-size":
                batch_size,
        }
    )


    # ================================================================
    # Run federated training
    # ================================================================

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )


    # ================================================================
    # Save final global model
    # ================================================================

    final_model = (
        LogisticRegressionModel(
            n_features=len(
                feature_columns
            )
        )
    )


    final_model.load_state_dict(
        result.arrays.to_torch_state_dict()
    )


    final_model.to(
        device
    )


    torch.save(
        {
            "model_state_dict":
                final_model.state_dict(),

            "model_type":
                "logistic_regression",

            "experiment":
                "F1",

            "architecture":
                "FedAvg",

            "num_rounds":
                num_rounds,

            "local_epochs":
                local_epochs,

            "batch_size":
                batch_size,

            "learning_rate":
                learning_rate,

            "seed":
                seed,

            "feature_columns":
                feature_columns,

            "classification_threshold":
                0.5,
        },
        results_dir
        / "f1_model.pt",
    )


    # ================================================================
    # Save round-level global metrics
    # ================================================================

    round_history_df = (
        pd.DataFrame(
            round_history
        )
    )


    round_history_df.to_csv(
        results_dir
        / "f1_round_global_metrics.csv",
        index=False,
    )


    # ================================================================
    # Save aggregated client training history
    # ================================================================

    client_train_rows = []


    for (
        round_number,
        metric_record,
    ) in result.train_metrics_clientapp.items():

        row = {
            "round":
                int(
                    round_number
                )
        }

        for (
            key,
            value,
        ) in metric_record.items():

            row[
                key
            ] = value

        client_train_rows.append(
            row
        )


    pd.DataFrame(
        client_train_rows
    ).to_csv(
        results_dir
        / "f1_aggregated_train_metrics.csv",
        index=False,
    )


    # ================================================================
    # Final predictions
    # ================================================================

    probabilities = predict_model(
        model=final_model,
        X=X_test,
        device=device,
    )


    predictions = (
        probabilities
        > 0.5
    ).astype(
        int
    )


    final_metrics = binary_metrics(
        y_true=y_test,
        probabilities=probabilities,
        predictions=predictions,
    )


    # ================================================================
    # Prediction table
    # ================================================================

    prediction_columns = [
        "subject_id",
        "stay_id",
        "hospital",
        "race",
        "insurance",
        "los",
    ]


    predictions_df = (
        test_data[
            prediction_columns
        ]
        .copy()
    )


    predictions_df[
        "y_true"
    ] = y_test.astype(
        int
    )


    predictions_df[
        "probability"
    ] = probabilities


    predictions_df[
        "y_pred"
    ] = predictions


    predictions_df.to_csv(
        results_dir
        / "f1_test_predictions.csv.gz",
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )


    # ================================================================
    # Global metrics
    # ================================================================

    global_metrics_df = (
        pd.DataFrame(
            [
                {
                    "model":
                        "F1_FedAvg_Logistic_Regression",

                    "rounds":
                        num_rounds,

                    "local_epochs":
                        local_epochs,

                    "batch_size":
                        batch_size,

                    "learning_rate":
                        learning_rate,

                    "classification_threshold":
                        0.5,

                    **final_metrics,
                }
            ]
        )
    )


    global_metrics_df.to_csv(
        results_dir
        / "f1_global_metrics.csv",
        index=False,
    )


    # ================================================================
    # Hospital metrics
    # ================================================================

    hospital_metrics = (
        evaluate_groups(
            predictions_df,
            [
                "hospital",
            ],
        )
    )


    hospital_metrics.to_csv(
        results_dir
        / "f1_hospital_metrics.csv",
        index=False,
    )


    # ================================================================
    # Race
    # ================================================================

    race_predictions = (
        predictions_df.copy()
    )


    race_predictions[
        "race"
    ] = (
        race_predictions[
            "race"
        ]
        .fillna(
            "<MISSING>"
        )
        .astype(str)
    )


    race_metrics = (
        evaluate_groups(
            race_predictions,
            [
                "race",
            ],
        )
    )


    race_metrics[
        "eligible_n_ge_100"
    ] = (
        race_metrics[
            "n"
        ]
        >= 100
    )


    race_metrics.to_csv(
        results_dir
        / "f1_race_metrics.csv",
        index=False,
    )


    # ================================================================
    # Insurance
    # ================================================================

    insurance_predictions = (
        predictions_df.copy()
    )


    insurance_predictions[
        "insurance"
    ] = (
        insurance_predictions[
            "insurance"
        ]
        .fillna(
            "<MISSING>"
        )
        .astype(str)
    )


    insurance_metrics = (
        evaluate_groups(
            insurance_predictions,
            [
                "insurance",
            ],
        )
    )


    insurance_metrics[
        "eligible_n_ge_100"
    ] = (
        insurance_metrics[
            "n"
        ]
        >= 100
    )


    insurance_metrics.to_csv(
        results_dir
        / "f1_insurance_metrics.csv",
        index=False,
    )


    # ================================================================
    # Hospital x Race
    # ================================================================

    hospital_race_metrics = (
        evaluate_groups(
            race_predictions,
            [
                "hospital",
                "race",
            ],
        )
    )


    hospital_race_metrics[
        "eligible_n_ge_100"
    ] = (
        hospital_race_metrics[
            "n"
        ]
        >= 100
    )


    hospital_race_metrics.to_csv(
        results_dir
        / "f1_hospital_race_metrics.csv",
        index=False,
    )


    # ================================================================
    # Hospital x Insurance
    # ================================================================

    hospital_insurance_metrics = (
        evaluate_groups(
            insurance_predictions,
            [
                "hospital",
                "insurance",
            ],
        )
    )


    hospital_insurance_metrics[
        "eligible_n_ge_100"
    ] = (
        hospital_insurance_metrics[
            "n"
        ]
        >= 100
    )


    hospital_insurance_metrics.to_csv(
        results_dir
        / "f1_hospital_insurance_metrics.csv",
        index=False,
    )


    # ================================================================
    # Metadata
    # ================================================================

    metadata = {
        "experiment":
            "F1",

        "model":
            "PyTorch logistic regression",

        "aggregation":
            "FedAvg",

        "aggregation_weight":
            "local training examples",

        "num_clients":
            5,

        "client_participation":
            "all clients every round",

        "rounds":
            num_rounds,

        "local_epochs":
            local_epochs,

        "batch_size":
            batch_size,

        "learning_rate":
            learning_rate,

        "privacy":
            "none",

        "seed":
            seed,

        "training_unit":
            "ICU stay",

        "split_unit":
            "patient",

        "patient_disjoint_train_test":
            True,
    }


    with open(
        results_dir
        / "f1_experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )


    # ================================================================
    # Console summary
    # ================================================================

    print()
    print("=" * 80)
    print("F1 FINAL GLOBAL RESULTS")
    print("=" * 80)

    print(
        global_metrics_df[
            [
                "roc_auc",
                "pr_auc",
                "accuracy",
                "long_precision",
                "long_recall",
                "long_f1",
                "FPR",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ]
        .round(
            5
        )
        .to_string(
            index=False
        )
    )


    print()
    print("=" * 80)
    print("F1 FINAL HOSPITAL RESULTS")
    print("=" * 80)

    print(
        hospital_metrics[
            [
                "hospital",
                "n",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "long_recall",
                "FPR",
                "TN",
                "FP",
                "FN",
                "TP",
            ]
        ]
        .round(
            5
        )
        .to_string(
            index=False
        )
    )


    print()
    print("=" * 80)
    print("F1 COMPLETE")
    print("=" * 80)

    print(
        f"Results directory:\n"
        f"  {results_dir}"
    )