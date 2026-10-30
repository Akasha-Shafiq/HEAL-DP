from __future__ import annotations

import json
from pathlib import Path

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

from federated_learning.federated.representation.task import (
    SUPPORTED_STRATEGIES,
)


app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:

    # =================================================================
    # Configuration
    # =================================================================

    project_root = Path(
        str(
            context.run_config[
                "project-root"
            ]
        )
    )


    strategy_name = str(
        context.run_config[
            "training-strategy"
        ]
    )


    if strategy_name not in SUPPORTED_STRATEGIES:

        raise ValueError(
            f"Unsupported strategy: {strategy_name}"
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


    clipping_norm = float(
        context.run_config[
            "clipping-norm"
        ]
    )


    if clipping_norm <= 0:

        raise ValueError(
            f"clipping-norm must be > 0, got {clipping_norm}"
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


    # =================================================================
    # Results directory
    # =================================================================

    clipping_tag = (
        f"C_{clipping_norm:.1f}"
        .replace(
            ".",
            "p",
        )
    )


    results_dir = (
        project_root
        / "results"
        / "federated_learning"
        / "patient_clipping"
        / strategy_name
        / clipping_tag
    )


    if num_rounds != 30:

        results_dir = (
            results_dir
            / f"smoke_{num_rounds}_rounds"
        )


    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # =================================================================
    # ORIGINAL frozen global test population
    # =================================================================

    (
        test_data,
        X_test,
        y_test,
        feature_columns,
    ) = load_global_test(
        project_root
    )


    assert len(
        test_data
    ) == 14979


    assert (
        test_data[
            "subject_id"
        ].nunique()
        == 10864
    )


    # =================================================================
    # Identical global initialization
    # =================================================================

    global_model = LogisticRegressionModel(
        n_features=len(
            feature_columns
        )
    )


    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )


    round_history = []


    # =================================================================
    # Server-side global evaluation
    # =================================================================

    def global_evaluate(
        server_round,
        arrays,
    ):

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
            model,
            X_test,
            device,
        )


        predictions = (
            probabilities
            > 0.5
        ).astype(
            int
        )


        metrics = binary_metrics(
            y_test,
            probabilities,
            predictions,
        )


        round_history.append(
            {
                "round":
                    int(
                        server_round
                    ),
                **metrics,
            }
        )


        print(
            f"[{strategy_name} | "
            f"CLIP-ONLY C={clipping_norm:g} | "
            f"Global round {server_round:02d}] "
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


    # =================================================================
    # FedAvg
    # =================================================================
    #
    # "num-examples" returned by ClientApp is PATIENT COUNT.
    #
    # Therefore aggregation remains patient-count weighted.
    # =================================================================

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_train_nodes=5,
        min_available_nodes=5,
        weighted_by_key="num-examples",
    )


    # =================================================================
    # Client training configuration
    # =================================================================

    train_config = ConfigRecord(
        {
            "learning-rate":
                learning_rate,

            "local-epochs":
                local_epochs,

            "batch-size":
                batch_size,

            "clipping-norm":
                clipping_norm,
        }
    )


    # =================================================================
    # Federated training
    # =================================================================

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )


    # =================================================================
    # Final model
    # =================================================================

    final_model = LogisticRegressionModel(
        n_features=len(
            feature_columns
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

            "training_strategy":
                strategy_name,

            "experiment_type":
                "patient_gradient_clipping_control",

            "rounds":
                num_rounds,

            "local_epochs":
                local_epochs,

            "batch_size":
                batch_size,

            "learning_rate":
                learning_rate,

            "clipping_norm":
                clipping_norm,

            "noise_multiplier":
                0.0,

            "differential_privacy":
                False,

            "seed":
                seed,

            "feature_columns":
                feature_columns,
        },
        results_dir
        / "model.pt",
    )


    # =================================================================
    # Round-wise global metrics
    # =================================================================

    pd.DataFrame(
        round_history
    ).to_csv(
        results_dir
        / "round_global_metrics.csv",
        index=False,
    )


    # =================================================================
    # Final predictions
    # =================================================================

    probabilities = predict_model(
        final_model,
        X_test,
        device,
    )


    predictions = (
        probabilities
        > 0.5
    ).astype(
        int
    )


    final_metrics = binary_metrics(
        y_test,
        probabilities,
        predictions,
    )


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
        / "test_predictions.csv.gz",
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )


    # =================================================================
    # Global metrics
    # =================================================================

    global_df = pd.DataFrame(
        [
            {
                "training_strategy":
                    strategy_name,

                "experiment_type":
                    "patient_gradient_clipping_control",

                "rounds":
                    num_rounds,

                "local_epochs":
                    local_epochs,

                "batch_size":
                    batch_size,

                "learning_rate":
                    learning_rate,

                "clipping_norm":
                    clipping_norm,

                "noise_multiplier":
                    0.0,

                "differential_privacy":
                    False,

                **final_metrics,
            }
        ]
    )


    global_df.to_csv(
        results_dir
        / "global_metrics.csv",
        index=False,
    )


    # =================================================================
    # Hospital metrics
    # =================================================================

    hospital_metrics = evaluate_groups(
        predictions_df,
        [
            "hospital",
        ],
    )


    hospital_metrics.to_csv(
        results_dir
        / "hospital_metrics.csv",
        index=False,
    )


    # =================================================================
    # Race metrics
    # =================================================================

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


    race_metrics = evaluate_groups(
        race_predictions,
        [
            "race",
        ],
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
        / "race_metrics.csv",
        index=False,
    )


    # =================================================================
    # Insurance metrics
    # =================================================================

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


    insurance_metrics = evaluate_groups(
        insurance_predictions,
        [
            "insurance",
        ],
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
        / "insurance_metrics.csv",
        index=False,
    )


    # =================================================================
    # Hospital x Race metrics
    # =================================================================

    hospital_race = evaluate_groups(
        race_predictions,
        [
            "hospital",
            "race",
        ],
    )


    hospital_race[
        "eligible_n_ge_100"
    ] = (
        hospital_race[
            "n"
        ]
        >= 100
    )


    hospital_race.to_csv(
        results_dir
        / "hospital_race_metrics.csv",
        index=False,
    )


    # =================================================================
    # Hospital x Insurance metrics
    # =================================================================

    hospital_insurance = evaluate_groups(
        insurance_predictions,
        [
            "hospital",
            "insurance",
        ],
    )


    hospital_insurance[
        "eligible_n_ge_100"
    ] = (
        hospital_insurance[
            "n"
        ]
        >= 100
    )


    hospital_insurance.to_csv(
        results_dir
        / "hospital_insurance_metrics.csv",
        index=False,
    )


    # =================================================================
    # Metadata
    # =================================================================

    metadata = {
        "experiment":
            "patient_gradient_clipping_control",

        "experiment_type":
            "patient_gradient_clipping_control",

        "training_strategy":
            strategy_name,

        "aggregation":
            "patient-count weighted FedAvg",

        "num_clients":
            5,

        "all_clients_every_round":
            True,

        "rounds":
            num_rounds,

        "local_epochs":
            local_epochs,

        "batch_size":
            batch_size,

        "learning_rate":
            learning_rate,

        "clipping_norm":
            clipping_norm,

        "noise_multiplier":
            0.0,

        "differential_privacy":
            False,

        "privacy":
            "none",

        "privacy_unit":
            "subject_id",

        "sampling_unit":
            "patient",

        "sampling_method":
            "Poisson",

        "seed":
            seed,

        "test_rows":
            14979,

        "test_patients":
            10864,

        "test_population":
            "original frozen H1-H5 held-out test union",
    }


    with open(
        results_dir
        / "experiment_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
        )


    # =================================================================
    # Console output
    # =================================================================

    print()
    print("=" * 80)

    print(
        f"{strategy_name.upper()} "
        f"CLIP-ONLY C={clipping_norm:g} "
        f"FINAL GLOBAL RESULTS"
    )

    print("=" * 80)


    print(
        global_df[
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

    print(
        f"{strategy_name.upper()} "
        f"CLIP-ONLY C={clipping_norm:g} "
        f"HOSPITAL RESULTS"
    )

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

    print(
        f"Results saved to:\n"
        f"  {results_dir}"
    )