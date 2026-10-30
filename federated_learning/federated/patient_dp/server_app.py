from __future__ import annotations



import json

import math

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





# =====================================================================

# Frozen patient-level DP accounting configuration

# =====================================================================



ACCOUNTING_METHOD = "RDP"



ACCOUNTING_DELTA = 1e-5



ACCOUNTING_ROUNDS = 30



ACCOUNTING_LOCAL_EPOCHS = 1



ACCOUNTING_BATCH_SIZE = 256





# ---------------------------------------------------------------------

# Hospital-specific noise multipliers calibrated previously for:

#

#     patient-level add/remove adjacency

#     Poisson patient sampling

#     30 Flower rounds

#     1 local epoch / round

#     nominal patient batch size = 256

#     delta = 1e-5

#

# H1-H3:

#     N = 8,690

#     steps/round = 34

#     q = 1/34

#

# H4:

#     N = 2,172

#     steps/round = 9

#     q = 1/9

#

# H5:

#     N = 15,208

#     steps/round = 60

#     q = 1/60

# ---------------------------------------------------------------------



CALIBRATED_NOISE_MULTIPLIERS = {



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

        CALIBRATED_NOISE_MULTIPLIERS.keys()

    )

)





# =====================================================================

# Utility

# =====================================================================



def value_tag(

    value: float,

) -> str:



    return (

        f"{value:.1f}"

        .replace(

            ".",

            "p",

        )

    )





# =====================================================================

# Server

# =====================================================================



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



    target_epsilon = float(

        context.run_config[

            "target-epsilon"

        ]

    )



    delta = float(

        context.run_config[

            "delta"

        ]

    )



    seed = int(

        context.run_config[

            "seed"

        ]

    )



    replicate_id = int(

        context.run_config[

            "replicate-id"

        ]

    )



    if replicate_id < 1:

        raise ValueError(

            "replicate-id must be >= 1, "

            f"got {replicate_id}"

        )



    # =================================================================

    # Validate DP accounting contract

    # =================================================================



    if clipping_norm <= 0:



        raise ValueError(

            f"clipping-norm must be > 0, "

            f"got {clipping_norm}"

        )



    if target_epsilon not in CALIBRATED_NOISE_MULTIPLIERS:



        raise ValueError(

            "Unsupported target epsilon. "

            f"Received {target_epsilon}; "

            f"supported values are "

            f"{SUPPORTED_EPSILONS}."

        )



    if not math.isclose(

        delta,

        ACCOUNTING_DELTA,

        rel_tol=0.0,

        abs_tol=1e-15,

    ):



        raise ValueError(

            "The frozen accounting calibration requires "

            f"delta={ACCOUNTING_DELTA}, "

            f"but received {delta}."

        )



    if batch_size != ACCOUNTING_BATCH_SIZE:



        raise ValueError(

            "The frozen accounting calibration requires "

            f"batch-size={ACCOUNTING_BATCH_SIZE}, "

            f"but received {batch_size}."

        )



    if local_epochs != ACCOUNTING_LOCAL_EPOCHS:



        raise ValueError(

            "The frozen accounting calibration requires "

            f"local-epochs={ACCOUNTING_LOCAL_EPOCHS}, "

            f"but received {local_epochs}."

        )



    if num_rounds <= 0:



        raise ValueError(

            "num-server-rounds must be positive."

        )



    # ---------------------------------------------------------------

    # The current sigma table was calibrated for at most 30 rounds.

    #

    # Fewer rounds are allowed for smoke/validation runs.

    # More than 30 rounds would exceed the calibrated composition

    # and therefore must not be silently permitted.

    # ---------------------------------------------------------------



    if num_rounds > ACCOUNTING_ROUNDS:



        raise ValueError(

            "This noise calibration supports at most "

            f"{ACCOUNTING_ROUNDS} rounds, "

            f"but received {num_rounds}. "

            "Recalibrate sigma before running more rounds."

        )



    full_accounting_run = (

        num_rounds

        == ACCOUNTING_ROUNDS

    )



    # ---------------------------------------------------------------

    # For a 30-round run, target_epsilon is the calibrated cumulative

    # epsilon target.

    #

    # For a shorter smoke run, the same target_epsilon identifies the

    # sigma table being used, but we deliberately DO NOT report the

    # smoke run as having consumed that full epsilon.

    # ---------------------------------------------------------------



    reported_total_epsilon = (

        float(

            target_epsilon

        )

        if full_accounting_run

        else None

    )



    reported_total_delta = (

        float(

            delta

        )

        if full_accounting_run

        else None

    )



    # The 30-round target remains a valid loose upper bound for runs

    # using fewer than 30 composed DP steps/round blocks.

    privacy_upper_bound_epsilon = float(

        target_epsilon

    )



    privacy_upper_bound_delta = float(

        delta

    )



    selected_sigma_table = (

        CALIBRATED_NOISE_MULTIPLIERS[

            target_epsilon

        ]

    )



    # =================================================================

    # Public experimental initialization

    # =================================================================

    #

    # This seed controls model initialization / ordinary experiment

    # reproducibility only.

    #

    # DP sampling and Gaussian-noise randomness are generated privately

    # inside ClientApp/task.py and are NOT derived from this seed.

    # =================================================================



    set_seed(

        seed

    )



    device = torch.device(

        "cpu"

    )



    # =================================================================

    # Results directory

    # =================================================================



    epsilon_tag = (

        "eps_"

        + value_tag(

            target_epsilon

        )

    )



    clipping_tag = (

        "C_"

        + value_tag(

            clipping_norm

        )

    )



    results_dir = (

        project_root

        / "results"

        / "federated_learning"

        / "patient_dp"

        / strategy_name

        / epsilon_tag

        / clipping_tag

    )



    # ---------------------------------------------------------------

    # Replicate handling

    #

    # Existing full experiments are treated as replicate 1 and remain

    # in their original configuration directory.

    #

    # Additional independent repetitions are stored under rep_2,

    # rep_3, ... so validated first-run outputs are not overwritten.

    # ---------------------------------------------------------------



    if not full_accounting_run:



        results_dir = (

            results_dir

            / f"smoke_{num_rounds}_rounds"

        )



    if replicate_id > 1:



        results_dir = (

            results_dir

            / f"rep_{replicate_id}"

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



                "replicate_id":

                    replicate_id,



                "target_epsilon_30_rounds":

                    float(

                        target_epsilon

                    ),



                "delta":

                    float(

                        delta

                    ),



                "clipping_norm":

                    float(

                        clipping_norm

                    ),



                **metrics,

            }

        )



        print(

            f"[{strategy_name} | "

            f"PATIENT-DP | "

            f"rep={replicate_id} | "

            f"eps_target={target_epsilon:g} | "

            f"delta={delta:.1e} | "

            f"C={clipping_norm:g} | "

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

    # ClientApp returns:

    #

    #     num-examples = training PATIENT count

    #

    # Therefore hospital aggregation remains patient-count weighted.

    #

    # Each hospital performs its own patient-level DP mechanism before

    # its model parameters are returned. FedAvg is post-processing of

    # those DP client outputs.

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



            "target-epsilon":

                target_epsilon,



            "delta":

                delta,

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

                "patient_level_dp",



            "replicate_id":

                replicate_id,



            "differential_privacy":

                True,



            "privacy_unit":

                "subject_id",



            "adjacency":

                "add_remove_one_patient",



            "accountant":

                ACCOUNTING_METHOD,



            "target_epsilon_30_rounds":

                target_epsilon,



            "delta":

                delta,



            "reported_total_epsilon":

                reported_total_epsilon,



            "reported_total_delta":

                reported_total_delta,



            "privacy_accounting_complete":

                full_accounting_run,



            "privacy_upper_bound_epsilon":

                privacy_upper_bound_epsilon,



            "privacy_upper_bound_delta":

                privacy_upper_bound_delta,



            "rounds":

                num_rounds,



            "accounting_rounds":

                ACCOUNTING_ROUNDS,



            "local_epochs":

                local_epochs,



            "batch_size":

                batch_size,



            "learning_rate":

                learning_rate,



            "clipping_norm":

                clipping_norm,



            "noise_multipliers_by_hospital":

                selected_sigma_table,



            "public_experiment_seed":

                seed,



            "private_randomness":

                "fresh_client_side_sampling_and_gaussian_noise",



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

                    "patient_level_dp",



                "replicate_id":

                    replicate_id,



                "differential_privacy":

                    True,



                "privacy_unit":

                    "subject_id",



                "adjacency":

                    "add_remove_one_patient",



                "accountant":

                    ACCOUNTING_METHOD,



                "target_epsilon_30_rounds":

                    target_epsilon,



                "delta":

                    delta,



                "reported_total_epsilon":

                    reported_total_epsilon,



                "reported_total_delta":

                    reported_total_delta,



                "privacy_accounting_complete":

                    full_accounting_run,



                "rounds":

                    num_rounds,



                "accounting_rounds":

                    ACCOUNTING_ROUNDS,



                "local_epochs":

                    local_epochs,



                "batch_size":

                    batch_size,



                "learning_rate":

                    learning_rate,



                "clipping_norm":

                    clipping_norm,



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

        .astype(

            str

        )

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

        .astype(

            str

        )

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

    # Privacy calibration table for this run

    # =================================================================



    calibration_rows = []



    hospital_accounting = {

        "H1": {

            "n_patients": 8690,

            "steps_per_round": 34,

        },

        "H2": {

            "n_patients": 8690,

            "steps_per_round": 34,

        },

        "H3": {

            "n_patients": 8690,

            "steps_per_round": 34,

        },

        "H4": {

            "n_patients": 2172,

            "steps_per_round": 9,

        },

        "H5": {

            "n_patients": 15208,

            "steps_per_round": 60,

        },

    }



    for hospital, accounting_info in (

        hospital_accounting.items()

    ):



        n_patients = int(

            accounting_info[

                "n_patients"

            ]

        )



        steps_per_round = int(

            accounting_info[

                "steps_per_round"

            ]

        )



        q = (

            1.0

            / steps_per_round

        )



        sigma = float(

            selected_sigma_table[

                hospital

            ]

        )



        calibration_rows.append(

            {

                "hospital":

                    hospital,



                "replicate_id":

                    replicate_id,



                "n_patients":

                    n_patients,



                "steps_per_round":

                    steps_per_round,



                "sample_rate":

                    q,



                "expected_patient_batch":

                    n_patients * q,



                "target_epsilon_30_rounds":

                    target_epsilon,



                "delta":

                    delta,



                "noise_multiplier":

                    sigma,



                "clipping_norm":

                    clipping_norm,



                "gaussian_std_on_clipped_sum":

                    sigma

                    * clipping_norm,



                "configured_run_rounds":

                    num_rounds,



                "calibration_rounds":

                    ACCOUNTING_ROUNDS,



                "privacy_accounting_complete":

                    full_accounting_run,

            }

        )



    pd.DataFrame(

        calibration_rows

    ).to_csv(

        results_dir

        / "privacy_calibration.csv",

        index=False,

    )



    # =================================================================

    # Metadata

    # =================================================================



    if full_accounting_run:



        accounting_status = (

            "complete_30_round_calibrated_run"

        )



    else:



        accounting_status = (

            "smoke_or_partial_run_"

            "using_30_round_calibrated_sigma;_"

            "exact_partial_run_epsilon_not_reported"

        )



    metadata = {

        "experiment":

            "patient_level_dp",



        "experiment_type":

            "patient_level_dp",



        "replicate_id":

            replicate_id,



        "training_strategy":

            strategy_name,



        "differential_privacy":

            True,



        "privacy_unit":

            "subject_id",



        "adjacency":

            "add_remove_one_patient",



        "sampling_unit":

            "patient",



        "sampling_method":

            "Poisson",



        "patient_gradient_definition":

            (

                "mean retained-stay loss gradient "

                "within each patient, then one L2 clip"

            ),



        "aggregation":

            "patient-count weighted FedAvg",



        "num_clients":

            5,



        "all_clients_every_round":

            True,



        "accountant":

            ACCOUNTING_METHOD,



        "target_epsilon_30_rounds":

            target_epsilon,



        "delta":

            delta,



        "reported_total_epsilon":

            reported_total_epsilon,



        "reported_total_delta":

            reported_total_delta,



        "privacy_upper_bound_epsilon":

            privacy_upper_bound_epsilon,



        "privacy_upper_bound_delta":

            privacy_upper_bound_delta,



        "privacy_accounting_complete":

            full_accounting_run,



        "accounting_status":

            accounting_status,



        "rounds":

            num_rounds,



        "accounting_rounds":

            ACCOUNTING_ROUNDS,



        "local_epochs":

            local_epochs,



        "batch_size":

            batch_size,



        "learning_rate":

            learning_rate,



        "clipping_norm":

            clipping_norm,



        "noise_multipliers_by_hospital":

            selected_sigma_table,



        "gaussian_noise_location":

            (

                "client-side clipped patient-gradient sum "

                "before expected-batch normalization"

            ),



        "normalization":

            "expected_poisson_patient_batch_N_times_q",



        "public_experiment_seed":

            seed,



        "private_randomness":

            (

                "fresh system-seeded client-side Poisson "

                "sampling and Gaussian noise"

            ),



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

        f"PATIENT-DP "

        f"REP={replicate_id} "

        f"EPS_TARGET={target_epsilon:g} "

        f"DELTA={delta:.1e} "

        f"C={clipping_norm:g} "

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

        f"PATIENT-DP "

        f"REP={replicate_id} "

        f"EPS_TARGET={target_epsilon:g} "

        f"DELTA={delta:.1e} "

        f"C={clipping_norm:g} "

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



    if full_accounting_run:



        print(

            "Privacy accounting: "

            f"30-round calibrated target "

            f"(epsilon={target_epsilon:g}, "

            f"delta={delta:.1e})."

        )



    else:



        print(

            "Privacy accounting: SMOKE/PARTIAL RUN. "

            f"The clients use sigma values calibrated for "

            f"the {ACCOUNTING_ROUNDS}-round "

            f"epsilon={target_epsilon:g}, "

            f"delta={delta:.1e} experiment. "

            "This run is not reported as having consumed "

            "the full target epsilon; exact partial-run "

            "epsilon is not computed here."

        )



    print()



    print(

        f"Results saved to:\n"

        f"  {results_dir}"

    )