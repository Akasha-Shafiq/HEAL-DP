# Privacy-Enhancing Technologies for Healthcare Analytics

**Research code for central differential privacy, homomorphic-encryption-assisted analytics, and patient-level differentially private federated learning.**

> **Research prototype — not a production healthcare service.** This repository is intended for controlled experimentation in appropriately governed environments. It does not include MIMIC-IV data, patient-level derived datasets, hospital partitions, trained model weights, cryptographic secret keys, or experimental outputs derived from restricted records. Independent data access and disclosure approval remain the researcher's responsibility.

## Overview

The project studies three complementary approaches for limiting information exposure during healthcare analytics:

1. **Central Differential Privacy (CDP):** Five hospital-node services calculate approved local statistics. A trusted coordinator combines the statistics and applies differential privacy before release, with query registration, privacy-budget accounting, and audit events.
2. **Homomorphic Encryption with Differential Privacy (HE + DP):** An experimental encrypted-aggregation pathway protects selected intermediate statistics from the aggregation service. BFV and CKKS support the query-specific arithmetic; a separate release service participates in the controlled decryption/DP workflow. The intended released statistics follow the same Q001–Q007 query definitions as the CDP reference implementation.
3. **Federated Learning with Patient-Level DP (FL + DP):** A fixed feature representation and patient-disjoint hospital partitions support controlled comparisons among a centralised logistic-regression baseline (F0), federated training without DP (F1), and federated training with patient-level clipping and calibrated Gaussian noise (F2).

These approaches address **different trust boundaries**. Central DP limits information disclosed by a trusted coordinator; HE protects specified values during aggregation; and patient-level DP-FL limits the influence of a patient's bounded contribution under the stated training and accounting assumptions. Combining technologies does not, by itself, establish end-to-end security or a privacy guarantee beyond those assumptions.

## Repository structure

```text
.
├── coordinator/                  # Central-DP coordinator, approved queries, budgeting, audit
├── node/                         # Hospital-node statistics and local query logic
├── he_pipeline/                  # Encrypted aggregation, keys, noise/release services, tests
├── federated_learning/           # Data preparation, F0, F1, F2, evaluations, accounting
│   ├── centralized/              # F0: non-private centralised reference
│   ├── federated/                # F1 and additional non-private research controls
│   │   ├── patient_dp/           # F2: patient-level DP training
│   │   ├── patient_clipping/     # Clipping-only control; NOT a DP experiment
│   │   ├── patient_control/      # Sampling/patient-control study; NOT a DP claim
│   │   └── representation/       # Patient-representation experiments
│   ├── dp/                       # Patient contribution and accounting studies
│   └── comparison/               # Cross-experiment comparison code
├── experiments/                  # CDP utility, robustness and comparison scripts
├── experiments_disclosure/       # Experimental Q008/Q009 disclosure-risk evaluation
├── modelling_baselines/
│   ├── A1_No_imputation_feature_reduction.ipynb
│   ├── A2_Laboratory_Feature_Reduction.ipynb
│   └── A3_Centralized_Baseline.ipynb
├── requirements-cdp.txt
├── requirements-he.txt
└── pyproject.toml                # FL packaging and Flower configuration
```


## Data access and handling

The code was developed for research involving MIMIC-IV-derived cohorts. **No data access is conferred by this repository.** Obtain the appropriate permissions and comply with the relevant data-use terms before obtaining or processing restricted data.


## Reproducing the modelling and FL experiments

### A1–A3: feature reduction and centralised modelling

The approved notebooks comprise medication feature reduction (A1), laboratory feature reduction (A2), and the centralised logistic-regression baseline (A3). Their private working directory is selected with `MIMIC_FAIRNESS_WORKDIR`.

The A1–A3 notebooks implement the feature-reduction and centralised logistic-regression experiments. They use a configurable local working directory to access the required datasets and intermediate preprocessing files.

Before executing the notebooks, specify the location of your research workspace using the MIMIC_FAIRNESS_WORKDIR environment variable.

The model predicts length of stay greater than four days using a 351-feature representation (193 binary features and 158 continuous features). The FL model follows the corresponding linear binary-classification design. The feature-order manifest and preprocessing specification form part of the required experimental inputs. The notebooks follow the data preparation and feature specifications used in the original study. Researchers should ensure that the required input files are available in the expected format.

### Fixed experimental artifacts and patient partitioning

To reduce unintended variation between F0, F1 and F2, data preparation is separated from model execution. The feature list, preprocessing specification, patient-to-hospital assignments, patient-disjoint training/test assignments, and materialised hospital partitions are **generated before training and held fixed within each controlled comparison**.

Relevant preparation and validation source includes:

- `federated_learning/create_hospital_assignment.py`
- `federated_learning/create_local_train_test_splits.py`
- `federated_learning/materialize_partitions.py`
- `federated_learning/validate_config.py`

The original-study assertions, including cohort and partition sizes, are retained to detect deviations from the studied configuration. Alternative datasets require their own validated preparation procedure; they should not silently bypass those assertions.

**A3 and F0 are not interchangeable result sets.** A3 used a two-stage global patient-level split. F0/F1/F2 use hospital-first, fixed patient partitions. Although the model specification is carried forward, the exact training/test patient assignments are not necessarily the same, so A3-vs-F0/F1/F2 scores are not a fully controlled split-matched comparison.

### F0, F1 and F2

| Experiment | Method | Privacy interpretation |
|---|---|---|
| **F0** | Centralised logistic-regression baseline | No training DP |
| **F1** | Federated logistic-regression training | No training DP |
| **F2** | Federated training with bounded per-patient contributions and Gaussian noise | Patient-level DP **subject to the stated sampling, accounting, partition and threat-model assumptions** |

The representation, patient-control and clipping-only subdirectories are experimental controls. **Clipping alone and sampling alone must not be described as differential privacy.**

The Flower app configuration in `pyproject.toml` selects `strategy_b_k5`, 30 server rounds, one local epoch per round, batch size 256, clipping norm 2.0, target epsilon 8.0 and delta `1e-5` by default. These are study-specific settings, not recommended universal privacy parameters.

The F2 implementation uses calibrated hospital-specific noise multipliers and patient-count assumptions. The client includes a guard against unsupported hospital training counts.
on script.

## Python environments

Python **3.11** is the declared FL interpreter range (`>=3.11,<3.12`). Use **separate environments** for the three components to avoid dependency conflicts.

### Central DP

```powershell
python -m venv .venv-cdp
.\.venv-cdp\Scripts\python.exe -m pip install -r requirements-cdp.txt
.\.venv-cdp\Scripts\python.exe -m pip check
```

The verified CDP dependencies include FastAPI, Uvicorn, HTTPX, pandas, diffprivlib, scikit-learn and OpenDP. Hospital nodes must only be used with authorised local datasets on an appropriately isolated network.

### HE + DP

```powershell
python -m venv .venv-he
.\.venv-he\Scripts\python.exe -m pip install -r requirements-he.txt
.\.venv-he\Scripts\python.exe -m pip check
```

HE uses TenSEAL together with the platform's query, DP and service dependencies. Cryptographic contexts are generated locally and **private context files must not be committed**. Running the HE service graph requires an explicitly configured and restricted trust boundary; do not expose decryption services or exact-statistic endpoints to untrusted callers.

### Federated learning and patient-level DP

```powershell
python -m venv .venv-fl
.\.venv-fl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-fl\Scripts\python.exe -m pip install -e .
.\.venv-fl\Scripts\python.exe -m pip check
```

From the repository root, `pip install -e .` installs the FL project defined in `pyproject.toml`. This command does **not** replace the separate CDP/HE requirements. The Flower app configuration uses `project-root = "."`; run it from the intended repository context, and validate working-directory resolution in the chosen Flower runtime.

The FL dependency smoke check performed on 29 October 2026 reported Flower 1.39.0, Opacus 1.6.0, PyTorch 2.15.0, NumPy 2.4.6, pandas 3.1.0 and scikit-learn 1.9.1. These are reported environment versions, **not a full dependency lockfile**; NumPy, pandas, scikit-learn and PyTorch remain unpinned in `pyproject.toml`.

## Central-DP query interface

The CDP coordinator implements the following approved Q001–Q007 analytics:

| Query | Operation |
|---|---|
| Q001 | Unique-patient count |
| Q002 | Admission-type histogram |
| Q003 | Bounded mean length of stay |
| Q004 | Race-by-admission-type contingency table |
| Q005 | Bounded mean length of stay by race |
| Q006 | Predefined rare-subgroup count |
| Q007 | Bounded mean length of stay for the predefined subgroup |

The coordinator exposes `/health`, `/queries`, `/privacy/budget` and DP release routes such as `/query/q001/dp?epsilon=0.1`, when correctly configured and running. The permitted query domains, contribution rules, bounds and privacy parameters are defined in the source. Repeated releases consume privacy budget.

The trusted coordinator can access **exact local aggregates**, even when only DP results are released externally. Exact coordinator baseline routes are disabled by default in the reviewed configuration (`CDP_ENABLE_EXACT_BASELINES=false`), but hospital nodes still expose exact local statistics. Never expose the nodes or any exact-result/decryption services directly on a public network.


## HE + DP research pathway

`he_pipeline/` provides query adapters, public/private cryptographic-context management, encrypted hospital aggregation, coordinator and release services, and evaluation/test code. BFV and CKKS serve different supported arithmetic needs. The encrypted pathway is assessed relative to the same underlying Q001–Q007 statistical queries as central DP; encryption is not itself a DP mechanism.

The release service is separated from the coordinator and its internal routes are token-guarded in the reviewed source. This is **not** proof of end-to-end authorization, budget-bound decryption or a malicious-party-secure cryptographic protocol. Network isolation, key lifecycle, client authentication, atomic budget/release binding and failure behaviour require independent validation before use with restricted data.

Automated HE unit tests were reported as **23 passed** in the isolated review environment. They do not constitute comprehensive security evaluation or a complete multi-service experiment.

## Q008/Q009 disclosure-risk experiments

`experiments_disclosure/disclosure_evaluation/` contains experimental disclosure-risk evaluations separate from the registered Central-DP Q001–Q007 service interface. The code includes:

- **Q008 (rare-cohort prolonged-stay analysis):** An exact patient-level histogram reference, DP mechanisms, controlled synthetic cohorts, attribute inference and auxiliary-knowledge evaluations.
- **Q009 (overlapping-query differencing):** Exact and DP-controlled differencing experiments on constructed comparison cohorts, with query-composition evaluation.
- **Comparison and plotting:** Scripts that process locally generated experimental output into summaries and figures.

**Exact evaluation is not a DP release.** In particular, `q008_prolonged_stay/run_exact_q008.py` reads authorised local hospital `admissions.csv.gz` files and writes exact JSON/CSV outputs. It is intended for controlled research evaluation only. Do not expose those outputs publicly or interpret them as privacy-preserving releases. The Q009 controlled target identifier is a synthetic experiment fixture, not a patient identifier from MIMIC-IV.

This repository does **not** include a supported Docker Compose deployment; the research components are managed through separate Python environments.
