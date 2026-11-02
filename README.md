# HEAL-DP: Homomorphic Encryption Aggregated Learning with Differential Privacy

*Unlocking the value of health data without compromising privacy*

HEAL-DP is a research framework investigating how **Differential Privacy (DP)**, **Homomorphic Encryption (HE)**, and **Federated Learning (FL)** can support statistical analysis and predictive modelling across distributed healthcare data holders while limiting information exposure.

This repository contains the research implementations, modelling notebooks, and evaluation code. The accompanying technical report describes the architecture, experimental methodology, findings, trade-offs, and limitations. Only synthetic data is used in the demonstrator for illustration purposes. 

## Research overview

Healthcare data can support research, service planning, and predictive modelling, but its sensitivity creates challenges for data sharing and collaboration across institutions. HEAL-DP examines three complementary approaches:

| Research component | Purpose |
| --- | --- |
| **Central Differential Privacy** | Controlled release of aggregate statistics through privacy-budget accounting and query governance. |
| **Homomorphic Encryption + DP** | Protection of selected intermediate aggregates during computation, combined with differentially private statistical releases. |
| **Federated Learning + patient-level DP** | Collaborative predictive modelling across simulated hospital nodes, evaluated against centralised and non-private federated baselines. |

The research was evaluated in a **five-hospital simulated setting using authorised MIMIC-IV data**. HEAL-DP is an experimental research implementation, not a production clinical system.

## Report and research results

**[Read the HEAL-DP Technical Report](report/HEAL-DP_Report.pdf)**

The technical report is the primary reference for the system architecture, privacy assumptions, study design, experimental results, and limitations. It covers statistical utility and disclosure-risk evaluation, encrypted computation and computational overhead, and the privacy–utility behaviour of federated predictive modelling.

**The reported research results originate from experiments performed using authorised MIMIC-IV data.** The repository does not distribute MIMIC-IV patient records, patient-level datasets, hospital partitions, model weights, or restricted experimental outputs.

## Demonstrator

 HEAL-DP framework is available here:

 **[Launch the HEAL-DP Demonstrator](https://heal-dp-demo.streamlit.app/)**

The demonstrator provides interactive exploration of the privacy-preserving
analytics pathways developed in this work, including Central Differential
Privacy, Homomorphic Encryption, and Federated Learning.

The public interface uses synthetic data for interactive exploration and
disclosure-reviewed experimental results for demonstrating the evaluated
privacy, utility, and computational trade-offs.

The demonstrator distinguishes between:

- **Research results:** Selected, disclosure-reviewed summaries and visualisations from HEAL-DP experiments conducted using **MIMIC-IV** data.
- **Interactive demonstration:** Synthetic healthcare data generated using [NHSSynth](https://github.com/nhsengland/NHSSynth) to illustrate the analytical workflows and privacy-enhancing techniques.

**No MIMIC-IV patient-level data is shared through the public demonstrator.** The interactive workflows use synthetic data; any MIMIC-IV-derived findings made public must first undergo appropriate disclosure review. The Streamlit application is hosted separately, and its source code is not included in this repository.

## Repository contents

```text
coordinator/             Central-DP coordination, privacy accounting and audit logic
node/                    Hospital-node statistical query implementation
he_pipeline/             HE-assisted analytics, release services and automated tests
federated_learning/      Data preparation, centralised and federated training, DP accounting
modelling_baselines/     A1–A3 feature reduction and logistic-regression notebooks
experiments/             Statistical evaluation and comparison code
experiments_disclosure/  Disclosure-risk evaluation code
report/                  HEAL-DP technical report
```

The modelling workflow covers medication feature reduction (**A1**), laboratory feature reduction (**A2**), and centralised logistic regression (**A3**). The federated-learning experiments use a corresponding linear binary-classification model and fixed, patient-disjoint partitions for **F0** (centralised training), **F1** (non-private federated training), and **F2** (federated training with patient-level DP).

A3 uses a separate global patient-level split, whereas F0–F2 follow the fixed hospital-based partitioning protocol used for the federated experiments.

## Running the research code

The components have separate dependency configurations and should be run in appropriate Python environments:

| Component | Dependency configuration |
| --- | --- |
| Central DP | `requirements-cdp.txt` |
| HE + DP | `requirements-he.txt` |
| Federated Learning + DP | `pyproject.toml` (Python 3.11; Flower and Opacus) |

The A1–A3 notebooks use the `MIMIC_FAIRNESS_WORKDIR` environment variable to locate the research workspace. For example, in PowerShell:

```powershell
$env:MIMIC_FAIRNESS_WORKDIR = "workspace-path"
```

Researchers must supply the input data and preprocessing artifacts required by the original experimental protocol. The feature specification, patient partitions, and DP calibration reflect the study configuration; applying the implementation to other data or training settings requires validation and, where applicable, recalibration.

The dataset and corresponding access permissions are **not provided** by this repository. The code and demonstrator are intended for research and methodological evaluation, not clinical deployment.



## License

This project is licensed under the [MIT License](LICENSE).
