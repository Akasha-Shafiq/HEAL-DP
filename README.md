# HEAL-DP: Homomorphic Encryption Aggregated Learning with Differential Privacy

**Unlocking the value of health data without compromising privacy**

HEAL-DP is a research framework for investigating how Privacy-Enhancing Technologies (PETs) can support healthcare analytics across distributed data holders. It examines the use of **Differential Privacy (DP)**, **Homomorphic Encryption (HE)**, and **Federated Learning (FL)** to reduce information exposure during statistical analysis and collaborative predictive modelling.

The repository provides the research implementations, modelling notebooks and evaluation code. The accompanying report presents the architecture, experimental methodology, results, limitations and discussion in detail.

## Research overview

Healthcare data can inform research, service planning and predictive modelling, but its sensitive nature makes data sharing and cross-institutional analysis challenging. HEAL-DP investigates complementary protections at different stages of an analytical workflow:

| Research component | Purpose |
| --- | --- |
| **Central Differential Privacy** | Controlled release of aggregate statistics with privacy-budget accounting and query governance. |
| **Homomorphic Encryption + DP** | Protection of selected intermediate aggregates during computation, alongside differentially private statistical releases. |
| **Federated Learning + patient-level DP** | Collaborative model training across simulated hospital nodes, evaluated against centralised and non-private federated baselines. |

The framework was evaluated in a **five-hospital simulated research setting using authorised MIMIC-IV data**. It is an experimental implementation, not a production clinical service.

## Technical report and research results

**[Read the HEAL-DP technical report](report/HEAL-DP_Technical_Report.pdf)** *(report PDF to be added to this repository)*

The report is the primary reference for the system architecture, privacy assumptions, experimental settings and findings. It includes evaluation of statistical utility and disclosure risk, encrypted computation and computational overhead, and the privacy–utility behaviour of federated predictive modelling.

**Research results are derived from experiments conducted on authorised MIMIC-IV data.** Selected, disclosure-reviewed study-level results may be presented in the report and interactive demonstrator. The repository does not include MIMIC-IV records, patient-level datasets, hospital partitions, model weights or restricted experimental outputs.

## Interactive demonstrator

A companion **Streamlit demonstrator** presents the HEAL-DP research approach through an interactive overview of the architecture, privacy-enhancing techniques and selected experimental findings.

**Streamlit app:** *Public link to be added after deployment.*

The demonstrator distinguishes two types of content:

- **Research findings:** Results from HEAL-DP experiments performed using authorised **MIMIC-IV** data. Only appropriately disclosure-reviewed summaries and visualisations should be displayed publicly; patient-level data is not accessible through the app.
- **Interactive synthetic-data demonstration:** Exploratory scenarios using **synthetic data generated with [NHS Synth (NHSSynth)](https://github.com/nhsengland/NHSSynth)**. These scenarios illustrate the workflow without providing access to MIMIC-IV data.

**No MIMIC-IV patient data is shared with or made available through the public demonstrator.**. Any MIMIC-IV-derived figures displayed publicly require appropriate disclosure review before deployment.

## Repository contents

```text
coordinator/           Central-DP coordination, privacy accounting and audit logic
node/                  Hospital-node statistical query implementation
he_pipeline/           HE-assisted analytics, release services and automated tests
federated_learning/    Data preparation, centralised and federated training, DP accounting
modelling_baselines/   A1–A3 feature reduction and logistic-regression notebooks
experiments/           Statistical evaluation and comparison code
experiments_disclosure/ Disclosure-risk evaluation code
report/                Technical report (to be added)
```

The modelling workflow includes medication feature reduction (A1), laboratory feature reduction (A2), and centralised logistic regression (A3). The FL experiments use a corresponding linear binary-classification model and fixed patient-disjoint partitions for **F0** (centralised), **F1** (federated) and **F2** (federated with patient-level DP). A3 uses a separate global patient-level split; F0–F2 follow the hospital-based experimental protocol.

## Running the research code

The components have separate Python dependency configurations and should be installed in separate environments:

| Component | Dependency file |
| --- | --- |
| Central DP | `requirements-cdp.txt` |
| HE + DP | `requirements-he.txt` |
| Federated learning + DP | `pyproject.toml` (Python 3.11; Flower and Opacus) |

The A1–A3 notebooks accept the environment variable `MIMIC_FAIRNESS_WORKDIR` to identify their local research workspace:

Researchers must supply authorised data and the preprocessing artifacts expected by the original experimental protocol. The existing patient-partition assumptions and calibrated DP parameters are study-specific; using different data or training settings requires further validation and, where applicable, recalibration.

## Data availability and responsible use

MIMIC-IV is a controlled-access healthcare dataset and **is not included** in this repository. Access to the underlying data must be obtained independently under the applicable terms. The public demonstrator uses synthetic data for interactive exploration; it does not expose restricted records or run public queries against the original MIMIC-IV hospital services.

The software is intended for research and methodological evaluation. Its experimental privacy and security assumptions should not be interpreted as a certification for clinical deployment.

## License

This project is licensed under the [MIT License](LICENSE).
