# MedixAI: Medical Machine Learning, Calibration & Uncertainty

An experimental machine learning project examining how **feature selection, model choice, probability calibration, and predictive uncertainty** interact in tabular medical classification tasks. The repository provides a reproducible Python pipeline for comparing methods across two public benchmark datasets.

> **Scope:** This is a research and educational project, not a clinical decision-support system. Its predictions and evaluation results should not be used for medical diagnosis or treatment.

## Overview

The project investigates several questions:

- How do different feature subsets affect classification performance?
- How do different machine learning models perform on the same datasets?
- Can probability calibration improve the reliability of predicted probabilities?
- How much predictive uncertainty is attributable to variation across fitted models versus uncertainty in their predictions?

The experiments cover data preparation, feature subset generation, hyperparameter searches, probability calibration, bootstrap-based uncertainty analysis, and comparative visualizations.

## Datasets

| Dataset | Task | Source in this repository |
| --- | --- | --- |
| **Pima Indians Diabetes** | Binary diabetes outcome classification | Downloaded by the data-loading code from a public CSV; invalid zero entries in selected clinical variables are treated as missing values and imputed using training-set medians |
| **Breast Cancer Wisconsin (Diagnostic)** | Binary breast-cancer classification | Loaded through `sklearn.datasets.load_breast_cancer` |

For both datasets, the experiment utilities use stratified **60% training / 15% validation / 15% calibration / 10% test** splits.

## Methods

**Classification models**

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multi-Layer Perceptron (MLP)

**Feature subset strategies**

- All features (baseline)
- Mutual Information and minimum-redundancy maximum-relevance (**mRMR**) rankings
- Correlation-based filtering
- Domain-informed feature groups
- Random feature subsets as controls

**Probability calibration**

- Platt scaling
- Isotonic regression
- Histogram binning with uniform or quantile bins

**Uncertainty analysis**

- Bootstrap-based comparison using repeated model fits
- Decomposition of predictive entropy into **aleatoric** and **epistemic** components, as implemented in the experimental code

**Evaluation measures** include ROC-AUC, F1, recall, precision, accuracy, Brier score, negative log-likelihood, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).

## Technologies

Python, scikit-learn, NumPy, pandas, SciPy, Matplotlib, Seaborn, and mRMR feature selection.

## Getting started

Clone the repository:

```bash
git clone https://github.com/eminamandzukic/MedixAI_project.git
cd MedixAI_project
```

On macOS or Linux, create a virtual environment and install dependencies with the provided setup script:

```bash
bash setup_venv.sh
```

Then run the complete experimental pipeline:

```bash
bash run_pipeline.sh
```

This runs feature subset generation, model experiments, result exports, uncertainty analysis, and visualizations for both datasets. **The full experiment search may take substantial computing time.** The Pima dataset is downloaded automatically when needed, so first-time runs require an internet connection.

For a more targeted experiment, generate the feature subsets and run selected models:

```bash
source .venv/bin/activate
python3 feature_extraction/generate_feature_subsets.py --dataset pima
python3 experiments/run_all_experiments.py --dataset pima --models lr rf
```

The model options are `lr`, `svm`, `rf`, `mlp`, and `all`.

The pipeline also supports `--skip-experiments` and `--skip-feature-extraction` when appropriate outputs from prior stages already exist.

## Project structure

```text
MedixAI_project/
├── experiments/            # Model searches, calibration, result exports, uncertainty
├── feature_extraction/     # Feature subset generation and selection
├── visualization/          # Comparative plots and visualization scripts
├── saved_models/           # Saved fitted models and calibrators
├── requirements.txt       # Python dependencies
├── setup_venv.sh           # Virtual-environment setup
├── run_pipeline.sh         # End-to-end experiment pipeline
└── report.pdf              # Project report
```

Generated results are stored under `results/` and plots under `visualization/figures/` when the corresponding scripts are run.

## Project report

For the detailed methodology, analysis, and findings, read the **[full project report](report.pdf)**.

## Important note

These experiments use publicly available benchmark datasets. They demonstrate and compare machine learning methods, but do **not** establish clinical validity, generalization to hospital populations, or suitability for real-world medical decisions.
