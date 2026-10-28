# Software Maintainability Prediction

Compare machine-learning regressors that estimate a class's future maintenance
effort (change) from object-oriented software metrics.

## Start in JupyterLab

Clone the repository, create a virtual environment, and install JupyterLab:

    git clone https://github.com/Rahul-Chaudhry/Software-Maintainence-Predictibility.git
    cd Software-Maintainence-Predictibility
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install jupyterlab
    jupyter lab

On Windows PowerShell, activate the environment with:

    .venv\Scripts\Activate.ps1

Open notebooks/baseline_models.ipynb. It begins with a commented,
one-package-per-line install cell. If a package is missing, remove the
leading # from its line and run that cell once.

## Notebook workflow

Choose QUES, UIMS, or all, then select Run All Cells. The notebook compares
five baseline models and saves the metrics CSV in outputs.

The evaluation uses a deterministic 80/20 train/test split and reports R2,
MAE, and RMSE for linear regression, ridge regression, K-nearest neighbors,
decision tree, and support-vector regression.

## Data

| Dataset | Rows | Description |
| --- | ---: | --- |
| QUES.csv | 71 | Quality Evaluation System classes |
| UIMS1.csv | 39 | User Interface Management System classes |

Each row includes ten object-oriented metrics and change, the number of lines
added, removed, or modified during maintenance. The datasets are small, so
the results are useful for comparison and learning, not production forecasts.

## Project layout

    data/              QUES and UIMS source datasets
    outputs/           Generated model metrics, ignored by Git
    notebooks/         All executable project workflows
    notebooks/legacy/  Original exploratory notebooks
    documentation/     Project report and presentation

## Original coursework

The original [report](documentation/report.pdf),
[presentation](documentation/project_presentation.pptx), and exploratory
notebooks remain available in notebooks/legacy. Some older neural-network
notebooks use retired Keras APIs; use the maintained baseline workflow above
for a current, reproducible starting point.
