# Software Maintainability Prediction

Compare machine-learning regressors that estimate a class's future maintenance
effort (`change`) from object-oriented software metrics.

## Quick start

```bash
git clone https://github.com/Rahul-Chaudhry/Software-Maintainence-Predictibility.git
cd Software-Maintainence-Predictibility
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

## Notebook workflow

Open `notebooks/baseline_models.ipynb`, choose `QUES`, `UIMS`, or `all`, and
select **Run All Cells**. The notebook compares five baseline models and saves
the metrics CSV in `outputs/`.

## Command-line workflow

```bash
python Code/run_baseline_models.py
python Code/run_baseline_models.py --dataset QUES --output outputs/ques_results.csv
```

The runner uses a deterministic 80/20 train/test split and reports R², MAE, and
RMSE for linear regression, ridge regression, K-nearest neighbors, decision
tree, and support-vector regression.

## Data

| Dataset | Rows | Description |
| --- | ---: | --- |
| `QUES.csv` | 71 | Quality Evaluation System classes |
| `UIMS1.csv` | 39 | User Interface Management System classes |

Each row includes ten object-oriented metrics and `change`, the number of lines
added, removed, or modified during maintenance. The datasets are small, so the
results are useful for comparison and learning—not production forecasts.

## Project layout

```text
data/          # QUES and UIMS source datasets
outputs/       # Generated model metrics (ignored by Git)
notebooks/     # Beginner-friendly Jupyter workflow
Code/          # Reproducible runner and original exploratory notebooks
tests/         # Automated tests
requirements.txt
```

## Test

```bash
python -m unittest discover -s tests
```

## Original coursework

The original [report](Report.pdf), [presentation](Project%20Presentation.pptx),
and exploratory notebooks remain available in `Code/`. Some older neural-network
notebooks use retired Keras APIs; use the maintained baseline workflow above for
a current, reproducible starting point.
