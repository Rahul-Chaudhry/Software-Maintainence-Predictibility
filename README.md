# Software Maintainability Prediction

Compare machine-learning regressors that estimate a class's future maintenance
effort (`change`) from object-oriented software metrics. The repository includes
two small research datasets and a reproducible baseline runner.

## Quick start

```bash
git clone https://github.com/Rahul-Chaudhry/Software-Maintainence-Predictibility.git
cd Software-Maintainence-Predictibility
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python Code/run_baseline_models.py
```

On Windows PowerShell, activate the virtual environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The command evaluates five regression models on both datasets, shows R², MAE,
and RMSE in the terminal, and saves the complete results to
`results/baseline_results.csv`.

## Run one dataset

```bash
python Code/run_baseline_models.py --dataset QUES --output results/ques_results.csv
python Code/run_baseline_models.py --dataset UIMS
```

Use `--dataset all` (the default) to evaluate both datasets.

## Datasets

| Dataset | Rows | Description |
| --- | ---: | --- |
| `QUES.csv` | 71 | Quality Evaluation System classes |
| `UIMS1.csv` | 39 | User Interface Management System classes |

Each row has ten object-oriented metrics (`dit`, `noc`, `mpc`, `rfc`, `lcom`,
`dac`, `wmc`, `nom`, `size2`, and `size1`) and the target `change`: the number
of lines added, removed, or modified during maintenance.

## Baseline models

The maintained runner applies the same deterministic 80/20 train/test split
(`random_state=0`) to each model:

- Linear regression
- Ridge regression
- K-nearest-neighbors regression
- Decision-tree regression
- Support-vector regression

The datasets are small, so the results are useful for comparison and learning,
not as reliable production-performance estimates.

## Original notebooks

The original exploratory Jupyter notebooks are preserved in `Code/`. They are
valuable coursework references, but several depend on retired Keras APIs and
expect their CSV files to be in the current folder. For an up-to-date and
reproducible starting point, use `Code/run_baseline_models.py`.

To inspect the original notebooks after installation, run:

```bash
jupyter lab
```

## Project layout

```text
Code/
  run_baseline_models.py  # Reproducible command-line comparison
  *.ipynb                 # Original exploratory notebooks
Data-Files/               # QUES and UIMS datasets
results/                  # Generated CSV evaluation results (ignored by Git)
tests/                    # Small regression tests
requirements.txt          # Required Python libraries
```

## Test

```bash
python -m unittest discover -s tests
```

## Coursework materials

The original [report](Report.pdf) and [presentation](Project%20Presentation.pptx)
remain available in the repository.
