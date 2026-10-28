"""Compare reproducible regression baselines for software-maintainability data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "QUES": PROJECT_ROOT / "data" / "QUES.csv",
    "UIMS": PROJECT_ROOT / "data" / "UIMS1.csv",
}
FEATURES = ("dit", "noc", "mpc", "rfc", "lcom", "dac", "wmc", "nom", "size2", "size1")
TARGET = "change"


def models() -> dict[str, object]:
    """Return small, explainable regression baselines."""
    return {
        "Linear regression": LinearRegression(),
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "K-nearest neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "Decision tree": DecisionTreeRegressor(random_state=0),
        "Support vector regression": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
    }


def load_dataset(name: str) -> pd.DataFrame:
    """Load an included dataset and verify its expected columns."""
    try:
        path = DATASETS[name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset: {name}") from error

    dataframe = pd.read_csv(path)
    missing = set(FEATURES).union({TARGET}).difference(dataframe.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
    return dataframe


def evaluate_dataset(name: str, test_size: float = 0.2) -> pd.DataFrame:
    """Fit each baseline using the same train/test split and return its metrics."""
    dataframe = load_dataset(name)
    features = dataframe.loc[:, list(FEATURES)]
    target = dataframe[TARGET]
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=test_size, random_state=0
    )

    rows = []
    for model_name, model in models().items():
        model.fit(train_features, train_target)
        prediction = model.predict(test_features)
        rows.append(
            {
                "Dataset": name,
                "Model": model_name,
                "R2": r2_score(test_target, prediction),
                "MAE": mean_absolute_error(test_target, prediction),
                "RMSE": np.sqrt(mean_squared_error(test_target, prediction)),
                "Train samples": len(train_target),
                "Test samples": len(test_target),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE", ignore_index=True)


def evaluate(dataset: str) -> pd.DataFrame:
    """Evaluate one dataset or both included datasets."""
    names = DATASETS if dataset == "all" else (dataset,)
    return pd.concat([evaluate_dataset(name) for name in names], ignore_index=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline models for software-maintainability prediction."
    )
    parser.add_argument(
        "--dataset", choices=("all", *DATASETS), default="all", help="Dataset to evaluate"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "baseline_results.csv",
        help="CSV file to create (default: outputs/baseline_results.csv)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = evaluate(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
