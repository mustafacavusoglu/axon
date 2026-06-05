"""
Shared utilities for model training scripts.
Provides dataset generators and common helpers.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    make_classification,
    make_regression,
    load_breast_cancer,
    load_diabetes,
    fetch_california_housing,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset registry — easy to extend
# ---------------------------------------------------------------------------

DATASETS = {
    "classification_synthetic": "generate_classification_synthetic",
    "regression_synthetic": "generate_regression_synthetic",
    "breast_cancer": "load_breast_cancer_dataset",
    "diabetes": "load_diabetes_dataset",
    "california_housing": "load_california_housing_dataset",
}


def generate_classification_synthetic(
    n_samples: int = 10_000,
    n_features: int = 20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=random_state,
    )
    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


def generate_regression_synthetic(
    n_samples: int = 10_000,
    n_features: int = 20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic regression dataset."""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        noise=0.1,
        random_state=random_state,
    )
    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


def load_breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Breast Cancer dataset (binary classification)."""
    data = load_breast_cancer()
    columns = [f"feature_{i}" for i in range(data.data.shape[1])]
    return pd.DataFrame(data.data, columns=columns), pd.Series(data.target, name="target")


def load_diabetes_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Diabetes dataset (regression)."""
    data = load_diabetes()
    columns = [f"feature_{i}" for i in range(data.data.shape[1])]
    return pd.DataFrame(data.data, columns=columns), pd.Series(data.target, name="target")


def load_california_housing_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """California Housing dataset (regression)."""
    data = fetch_california_housing()
    columns = [f"feature_{i}" for i in range(data.data.shape[1])]
    return pd.DataFrame(data.data, columns=columns), pd.Series(data.target, name="target")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset by name."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choices: {list(DATASETS.keys())}")
    fn_name = DATASETS[name]
    fn = globals()[fn_name]
    return fn()


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split with stratification when appropriate."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, task: str) -> dict:
    """Print and return evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        r2_score,
        mean_squared_error,
        mean_absolute_error,
    )

    metrics = {}
    if task == "classification":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    else:
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mae"] = mean_absolute_error(y_true, y_pred)

    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    return metrics


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments to a parser."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="classification_synthetic",
        choices=list(DATASETS.keys()),
        help="Dataset to use for training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory to save trained model.",
    )


def infer_task(dataset_name: str) -> str:
    """Guess task type from dataset name."""
    classification_datasets = {"classification_synthetic", "breast_cancer"}
    return "classification" if dataset_name in classification_datasets else "regression"
