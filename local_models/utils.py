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
    "credit_risk": "generate_credit_risk",
    "breast_cancer": "load_breast_cancer_dataset",
    "diabetes": "load_diabetes_dataset",
    "california_housing": "load_california_housing_dataset",
}


# ---------------------------------------------------------------------------
# 1) Credit Risk (synthetic binary classification) — business-friendly names
# ---------------------------------------------------------------------------

_CREDIT_FEATURE_NAMES = [
    "age",
    "annual_income",
    "credit_score",
    "debt_to_income_ratio",
    "employment_years",
    "num_open_accounts",
    "num_late_payments",
    "loan_amount",
    "loan_term_months",
    "has_cosigner",
    "home_ownership",
    "existing_customer_years",
    "num_dependents",
    "monthly_expenses",
    "savings_balance",
]

def generate_credit_risk(
    n_samples: int = 10_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic credit risk dataset (will_default: 0/1).

    Simulates a realistic loan-application dataset with named features
    like income, credit_score, debt_to_income_ratio etc.
    """
    rng = np.random.default_rng(random_state)

    n = n_samples
    age = rng.integers(21, 75, n).astype(float)
    annual_income = rng.lognormal(mean=10.8, sigma=0.4, size=n)  # ~50k median
    credit_score = np.clip(rng.normal(650, 100, n), 300, 850)
    debt_to_income_ratio = np.clip(rng.normal(0.35, 0.15, n), 0, 1.2)
    employment_years = np.clip(rng.exponential(5, n), 0, 40)
    num_open_accounts = rng.poisson(3, n).astype(float)
    num_late_payments = rng.poisson(1, n).astype(float)
    loan_amount = rng.lognormal(mean=10.0, sigma=0.8, size=n)  # ~22k median
    loan_term_months = rng.choice([12, 24, 36, 48, 60], n).astype(float)
    has_cosigner = rng.binomial(1, 0.25, n).astype(float)
    home_ownership = rng.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25]).astype(float)  # rent/own/mortgage
    existing_customer_years = np.clip(rng.exponential(3, n), 0, 20)
    num_dependents = rng.poisson(1, n).astype(float)
    monthly_expenses = rng.lognormal(mean=7.5, sigma=0.5, size=n)  # ~1.8k median
    savings_balance = rng.lognormal(mean=9.0, sigma=1.5, size=n)   # ~8k median

    features = np.column_stack([
        age, annual_income, credit_score, debt_to_income_ratio,
        employment_years, num_open_accounts, num_late_payments,
        loan_amount, loan_term_months, has_cosigner, home_ownership,
        existing_customer_years, num_dependents, monthly_expenses,
        savings_balance,
    ])

    # Target: default probability (~18% default rate)
    # Standardise key drivers to ~N(0,1) range before combining
    logit = (
        -0.003 * (credit_score - 650)                  # higher score → lower risk
        + 2.0 * (debt_to_income_ratio - 0.35)          # high DTI → higher risk
        + 1.2 * num_late_payments                      # late payments → higher risk
        + 0.8 * (loan_amount / (annual_income + 1) - 0.5)
        - 0.4 * employment_years                       # stable job → lower risk
        - 0.6 * savings_balance / 10_000               # savings → lower risk
        + 0.7 * (age < 25).astype(float)               # young → higher risk
        - 0.5                                                 # intercept: ~18% base rate
        + rng.normal(0, 1.0, n)                        # random noise
    )
    logit = np.clip(logit, -50, 50)
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)  # binary target

    df = pd.DataFrame(features, columns=_CREDIT_FEATURE_NAMES)
    return df, pd.Series(y, name="will_default")


# ---------------------------------------------------------------------------
# 2) Breast Cancer Wisconsin (binary classification) — real medical names
# ---------------------------------------------------------------------------

def load_breast_cancer_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Wisconsin Breast Cancer dataset (binary classification, 30 features)."""
    data = load_breast_cancer()
    # sklearn provides descriptive feature names like 'mean radius', 'mean texture', …
    # Clean them up: lowercase + replace spaces with underscores
    names = [n.replace(" ", "_") for n in data.feature_names]
    return pd.DataFrame(data.data, columns=names), pd.Series(data.target, name="diagnosis")


# ---------------------------------------------------------------------------
# 3) Diabetes (regression) — medical feature names
# ---------------------------------------------------------------------------

_DIABETES_FEATURE_NAMES = [
    "age", "sex", "bmi", "blood_pressure",
    "total_cholesterol", "ldl", "hdl", "tch_ldl_ratio",
    "ltg", "glucose",
]

def load_diabetes_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Diabetes progression dataset (regression, 10 features).

    Features are already mean-centered and scaled by sklearn.
    We map s1..s6 to semantically meaningful names.
    """
    data = load_diabetes()
    return pd.DataFrame(data.data, columns=_DIABETES_FEATURE_NAMES), pd.Series(data.target, name="progression")


# ---------------------------------------------------------------------------
# 4) California Housing (regression) — real-estate feature names
# ---------------------------------------------------------------------------

_CALIFORNIA_FEATURE_NAMES = [
    "median_income",
    "house_age",
    "avg_rooms",
    "avg_bedrooms",
    "population",
    "avg_occupancy",
    "latitude",
    "longitude",
]

def load_california_housing_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """California Housing dataset (regression, 8 features).

    Target = median house value in $100k units.
    """
    data = fetch_california_housing()
    return pd.DataFrame(data.data, columns=_CALIFORNIA_FEATURE_NAMES), pd.Series(data.target, name="median_house_value")


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
        default="credit_risk",
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
    classification_datasets = {"credit_risk", "breast_cancer"}
    return "classification" if dataset_name in classification_datasets else "regression"
