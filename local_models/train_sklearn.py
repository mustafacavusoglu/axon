#!/usr/bin/env python3
"""
scikit-learn model training script.

Supports multiple sklearn estimators selectable via CLI.

Usage:
    python train_sklearn.py --dataset breast_cancer --model random_forest
    python train_sklearn.py --dataset california_housing --model gradient_boosting
    python train_sklearn.py --dataset diabetes --model elastic_net

Available models:
    classification: random_forest, gradient_boosting, logistic_regression
    regression:     random_forest, gradient_boosting, elastic_net, ridge

Output:
    models/skl_<model>_<dataset>.pkl   → joblib-pickled model
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from utils import (
    add_common_args,
    evaluate_model,
    get_dataset,
    infer_task,
    split_data,
    MODELS_DIR,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_CLASSIFIERS = {
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingClassifier",
    "logistic_regression": "sklearn.linear_model.LogisticRegression",
}

_REGRESSORS = {
    "random_forest": "sklearn.ensemble.RandomForestRegressor",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingRegressor",
    "elastic_net": "sklearn.linear_model.ElasticNet",
    "ridge": "sklearn.linear_model.Ridge",
}


def build_model(model_name: str, task: str) -> object:
    """Instantiate the requested sklearn model with sensible defaults."""
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.linear_model import (
        LogisticRegression,
        ElasticNet,
        Ridge,
    )

    if task == "classification":
        mapping = {
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            ),
            "logistic_regression": LogisticRegression(
                max_iter=2000, random_state=42, n_jobs=-1
            ),
        }
    else:
        mapping = {
            "random_forest": RandomForestRegressor(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
            ),
            "elastic_net": ElasticNet(alpha=0.1, random_state=42),
            "ridge": Ridge(alpha=1.0, random_state=42),
        }

    if model_name not in mapping:
        available = list(mapping.keys())
        raise ValueError(f"Unknown model '{model_name}' for task='{task}'. Choices: {available}")

    return mapping[model_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a scikit-learn model.")
    add_common_args(parser)
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        help="Which sklearn estimator to use. See script docstring for choices.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X, y = get_dataset(args.dataset)
    task = infer_task(args.dataset)
    print(f"Dataset: {args.dataset}  |  Task: {task}  |  Shape: {X.shape}")

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = build_model(args.model, task)
    print(f"\nTraining {type(model).__name__}…")
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    if task == "classification" and hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)
        if y_pred_proba.shape[1] == 2:
            y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)

    evaluate_model(y_test.values, y_pred, task)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"skl_{args.model}_{args.dataset}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {save_path}")

    # Feature importance (tree-based models only)
    if hasattr(model, "feature_importances_"):
        print("\nTop-10 feature importances:")
        names = X.columns
        top_idx = np.argsort(model.feature_importances_)[::-1][:10]
        for idx in top_idx:
            print(f"  {names[idx]:20s}  {model.feature_importances_[idx]:.4f}")


if __name__ == "__main__":
    main()
