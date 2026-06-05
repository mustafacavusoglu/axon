#!/usr/bin/env python3
"""
LightGBM model training script.

Usage:
    python train_lgbm.py --dataset breast_cancer
    python train_lgbm.py --dataset regression_synthetic --test-size 0.15

Output:
    models/lgbm_<dataset>.pkl   → pickled LightGBM model
"""

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np

from utils import (
    add_common_args,
    evaluate_model,
    get_dataset,
    infer_task,
    split_data,
    MODELS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LightGBM model.")
    add_common_args(parser)
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting rounds.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=7,
        help="Maximum tree depth.",
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
    params: dict = {
        "objective": "binary" if task == "classification" else "regression",
        "metric": "binary_logloss" if task == "classification" else "rmse",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": min(2 ** args.max_depth, 256),
        "max_depth": args.max_depth,
        "verbose": -1,
        "seed": 42,
    }

    print("\nTraining LightGBM…")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=args.n_estimators,
        valid_sets=[valid_data],
        valid_names=["validation"],
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    if task == "classification":
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        # handle multi-class case (breast_cancer has classes 0/1)
        if len(np.unique(y_train)) > 2:
            y_pred = np.argmax(y_pred_proba, axis=1) if y_pred_proba.ndim > 1 else (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)

    evaluate_model(y_test.values, y_pred, task)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"lgbm_{args.dataset}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {save_path}")

    # Also show feature importance
    print("\nTop-10 feature importances (gain):")
    importance = model.feature_importance(importance_type="gain")
    names = model.feature_name()
    top_idx = np.argsort(importance)[::-1][:10]
    for idx in top_idx:
        print(f"  {names[idx]:20s}  {importance[idx]:.1f}")


if __name__ == "__main__":
    main()
