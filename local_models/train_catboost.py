#!/usr/bin/env python3
"""
CatBoost model training script.

Usage:
    python train_catboost.py --dataset breast_cancer
    python train_catboost.py --dataset california_housing --iterations 500 --depth 8

Output:
    models/cb_<dataset>.pkl   → pickled CatBoost model
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from utils import (
    add_common_args,
    evaluate_model,
    get_dataset,
    infer_task,
    split_data,
    MODELS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CatBoost model.")
    add_common_args(parser)
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Number of boosting iterations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=7,
        help="Tree depth.",
    )
    parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=3.0,
        help="L2 leaf regularization.",
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
    common_params: dict = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": 42,
        "verbose": 50,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    if task == "classification":
        model = CatBoostClassifier(**common_params)
    else:
        model = CatBoostRegressor(**common_params)

    print(f"\nTraining CatBoost ({type(model).__name__})…")
    train_pool = Pool(X_train, label=y_train)
    test_pool = Pool(X_test, label=y_test)

    model.fit(train_pool, eval_set=test_pool)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred_proba = model.predict(test_pool)
    if task == "classification":
        y_pred = (y_pred_proba > 0.5).astype(int)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = y_pred.ravel()
    else:
        y_pred = y_pred_proba.ravel()

    evaluate_model(y_test.values.ravel(), y_pred, task)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"cb_{args.dataset}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {save_path}")

    # Feature importance
    importance = model.get_feature_importance(type="PredictionValuesChange")
    names = X.columns.tolist()
    top_idx = np.argsort(importance)[::-1][:10]
    print("\nTop-10 feature importances (PredictionValuesChange):")
    for idx in top_idx:
        print(f"  {names[idx]:20s}  {importance[idx]:.1f}")


if __name__ == "__main__":
    main()
