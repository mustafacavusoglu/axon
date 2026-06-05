#!/usr/bin/env python3
"""
XGBoost model training script.

Usage:
    python train_xgboost.py --dataset breast_cancer
    python train_xgboost.py --dataset california_housing --n-estimators 300

Output:
    models/xgb_<dataset>.pkl   → pickled XGBoost model
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

from utils import (
    add_common_args,
    evaluate_model,
    get_dataset,
    infer_task,
    split_data,
    MODELS_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an XGBoost model.")
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
        help="Learning rate (eta).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum tree depth.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row subsample ratio.",
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
        "objective": "binary:logistic" if task == "classification" else "reg:squarederror",
        "eval_metric": "logloss" if task == "classification" else "rmse",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("\nTraining XGBoost…")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dtest, "validation")],
        verbose_eval=50,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    y_pred_proba = model.predict(dtest)
    if task == "classification":
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = y_pred_proba

    evaluate_model(y_test.values, y_pred, task)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / f"xgb_{args.dataset}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {save_path}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    print("\nTop-10 feature importances (gain):")
    for name, score in sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {name:20s}  {score:.1f}")


if __name__ == "__main__":
    main()
