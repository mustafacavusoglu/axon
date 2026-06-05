#!/usr/bin/env python3
"""
Convert pickled models to ONNX and create Triton Inference Server model repository.

Supported sources:
    - LightGBM  Booster  (.pkl)
    - XGBoost   Booster  (.pkl)
    - CatBoost  *Classifier / *Regressor  (.pkl)
    - sklearn   any estimator  (.pkl)

Output per model:
    model_repository/<name>/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt

Usage:
    # Convert all .pkl files under models/
    python convert_to_onnx.py

    # Convert a specific file
    python convert_to_onnx.py --input models/lgbm_breast_cancer.pkl

    # With explicit feature count / names
    python convert_to_onnx.py --input models/xgb_breast_cancer.pkl --n-features 30
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
REPO_DIR = BASE_DIR / "model_repository"

# ---------------------------------------------------------------------------
# Model-type detection
# ---------------------------------------------------------------------------

def detect_model_type(model: Any) -> str:
    """Return a short key identifying the framework of a loaded model."""
    qualname = type(model).__module__ + "." + type(model).__qualname__

    if "lightgbm" in qualname:
        return "lightgbm"
    if "xgboost" in qualname:
        return "xgboost"
    if "catboost" in qualname:
        return "catboost"
    if "sklearn" in qualname:
        return "sklearn"

    raise ValueError(f"Cannot detect model type from: {qualname}")


# ---------------------------------------------------------------------------
# Feature-name extraction
# ---------------------------------------------------------------------------

def extract_feature_names(model: Any, model_type: str, n_features: Optional[int]) -> List[str]:
    """Try to get feature names from the model, fall back to feature_0..feature_N."""
    names: Optional[List[str]] = None

    try:
        if model_type == "lightgbm":
            names = model.feature_name()
        elif model_type == "xgboost":
            names = model.feature_names
        elif model_type == "catboost":
            names = model.feature_names_
        elif model_type == "sklearn":
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
    except Exception:
        names = None

    if names and len(names) > 0:
        return names

    # Fallback
    if n_features is None:
        n_features = _guess_n_features(model, model_type)
    return [f"feature_{i}" for i in range(n_features)]


def _guess_n_features(model: Any, model_type: str) -> int:
    """Guess number of input features from the model object."""
    try:
        if model_type == "lightgbm":
            return model.num_feature()
        if model_type == "xgboost":
            return model.num_features()
        if model_type == "catboost":
            if hasattr(model, "tree_count_"):
                return model.tree_count_
            # fallback: try feature_importances_
            fi = model.get_feature_importance()
            return len(fi)
        if model_type == "sklearn":
            if hasattr(model, "n_features_in_"):
                return model.n_features_in_
            if hasattr(model, "feature_importances_"):
                return len(model.feature_importances_)
            if hasattr(model, "coef_"):
                return model.coef_.shape[-1] if model.coef_.ndim > 1 else len(model.coef_)
    except Exception:
        pass
    return 10  # safe default


# ---------------------------------------------------------------------------
# ONNX conversion dispatcher
# ---------------------------------------------------------------------------

def _make_dummy_input(n_features: int, batch_size: int = 1) -> np.ndarray:
    """Float32 dummy input for ONNX tracing."""
    return np.random.randn(batch_size, n_features).astype(np.float32)


def convert_lightgbm_to_onnx(model: Any, n_features: int, onnx_path: Path) -> None:
    """LightGBM Booster → ONNX via onnxmltools."""
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(model, initial_types=initial_type)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  [LightGBM → ONNX] {onnx_path}")


def convert_xgboost_to_onnx(model: Any, n_features: int, onnx_path: Path) -> None:
    """XGBoost Booster → ONNX via onnxmltools."""
    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    # XGBoost onnxmltools converter requires f0..fN style feature names.
    # Temporarily set them, then restore after conversion.
    original_names = model.feature_names
    model.feature_names = [f"f{i}" for i in range(n_features)]

    try:
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  [XGBoost → ONNX] {onnx_path}")
    finally:
        model.feature_names = original_names


def convert_catboost_to_onnx(model: Any, n_features: int, onnx_path: Path) -> None:
    """CatBoost model → ONNX via native CatBoost export."""
    # CatBoost has built-in ONNX export since v0.23
    model.save_model(
        str(onnx_path),
        format="onnx",
        export_parameters={
            "onnx_domain": "ai.catboost",
            "onnx_model_version": 1,
            "onnx_doc_string": "CatBoost model exported to ONNX",
        },
    )
    print(f"  [CatBoost → ONNX] {onnx_path}")


def convert_sklearn_to_onnx(model: Any, n_features: int, onnx_path: Path) -> None:
    """sklearn estimator → ONNX via skl2onnx."""
    from skl2onnx import convert_sklearn, to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=18,  # onnxruntime supports up to 21 stable
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  [sklearn → ONNX] {onnx_path}")


CONVERTERS = {
    "lightgbm": convert_lightgbm_to_onnx,
    "xgboost": convert_xgboost_to_onnx,
    "catboost": convert_catboost_to_onnx,
    "sklearn": convert_sklearn_to_onnx,
}


# ---------------------------------------------------------------------------
# config.pbtxt generation (Triton)
# ---------------------------------------------------------------------------

def _task_from_name(name: str) -> str:
    """Heuristic: 'breast_cancer', 'classification' → classification, else regression."""
    classification_keywords = ["breast_cancer", "classification"]
    for kw in classification_keywords:
        if kw in name.lower():
            return "classification"
    return "regression"


def _output_spec(task: str) -> str:
    if task == "classification":
        return (
            "    {\n"
            "      name: \"output_label\"\n"
            "      data_type: TYPE_INT64\n"
            "      dims: [ 1 ]\n"
            "    },\n"
            "    {\n"
            "      name: \"output_probability\"\n"
            "      data_type: TYPE_FP32\n"
            "      dims: [ 2 ]\n"
            "    }"
        )
    else:
        return (
            "    {\n"
            "      name: \"output\"\n"
            "      data_type: TYPE_FP32\n"
            "      dims: [ 1 ]\n"
            "    }"
        )


def generate_config_pbtxt(
    model_name: str,
    feature_names: List[str],
    task: str,
    max_batch_size: int = 32,
) -> str:
    """Generate Triton config.pbtxt content with explicit input feature schema."""

    n = len(feature_names)

    # Build the input schema table showing exact column→feature mapping
    schema_lines = []
    schema_lines.append(f"# │  col │ name{'':21s} │ dtype     │")
    schema_lines.append(f"# ├──────┼{'─'*24}┼───────────┤")
    for i, fn in enumerate(feature_names):
        schema_lines.append(f"# │ {i:4d} │ {fn:22s} │ TYPE_FP32 │")
    schema_lines.append(f"#")
    schema_lines.append(f"#   Total: {n} features, passed as a flat float32 array shape [1, {n}]")

    input_schema_block = "\n".join(schema_lines)

    config = f"""# Triton Inference Server model configuration
# Auto-generated — model: {model_name}

name: "{model_name}"
platform: "onnxruntime_onnx"

max_batch_size: {max_batch_size}

# ── Input schema ───────────────────────────────────────
# The ONNX model expects a single tensor "input" with all
# features concatenated in the exact order below:
#
{input_schema_block}

# ── Input ──────────────────────────────────────────────
input [
    {{
      name: "input"
      data_type: TYPE_FP32
      dims: [ {n} ]
    }}
]

# ── Output ─────────────────────────────────────────────
output [
{_output_spec(task)}
]

# ── Dynamic batching ───────────────────────────────────
dynamic_batching {{
  max_queue_delay_microseconds: 100
}}

# ── Instance group (CPU-only) ──────────────────────────
instance_group [
  {{
    count: 2
    kind: KIND_CPU
  }}
]
"""
    return config


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def convert_one(pkl_path: Path, n_features: Optional[int], model_name: Optional[str] = None) -> None:
    """Convert a single .pkl file to ONNX + config.pbtxt inside model_repository/."""

    print(f"\n{'='*60}")
    print(f"Processing: {pkl_path.name}")

    # 1. Load
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    model_type = detect_model_type(model)
    print(f"  Detected: {model_type}")

    # 2. Feature names
    feature_names = extract_feature_names(model, model_type, n_features)
    if n_features is None:
        n_features = len(feature_names)

    print(f"  Features: {n_features}  ({feature_names[0]} … {feature_names[-1]})")

    # 3. Model name
    if model_name is None:
        model_name = pkl_path.stem
    task = _task_from_name(model_name)

    # 4. Triton directory structure
    version_dir = REPO_DIR / model_name / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = version_dir / "model.onnx"

    # 5. Convert
    converter = CONVERTERS[model_type]
    converter(model, n_features, onnx_path)

    # 6. config.pbtxt
    config = generate_config_pbtxt(model_name, feature_names, task)
    config_path = REPO_DIR / model_name / "config.pbtxt"
    config_path.write_text(config)
    print(f"  [config.pbtxt] {config_path}")
    print(f"  Done ✓")


def convert_all(n_features: Optional[int] = None) -> None:
    """Convert every .pkl file under models/."""
    pkl_files = sorted(MODELS_DIR.glob("*.pkl"))
    if not pkl_files:
        print("No .pkl files found in models/")
        return

    print(f"Found {len(pkl_files)} model(s) to convert.")
    for p in pkl_files:
        try:
            convert_one(p, n_features)
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pickle models to ONNX + Triton config.")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to a single .pkl file. If not given, converts all under models/.",
    )
    parser.add_argument(
        "--n-features", "-n",
        type=int,
        default=None,
        help="Number of input features (auto-detected if omitted).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name for Triton (default: filename stem).",
    )
    args = parser.parse_args()

    if args.input:
        convert_one(Path(args.input), args.n_features, args.name)
    else:
        convert_all(args.n_features)

    print(f"\nAll done. Repository: {REPO_DIR}")


if __name__ == "__main__":
    main()
