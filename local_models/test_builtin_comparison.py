#!/usr/bin/env python3
"""
Compare HuggingFace Python pipeline vs Axon pipeline step by step.

Pipeline: text → tokenize → ONNX model → softmax → argmax → label
Verifies that Axon's builtin functions (tokenize, softmax, argmax) produce
identical results to Python's transformers library.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
REPO_DIR = BASE_DIR / "model_repository" / "nlp_ml"
AXON_BIN = Path(__file__).resolve().parents[1] / "axon-server"
LABEL_NAMES = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

TEST_TEXTS = [
    "I love this product, it's amazing!",
    "This is terrible, I hate it.",
    "The weather is okay today.",
    "Covid cases are increasing fast!",
    "Happy to see everyone today!",
]


def python_inference(tokenizer, model, text):
    import torch

    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    input_ids = tokens["input_ids"].numpy().tolist()[0]
    attention_mask = tokens["attention_mask"].numpy().tolist()[0]

    with torch.no_grad():
        logits = model(**tokens).logits.numpy()[0]

    probs = np.exp(logits - np.max(logits))
    probs = probs / probs.sum()

    label = int(np.argmax(probs))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits.tolist(),
        "probs": probs.tolist(),
        "label": label,
    }


def axon_infer(port, model_name, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v2/models/{model_name}/infer",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def axon_pipeline_inference(port, text):
    payload = {
        "inputs": [
            {"name": "text", "shape": [1], "datatype": "BYTES", "data": [text]}
        ]
    }
    result = axon_infer(port, "roberta_pipeline", payload)
    outputs = {}
    for out in result.get("outputs", []):
        outputs[out["name"]] = out["data"]
    return outputs


def axon_tokenizer_inference(port, text):
    payload = {
        "inputs": [
            {"name": "text", "shape": [1], "datatype": "BYTES", "data": [text]}
        ]
    }
    result = axon_infer(port, "roberta_tokenizer", payload)
    outputs = {}
    for out in result.get("outputs", []):
        outputs[out["name"]] = out["data"]
    return outputs


def wait_for_server(port, timeout=30):
    for _ in range(timeout):
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/v2/health/ready")
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def compare_arrays(name, py_arr, axon_arr, rtol=1e-4, atol=1e-5):
    py_np = np.array(py_arr, dtype=np.float64)
    ax_np = np.array(axon_arr, dtype=np.float64)
    if py_np.shape != ax_np.shape:
        print(f"  FAIL {name}: shape mismatch {py_np.shape} vs {ax_np.shape}")
        return False
    max_diff = np.max(np.abs(py_np - ax_np))
    match = np.allclose(py_np, ax_np, rtol=rtol, atol=atol)
    status = "PASS" if match else "FAIL"
    print(f"  {status} {name}: max_diff={max_diff:.8f}")
    if not match:
        print(f"    Python: {py_arr[:5]}...")
        print(f"    Axon:   {axon_arr[:5]}...")
    return match


def main():
    print("=" * 60)
    print("  Axon Builtin Functions vs Python Comparison Test")
    print("=" * 60)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print("\n[1] Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print(f"  Model: {MODEL_NAME}")

    print("\n[2] Starting axon-server...")
    port = 8910
    proc = subprocess.Popen(
        [
            str(AXON_BIN),
            "--model-repository", str(REPO_DIR),
            "--http-port", str(port),
            "--metrics-port", str(port + 1),
            "--grpc-port", "0",
            "--num-threads", "2",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "DYLD_LIBRARY_PATH": "/opt/homebrew/lib"},
    )

    if not wait_for_server(port):
        print("  ERROR: axon-server did not start")
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        if stderr:
            print(f"  stderr: {stderr[:500]}")
        proc.kill()
        proc.wait()
        sys.exit(1)
    print(f"  Server ready on port {port}")

    print("\n[3] Step-by-step comparison")
    print("-" * 60)

    total_pass = 0
    total_fail = 0

    try:
        for i, text in enumerate(TEST_TEXTS):
            print(f"\n  Text {i+1}: \"{text}\"")

            py = python_inference(tokenizer, model, text)

            axon_tok = axon_tokenizer_inference(port, text)
            axon_out = axon_pipeline_inference(port, text)

            print(f"\n  -- Step 1: Tokenization --")
            tok_match = (
                py["input_ids"] == [int(x) for x in axon_tok["input_ids"]]
                and py["attention_mask"] == [int(x) for x in axon_tok["attention_mask"]]
            )
            status = "PASS" if tok_match else "FAIL"
            print(f"  {status} token count: Python={len(py['input_ids'])}, Axon={len(axon_tok['input_ids'])}")
            if not tok_match:
                print(f"    Python ids: {py['input_ids'][:10]}...")
                print(f"    Axon ids:   {[int(x) for x in axon_tok['input_ids'][:10]]}...")
            if tok_match:
                total_pass += 1
            else:
                total_fail += 1

            print(f"\n  -- Step 2: Softmax Probabilities --")
            axon_probs = axon_out.get("probs", [])
            prob_match = compare_arrays("probs", py["probs"], axon_probs)
            if prob_match:
                total_pass += 1
            else:
                total_fail += 1

            print(f"\n  -- Step 3: Argmax Label --")
            axon_label = int(axon_out.get("label", [-1])[0])
            label_match = py["label"] == axon_label
            status = "PASS" if label_match else "FAIL"
            print(f"  {status} Python={LABEL_NAMES[py['label']]}({py['label']}), Axon={LABEL_NAMES[axon_label]}({axon_label})")
            if label_match:
                total_pass += 1
            else:
                total_fail += 1

            print(f"\n  Summary: Python probs={[f'{p:.4f}' for p in py['probs']]}")
            print(f"           Axon   probs={[f'{p:.4f}' for p in axon_probs]}")
            print("-" * 60)

    finally:
        proc.kill()
        proc.wait()

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {total_pass} passed, {total_fail} failed")
    print(f"  (out of {len(TEST_TEXTS)} texts x 3 checks = {len(TEST_TEXTS)*3} total)")
    print(f"{'=' * 60}")

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
