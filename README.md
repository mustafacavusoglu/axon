# Axon — CPU Inference Server

> [Türkçe dokümantasyon](README_TR.md) | [Wiki](https://github.com/mustafacavusoglu/axon/wiki)

Single-binary, Triton-compatible, CPU-first model serving.  
**Language:** Rust | **Runtime:** ONNX Runtime | **Transport:** gRPC + HTTP/REST (KServe v2)  
**BLS/Scripting:** Rhai | **Config:** YAML + pbtxt | **Ensemble:** Declarative model chaining

---

## Features

- **KServe v2 API** — HTTP + gRPC, full model management
- **Ensemble Pipelines** — Declarative multi-model chaining via config
- **BLS (Rhai Scripting)** — Custom pre/post processing logic
- **23 Builtin Functions** — ML, NLP, CV, tabular preprocessing/postprocessing
- **HuggingFace Tokenizer** — Native tokenizer.json support
- **Structured Logging** — JSON file (daily rotation) + OTEL + filtered stdout
- **Inference Timeout** — Configurable per-server timeout
- **Circuit Breaker** — Auto-skip models that fail to load
- **Prometheus Metrics** — Latency, throughput, inflight, queue wait
- **Docker Multi-arch** — linux/amd64 + linux/arm64

---

## Quick Start

### 1. Prerequisites
```bash
# Rust (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# ONNX Runtime (macOS)
brew install onnxruntime

# ONNX Runtime (Linux)
# Download from https://github.com/microsoft/onnxruntime/releases and copy to /usr/local/lib
```

### 2. Build
```bash
git clone https://github.com/mustafacavusoglu/axon.git
cd axon/inference-engine

# macOS
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib cargo build --release

# Linux
cargo build --release
```

### 3. Run with example model
```bash
./target/release/axon-server \
  --model-repository=../models \
  --model-control-mode=poll

# Health check
curl http://localhost:8000/v2/health/ready

# Inference — 8 housing features → price prediction
curl -s -X POST http://localhost:8000/v2/models/xgb_housing/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"median_income","shape":[1],"datatype":"FP32","data":[3.5]},
    {"name":"house_age","shape":[1],"datatype":"FP32","data":[25]},
    {"name":"avg_rooms","shape":[1],"datatype":"FP32","data":[6]},
    {"name":"avg_bedrooms","shape":[1],"datatype":"FP32","data":[1.1]},
    {"name":"population","shape":[1],"datatype":"FP32","data":[1500]},
    {"name":"avg_occupancy","shape":[1],"datatype":"FP32","data":[3]},
    {"name":"latitude","shape":[1],"datatype":"FP32","data":[34]},
    {"name":"longitude","shape":[1],"datatype":"FP32","data":[-118]}
  ]}'
```

### 4. Docker
```bash
docker compose -f docker-compose.turkish_safety.yml up -d
```

---

## CLI Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-repository` | `/models` | Path to model repository |
| `--model-control-mode` | `none` | `none` or `poll` (auto-reload) |
| `--repository-poll-secs` | `30` | Poll interval in seconds |
| `--http-port` | `8000` | HTTP REST port |
| `--grpc-port` | `8001` | gRPC port |
| `--metrics-port` | `8002` | Prometheus metrics port |
| `--inference-timeout-ms` | `30000` | Max inference time before 504 |
| `--num-threads` | `0` | Inference threads (0 = auto) |
| `--concurrency-per-model` | `4` | Max concurrent requests per model |
| `--log-level` | `info` | Log level: trace, debug, info, warn, error |
| `--log-dir` | `/tmp/logs/axon` | Log directory (JSON, daily rotation) |

---

## Builtin Functions (Rhai BLS)

23 builtin functions available in all Rhai scripts for pre/post processing:

### ML / Math
| Function | Signature | Description |
|----------|-----------|-------------|
| `softmax` | `(arr) → arr` | Logits to probabilities |
| `sigmoid` | `(arr) → arr` | Sigmoid activation |
| `argmax` | `(arr) → int` | Index of max value |
| `argmin` | `(arr) → int` | Index of min value |
| `topk` | `(arr, k) → arr` | Top-K values with indices |
| `threshold` | `(arr, val) → arr` | Binary thresholding |
| `clip` | `(arr, min, max) → arr` | Clamp values to range |

### NLP
| Function | Signature | Description |
|----------|-----------|-------------|
| `tokenize` | `(text) → map` | HuggingFace tokenizer (requires tokenizer.json) |
| `decode_tokens` | `(ids) → string` | Token IDs back to text |
| `pad_sequence` | `(arr, len, pad) → arr` | Pad/truncate to fixed length |
| `text_lower` | `(text) → string` | Lowercase conversion |
| `regex_replace` | `(text, pattern, repl) → string` | Regex text replacement |

### Tabular / Normalization
| Function | Signature | Description |
|----------|-----------|-------------|
| `normalize` | `(arr, method) → arr` | "minmax" or "l2" normalization |
| `standardize` | `(arr, mean, std) → arr` | Z-score standardization |
| `one_hot` | `(index, n) → arr` | One-hot encoding |
| `label_encode` | `(value, map) → int` | Categorical to numeric |
| `fill_missing` | `(arr, strategy) → arr` | Fill NaN: "zero", "mean", "median" |

### Computer Vision
| Function | Signature | Description |
|----------|-----------|-------------|
| `decode_image` | `(base64) → map` | Decode JPEG/PNG to pixel array |
| `resize_image` | `(pixels, sh, sw, dh, dw, c) → arr` | Bilinear resize |
| `normalize_image` | `(pixels, mean, std) → arr` | Per-channel normalization |
| `image_to_chw` | `(pixels, h, w, c) → arr` | HWC to CHW layout |
| `center_crop` | `(pixels, sh, sw, ch, cw, c) → arr` | Center crop |
| `grayscale` | `(pixels, h, w) → arr` | RGB to grayscale |
| `nms` | `(boxes, scores, iou) → arr` | Non-Maximum Suppression |

### Example: Sentiment Analysis Decoder
```rhai
fn execute(inputs) {
    let logits = inputs.get("logits").as_f64();
    let probs = softmax(logits);
    let label = argmax(probs);

    return #{
        "label": create_tensor_i64("label", [1], [label]),
        "probs": create_tensor_f64("probs", [3], probs),
    };
}
```

---

## Logging

Three-layer logging architecture, all non-blocking:

| Layer | Output | Content |
|-------|--------|---------|
| **stdout** | Terminal | Model loading, health, startup table only |
| **file** | `/tmp/logs/axon/axon-server.YYYY-MM-DD.json` | Everything (JSON, daily rotation) |
| **OTEL** | OTLP endpoint | Everything (if `OTEL_EXPORTER_OTLP_ENDPOINT` set) |

Inference tracing logs include `model`, `latency_ms`, `total_ms` for each request.

---

## Documentation

Full documentation on the **[Wiki](https://github.com/mustafacavusoglu/axon/wiki)**:

| Guide | Description |
|-------|-------------|
| [Config Reference](https://github.com/mustafacavusoglu/axon/wiki/Config-Reference) | YAML & pbtxt config format |
| [BLS / Scripting](https://github.com/mustafacavusoglu/axon/wiki/BLS-Scripting) | Rhai scripting engine |
| [Builtin Functions](https://github.com/mustafacavusoglu/axon/wiki/Builtin-Functions) | 23 preprocessing/postprocessing functions |
| [Ensemble Pipeline](https://github.com/mustafacavusoglu/axon/wiki/Ensemble-Pipeline) | Declarative model chaining |
| [Logging](https://github.com/mustafacavusoglu/axon/wiki/Logging) | Structured logging & OTEL |
| [Deployment](https://github.com/mustafacavusoglu/axon/wiki/Deployment) | CLI, Docker, Compose |
| [Metrics](https://github.com/mustafacavusoglu/axon/wiki/Metrics) | Prometheus metrics |

---

## Example Models

| Directory | Model | Platform | Description |
|-----------|-------|----------|-------------|
| `models/axon_turkish_safety/` | Turkish safety classifier | ensemble | BERT-based NLP pipeline (tokenizer → ONNX → decoder) |
| `models/cb_credit_risk/` | Credit risk classifier | onnxruntime_onnx | CatBoost (15 features → default probability) |
| `models/xgb_california_housing/` | Housing price regression | onnxruntime_onnx | XGBoost (8 features → price) |
| `models/skl_random_forest_breast_cancer/` | Breast cancer classifier | onnxruntime_onnx | sklearn Random Forest (30 features → benign/malignant) |
| `models/echo_preprocess/` | Echo preprocessor | script | Rhai script that passes input through |

Each model directory is self-contained at `models/<name>/` with this structure:
```
models/<name>/
├── config.pbtxt          # Model configuration (tracked)
└── 1/
    ├── model.onnx        # ONNX model file (gitignored — generate or download)
    ├── model.rhai        # Rhai script (tracked, for script/ensemble models)
    ├── tokenizer.json    # HuggingFace tokenizer (gitignored — see convert script)
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── vocab.txt
```

> **Note:** Large/generated files (`*.onnx`, `tokenizer.json`, `vocab.txt`, etc.) are gitignored.
> Generate them using the conversion scripts in `locals/scripts/` or download pre-trained models
> from HuggingFace.

```bash
# Pipeline inference — text to safety label
curl -s -X POST http://localhost:8000/v2/models/turkish_safety_pipeline/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"text","shape":[1],"datatype":"BYTES","data":["Bu çok güzel bir gün!"]}]}'
# → {"label": "SAFE", "score": 0.94}
```

---

## Upcoming Features

- Dynamic batching — accumulate requests into batches per model
- OpenVINO backend — Intel CPU optimization
- Model warmup — pre-warm ONNX sessions on load
- Authentication — API key + mTLS
- Binary tensor extension — raw bytes for large payloads
- Rate limiting middleware
- Model A/B traffic splitting
- LRU eviction — evict least-used model under memory pressure
- Swagger UI — OpenAPI 3.0 browsable docs
- NUMA-aware session pools — multi-socket server optimization
