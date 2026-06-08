# Axon — CPU Inference Server

> [Turkce dokumantasyon](README_TR.md)

Single-binary, Triton-compatible, CPU-first model serving.  
**Language:** Rust  
**Transport:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**BLS/Scripting:** Rhai (Python-like, Rust-native scripting language)  
**Target:** Kubernetes / Docker / Bare-metal

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

# Protobuf compiler (Linux)
apt-get install protobuf-compiler
```

### 2. Build from source
```bash
git clone https://github.com/mustafacavusoglu/axon.git
cd axon/inference-engine

# macOS
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib cargo build --release

# Linux
cargo build --release

# Binary: target/release/axon-server
```

### 3. Prepare model repository
```bash
# ONNX model layout:
# models/
# ├── model-name/
# │   ├── config.pbtxt
# │   ├── 1/
# │   │   └── model.onnx

# Script model layout:
# models/
# ├── pipeline/
# │   ├── config.pbtxt     # platform: "script"
# │   ├── 1/
# │   │   └── model.rhai
```

### 4. Start server
```bash
# macOS
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
  ./target/release/axon-server \
  --model-repository=./models \
  --model-control-mode=poll

# Linux
./target/release/axon-server \
  --model-repository=/models \
  --model-control-mode=poll

# Health check
curl http://localhost:8000/v2/health/ready
```

### 5. Inference
```bash
curl -s -X POST http://localhost:8000/v2/models/model-name/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"features","shape":[1],"datatype":"FP32","data":[42.5]}]}'
```

### Docker
```bash
docker run -v ./models:/models -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  mustdo12/axon-server:0.3.0 \
  --model-repository=/models \
  --model-control-mode=poll
```

### Docker Compose
```bash
docker-compose up
```

### Kubernetes
```bash
kubectl apply -f deploy/k8s/
```

Health check:
```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready
```

---

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-repository` | `/models` | Path to model repository |
| `--model-control-mode` | `none` | `none` or `poll` |
| `--repository-poll-secs` | `30` | Poll interval (when mode=poll) |
| `--http-port` | `8000` | HTTP/REST API port |
| `--grpc-port` | `8001` | gRPC API port |
| `--metrics-port` | `8002` | Prometheus metrics port |
| `--inference-timeout-ms` | `30000` | Per-request timeout |
| `--num-threads` | `0` (auto) | Worker threads (0 = CPU count) |
| `--concurrency-per-model` | `4` | Max concurrent inferences per model |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            axon-server (single binary)      │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ HTTP API │  │ gRPC API │  │ Metrics  │   │
│  │ :8000    │  │ :8001    │  │ :8002    │   │
│  └────┬─────┘  └────┬─────┘  └──────────┘   │
│       │              │                      │
│       └──────┬───────┘                      │
│              ▼                              │
│  ┌─────────────────────────────┐            │
│  │     Session Pool            │            │
│  │  ONNX Runtime (CPU)         │            │
│  │  per-model concurrency      │            │
│  └─────────────────────────────┘            │
│              │                              │
│              ▼                              │
│  ┌─────────────────────────────┐            │
│  │    Model Repository         │            │
│  │    /models/<name>/<ver>/    │            │
│  └─────────────────────────────┘            │
└─────────────────────────────────────────────┘
```

---

## HTTP API (KServe v2)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v2/health/live` | Liveness probe |
| GET | `/v2/health/ready` | Readiness probe |
| GET | `/v2` | Server metadata |
| GET | `/v2/models` | List loaded models |
| GET | `/v2/models/{name}` | Model metadata |
| GET | `/v2/models/{name}/versions/{ver}` | Version metadata |
| POST | `/v2/models/{name}/infer` | Inference (latest version) |
| POST | `/v2/models/{name}/versions/{ver}/infer` | Inference (specific version) |
| POST | `/v2/models/{name}/load` | Load model |
| POST | `/v2/models/{name}/unload` | Unload model |
| POST | `/v2/repository/index` | Repository index |

### Inference Example
```bash
curl -s -X POST http://localhost:8000/v2/models/lgbm_credit_risk/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "features", "shape": [1, 30], "datatype": "FP32", "data": [1.0, 2.0, ...]}
    ]
  }'
```

Response:
```json
{
  "id": "",
  "model_name": "lgbm_credit_risk",
  "model_version": "1",
  "outputs": [
    {"name": "label", "datatype": "INT64", "shape": [1], "data": [1]},
    {"name": "probabilities", "datatype": "FP32", "shape": [1, 2], "data": [0.23, 0.77]}
  ]
}
```

---

## gRPC API

KServe-compatible `GRPCInferenceService`:
- `ServerLive` / `ServerReady` / `ModelReady`
- `ServerMetadata` / `ModelMetadata`
- `ModelInfer`

Connect on port 8001 using the KServe proto definitions.

---

## Model Repository

Triton-compatible layout:
```
/models/
├── my-model/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
└── another-model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### config.pbtxt
```
name: "my-model"
platform: "onnxruntime_onnx"
max_batch_size: 8

input {
  name: "features"
  data_type: TYPE_FP32
  dims: [30]
}

output {
  name: "probabilities"
  data_type: TYPE_FP32
  dims: [2]
}

instance_group {
  count: 4
  kind: "KIND_CPU"
}
```

## BLS / Script Models (Rhai)

Use `platform: "script"` for preprocess, postprocess, or BLS (Business Logic Scripting).
The scripting language is [Rhai](https://rhai.rs) — a Python-like, Rust-native embedded language.
The inference engine contains **zero model-specific functions** — all business logic lives in `model.rhai`.

### Script Model Layout
```
models/
├── pipeline/
│   ├── config.pbtxt      # platform: "script"
│   ├── 1/
│   │   └── model.rhai    # Script file
```

### config.pbtxt (script model)
```
name: "pipeline"
platform: "script"
max_batch_size: 1

input {
  name: "features"
  data_type: TYPE_FP32
  dims: [1]
}
output {
  name: "result"
  data_type: TYPE_FP32
  dims: [1]
}

instance_group { count: 2 kind: KIND_CPU }
```

### Rhai API

Functions provided by the inference engine (tensor ops and BLS only):

| Function | Description |
|----------|-------------|
| `tensor.name` | Tensor name (getter) |
| `tensor.shape` | Shape array (getter) |
| `tensor.datatype` | Data type: "FP32", "INT64", "BYTES" (getter) |
| `tensor.as_f64()` | Returns data as f64 array |
| `tensor.as_i64()` | Returns data as i64 array |
| `tensor.as_string()` | Returns string data |
| `create_tensor_f64(name, shape, data)` | Create FP32 tensor |
| `create_tensor_i64(name, shape, data)` | Create INT64 tensor |
| `create_tensor_string(name, shape, data)` | Create string tensor |
| `infer(model_name, inputs)` | BLS: call inference on another model |

### Example 1: ML Preprocess (fill null and -1 with 0)

`ml_model/preprocess_ml/1/model.rhai` — replaces -1 and negative values with 0 before inference:

```rhai
fn clean(values) {
    let cleaned = [];
    for v in values {
        if v < 0.0 { cleaned.push(0.0); }
        else { cleaned.push(v); }
    }
    return cleaned;
}

fn execute(inputs) {
    let names = ["median_income","house_age","avg_rooms","avg_bedrooms",
                 "population","avg_occupancy","latitude","longitude"];
    let cleaned = #{};
    for name in names {
        let t = inputs.get(name);
        cleaned[name] = create_tensor_f64(name, t.shape, clean(t.as_f64()));
    }
    return infer("xgb_housing", cleaned);
}
```

```bash
# Data with -1 values (preprocess auto-fills with 0)
curl -s -X POST http://localhost:8000/v2/models/preprocess_ml/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"median_income","shape":[1],"datatype":"FP32","data":[-1]},
    {"name":"house_age","shape":[1],"datatype":"FP32","data":[20]},
    {"name":"avg_rooms","shape":[1],"datatype":"FP32","data":[-1]},
    {"name":"avg_bedrooms","shape":[1],"datatype":"FP32","data":[1]},
    {"name":"population","shape":[1],"datatype":"FP32","data":[1200]},
    {"name":"avg_occupancy","shape":[1],"datatype":"FP32","data":[-1]},
    {"name":"latitude","shape":[1],"datatype":"FP32","data":[34]},
    {"name":"longitude","shape":[1],"datatype":"FP32","data":[-118]}
  ]}'
```

### Example 2: NLP NER Pipeline (attention mask + decode)

`nlp_model/ner_pipeline/1/model.rhai` — receives pre-tokenized input_ids, adds attention mask, calls BERT via BLS, decodes logits to entity labels:

```rhai
fn execute(inputs) {
    let ids = inputs.get("input_ids");
    let n = ids.shape[1];

    let mask_data = []; let type_data = []; let i = 0;
    while i < n { mask_data.push(1); type_data.push(0); i += 1; }

    let result = infer("ner_model", #{
        "input_ids": ids,
        "attention_mask": create_tensor_i64("mask", [1,n], mask_data),
        "token_type_ids": create_tensor_i64("type", [1,n], type_data),
    });

    let logits = result.get("logits").as_f64();
    let labels = ["O","B-PER","I-PER","B-ORG","I-ORG","B-LOC","I-LOC"];

    let output = "";
    for pos in 1..n-1 {
        let best = 0; let best_score = logits[pos*7];
        for j in 1..7 { if logits[pos*7+j] > best_score { best = j; best_score = logits[pos*7+j]; } }
        if labels[best] != "O" {
            if output != "" { output += "; "; }
            output += "token " + pos + " (id=" + ids.as_i64()[pos] + "): " + labels[best];
        }
    }
    return #{ "entities": create_tensor_string("entities", [1], [output]) };
}
```

```bash
# Pre-tokenized IDs from HuggingFace tokenizer
curl -s -X POST http://localhost:8000/v2/models/ner_pipeline/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"input_ids","shape":[1,13],"datatype":"INT64","data":[2,3222,11,2054,4611,4542,16,2673,11,69,6128,18,3]}]}'
# Output: "token 1 (id=3222): B-LOC; token 5 (id=4542): B-PER; token 7 (id=2673): B-LOC"
```

---

## Metrics

Prometheus metrics at `:8002/metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `axon_requests_total` | counter | model, status | Requests by model and HTTP status |
| `axon_inference_duration_seconds` | histogram | model | End-to-end inference latency |
| `axon_inflight_requests` | gauge | model | Currently processing requests |
| `axon_queue_wait_seconds` | histogram | model | Time waiting for concurrency permit |
| `axon_models_loaded` | gauge | — | Number of loaded models |
| `axon_model_info` | gauge | model, version | Model inventory (1=ready) |
| `axon_model_load_duration_seconds` | histogram | model | Time to load model from disk |
| `axon_model_load_errors_total` | counter | model | Model load failures |
| `axon_circuit_breaker_trips_total` | counter | — | Circuit breaker activations |
| `axon_server_info` | gauge | version | Server version identifier |

### Key Alerts (Grafana examples)
```promql
# P99 latency > 500ms
histogram_quantile(0.99, rate(axon_inference_duration_seconds_bucket[5m])) > 0.5

# Model saturation (inflight near concurrency limit)
axon_inflight_requests / 4 > 0.8

# Error rate > 5%
rate(axon_requests_total{status=~"5.."}[5m]) / rate(axon_requests_total[5m]) > 0.05
```

---

## Development

```bash
# Build
cd inference-engine && cargo build --release

# Test
cargo test

# Run locally (requires ONNX Runtime)
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
  ./target/release/axon-server --model-repository=./local_models/model_repository
```

### Dependencies
- **Rust** (stable)
- **ONNX Runtime** — `brew install onnxruntime` (macOS) or download from GitHub
- **Rhai** — embedded scripting engine (auto-fetched by cargo)

---

## Upcoming Features

- ~~BLS / Scripting~~ — Preprocess/postprocess/BLS via Rhai scripting engine ✅
- Ensemble pipelines — chain models via config.pbtxt (A output -> B input)
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

---
