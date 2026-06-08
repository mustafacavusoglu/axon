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

### Script Model Layout
```
/models/
├── preprocess_model/
│   ├── config.pbtxt      # platform: "script"
│   ├── 1/
│   │   └── model.rhai    # Script file
```

### config.pbtxt (script model)
```
name: "preprocess_model"
platform: "script"
max_batch_size: 1

input {
  name: "text"
  data_type: TYPE_STRING
  dims: [1]
}
output {
  name: "processed"
  data_type: TYPE_FP32
  dims: [1, 2]
}
```

### model.rhai API

```rhai
fn execute(inputs) {
    // inputs: object map (name -> Tensor)

    // Reading tensors
    let t = inputs.get("text");
    let text = t.as_string();
    let shape = t.shape;
    let dtype = t.datatype;
    let data = t.as_f64();     // as f64 array
    let ints = t.as_i64();     // as i64 array

    // Creating tensors
    let new_tensor = create_tensor_f64("name", shape_vals, data_vals);

    // BLS: inference on another model
    let result = infer("other_model", #{
        "input_name": some_tensor,
    });

    // Return result
    return #{ "output_name": result.get("output_name") };
}
```

### BLS Example

Preprocess + BLS calling xgb_california_housing (`model.rhai`):
```rhai
fn execute(inputs) {
    let income = inputs.get("median_income");
    let vals = income.as_f64();
    let scaled = [];
    for v in vals {
        scaled.push(v * 1.5);
    }
    let scaled_income = create_tensor_f64("median_income", income.shape, scaled);

    let mod_inputs = #{
        "median_income": scaled_income,
        "house_age": inputs.get("house_age"),
        "avg_rooms": inputs.get("avg_rooms"),
        "avg_bedrooms": inputs.get("avg_bedrooms"),
        "population": inputs.get("population"),
        "avg_occupancy": inputs.get("avg_occupancy"),
        "latitude": inputs.get("latitude"),
        "longitude": inputs.get("longitude"),
    };

    return infer("xgb_california_housing", mod_inputs);
}
```

### Available Rhai API Functions

| Function | Description |
|----------|-------------|
| `tensor.name` | Tensor name (getter) |
| `tensor.shape` | Shape array (getter) |
| `tensor.datatype` | Data type: "FP32", "INT64", "BYTES" (getter) |
| `tensor.as_f64()` | Returns tensor data as f64 array |
| `tensor.as_i64()` | Returns tensor data as i64 array |
| `tensor.as_string()` | Returns string tensor data |
| `create_tensor_f64(name, shape, data)` | Create FP32 tensor |
| `create_tensor_i64(name, shape, data)` | Create INT64 tensor |
| `infer(model_name, inputs)` | BLS: call inference on another model |

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
