# Axon — CPU Inference Server

> [Turkce dokumantasyon](README_TR.md)

Single-binary, Triton-compatible, CPU-first model serving.  
**Language:** Rust  
**Transport:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**BLS/Scripting:** Rhai (Python-like, Rust-native scripting language)  
**Config:** YAML (`config.yaml`) or Triton-format (`config.pbtxt`)  
**Ensemble:** Declarative model chaining via config (no scripting needed)  
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
# │   ├── config.yaml        # or config.pbtxt
# │   ├── 1/
# │   │   └── model.onnx

# Script model layout (BLS):
# models/
# ├── pipeline/
# │   ├── config.yaml        # platform: "script"
# │   ├── 1/
# │   │   └── model.rhai

# Ensemble model layout (declarative, no scripting):
# models/
# ├── ensemble-name/
# │   ├── config.yaml        # platform: "ensemble"
# │   ├── 1/                 # version dir (no model file)
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

### Config Format — YAML & pbtxt

Axon supports two config formats. **YAML is recommended** for readability; **pbtxt** is kept for Triton compatibility.

When both `config.yaml` and `config.pbtxt` exist, **YAML takes priority**.

#### YAML (recommended)
```yaml
name: my-model
platform: onnxruntime_onnx
max_batch_size: 8

inputs:
  - name: features
    data_type: TYPE_FP32
    dims: [30]

outputs:
  - name: probabilities
    data_type: TYPE_FP32
    dims: [2]

instance_groups:
  - count: 4
    kind: KIND_CPU
```

#### pbtxt (Triton-compatible fallback)
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

### Example 1: ML Model (direct inference)

`ml_model/xgb_housing/` — simple XGBoost housing price predictor, 8 features → 1 output:

```yaml
# config.yaml
name: xgb_housing
platform: onnxruntime_onnx
max_batch_size: 8

inputs:
  - name: median_income
    data_type: TYPE_FP32
    dims: [1]
  # ... 7 more features

outputs:
  - name: variable
    data_type: TYPE_FP32
    dims: [1, 1]

instance_groups:
  - count: 2
    kind: KIND_CPU
```

```bash
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

### Example 2: NLP Pipeline (tokenizer → ner_model → decoder)

`nlp_model/` contains four models that demonstrate BLS and ensemble chaining:

| Model | Type | Description |
|-------|------|-------------|
| `tokenizer` | Script (BLS) | Text → input_ids, attention_mask, token_type_ids, words |
| `ner_model` | ONNX | BERT-based NER (3 token inputs → logits) |
| `decoder` | Script (BLS) | logits + words → named entities (B-PER, B-LOC…) |
| `pipeline` | Ensemble | Declarative 3-step: tokenizer → ner_model → decoder |

**Tokenizer `model.rhai`** — reads vocab, tokenizes text, outputs words for downstream decoding:

```rhai
fn execute(inputs) {
    let text = inputs.get("text").as_string();
    // ... build vocab map, split words, convert to token IDs
    return #{
        "input_ids": create_tensor_i64("input_ids", [1, n], input_ids),
        "attention_mask": create_tensor_i64("attention_mask", [1, n], attention_mask),
        "token_type_ids": create_tensor_i64("token_type_ids", [1, n], token_type_ids),
        "words": create_tensor_string("words", [1], [word_str]),
    };
}
```

```bash
# Tokenize plain text
curl -s -X POST http://localhost:8000/v2/models/tokenizer/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"text","shape":[1],"datatype":"BYTES","data":["John lives in Paris"]}]}'

# Direct NER model inference
curl -s -X POST http://localhost:8000/v2/models/ner_model/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"input_ids","shape":[1,6],"datatype":"INT64","data":[2,7255,1,2091,7700,3]},
    {"name":"attention_mask","shape":[1,6],"datatype":"INT64","data":[1,1,1,1,1,1]},
    {"name":"token_type_ids","shape":[1,6],"datatype":"INT64","data":[0,0,0,0,0,0]}
  ]}'
```

---

## Ensemble Pipelines

Declarative model chaining via `config.pbtxt` / `config.yaml` — **no scripting required**.  
Use `platform: "ensemble"` and define `ensemble_scheduling` steps with `input_map` / `output_map`.

Each step maps tensors between models: step outputs flow into later step inputs via the ensemble tensor pool. Intermediate tensors (like `ner_logits`) pass between steps transparently.

### Example: NLP Pipeline (3-step ensemble)

`nlp_model/pipeline/config.pbtxt` — 3-step ensemble that chains `tokenizer` → `ner_model` → `decoder`:

```protobuf
name: "pipeline"
platform: "ensemble"
max_batch_size: 1

input  { name: "raw_text"  data_type: TYPE_STRING  dims: [1] }
output { name: "entities"  data_type: TYPE_STRING  dims: [1] }

ensemble_scheduling {
  step [
    {   # Step 1: Text → token IDs + words
      model_name: "tokenizer"
      model_version: -1
      input_map  [ { key: "text"  value: "raw_text" } ]
      output_map [
        { key: "input_ids"       value: "ids" }
        { key: "attention_mask"  value: "mask" }
        { key: "token_type_ids"  value: "types" }
        { key: "words"           value: "words" }
      ]
    }
    {   # Step 2: Token IDs → NER logits
      model_name: "ner_model"
      model_version: -1
      input_map [
        { key: "input_ids"       value: "ids" }
        { key: "attention_mask"  value: "mask" }
        { key: "token_type_ids"  value: "types" }
      ]
      output_map [ { key: "logits"  value: "ner_logits" } ]
    }
    {   # Step 3: logits + words → named entities
      model_name: "decoder"
      model_version: -1
      input_map [
        { key: "logits"  value: "ner_logits" }
        { key: "words"   value: "words" }
      ]
      output_map [ { key: "entities"  value: "entities" } ]
    }
  ]
}

instance_group { count: 1  kind: KIND_CPU }
```

YAML is cleaner:
```yaml
name: pipeline
platform: ensemble

inputs:
  - name: raw_text
    data_type: TYPE_STRING
    dims: [1]
outputs:
  - name: entities
    data_type: TYPE_STRING
    dims: [1]

ensemble_scheduling:
  steps:
    - model_name: tokenizer
      input_map: [{ key: text, value: raw_text }]
      output_map:
        - { key: input_ids, value: ids }
        - { key: attention_mask, value: mask }
        - { key: words, value: words }
    - model_name: ner_model
      input_map:
        - { key: input_ids, value: ids }
        - { key: attention_mask, value: mask }
      output_map: [{ key: logits, value: ner_logits }]
    - model_name: decoder
      input_map:
        - { key: logits, value: ner_logits }
        - { key: words, value: words }
      output_map: [{ key: entities, value: entities }]
```

```bash
# Ensemble pipeline (text → named entities, declarative — no scripting)
curl -s -X POST http://localhost:8000/v2/models/pipeline/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"raw_text","shape":[1],"datatype":"BYTES","data":["John lives in Paris"]}]}'
# → "John: B-PER; lives: I-PER; Paris: B-LOC"
```

### Directory layout

```
nlp_model/
├── tokenizer/
│   ├── config.yaml           # platform: "script"
│   ├── 1/model.rhai          # text → token IDs + words
│   └── 1/vocab.txt
├── ner_model/
│   ├── config.yaml           # platform: "onnxruntime_onnx"
│   └── 1/model.onnx
├── decoder/
│   ├── config.yaml           # platform: "script"
│   └── 1/model.rhai          # logits + words → named entities
└── pipeline/                 # Ensemble: tokenizer → ner_model → decoder
    ├── config.yaml
    └── 1/                    # version dir (empty)
```

### How it works
1. **Top-level inputs/outputs** define the ensemble's external API
2. **`input_map`**: maps ensemble tensor → step input (`key` = step input name, `value` = ensemble tensor name)
3. **`output_map`**: maps step output → ensemble tensor (`key` = step output name, `value` = ensemble tensor name)
4. Steps run sequentially; each step's outputs become available for later steps' inputs
5. `model_version: -1` always uses the latest version of the model

### Available formats
The config parser supports both block and list format:
- `step { ... }` / `step [ { ... } { ... } ]`
- `input_map { key: "x" value: "y" }` / `input_map [ { key: "x" value: "y" } ]`
- `step { ... }` / `step [ { ... } { ... } ]`
- `input_map { key: "x" value: "y" }` / `input_map [ { key: "x" value: "y" } ]`

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
- ~~Ensemble pipelines~~ — Declarative model chaining via config (pbtxt + YAML, no scripting) ✅
- ~~YAML config~~ — config.yaml support alongside config.pbtxt (serde_yaml) ✅
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
