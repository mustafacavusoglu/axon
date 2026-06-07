# Axon — CPU Inference Server

> [Turkce dokumantasyon](README_TR.md)

Single-binary, Triton-compatible, CPU-first model serving.  
**Language:** Rust  
**Transport:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**Target:** Kubernetes / Docker / Bare-metal

---

## Quick Start

### Binary
```bash
axon-server \
  --model-repository=/models \
  --model-control-mode=poll \
  --repository-poll-secs=30 \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002
```

### Docker
```bash
docker run -v ./models:/models -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  mustdo12/axon-server:0.2.0 \
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
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ HTTP API │  │ gRPC API │  │ Metrics  │  │
│  │ :8000    │  │ :8001    │  │ :8002    │  │
│  └────┬─────┘  └────┬─────┘  └──────────┘  │
│       │              │                      │
│       └──────┬───────┘                      │
│              ▼                              │
│  ┌─────────────────────────────┐            │
│  │     Session Pool            │            │
│  │  ONNX Runtime (CPU)        │            │
│  │  per-model concurrency     │            │
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

---

## Metrics

Prometheus metrics at `:8002/metrics`:
- `axon_requests_total` — Total inference requests
- `axon_models_loaded` — Number of loaded models
- `axon_inference_latency_ms{model="..."}` — Per-model latency histogram

---

## Development

```bash
# Build
cd inference-engine && cargo build --release

# Run locally
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
  ./target/release/axon-server --model-repository=./local_models/model_repository

# Test
cargo test
```

---

## Upcoming Features

- Dynamic batching — accumulate requests into batches per model
- Ensemble pipelines — chain models (A output -> B input)
- OpenVINO backend — Intel CPU optimization
- Model warmup — pre-warm ONNX sessions on load
- Authentication — API key + mTLS
- Binary tensor extension — raw bytes for large payloads
- Graceful rolling update — zero dropped requests
- Rate limiting middleware
- Model A/B traffic splitting
- LRU eviction — evict least-used model under memory pressure
- Swagger UI — OpenAPI 3.0 browsable docs
- NUMA-aware session pools — multi-socket server optimization

---
