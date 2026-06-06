# Axon — CPU Inference Server


> [Türkçe dökümantasyon](README_TR.md)


Triton-compatible, CPU-first model serving infrastructure.  
**Control Plane:** Go · **Inference Engine:** Rust  
**Transport:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**Target:** Kubernetes

---

## Quick Start

### Local
```bash
# Prerequisites: Rust, Go, ONNX Runtime (brew install onnxruntime)
./run.sh
```

### Docker
```bash
docker-compose up --build
```

### Kubernetes
```bash
kubectl apply -f deploy/k8s/
```

Then hit the server:
```bash
curl http://localhost:8080/v2/health/live
curl http://localhost:8080/v2/models
```

---

## Architecture

```
┌──────────────────────────────────────────┐
│              Kubernetes Pod              │
│                                          │
│  ┌────────────┐  ┌───────────────────┐   │
│  │ Go CP      │  │ Rust Engine       │   │
│  │ :8080 HTTP │  │ :unix socket      │   │
│  │ :8001 gRPC │──│ ONNX Runtime      │   │
│  └────────────┘  └───────────────────┘   │
│         │                  │             │
│         └──── /models ─────┘             │
└──────────────────────────────────────────┘
```

| Component | Language | Role |
|-----------|----------|------|
| Control Plane | Go | API, model registry, batching, health checks |
| Inference Engine | Rust | ONNX Runtime sessions, tensor execution |
| IPC | gRPC over Unix socket | Go ↔ Rust communication |

---

## Inference

See [sample-request.md](sample-request.md) for ready-to-use curl commands.

```bash
curl -s -X POST http://localhost:8080/v2/models/lgbm_credit_risk/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"age","shape":[1],"datatype":"FP32","data":[25.0]},
    {"name":"bmi","shape":[1],"datatype":"FP32","data":[22.5]}
  ]}'
```

Response (KServe v2):
```json
{
  "outputs": [
    {"name": "label",         "datatype": "INT64", "shape": [1],    "data": [1]},
    {"name": "probabilities", "datatype": "FP32",  "shape": [1, 2], "data": [0.23, 0.77]}
  ]
}
```

---

## Model Repository

Triton-compatible layout:
```
/models/
└── my-model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

---

## Development

```bash
make build        # Build both
make test         # Run all tests
make proto        # Regenerate protobuf code (buf)
```

---

## Upcoming Features

- Dynamic batching — accumulate requests into batches per model
- Ensemble pipelines — chain models (A output → B input)
- OpenVINO backend — Intel CPU optimization
- Model warmup — pre-warm ONNX sessions on load
- Authentication — API key + mTLS
- Binary tensor extension — raw bytes for large payloads
- Graceful rolling update — zero dropped requests
- Rate limiting middleware
- Model A/B traffic splitting
- LRU eviction — evict least-used model under memory pressure
- FSWatcher — hot reload on model file changes
- Swagger UI — OpenAPI 3.0 browsable docs
- Distributed trace propagation — Go ↔ Rust span linking
- NUMA-aware session pools — multi-socket server optimization

---
