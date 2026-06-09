# Axon — CPU Inference Server

> [Türkçe dokümantasyon](README_TR.md) | [Wiki](https://github.com/mustafacavusoglu/axon/wiki)

Single-binary, Triton-compatible, CPU-first model serving.  
**Language:** Rust | **Runtime:** ONNX Runtime | **Transport:** gRPC + HTTP/REST (KServe v2)  
**BLS/Scripting:** Rhai | **Config:** YAML + pbtxt | **Ensemble:** Declarative model chaining

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
# Start server with ML model
./target/release/axon-server \
  --model-repository=../ml_model \
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
docker run -v ./models:/models -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  mustdo12/axon-server:latest \
  --model-repository=/models --model-control-mode=poll
```

---

## Documentation

Full documentation on the **[Wiki](https://github.com/mustafacavusoglu/axon/wiki)**:

| Guide | Description |
|-------|-------------|
| [Config Reference](https://github.com/mustafacavusoglu/axon/wiki/Config-Reference) | YAML & pbtxt config format |
| [BLS / Scripting](https://github.com/mustafacavusoglu/axon/wiki/BLS-Scripting) | Rhai scripting engine |
| [Ensemble Pipeline](https://github.com/mustafacavusoglu/axon/wiki/Ensemble-Pipeline) | Declarative model chaining |
| [Deployment](https://github.com/mustafacavusoglu/axon/wiki/Deployment) | CLI, Docker, Compose |
| [Metrics](https://github.com/mustafacavusoglu/axon/wiki/Metrics) | Prometheus metrics |

---

## Example Models

| Directory | Models | Description |
|-----------|--------|-------------|
| `ml_model/` | `xgb_housing` | Simple ONNX model (8 features → price) |
| `nlp_model/` | `tokenizer`, `ner_model`, `decoder`, `pipeline` | BLS + Ensemble NLP pipeline |

```bash
# NLP ensemble — text to named entities (no scripting)
curl -s -X POST http://localhost:8000/v2/models/pipeline/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"raw_text","shape":[1],"datatype":"BYTES","data":["John lives in Paris"]}]}'
# → "John: B-PER; lives: I-PER; Paris: B-LOC"
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
