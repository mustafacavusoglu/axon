# Axon — CPU Inference Server

> [English documentation](README.md) | [Wiki](https://github.com/mustafacavusoglu/axon/wiki)

Tek binary, Triton uyumlu, CPU öncelikli model sunum sistemi.  
**Dil:** Rust | **Runtime:** ONNX Runtime | **İletişim:** gRPC + HTTP/REST (KServe v2)  
**BLS/Scripting:** Rhai | **Config:** YAML + pbtxt | **Ensemble:** Declarative model zincirleme

---

## Hızlı Başlangıç

### 1. Gereksinimler
```bash
# Rust (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# ONNX Runtime (macOS)
brew install onnxruntime

# ONNX Runtime (Linux)
# https://github.com/microsoft/onnxruntime/releases adresinden indir, /usr/local/lib'e kopyala
```

### 2. Derleme
```bash
git clone https://github.com/mustafacavusoglu/axon.git
cd axon/inference-engine

# macOS
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib cargo build --release

# Linux
cargo build --release
```

### 3. Örnek modelle çalıştır
```bash
# Sunucuyu ML modeliyle başlat
./target/release/axon-server \
  --model-repository=../ml_model \
  --model-control-mode=poll

# Sağlık kontrolü
curl http://localhost:8000/v2/health/ready

# Inference — 8 ev özelliği → fiyat tahmini
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

## Dokümantasyon

Detaylı dökümantasyon **[Wiki](https://github.com/mustafacavusoglu/axon/wiki)** sayfasında:

| Rehber | Açıklama |
|--------|----------|
| [Config Referansı](https://github.com/mustafacavusoglu/axon/wiki/Config-Reference) | YAML & pbtxt config formatı |
| [BLS / Scripting](https://github.com/mustafacavusoglu/axon/wiki/BLS-Scripting) | Rhai scripting engine |
| [Ensemble Pipeline](https://github.com/mustafacavusoglu/axon/wiki/Ensemble-Pipeline) | Declarative model zincirleme |
| [Deployment](https://github.com/mustafacavusoglu/axon/wiki/Deployment) | CLI, Docker, Compose |
| [Metrics](https://github.com/mustafacavusoglu/axon/wiki/Metrics) | Prometheus metrikleri |

---

## Örnek Modeller

| Dizin | Modeller | Açıklama |
|-------|----------|----------|
| `ml_model/` | `xgb_housing` | Basit ONNX modeli (8 özellik → fiyat) |
| `nlp_model/` | `tokenizer`, `ner_model`, `decoder`, `pipeline` | BLS + Ensemble NLP pipeline |

```bash
# NLP ensemble — metinden entity etiketlerine (script yok)
curl -s -X POST http://localhost:8000/v2/models/pipeline/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"raw_text","shape":[1],"datatype":"BYTES","data":["John lives in Paris"]}]}'
# → "John: B-PER; lives: I-PER; Paris: B-LOC"
```

---

## Sıradaki Özellikler

- Dynamic batching — modele özel istekleri biriktirip toplu işleme
- OpenVINO backend — Intel CPU optimizasyonu
- Model warmup — ilk yüklemede ONNX oturumlarını ısıtma
- Authentication — API key + mTLS
- Binary tensor extension — büyük veriler için raw bytes
- Rate limiting middleware
- Model A/B traffic splitting
- LRU eviction — bellek baskısında en az kullanılan modeli kaldırma
- Swagger UI — OpenAPI 3.0 tarayıcı tabanlı dökümantasyon
- NUMA-aware session pools — çok soketli sunucu optimizasyonu
