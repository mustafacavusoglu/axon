# Axon — CPU Inference Server

> [English documentation](README.md)

Tek binary, Triton uyumlu, CPU oncelikli model sunum sistemi.  
**Dil:** Rust  
**Iletisim:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**Hedef:** Kubernetes / Docker / Bare-metal

---

## Hizli Baslangic

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

Saglik kontrolu:
```bash
curl http://localhost:8000/v2/health/live
curl http://localhost:8000/v2/health/ready
```

---

## CLI Parametreleri

| Parametre | Varsayilan | Aciklama |
|-----------|-----------|----------|
| `--model-repository` | `/models` | Model deposu yolu |
| `--model-control-mode` | `none` | `none` veya `poll` |
| `--repository-poll-secs` | `30` | Polling araligi (mode=poll iken) |
| `--http-port` | `8000` | HTTP/REST API portu |
| `--grpc-port` | `8001` | gRPC API portu |
| `--metrics-port` | `8002` | Prometheus metrics portu |
| `--inference-timeout-ms` | `30000` | Istek basi timeout |
| `--num-threads` | `0` (otomatik) | Worker thread sayisi (0 = CPU sayisi) |
| `--concurrency-per-model` | `4` | Model basi maks esanli inference |

---

## Mimari

```
┌─────────────────────────────────────────────┐
│            axon-server (tek binary)         │
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
│  │  model basi concurrency     │            │
│  └─────────────────────────────┘            │
│              │                              │
│              ▼                              │
│  ┌─────────────────────────────┐            │
│  │    Model Repository         │            │
│  │    /models/<isim>/<ver>/    │            │
│  └─────────────────────────────┘            │
└─────────────────────────────────────────────┘
```

---

## HTTP API (KServe v2)

| Metot | Endpoint | Aciklama |
|-------|----------|----------|
| GET | `/v2/health/live` | Liveness probe |
| GET | `/v2/health/ready` | Readiness probe |
| GET | `/v2` | Sunucu metadata |
| GET | `/v2/models` | Yuklu modelleri listele |
| GET | `/v2/models/{isim}` | Model metadata |
| GET | `/v2/models/{isim}/versions/{ver}` | Versiyon metadata |
| POST | `/v2/models/{isim}/infer` | Inference (son versiyon) |
| POST | `/v2/models/{isim}/versions/{ver}/infer` | Inference (belirli versiyon) |
| POST | `/v2/models/{isim}/load` | Model yukle |
| POST | `/v2/models/{isim}/unload` | Model kaldir |
| POST | `/v2/repository/index` | Depo indeksi |

### Inference Ornegi
```bash
curl -s -X POST http://localhost:8000/v2/models/lgbm_credit_risk/infer \
  -H 'Content-Type: application/json' \
  -d '{
    "inputs": [
      {"name": "features", "shape": [1, 30], "datatype": "FP32", "data": [1.0, 2.0, ...]}
    ]
  }'
```

Yanit:
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

KServe uyumlu `GRPCInferenceService`:
- `ServerLive` / `ServerReady` / `ModelReady`
- `ServerMetadata` / `ModelMetadata`
- `ModelInfer`

Port 8001 uzerinden KServe proto tanimlariyla baglanin.

---

## Model Deposu

Triton uyumlu dizin yapisi:
```
/models/
├── benim-modelim/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/
│       └── model.onnx
└── diger-model/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

### config.pbtxt
```
name: "benim-modelim"
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

## Metrikler

Prometheus metrikleri `:8002/metrics` uzerinde:
- `axon_requests_total` — Toplam inference istegi
- `axon_models_loaded` — Yuklu model sayisi
- `axon_inference_latency_ms{model="..."}` — Model basi latency histogrami

---

## Gelistirme

```bash
# Derleme
cd inference-engine && cargo build --release

# Lokal calistirma
ORT_DYLIB_PATH=/path/to/libonnxruntime.dylib \
  ./target/release/axon-server --model-repository=./local_models/model_repository

# Test
cargo test
```

---

## Siradaki Ozellikler

- Dynamic batching — modele ozel istekleri biriktirip toplu isleme
- Ensemble pipelines — modelleri zincirleme (A ciktisi -> B girdisi)
- OpenVINO backend — Intel CPU optimizasyonu
- Model warmup — ilk yuklemede ONNX oturumlarini isitma
- Authentication — API key + mTLS
- Binary tensor extension — buyuk veriler icin raw bytes
- Graceful rolling update — sifir istek kaybiyla yeniden baslatma
- Rate limiting middleware
- Model A/B traffic splitting
- LRU eviction — bellek baskisinda en az kullanilan modeli kaldirma
- Swagger UI — OpenAPI 3.0 tarayici tabanli dokumantasyon
- NUMA-aware session pools — cok soketli sunucu optimizasyonu

---
