# Axon — CPU Inference Server

> [English documentation](README.md)

Tek binary, Triton uyumlu, CPU oncelikli model sunum sistemi.  
**Dil:** Rust  
**Iletisim:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**BLS/Scripting:** Rhai (Python benzeri Rust-native script dili)  
**Hedef:** Kubernetes / Docker / Bare-metal

---

## Hizli Baslangic

### 1. Gereksinimler
```bash
# Rust (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# ONNX Runtime (macOS)
brew install onnxruntime

# ONNX Runtime (Linux)
# https://github.com/microsoft/onnxruntime/releases adresinden indirip /usr/local/lib'e kopyalayin

# Protobuf derleyici (Linux)
apt-get install protobuf-compiler
```

### 2. Build (kaynaktan derleme)
```bash
git clone https://github.com/mustafacavusoglu/axon.git
cd axon/inference-engine

# macOS
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib cargo build --release

# Linux
cargo build --release

# Binary: target/release/axon-server
```

### 3. Model deposu hazirlama
```bash
# ONNX modeli icin dizin yapisi:
# models/
# ├── model-adi/
# │   ├── config.pbtxt
# │   ├── 1/
# │   │   └── model.onnx

# Script modeli icin:
# models/
# ├── pipeline/
# │   ├── config.pbtxt     # platform: "script"
# │   ├── 1/
# │   │   └── model.rhai
```

### 4. Sunucuyu calistirma
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

# Saglik kontrolu
curl http://localhost:8000/v2/health/ready
```

### 5. Inference
```bash
curl -s -X POST http://localhost:8000/v2/models/model-adi/infer \
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

## BLS / Script Modelleri (Rhai)

`platform: "script"` ile preprocess, postprocess veya BLS (Business Logic Scripting) islemleri yapabilirsiniz.
Script dili olarak Python benzeri sentaksa sahip, Rust-native [Rhai](https://rhai.rs) kullanilir.

### Script Model Dizini
```
/models/
├── preprocess_model/
│   ├── config.pbtxt      # platform: "script"
│   ├── 1/
│   │   └── model.rhai    # Script dosyasi
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
    // inputs: obje haritasi (isim -> Tensor)

    // Tensor okuma
    let t = inputs.get("text");
    let text = t.as_string();
    let shape = t.shape;
    let dtype = t.datatype;
    let data = t.as_f64();     // f64 dizisi olarak
    let ints = t.as_i64();     // i64 dizisi olarak

    // Tensor olusturma
    let new_tensor = create_tensor_f64("name", shape_vals, data_vals);

    // BLS: baska bir modele inference
    let result = infer("diger_model", #{
        "input_name": some_tensor,
    });

    // Sonucu dondur
    return #{ "output_name": result.get("output_name") };
}
```

### BLS Ornegi

Preprocess + BLS ile xgb_california_housing modelini cagiran ornek (`model.rhai`):
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

Inference cagrisi:
```bash
curl -s -X POST http://localhost:8000/v2/models/preprocess_housing/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"median_income","shape":[1],"datatype":"FP32","data":[3.5]},
    {"name":"house_age","shape":[1],"datatype":"FP32","data":[20]},
    {"name":"avg_rooms","shape":[1],"datatype":"FP32","data":[5]},
    {"name":"avg_bedrooms","shape":[1],"datatype":"FP32","data":[1]},
    {"name":"population","shape":[1],"datatype":"FP32","data":[1200]},
    {"name":"avg_occupancy","shape":[1],"datatype":"FP32","data":[3]},
    {"name":"latitude","shape":[1],"datatype":"FP32","data":[34]},
    {"name":"longitude","shape":[1],"datatype":"FP32","data":[-118]}
  ]}'
```

### Mevcut Rhai API Fonksiyonlari

| Fonksiyon | Aciklama |
|-----------|----------|
| `tensor.name` | Tensor ismi (getter) |
| `tensor.shape` | Shape dizisi (getter) |
| `tensor.datatype` | Veri tipi: "FP32", "INT64", "BYTES" (getter) |
| `tensor.as_f64()` | Tensor verisini f64 dizisi olarak dondurur |
| `tensor.as_i64()` | Tensor verisini i64 dizisi olarak dondurur |
| `tensor.as_string()` | String tensor verisini dondurur |
| `create_tensor_f64(name, shape, data)` | FP32 tensor olusturur |
| `create_tensor_i64(name, shape, data)` | INT64 tensor olusturur |
| `infer(model_name, inputs)` | BLS: baska bir modele inference yapar |

---

## Metrikler

Prometheus metrikleri `:8002/metrics` uzerinde:

| Metrik | Tip | Label | Aciklama |
|--------|-----|-------|----------|
| `axon_requests_total` | counter | model, status | Model ve HTTP status bazinda istekler |
| `axon_inference_duration_seconds` | histogram | model | Uctan uca inference suresi |
| `axon_inflight_requests` | gauge | model | Su an islenen istek sayisi |
| `axon_queue_wait_seconds` | histogram | model | Concurrency permit bekleme suresi |
| `axon_models_loaded` | gauge | — | Yuklu model sayisi |
| `axon_model_info` | gauge | model, version | Model envanteri (1=hazir) |
| `axon_model_load_duration_seconds` | histogram | model | Diskten model yukleme suresi |
| `axon_model_load_errors_total` | counter | model | Model yukleme hatalari |
| `axon_circuit_breaker_trips_total` | counter | — | Circuit breaker aktiflesmeleri |
| `axon_server_info` | gauge | version | Server versiyon bilgisi |

### Ornek Alertler (Grafana)
```promql
# P99 latency > 500ms
histogram_quantile(0.99, rate(axon_inference_duration_seconds_bucket[5m])) > 0.5

# Model dolulugu (inflight yaklasik limit)
axon_inflight_requests / 4 > 0.8

# Hata orani > %5
rate(axon_requests_total{status=~"5.."}[5m]) / rate(axon_requests_total[5m]) > 0.05
```

---

## Gelistirme

```bash
# Derleme
cd inference-engine && cargo build --release

# Test
cargo test

# Lokal calistirma (ONNX Runtime gerektirir)
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
  ./target/release/axon-server --model-repository=./local_models/model_repository
```

### Bagimliliklar
- **Rust** (stable)
- **ONNX Runtime** — `brew install onnxruntime` (macOS) veya GitHub'dan indirin
- **Rhai** — embedded scripting engine (cargo otomatik ceker)

---

## Siradaki Ozellikler

- ~~BLS / Scripting~~ — Preprocess/postprocess/BLS Rhai scripting engine ✅
- Ensemble pipelines — modelleri zincirleme (A ciktisi -> B girdisi, config.pbtxt tabanli)
- Dynamic batching — modele ozel istekleri biriktirip toplu isleme
- OpenVINO backend — Intel CPU optimizasyonu
- Model warmup — ilk yuklemede ONNX oturumlarini isitma
- Authentication — API key + mTLS
- Binary tensor extension — buyuk veriler icin raw bytes
- Rate limiting middleware
- Model A/B traffic splitting
- LRU eviction — bellek baskisinda en az kullanilan modeli kaldirma
- Swagger UI — OpenAPI 3.0 tarayici tabanli dokumantasyon
- NUMA-aware session pools — cok soketli sunucu optimizasyonu

---
