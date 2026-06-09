# Axon — CPU Inference Server

> [English documentation](README.md)

Tek binary, Triton uyumlu, CPU oncelikli model sunum sistemi.  
**Dil:** Rust  
**Iletisim:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**BLS/Scripting:** Rhai (Python benzeri Rust-native script dili)  
**Config:** YAML (`config.yaml`) veya Triton formatı (`config.pbtxt`)  
**Ensemble:** Declarative model zincirleme (script gerektirmez)  
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
# │   ├── config.yaml        # veya config.pbtxt
# │   ├── 1/
# │   │   └── model.onnx

# Script modeli icin (BLS):
# models/
# ├── pipeline/
# │   ├── config.yaml        # platform: "script"
# │   ├── 1/
# │   │   └── model.rhai

# Ensemble modeli icin (declarative, script gerektirmez):
# models/
# ├── ensemble-adi/
# │   ├── config.yaml        # platform: "ensemble"
# │   ├── 1/                 # versiyon dizini (dosya gerekmez)
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

### Config Formati — YAML & pbtxt

Axon iki config formatini destekler. **YAML** okunabilirlik acisindan onerilir; **pbtxt** Triton uyumlulugu icin korunur.

Ayni dizinde hem `config.yaml` hem `config.pbtxt` varsa, **YAML onceliklidir**.

#### YAML (onerilen)
```yaml
name: benim-modelim
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

#### pbtxt (Triton uyumlu fallback)
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
Inference engine'e model-ozel hicbir fonksiyon eklenmez — tum is mantigi `model.rhai` icindedir.

### Script Model Dizini
```
models/
├── pipeline/
│   ├── config.yaml         # veya config.pbtxt — platform: "script"
│   ├── 1/
│   │   └── model.rhai    # Script dosyasi
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

Inference engine tarafindan saglanan fonksiyonlar (sadece tensor ve BLS):

| Fonksiyon | Aciklama |
|-----------|----------|
| `tensor.name` | Tensor ismi (getter) |
| `tensor.shape` | Shape dizisi (getter) |
| `tensor.datatype` | Veri tipi: "FP32", "INT64", "BYTES" (getter) |
| `tensor.as_f64()` | Veriyi f64 dizisi olarak dondurur |
| `tensor.as_i64()` | Veriyi i64 dizisi olarak dondurur |
| `tensor.as_string()` | String veriyi dondurur |
| `create_tensor_f64(name, shape, data)` | FP32 tensor olusturur |
| `create_tensor_i64(name, shape, data)` | INT64 tensor olusturur |
| `create_tensor_string(name, shape, data)` | String tensor olusturur |
| `infer(model_name, inputs)` | BLS: baska bir modele inference yapar |

### Ornek 1: ML Preprocess (null ve -1 temizleme)

`ml_model/preprocess_ml/1/model.rhai` — gelen ozelliklerdeki -1 ve negatif degerleri 0 ile doldurup modele gonderir:

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
# -1'li veri (preprocess otomatik 0 yapar)
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

### Ornek 2: NLP NER Pipeline (attention mask + decode)

`nlp_model/ner_pipeline/1/model.rhai` — onceden tokenize edilmis input_ids alir, attention mask ekler, BERT modeline BLS yapar, logits'leri cozup entity etiketlerini dondurur:

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
# HuggingFace tokenizer ile onceden tokenize edilmis ID'ler
curl -s -X POST http://localhost:8000/v2/models/ner_pipeline/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"input_ids","shape":[1,13],"datatype":"INT64","data":[2,3222,11,2054,4611,4542,16,2673,11,69,6128,18,3]}]}'
# Cikti: "token 1 (id=3222): B-LOC; token 5 (id=4542): B-PER; token 7 (id=2673): B-LOC"
```

---

## Ensemble Pipeline

Declarative model zincirleme — `config.pbtxt` / `config.yaml` ile, **hic script yazmadan**.  
`platform: "ensemble"` kullanip, `ensemble_scheduling` adimlarini `input_map` / `output_map` ile tanimlarsiniz.

Her adim, tensor'lari modeller arasi aktarir: bir step'in ciktisi sonraki step'in girdisine map'lenir.

### Ornek: NLP Pipeline (tokenizer → NER)

`nlp_model/nlp_ensemble/config.pbtxt` — tokenizer ciktisini NER modeline zincirleyen 2 adimli ensemble:

```protobuf
name: "nlp_ensemble"
platform: "ensemble"
max_batch_size: 1

input  { name: "raw_text"  data_type: TYPE_STRING  dims: [1] }
output { name: "entities"  data_type: TYPE_STRING  dims: [1] }

ensemble_scheduling {
  step [
    {   # Adim 1: Text → token ID'leri + attention mask
      model_name: "tokenizer_model"
      model_version: -1
      input_map  [ { key: "text"  value: "raw_text" } ]
      output_map [
        { key: "input_ids"       value: "ids" }
        { key: "attention_mask"  value: "mask" }
        { key: "token_type_ids"  value: "types" }
      ]
    }
    {   # Adim 2: Token ID'leri → NER logits
      model_name: "ner_model"
      model_version: -1
      input_map [
        { key: "input_ids"       value: "ids" }
        { key: "attention_mask"  value: "mask" }
        { key: "token_type_ids"  value: "types" }
      ]
      output_map [ { key: "logits"  value: "entities" } ]
    }
  ]
}

instance_group { count: 1  kind: KIND_CPU }
```

YAML daha okunabilir:
```yaml
ensemble_scheduling:
  steps:
    - model_name: tokenizer_model
      input_map: [{ key: text, value: raw_text }]
      output_map:
        - { key: input_ids, value: ids }
        - { key: attention_mask, value: mask }
```

### Nasil calisir
1. **En ustteki inputs/outputs** ensemble'in disariya actigi API'yi tanimlar
2. **`input_map`**: ensemble tensor → step girdisi (`key` = step'in bekledigi isim, `value` = ensemble havuzundaki tensor)
3. **`output_map`**: step ciktisi → ensemble tensor (`key` = step'in urettigi isim, `value` = ensemble havuzuna yazilacak isim)
4. Adimlar sirayla calisir; her adimin ciktisi sonraki adimlarin girdisine map'lenebilir
5. `model_version: -1` her zaman modelin en son versiyonunu kullanir

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
- ~~Ensemble pipelines~~ — Declarative model zincirleme (pbtxt + YAML, script gerektirmez) ✅
- ~~YAML config~~ — config.yaml destegi (config.pbtxt ile birlikte, serde_yaml) ✅
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
