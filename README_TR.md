# Axon — CPU Inference Server

> [English documentation](README.md) | [Wiki](https://github.com/mustafacavusoglu/axon/wiki)

Tek binary, Triton uyumlu, CPU öncelikli model sunum sistemi.  
**Dil:** Rust | **Runtime:** ONNX Runtime | **İletişim:** gRPC + HTTP/REST (KServe v2)  
**BLS/Scripting:** Rhai | **Config:** YAML + pbtxt | **Ensemble:** Declarative model zincirleme

---

## Özellikler

- **KServe v2 API** — HTTP + gRPC, tam model yönetimi
- **Ensemble Pipeline** — Config ile declarative çok-model zincirleme
- **BLS (Rhai Scripting)** — Özel pre/post processing mantığı
- **23 Builtin Fonksiyon** — ML, NLP, CV, tabular preprocessing/postprocessing
- **HuggingFace Tokenizer** — Native tokenizer.json desteği
- **Yapısal Loglama** — JSON dosya (günlük rotation) + OTEL + filtrelenmiş stdout
- **Inference Timeout** — Sunucu bazında ayarlanabilir timeout
- **Circuit Breaker** — Yüklenemeyen modelleri otomatik atlama
- **Prometheus Metrikleri** — Gecikme, throughput, inflight, kuyruk bekleme
- **Docker Multi-arch** — linux/amd64 + linux/arm64

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

## CLI Referansı

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--model-repository` | `/models` | Model deposu yolu |
| `--model-control-mode` | `none` | `none` veya `poll` (otomatik yeniden yükleme) |
| `--repository-poll-secs` | `30` | Tarama aralığı (saniye) |
| `--http-port` | `8000` | HTTP REST portu |
| `--grpc-port` | `8001` | gRPC portu |
| `--metrics-port` | `8002` | Prometheus metrik portu |
| `--inference-timeout-ms` | `30000` | Maks inference süresi (504 döner) |
| `--num-threads` | `0` | Inference thread sayısı (0 = otomatik) |
| `--concurrency-per-model` | `4` | Model başına maks eşzamanlı istek |
| `--log-level` | `info` | Log seviyesi: trace, debug, info, warn, error |
| `--log-dir` | `/tmp/logs/axon` | Log dizini (JSON, günlük rotation) |

---

## Builtin Fonksiyonlar (Rhai BLS)

Tüm Rhai scriptlerinde kullanılabilir 23 hazır fonksiyon:

### ML / Matematik
| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `softmax` | `(arr) → arr` | Logit → olasılık dönüşümü |
| `sigmoid` | `(arr) → arr` | Sigmoid aktivasyon |
| `argmax` | `(arr) → int` | Maks değerin indeksi |
| `argmin` | `(arr) → int` | Min değerin indeksi |
| `topk` | `(arr, k) → arr` | Top-K değer ve indeksler |
| `threshold` | `(arr, val) → arr` | Binary eşikleme |
| `clip` | `(arr, min, max) → arr` | Değerleri aralığa sınırla |

### NLP
| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `tokenize` | `(text) → map` | HuggingFace tokenizer (tokenizer.json gerekli) |
| `decode_tokens` | `(ids) → string` | Token ID'lerini metne çevir |
| `pad_sequence` | `(arr, len, pad) → arr` | Sabit uzunluğa padding/truncation |
| `text_lower` | `(text) → string` | Küçük harfe çevirme |
| `regex_replace` | `(text, pattern, repl) → string` | Regex metin değiştirme |

### Tabular / Normalizasyon
| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `normalize` | `(arr, method) → arr` | "minmax" veya "l2" normalizasyon |
| `standardize` | `(arr, mean, std) → arr` | Z-score standardizasyon |
| `one_hot` | `(index, n) → arr` | One-hot encoding |
| `label_encode` | `(value, map) → int` | Kategorik → sayısal |
| `fill_missing` | `(arr, strategy) → arr` | NaN doldurma: "zero", "mean", "median" |

### Bilgisayar Görüsü (CV)
| Fonksiyon | İmza | Açıklama |
|-----------|------|----------|
| `decode_image` | `(base64) → map` | JPEG/PNG decode → piksel array |
| `resize_image` | `(pixels, sh, sw, dh, dw, c) → arr` | Bilinear boyut değiştirme |
| `normalize_image` | `(pixels, mean, std) → arr` | Kanal bazlı normalizasyon |
| `image_to_chw` | `(pixels, h, w, c) → arr` | HWC → CHW layout dönüşümü |
| `center_crop` | `(pixels, sh, sw, ch, cw, c) → arr` | Merkez kırpma |
| `grayscale` | `(pixels, h, w) → arr` | RGB → gri tonlama |
| `nms` | `(boxes, scores, iou) → arr` | Non-Maximum Suppression |

### Örnek: Sentiment Analizi Decoder
```rhai
fn execute(inputs) {
    let logits = inputs.get("logits").as_f64();
    let probs = softmax(logits);
    let label = argmax(probs);

    return #{
        "label": create_tensor_i64("label", [1], [label]),
        "probs": create_tensor_f64("probs", [3], probs),
    };
}
```

---

## Loglama

Üç katmanlı loglama mimarisi, tamamı non-blocking:

| Katman | Çıktı | İçerik |
|--------|-------|--------|
| **stdout** | Terminal | Sadece model yükleme, health, başlangıç tablosu |
| **file** | `/tmp/logs/axon/axon-server.YYYY-MM-DD.json` | Her şey (JSON, günlük rotation) |
| **OTEL** | OTLP endpoint | Her şey (`OTEL_EXPORTER_OTLP_ENDPOINT` set ise) |

Inference tracing logları her istek için `model`, `latency_ms`, `total_ms` içerir.

---

## Dokümantasyon

Detaylı dokümantasyon **[Wiki](https://github.com/mustafacavusoglu/axon/wiki)** sayfasında:

| Rehber | Açıklama |
|--------|----------|
| [Config Referansı](https://github.com/mustafacavusoglu/axon/wiki/Config-Reference) | YAML & pbtxt config formatı |
| [BLS / Scripting](https://github.com/mustafacavusoglu/axon/wiki/BLS-Scripting) | Rhai scripting engine |
| [Builtin Fonksiyonlar](https://github.com/mustafacavusoglu/axon/wiki/Builtin-Functions) | 23 preprocessing/postprocessing fonksiyon |
| [Ensemble Pipeline](https://github.com/mustafacavusoglu/axon/wiki/Ensemble-Pipeline) | Declarative model zincirleme |
| [Loglama](https://github.com/mustafacavusoglu/axon/wiki/Logging) | Yapısal loglama & OTEL |
| [Deployment](https://github.com/mustafacavusoglu/axon/wiki/Deployment) | CLI, Docker, Compose |
| [Metrikler](https://github.com/mustafacavusoglu/axon/wiki/Metrics) | Prometheus metrikleri |

---

## Örnek Modeller

| Dizin | Modeller | Açıklama |
|-------|----------|----------|
| `ml_model/` | `xgb_housing` | Basit ONNX modeli (8 özellik → fiyat) |
| `nlp_model/` | `tokenizer`, `ner_model`, `decoder`, `pipeline` | BLS + Ensemble NLP pipeline |

```bash
# NLP ensemble — metinden entity etiketlerine
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
- Swagger UI — OpenAPI 3.0 tarayıcı tabanlı dokümantasyon
- NUMA-aware session pools — çok soketli sunucu optimizasyonu
