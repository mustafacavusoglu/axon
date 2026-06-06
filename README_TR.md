# Axon — CPU Inference Server

> [English documentation](README.md)

Triton uyumlu, CPU öncelikli model sunum altyapısı.  
**Control Plane:** Go · **Inference Engine:** Rust  
**İletişim:** gRPC + HTTP/REST (KServe v2)  
**Runtime:** ONNX Runtime  
**Hedef:** Kubernetes

---

## Hızlı Başlangıç

### Lokal
```bash
# Gereksinimler: Rust, Go, ONNX Runtime (brew install onnxruntime)
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

Sunucuya istek at:
```bash
curl http://localhost:8080/v2/health/live
curl http://localhost:8080/v2/models
```

---

## Mimari

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

| Bileşen | Dil | Görev |
|---------|-----|-------|
| Control Plane | Go | API, model kaydı, batch, sağlık kontrolü |
| Inference Engine | Rust | ONNX Runtime oturumları, tensor çalıştırma |
| IPC | gRPC (Unix socket) | Go ↔ Rust iletişimi |

---

## Inference

Hazır curl komutları için [sample-request.md](sample-request.md) dosyasına bak.

```bash
curl -s -X POST http://localhost:8080/v2/models/lgbm_credit_risk/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[
    {"name":"age","shape":[1],"datatype":"FP32","data":[25.0]},
    {"name":"bmi","shape":[1],"datatype":"FP32","data":[22.5]}
  ]}'
```

Yanıt (KServe v2):
```json
{
  "outputs": [
    {"name": "label",         "datatype": "INT64", "shape": [1],    "data": [1]},
    {"name": "probabilities", "datatype": "FP32",  "shape": [1, 2], "data": [0.23, 0.77]}
  ]
}
```

---

## Model Deposu

Triton uyumlu dizin yapısı:
```
/models/
└── benim-modelim/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

---

## Geliştirme

```bash
make build        # İkisini de derle
make test         # Tüm testleri çalıştır
make proto        # Protobuf kodunu yeniden üret (buf)
```

---

