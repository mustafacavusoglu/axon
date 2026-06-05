# DONE.md — Axon Inference Server

## Aşama 1: Proje İskeleti ve Protolar

- [x] Dizin yapısı oluşturuldu (proto/, control-plane/, inference-engine/, deploy/k8s/)
- [x] `proto/inference/kfs/kserve_grpc.proto` — KServe v2 public gRPC API tanımı
- [x] `proto/inference/engine/v1/inference_internal.proto` — Go ↔ Rust internal protokol
- [x] `buf.yaml` + `buf.gen.yaml` — proto codegen yapılandırması
- [x] `control-plane/go.mod` — Go modül başlatıldı (`github.com/mustafacavusoglu/axon/control-plane`)
- [x] `inference-engine/Cargo.toml` — Rust crate başlatıldı
- [x] Proto codegen (buf generate) — Go gRPC stub'ları oluşturuldu
- [x] `Makefile` — build, test, docker hedefleri
- [x] `.gitignore`
- [x] `git init`

## Aşama 2: Rust Inference Engine

- [x] `config/mod.rs` — Ortam değişkenlerinden config (socket_path, num_threads, arena_size_mb)
- [x] `session/pool.rs` — `SessionPool` (DashMap tabanlı multi-model yönetimi)
- [x] `session/runner.rs` — ONNX Runtime session sarma, `ModelRunner::load()`, `ModelRunner::run()`
- [x] `arena/mod.rs` — `TensorArena` pre-allocated slab allocator (testli)
- [x] `metrics/mod.rs` — Prometheus counter/histogram (request_count, latency, model_count)
- [x] `server.rs` — tonic gRPC server (Unix socket), `InferenceEngine` trait implementasyonu
- [x] `main.rs` — Rayon thread pool init, tokio runtime, signal handling
- [x] `build.rs` — tonic-build ile proto derleme
- [x] `cargo check` — sıfır hata
- [x] `cargo test` — 2/2 test geçti

## Aşama 3: Go Control Plane

- [x] `internal/config/config.go` — Viper ile env+YAML config
- [x] `internal/manager/config_parser.go` — Lightweight config.pbtxt parser
- [x] `internal/manager/registry.go` — Thread-safe ModelRegistry (state machine: LOADING→READY→UNLOADING→ERROR)
- [x] `internal/manager/lifecycle.go` — LoadModel, UnloadModel, LoadAllFromRepo
- [x] `internal/client/inference_client.go` — Rust engine gRPC client (Unix socket)
- [x] `internal/api/http/handler.go` — KServe v2 HTTP endpoint'leri:
  - `GET /v2/health/live`, `GET /v2/health/ready`
  - `GET /v2/models`, `GET /v2/models/{name}`
  - `GET /v2/models/{name}/versions/{version}`
  - `POST /v2/models/{name}/load`, `POST /v2/models/{name}/unload`
  - `POST /v2/models/{name}/infer`
- [x] `internal/api/http/middleware.go` — RequestID, Logger, Recover middleware (fiber v2)
- [x] `internal/api/grpc/server.go` — Public gRPC server (KServe v2 GRPCInferenceService)
- [x] `internal/metrics/prometheus.go` — Prometheus collectors
- [x] `internal/health/checker.go` — Liveness/readiness checker
- [x] `cmd/server/main.go` — Entry point, graceful shutdown (SIGTERM/SIGINT)
- [x] `go build ./...` — sıfır hata
- [x] `go vet ./...` — sıfır uyarı
- [x] `GET /metrics` — Prometheus metrics endpoint (adaptor)

## Aşama 4: Docker + K8s Altyapısı

- [x] `Dockerfile.control-plane` — Multi-stage Go build
- [x] `Dockerfile.inference-engine` — Multi-stage Rust build
- [x] `docker-compose.yml` — Local dev (sidecar: control-plane + inference-engine)
- [x] `deploy/k8s/deployment.yaml` — Sidecar pod deployment
- [x] `deploy/k8s/service.yaml` — External + headless service
- [x] `deploy/k8s/pvc.yaml` — Model store PVC

## Bug Fixes & Hardening

- [x] Config parser: array sözdizimi desteği (multi-input, multi-output)
- [x] Config parser: brace derinlik takibi, iç içe yapılar için doğru section yönetimi
- [x] Config parser: `currentSection` boşken section-specific case'lerin önce işlenmesi
- [x] Config parser: `dynamic_batching` section init (nil check fix)
- [x] Config parser: `{` aynı satırda section declaration ile gelince brace depth artışı
- [x] Config parser test: `lgbm_breast_cancer/config.pbtxt` başarıyla parse ediliyor
- [x] Viper `SetEnvPrefix` + `BindEnv` çakışması → manuel `os.Getenv` ile değiştirildi
- [x] LifecycleManager'a `repoPath` eklendi (hardcoded `/models` kaldırıldı)
- [x] Prometheus `MustRegister` duplicate hatası → `promauto` auto-register
- [x] Rust engine: INT64 output desteği (TensorData enum: F32/I64)
- [x] Rust engine: output dtype auto-detection (f32 fallback → i64)
- [x] HTTP handler: JSON `data` array desteği (FP32, INT64, INT32)
- [x] HTTP response: `data` array formatında çıktı (raw bytes → float64 slice)
- [x] `run.sh` — Local development script (build + run both services)
- [x] `control-plane/internal/manager/config_parser_test.go` — Unit test
