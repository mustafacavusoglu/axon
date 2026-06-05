# CPU Inference Server — Architecture Document

> Triton-compatible, CPU-first model serving infrastructure.  
> **Control Plane:** Go · **Inference Engine:** Rust  
> **Transport:** gRPC (primary) + HTTP/REST (Swagger UI)  
> **Runtime:** ONNX Runtime via `ort` crate  
> **Target:** Kubernetes, horizontally scalable

---

## Table of Contents

1. [Overview & Goals](#1-overview--goals)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Component Deep-Dive](#3-component-deep-dive)
   - 3.1 [Go Control Plane](#31-go-control-plane)
   - 3.2 [Rust Inference Engine](#32-rust-inference-engine)
   - 3.3 [IPC Layer — Go ↔ Rust](#33-ipc-layer--go--rust)
   - 3.4 [Model Repository](#34-model-repository)
4. [Model Configuration (`config.pbtxt`)](#4-model-configuration-configpbtxt)
5. [Request Lifecycle](#5-request-lifecycle)
6. [Dynamic Batching](#6-dynamic-batching)
7. [Model Lifecycle & Registry](#7-model-lifecycle--registry)
8. [API Design](#8-api-design)
9. [Kubernetes Architecture](#9-kubernetes-architecture)
10. [Scalability Patterns](#10-scalability-patterns)
11. [Observability](#11-observability)
12. [Project Directory Structure](#12-project-directory-structure)
13. [Technology Decision Records (ADR)](#13-technology-decision-records-adr)
14. [Roadmap](#14-roadmap)

---

## 1. Overview & Goals

### What This Is

A production-grade, CPU-first model serving system modeled after NVIDIA Triton Inference Server. It exposes identical API conventions (HTTP/REST + gRPC, KServe v2 protocol) so existing Triton clients work without modification.

### Design Goals

| Goal | Detail |
|---|---|
| **CPU-first** | No GPU dependency; SIMD/AVX paths exposed via ONNX Runtime |
| **Triton-compatible layout** | `model/<version>/model.onnx` + `config.pbtxt` |
| **Multi-model serving** | Hundreds of models concurrently with LRU eviction |
| **Dynamic batching** | Accumulate requests up to `max_batch_size` or deadline |
| **Zero-GC hot path** | Rust handles all inference; Go never touches tensor memory |
| **Kubernetes-native** | HPA, PDB, liveness/readiness probes, graceful drain |
| **Swagger UI** | Full OpenAPI 3.0 spec, browsable at `/swagger` |
| **Observability** | Prometheus metrics, structured JSON logs, tracing |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          Kubernetes Pod                          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               Go Control Plane  (:8080 HTTP, :8001 gRPC) │    │
│  │                                                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │    │
│  │  │ HTTP     │  │ gRPC     │  │  Model Manager       │  │    │
│  │  │ Server   │  │ Server   │  │  (Registry + FSWatch)│  │    │
│  │  │ +Swagger │  │          │  └──────────────────────┘  │    │
│  │  └────┬─────┘  └────┬─────┘                            │    │
│  │       │              │                                  │    │
│  │  ┌────▼──────────────▼──────┐  ┌────────────────────┐  │    │
│  │  │     Request Router       │  │  Dynamic Batcher   │  │    │
│  │  │  (validate, auth, route) ├─►│  (per-model queue) │  │    │
│  │  └──────────────────────────┘  └────────┬───────────┘  │    │
│  └────────────────────────────────────────-│───────────────┘    │
│                                            │ Unix Socket / gRPC  │
│  ┌─────────────────────────────────────────▼─────────────────┐  │
│  │            Rust Inference Engine  (:9001 internal)        │  │
│  │                                                           │  │
│  │  ┌─────────────────────┐    ┌───────────────────────┐    │  │
│  │  │  Session Pool       │    │  Thread Pool          │    │  │
│  │  │  (per model)        │    │  (rayon, core affinity│    │  │
│  │  │  ONNX Runtime       │    │   NUMA-aware)         │    │  │
│  │  └─────────────────────┘    └───────────────────────┘    │  │
│  │                                                           │  │
│  │  ┌─────────────────────────────────────────────────────┐ │  │
│  │  │  Tensor Arena (pre-allocated slab allocator)        │ │  │
│  │  └─────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Shared Volume Mount: /models  (PVC or emptyDir + init-container)│
└──────────────────────────────────────────────────────────────────┘
         │                    │
    Prometheus             Clients
    ServiceMonitor     (HTTP/gRPC)
```

### Pod Sidecar Pattern

```
Pod
├── container: go-control-plane    (port 8080 HTTP, 8001 gRPC)
│     image: project/control-plane:v1
│     volumeMount: /models, /run/inference.sock
│
└── container: rust-inference      (Unix socket: /run/inference.sock)
      image: project/inference-engine:v1
      volumeMount: /models, /run/inference.sock
```

Both containers share:
- `/models` — model repository (PVC or projected volume)
- `/run/inference.sock` — Unix domain socket (IPC)

---

## 3. Component Deep-Dive

### 3.1 Go Control Plane

**Responsibilities:** API surface, model lifecycle, request validation, batching coordination, Kubernetes health, metrics exposition.

**Key packages:**

```
internal/
  api/          — HTTP + gRPC handlers
  manager/      — model registry, config.pbtxt parser, FSWatcher
  scheduler/    — dynamic batching, per-model queues
  client/       — gRPC stub for Rust engine (generated from proto)
  metrics/      — Prometheus collectors
  health/       — liveness + readiness logic
  config/       — server config (env + YAML)
```

#### HTTP Server

- Framework: `fiber` v3
- Swagger: `swaggo/swag` + `gofiber/swagger`
- Middleware stack: `RequestID → Logger → Recover → CORS → Auth → RateLimit`


```go 
// Example route registration
app := fiber.New()
app.Use(middleware.RequestID())
app.Use(middleware.Logger())

// KServe v2 HTTP inference protocol
v2 := app.Group("/v2")
v2.Get("/health/live",               h.Live)
v2.Get("/health/ready",              h.Ready)
v2.Get("/models",                    h.ListModels)
v2.Get("/models/:name",              h.ModelMeta)
v2.Get("/models/:name/versions/:v",  h.ModelVersion)
v2.Post("/models/:name/load",        h.LoadModel)
v2.Post("/models/:name/unload",      h.UnloadModel)
v2.Post("/models/:name/infer",       h.Infer)

// Management
app.Get("/swagger/*", swagger.HandlerDefault)
app.Get("/metrics",   adaptor.HTTPHandler(promhttp.Handler()))
```

#### gRPC Server

Implements the **KServe v2 gRPC inference protocol** (same as Triton):

```protobuf
// kserve_grpc.proto (standard, Triton-compatible)
service GRPCInferenceService {
  rpc ServerLive(ServerLiveRequest)           returns (ServerLiveResponse);
  rpc ServerReady(ServerReadyRequest)         returns (ServerReadyResponse);
  rpc ModelReady(ModelReadyRequest)           returns (ModelReadyResponse);
  rpc ServerMetadata(ServerMetadataRequest)   returns (ServerMetadataResponse);
  rpc ModelMetadata(ModelMetadataRequest)     returns (ModelMetadataResponse);
  rpc ModelInfer(ModelInferRequest)           returns (ModelInferResponse);
  rpc ModelStreamInfer(ModelInferRequest)     returns (stream ModelInferResponse);
}
```

#### Model Manager

```go
type ModelRegistry struct {
    mu     sync.RWMutex
    models map[string]*ModelEntry  // key: "name@version"
}

type ModelEntry struct {
    Config    *ModelConfig     // parsed config.pbtxt
    State     ModelState       // LOADING | READY | UNLOADING | ERROR
    LoadedAt  time.Time
    LastUsed  time.Time        // for LRU eviction
    Version   int
}

type ModelState int
const (
    StateLoading   ModelState = iota
    StateReady
    StateUnloading
    StateError
)
```

FSWatcher (via `fsnotify`) watches `/models` and automatically:
- Detects new model directories → triggers load
- Detects `config.pbtxt` changes → triggers reload
- Detects removal → triggers graceful unload

#### Dynamic Batcher (Go-side)

```go
type BatchQueue struct {
    modelName  string
    maxBatch   int
    maxDelayUs int64          // from config.pbtxt

    queue      chan *PendingRequest
    done       chan struct{}
}

type PendingRequest struct {
    inputs   []*InferInput
    respChan chan *InferResult
    deadline time.Time
}
```

The batcher accumulates requests into a batch until either:
- `preferred_batch_size` is reached, OR
- `max_queue_delay_microseconds` elapses

Then dispatches the entire batch to Rust via a single gRPC call.

---

### 3.2 Rust Inference Engine

**Responsibilities:** ONNX Runtime session management, tensor execution, thread pool management, memory allocation.

**Key crates:**

```toml
[dependencies]
ort            = { version = "2", features = ["load-dynamic"] }
tonic          = "0.12"
tokio          = { version = "1", features = ["full"] }
rayon          = "1"
ndarray        = "0.16"
dashmap        = "6"       # concurrent model session map
bytes          = "1"
prometheus     = "0.14"
tracing        = "0.1"
tracing-subscriber = { version = "0.3", features = ["json"] }
```

#### Session Pool

```rust
pub struct SessionPool {
    sessions: Arc<DashMap<String, Arc<ModelSession>>>,
}

pub struct ModelSession {
    name: String,
    version: u32,
    session: Arc<ort::Session>,
    config: ModelConfig,
    // Semaphore limits concurrent executions per model
    // respects instance_group.count from config.pbtxt
    concurrency: Arc<Semaphore>,
}
```

Each model can have `instance_group.count` concurrent ONNX sessions. This maps directly to Triton's `instance_group` config field.

#### Thread Pool Strategy

```rust
// Rayon global pool: physical cores only (no hyperthreading)
rayon::ThreadPoolBuilder::new()
    .num_threads(num_cpus::get_physical())
    .thread_name(|i| format!("infer-worker-{}", i))
    .build_global()
    .unwrap();
```

For NUMA-aware execution on multi-socket servers, each socket gets its own rayon pool and session pool.

#### Tensor Arena

Pre-allocated memory slabs avoid per-request heap allocation in the hot path:

```rust
pub struct TensorArena {
    slab: Vec<u8>,
    cursor: AtomicUsize,
    capacity: usize,
}
```

Cleared between batch executions. Sized at startup based on `max_batch_size × max_tensor_bytes` from config.

#### Inference gRPC Handler

```rust
#[tonic::async_trait]
impl InferenceEngine for InferenceEngineImpl {
    async fn infer(
        &self,
        request: Request<BatchInferRequest>,
    ) -> Result<Response<BatchInferResponse>, Status> {
        let req = request.into_inner();
        let session = self.pool.get(&req.model_name, req.version)?;

        // Acquire concurrency slot
        let _permit = session.concurrency.acquire().await?;

        // Execute on rayon thread pool (blocking, CPU-bound)
        let result = tokio::task::spawn_blocking(move || {
            session.run_batch(&req.inputs)
        }).await??;

        Ok(Response::new(result))
    }
}
```

---

### 3.3 IPC Layer — Go ↔ Rust

**Transport:** Unix Domain Socket (same Pod) via gRPC.

Why Unix socket over TCP localhost:
- No kernel TCP stack overhead
- Lower latency for large tensor payloads
- No port conflict management

**Internal proto (not the public KServe proto):**

```protobuf
// inference_internal.proto
syntax = "proto3";
package inference.internal.v1;

service InferenceEngine {
  rpc BatchInfer   (BatchInferRequest)    returns (BatchInferResponse);
  rpc LoadModel    (LoadModelRequest)     returns (LoadModelResponse);
  rpc UnloadModel  (UnloadModelRequest)   returns (UnloadModelResponse);
  rpc ModelStatus  (ModelStatusRequest)   returns (ModelStatusResponse);
  rpc Healthcheck  (HealthRequest)        returns (HealthResponse);
}

message BatchInferRequest {
  string model_name    = 1;
  uint32 version       = 2;
  repeated InferInput inputs  = 3;
  uint32 batch_size    = 4;
}

message InferInput {
  string      name      = 1;
  bytes       data      = 2;
  repeated int64 shape = 3;
  DataType    dtype     = 4;
}

message BatchInferResponse {
  repeated InferOutput outputs = 1;
  double latency_ms            = 2;
}

enum DataType {
  TYPE_INVALID = 0;
  TYPE_FP32    = 1;
  TYPE_FP64    = 2;
  TYPE_INT32   = 3;
  TYPE_INT64   = 4;
  TYPE_INT8    = 5;
  TYPE_UINT8   = 6;
  TYPE_BOOL    = 7;
  TYPE_STRING  = 8;
}
```

**Tensor encoding:** Raw bytes (`bytes data`) in row-major order. Shape is sent alongside. This avoids double serialization and is zero-copy on the Rust side with `bytes::Bytes`.

---

### 3.4 Model Repository

Triton-compatible layout:

```
/models/
├── resnet50/
│   ├── config.pbtxt
│   └── 1/
│       └── model.onnx
│
├── bert-classifier/
│   ├── config.pbtxt
│   ├── 1/
│   │   └── model.onnx
│   └── 2/                  ← newer version
│       └── model.onnx
│
└── yolov8-nano/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

Version selection policy (from `config.pbtxt`):
- `version_policy: { latest { num_versions: 1 } }` — serve only latest
- `version_policy: { all {} }` — serve all versions
- `version_policy: { specific { versions: [1, 3] } }` — explicit

---

## 4. Model Configuration (`config.pbtxt`)

Full Protobuf Text Format schema supported:

```protobuf
# /models/my-classifier/config.pbtxt

name: "my-classifier"
platform: "onnxruntime_onnx"
max_batch_size: 64

# Input tensors
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1, 512 ]      # -1 = dynamic (batch) dim is implicit
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1, 512 ]
  }
]

# Output tensors
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ -1, 1000 ]
  }
]

# How many concurrent ONNX sessions per replica
instance_group [
  {
    count: 2
    kind: KIND_CPU
    # Optional: pin to CPU set
    # cpuset: "0-7"
  }
]

# Dynamic batching config
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 500
}

# Version policy
version_policy: {
  latest { num_versions: 1 }
}

# Optional: model warmup
model_warmup [
  {
    name: "warmup_request"
    batch_size: 1
    inputs: {
      key: "input_ids"
      value: { data_type: TYPE_INT64, dims: [1, 512], zero_data: true }
    }
  }
]
```

**Go-side parser** uses `google.golang.org/protobuf/encoding/prototext` with a hand-written `.proto` schema matching Triton's `ModelConfig`.

---

## 5. Request Lifecycle

```
Client (HTTP or gRPC)
  │
  ▼
[Go] HTTP Handler / gRPC Handler
  │ 1. Parse request (KServe v2 format)
  │ 2. Validate model name + version exists (registry lookup)
  │ 3. Validate input shapes & dtypes against config.pbtxt
  │
  ▼
[Go] Dynamic Batcher (per-model BatchQueue)
  │ 4. Enqueue PendingRequest with response channel
  │ 5. Wait: either batch full OR max_queue_delay elapsed
  │
  ▼
[Go] Batcher flushes → calls Rust via internal gRPC
  │ 6. Serialize tensors to bytes (row-major, no copy for []byte inputs)
  │ 7. Send BatchInferRequest over Unix socket
  │
  ▼
[Rust] InferenceEngine.BatchInfer()
  │ 8. Acquire concurrency permit (semaphore)
  │ 9. Dispatch to rayon thread pool (blocking inference)
  │ 10. ONNX Runtime session.run(inputs)
  │ 11. Serialize outputs → BatchInferResponse
  │
  ▼
[Go] Batcher receives response
  │ 12. Fan-out: split batch response → per-request result
  │ 13. Write to each PendingRequest.respChan
  │
  ▼
[Go] Handler sends response to client
  │ 14. HTTP: JSON response (KServe v2)
  │     gRPC: ModelInferResponse protobuf
```

**Error propagation:** Any error at step 8-11 is returned to all requests in the batch with the same error code. Partial batch failures are not supported in v1 (all-or-nothing).

---

## 6. Dynamic Batching

### Batcher State Machine

```
        enqueue()
           │
     ┌─────▼──────┐
     │   WAITING  │◄──── new requests arriving
     └─────┬──────┘
           │
     ┌─────▼──────────────────────────────┐
     │  Is batch_size >= preferred[0] ?   │──YES──►  FLUSH
     │  OR deadline exceeded?             │
     └─────┬──────────────────────────────┘
           │ NO
           │ (keep accumulating)
           ▼
         FLUSH ──► send to Rust ──► fan-out results
```

### Batching Config Interaction

```
preferred_batch_size: [8, 16, 32]
max_queue_delay_microseconds: 500
max_batch_size: 64
```

Logic:
1. Accumulate until we hit `preferred_batch_size[0]` (8) — flush early
2. If we keep getting requests before the flush: grow to `16`, then `32`
3. Hard cap at `max_batch_size` (64) regardless
4. Deadline `500µs` — flush whatever we have even if batch is 1

### Go Implementation Sketch

```go
func (b *BatchQueue) run() {
    timer := time.NewTimer(0)
    var batch []*PendingRequest

    for {
        select {
        case req := <-b.queue:
            batch = append(batch, req)
            // Check preferred batch sizes in order
            if b.shouldFlush(len(batch)) {
                b.flush(batch)
                batch = batch[:0]
                timer.Reset(time.Duration(b.maxDelayUs) * time.Microsecond)
            }

        case <-timer.C:
            if len(batch) > 0 {
                b.flush(batch)
                batch = batch[:0]
            }
            timer.Reset(time.Duration(b.maxDelayUs) * time.Microsecond)

        case <-b.done:
            return
        }
    }
}
```

---

## 7. Model Lifecycle & Registry

### States & Transitions

```
         load request
              │
        ┌─────▼──────┐
        │  LOADING   │──── config parse ──► session init (Rust)
        └─────┬──────┘
              │ success           failure
        ┌─────▼──────┐       ┌──────────┐
        │   READY    │       │  ERROR   │
        └─────┬──────┘       └──────────┘
              │ unload request
        ┌─────▼──────┐
        │ UNLOADING  │──── drain in-flight ──► Rust unload
        └─────┬──────┘
              │
           REMOVED
```

### LRU Eviction

When total loaded model memory exceeds `max_model_memory_bytes` (config):
1. Find READY model with oldest `LastUsed`
2. If no in-flight requests: evict
3. Else: skip and try next LRU candidate
4. New model loads into reclaimed slot

```go
type LRUEviction struct {
    registry    *ModelRegistry
    maxMemBytes int64
    mu          sync.Mutex
}
```

### Multi-Version Handling

```
GET  /v2/models/bert-classifier          → serves latest (v2)
GET  /v2/models/bert-classifier/versions/1  → serves v1 explicitly
POST /v2/models/bert-classifier/infer    → routes to latest
POST /v2/models/bert-classifier/infer    → with version field in body → routes to that version
```

Both versions can be loaded simultaneously. Each version is an independent `ModelEntry` in the registry.

---

## 8. API Design

### HTTP Endpoints (OpenAPI 3.0)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v2/health/live` | Liveness probe |
| GET | `/v2/health/ready` | Readiness probe (all READY models) |
| GET | `/v2` | Server metadata |
| GET | `/v2/models` | List all loaded models |
| GET | `/v2/models/{name}` | Model metadata |
| GET | `/v2/models/{name}/versions/{version}` | Version metadata |
| POST | `/v2/models/{name}/load` | Load model (admin) |
| POST | `/v2/models/{name}/unload` | Unload model (admin) |
| POST | `/v2/models/{name}/infer` | Run inference |
| GET | `/metrics` | Prometheus metrics |
| GET | `/swagger/*` | Swagger UI |

### Inference Request/Response (KServe v2)

```json
// POST /v2/models/resnet50/infer
{
  "id": "req-123",
  "inputs": [
    {
      "name": "image",
      "shape": [1, 3, 224, 224],
      "datatype": "FP32",
      "data": [0.1, 0.2, ...]
    }
  ],
  "outputs": [
    { "name": "probabilities" }
  ]
}

// Response
{
  "id": "req-123",
  "model_name": "resnet50",
  "model_version": "1",
  "outputs": [
    {
      "name": "probabilities",
      "shape": [1, 1000],
      "datatype": "FP32",
      "data": [0.001, 0.003, ...]
    }
  ]
}
```

**Binary extension:** For large tensors, data can be sent as raw bytes in the request body with `Content-Type: application/octet-stream` and metadata in a header — same as Triton's binary extension.

### Swagger UI Setup (Go)

```go
// @title           CPU Inference Server
// @version         1.0
// @description     Triton-compatible CPU model serving API
// @host            localhost:8080
// @BasePath        /v2

// @tag.name inference
// @tag.name management
// @tag.name health

// main.go
import "github.com/gofiber/swagger"
app.Get("/swagger/*", swagger.HandlerDefault)
```

---

## 9. Kubernetes Architecture

### Pod Layout

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: inference-server
  template:
    spec:
      # Co-locate on same NUMA node if possible
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule

      initContainers:
        # Pull models from S3/GCS into shared volume
        - name: model-puller
          image: project/model-puller:v1
          env:
            - name: MODEL_BUCKET
              value: s3://my-models
          volumeMounts:
            - name: models
              mountPath: /models

      containers:
        - name: control-plane
          image: project/control-plane:v1
          ports:
            - containerPort: 8080   # HTTP
            - containerPort: 8001   # gRPC
          env:
            - name: INFERENCE_SOCKET
              value: /run/inference.sock
            - name: MODEL_REPO_PATH
              value: /models
            - name: MAX_MODEL_MEMORY_GB
              value: "8"
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2000m"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /v2/health/live
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 5
          readinessProbe:
            httpGet:
              path: /v2/health/ready
              port: 8080
            initialDelaySeconds: 20
            periodSeconds: 5
          volumeMounts:
            - name: models
              mountPath: /models
            - name: socket-dir
              mountPath: /run

        - name: inference-engine
          image: project/inference-engine:v1
          env:
            - name: SOCKET_PATH
              value: /run/inference.sock
            - name: NUM_THREADS
              value: "0"   # 0 = auto-detect physical cores
            - name: ARENA_SIZE_MB
              value: "4096"
          resources:
            requests:
              cpu: "4000m"
              memory: "4Gi"
            limits:
              cpu: "16000m"
              memory: "16Gi"
          volumeMounts:
            - name: models
              mountPath: /models
              readOnly: true
            - name: socket-dir
              mountPath: /run

      volumes:
        - name: models
          persistentVolumeClaim:
            claimName: model-store-pvc
        - name: socket-dir
          emptyDir: {}
```

### Services

```yaml
# External service (clients)
apiVersion: v1
kind: Service
metadata:
  name: inference-server
spec:
  selector:
    app: inference-server
  ports:
    - name: http
      port: 80
      targetPort: 8080
    - name: grpc
      port: 8001
      targetPort: 8001

---
# Headless service for gRPC client-side load balancing
apiVersion: v1
kind: Service
metadata:
  name: inference-server-headless
spec:
  clusterIP: None
  selector:
    app: inference-server
  ports:
    - name: grpc
      port: 8001
```

### HPA (Horizontal Pod Autoscaler)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: inference-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
    # CPU utilization (Rust inference engine)
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70

    # Custom metric: batcher queue depth (via Prometheus adapter)
    - type: Pods
      pods:
        metric:
          name: inference_queue_depth
        target:
          type: AverageValue
          averageValue: "50"
```

### PodDisruptionBudget

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-server-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: inference-server
```

### Graceful Shutdown

On `SIGTERM`:
1. Go control plane: mark readiness probe `/v2/health/ready` → 503 (removed from Service LB)
2. Wait `DRAIN_TIMEOUT` (default 30s) for in-flight requests to complete
3. Send `UnloadAll` to Rust engine
4. Rust: drain semaphore permits, flush ONNX sessions
5. Exit 0

```go
// Go signal handler
quit := make(chan os.Signal, 1)
signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)
<-quit

ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
server.Shutdown(ctx)
```

---

## 10. Scalability Patterns

### Horizontal Scaling (Stateless Pods)

Each pod is fully independent. Model store is shared via PVC (ReadOnlyMany) or object storage. No pod-to-pod communication.

```
Client
  │
  ▼
Ingress / LB
  ├──► Pod-1 (2 model sessions)
  ├──► Pod-2 (2 model sessions)
  └──► Pod-3 (2 model sessions)
            ↑ all read from same PVC
```

### gRPC Client-Side Load Balancing

For internal gRPC clients (other services calling the inference server):

```go
conn, err := grpc.Dial(
    "dns:///inference-server-headless:8001",
    grpc.WithDefaultServiceConfig(`{"loadBalancingPolicy":"round_robin"}`),
)
```

### Large Model Strategy

For models too large to load on every pod, use a **model shard deployment**:

```
Deployment-A: loads models [resnet50, vgg16, ...]
Deployment-B: loads models [bert-large, llama-7b, ...]

Ingress routes /v2/models/resnet50/infer → Deployment-A
              /v2/models/bert-large/infer → Deployment-B
```

Routing map is managed via a `ConfigMap` + a lightweight router service.

### Queue Depth Metric Export

```go
// Exposed as Prometheus gauge for HPA custom metrics
inferenceQueueDepth.With(prometheus.Labels{
    "model": modelName,
}).Set(float64(queue.Len()))
```

---

## 11. Observability

### Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `inference_requests_total` | Counter | `model`, `version`, `status` | Total inference requests |
| `inference_latency_ms` | Histogram | `model`, `version` | End-to-end latency |
| `inference_engine_latency_ms` | Histogram | `model`, `version` | Rust execution latency |
| `inference_queue_depth` | Gauge | `model` | Pending requests in batcher |
| `inference_batch_size` | Histogram | `model` | Actual flushed batch sizes |
| `model_load_total` | Counter | `model`, `status` | Load/unload events |
| `model_memory_bytes` | Gauge | `model`, `version` | Loaded model memory |
| `inference_engine_threads_active` | Gauge | — | Active rayon threads |

### Structured Logging

Both Go and Rust emit JSON logs to stdout (collected by Fluentd/Loki):

```json
{
  "timestamp": "2025-01-15T10:23:45.123Z",
  "level": "info",
  "service": "control-plane",
  "request_id": "req-abc123",
  "model": "resnet50",
  "version": 1,
  "batch_size": 16,
  "queue_wait_ms": 0.23,
  "total_latency_ms": 12.4,
  "status": "ok"
}
```

### Distributed Tracing

Go: `go.opentelemetry.io/otel` + OTLP exporter (Jaeger/Tempo)

Trace spans:
- `http.request` → `batcher.wait` → `grpc.internal_call` → `onnx.run`

Rust: `tracing` crate + `opentelemetry-otlp`

---

## 12. Project Directory Structure

```
cpu-inference-server/
│
├── proto/
│   ├── kserve_grpc.proto          # Public gRPC API (KServe v2)
│   └── inference_internal.proto   # Internal Go↔Rust protocol
│
├── control-plane/                 # Go module
│   ├── go.mod
│   ├── main.go
│   ├── cmd/
│   │   └── server/
│   │       └── main.go
│   ├── internal/
│   │   ├── api/
│   │   │   ├── http/
│   │   │   │   ├── handler.go
│   │   │   │   ├── middleware.go
│   │   │   │   └── swagger.go
│   │   │   └── grpc/
│   │   │       ├── server.go
│   │   │       └── handler.go
│   │   ├── manager/
│   │   │   ├── registry.go        # ModelRegistry
│   │   │   ├── watcher.go         # FSNotify watcher
│   │   │   ├── config_parser.go   # config.pbtxt parser
│   │   │   └── lifecycle.go       # Load/unload state machine
│   │   ├── scheduler/
│   │   │   ├── batcher.go         # Dynamic batching
│   │   │   ├── queue.go           # Per-model queues
│   │   │   └── router.go          # Route to correct batcher
│   │   ├── client/
│   │   │   └── inference_client.go # gRPC stub for Rust engine
│   │   ├── metrics/
│   │   │   └── prometheus.go
│   │   ├── health/
│   │   │   └── checker.go
│   │   └── config/
│   │       └── config.go          # Server config (viper)
│   └── docs/                      # swaggo generated
│       ├── docs.go
│       └── swagger.yaml
│
├── inference-engine/              # Rust crate (workspace)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── server.rs              # tonic gRPC server
│   │   ├── session/
│   │   │   ├── mod.rs
│   │   │   ├── pool.rs            # SessionPool
│   │   │   └── runner.rs          # ONNX execution
│   │   ├── batching/
│   │   │   └── tensor.rs          # Tensor assembly/disassembly
│   │   ├── arena/
│   │   │   └── mod.rs             # TensorArena allocator
│   │   ├── config/
│   │   │   └── mod.rs             # Config parsing
│   │   └── metrics/
│   │       └── mod.rs
│   └── build.rs                   # tonic-build for proto codegen
│
├── model-puller/                  # Init container (Go or shell)
│   └── ...
│
├── deploy/
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── hpa.yaml
│   │   ├── pdb.yaml
│   │   ├── pvc.yaml
│   │   └── configmap.yaml
│   └── helm/
│       └── cpu-inference/
│           ├── Chart.yaml
│           ├── values.yaml
│           └── templates/
│
├── Makefile                       # build, generate, docker, deploy
├── docker-compose.yml             # local dev
├── Dockerfile.control-plane
├── Dockerfile.inference-engine
└── README.md
```

---

## 13. Technology Decision Records (ADR)

### ADR-001: Go for Control Plane

**Decision:** Go handles all API, model management, and scheduling.  
**Rationale:** Goroutines map naturally to concurrent model management. gRPC ecosystem (`google.golang.org/grpc`) is most mature here. Fast iteration with `fiber` + `swaggo`. No GC impact on inference latency since Go never touches tensor memory.  
**Rejected:** Python (too slow for serving layer), Rust (unnecessary for this layer).

### ADR-002: Rust for Inference Engine

**Decision:** Rust handles all ONNX Runtime calls.  
**Rationale:** Zero GC guarantees on hot path. `ort` crate wraps ONNX Runtime C API with production-grade safety. `rayon` provides ergonomic CPU parallelism. Memory layout control for tensor arena.  
**Rejected:** C++ (memory safety risk for long-running server), Go+CGo (CGo overhead on every inference call).

### ADR-003: Unix Domain Socket for IPC

**Decision:** Internal Go↔Rust communication over UDS, not TCP loopback.  
**Rationale:** Eliminates TCP overhead on hot path. Both processes run in same Pod. Zero port management. Benchmarks show ~15% lower p99 latency vs TCP loopback for 1MB tensor payloads.  
**Rejected:** Shared memory (complex ownership, harder to version), TCP (unnecessary overhead).

### ADR-004: KServe v2 Protocol

**Decision:** Implement KServe v2 HTTP + gRPC inference protocol.  
**Rationale:** Drop-in compatibility with Triton clients. Well-specified open standard. Binary tensor extension covers large payload use cases.  
**Rejected:** Custom protocol (breaks client ecosystem).

### ADR-005: Sidecar Pod Pattern

**Decision:** Go and Rust run as separate containers in the same Pod.  
**Rationale:** Independent Docker build/push cycles. Independent scaling of control-plane vs inference-engine is NOT needed (they're coupled). Shared UDS via emptyDir volume. Clean separation of concerns.  
**Rejected:** Single binary via CGo (complex, merges concerns), separate Deployments (adds network hop for IPC).

---

## 14. Roadmap

### v1.0 — Core (Current Scope)
- [ ] config.pbtxt parser
- [ ] Single model load + gRPC inference
- [ ] Multi-model registry with LRU eviction
- [ ] Dynamic batching scheduler
- [ ] KServe v2 HTTP + gRPC
- [ ] Swagger UI
- [ ] Kubernetes manifests + HPA
- [ ] Prometheus metrics

### v1.1 — Hardening
- [ ] Model warmup on load
- [ ] Circuit breaker (per-model error rate)
- [ ] Authentication (API key + mTLS)
- [ ] Binary tensor protocol extension
- [ ] Graceful rolling update with zero dropped requests

### v2.0 — Advanced
- [ ] Ensemble pipelines (chain models)
- [ ] NUMA-aware session pools (multi-socket servers)
- [ ] OpenVINO backend (Intel-optimized)
- [ ] llama.cpp backend for LLM serving
- [ ] Model A/B traffic splitting
- [ ] GPU backend (CUDA via `cudarc` crate)

---

*Document version: 1.0 — Last updated: 2025*
