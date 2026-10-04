# CLI Reference

```
axon-server [OPTIONS]
```

---

## Server Configuration

### `--model-repository <PATH>`
**Default:** `/models`

Path to the model repository directory. Each subdirectory is treated as a model.

```bash
axon-server --model-repository=/opt/models
```

### `--model-control-mode <MODE>`
**Default:** `none`

Model loading strategy:
- `none` — Load models once at startup, no hot-reload
- `poll` — Periodically scan repository; changed models are reloaded (built first, then swapped in — no downtime, a failed reload keeps the old version), removed models are unloaded
- `explicit` — Load once at startup; `POST /v2/models/{name}/load` and `/unload` are enabled (they return 403 in the other modes)

```bash
axon-server --model-control-mode=poll
```

### `--repository-poll-secs <SECONDS>`
**Default:** `30`

How often to scan the model repository for changes (only with `--model-control-mode=poll`).

---

## Network

### `--host <ADDR>`
**Default:** `0.0.0.0` · **Env:** `AXON_HOST`

Address the HTTP, gRPC and metrics servers bind to (e.g. `127.0.0.1` behind a sidecar proxy).

### `--http-port <PORT>`
**Default:** `8000`

HTTP REST API port (KServe v2 protocol).

### `--grpc-port <PORT>`
**Default:** `8001`

gRPC API port (KServe v2 protocol).

### `--metrics-port <PORT>`
**Default:** `8002`

Prometheus metrics endpoint port (`/metrics`).

```bash
axon-server --http-port=9000 --grpc-port=9001 --metrics-port=9002
```

---

## Performance

### `--num-threads <N>`
**Default:** `0` (auto-detect physical cores)

Size of the global compute admission semaphore (max requests computing at once across all models). Set to 0 to auto-detect.

### `--intra-op-threads <N>`
**Default:** `1`

ONNX Runtime intra-op threads per model instance. `1` maximises throughput with many concurrent requests; raise it to cut single-request latency of large models.

### `--concurrency-per-model <N>`
**Default:** `4`

Maximum concurrent inference requests per model. Can be overridden per-model via `instance_groups` in model config.

### `--inference-timeout-ms <MS>`
**Default:** `30000`

Maximum time (ms) for a single inference request. Returns HTTP 504 / gRPC DEADLINE_EXCEEDED on timeout.

```bash
axon-server --inference-timeout-ms=10000 --num-threads=8
```

---

## GPU

### `--device <DEVICE>`
**Default:** `cpu` · **Env:** `AXON_DEVICE`

Default execution device for ONNX models:
- `cpu` — CPU only (`KIND_GPU` models still request CUDA)
- `auto` — CUDA when available, otherwise CPU (logged as a warning)
- `cuda` — CUDA; models fail to load if no GPU is usable
- `tensorrt` — TensorRT with CUDA and CPU fallback per node (needs the image built with `INSTALL_TENSORRT=true`)

Per-model override via `instance_group.kind`: `KIND_CPU` (always CPU), `KIND_GPU` (GPU required), `KIND_AUTO` (follow `--device`), and `gpus: [0, 1]` to pin instances to GPUs.

### `--gpu-device-ids <IDS>`
**Default:** `0` · **Env:** `AXON_GPU_DEVICE_IDS`

Comma separated GPU ids used when a model does not list `gpus`. Instances are spread round-robin.

### `--gpu-mem-limit-mb <MiB>`
**Default:** `0` (unlimited) · **Env:** `AXON_GPU_MEM_LIMIT_MB`

Per-instance CUDA memory arena limit.

### `--trt-fp16` / `--trt-cache-dir <PATH>`
**Env:** `AXON_TRT_FP16`, `AXON_TRT_CACHE_DIR`

Enable FP16 TensorRT kernels; persist engine and timing caches so restarts skip the (slow) engine build.

### `--model-load-timeout-secs <SECONDS>`
**Default:** 10 (CPU), 120 (CUDA), 900 (TensorRT)

Timeout for creating one ONNX Runtime session.

```bash
axon-server --device=cuda --gpu-device-ids=0,1 --gpu-mem-limit-mb=8192
```

---

## Security

### `--api-key <KEY>`
**Env:** `AXON_API_KEY`

When set, every HTTP and gRPC endpoint except `/v2/health/live`, `/v2/health/ready`, gRPC `ServerLive` and `ServerReady` requires `Authorization: Bearer <KEY>` (HTTP header or gRPC metadata). Prefer the environment variable so the key does not show up in process listings.

```bash
AXON_API_KEY=change-me axon-server
curl -H "Authorization: Bearer change-me" http://localhost:8000/v2/models
```

---

## Web Console

### `--ui`
**Default:** off · **Env:** `AXON_UI`

Serves the built-in web console at `/ui` on the HTTP port. The static assets are public; API calls made by the page use the key typed into the console when `--api-key` is set.

```bash
axon-server --ui --api-key=change-me
```

---

## Logging

### `--log-level <LEVEL>`
**Default:** `info`

Minimum log level for file output. Options: `trace`, `debug`, `info`, `warn`, `error`.

Stdout always shows only model loading and health events regardless of this setting.

### `--log-dir <PATH>`
**Default:** `/tmp/logs/axon`

Directory for JSON log files with daily rotation. Created automatically if it doesn't exist.

```bash
axon-server --log-level=debug --log-dir=/var/log/axon
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enable OpenTelemetry trace export to this gRPC endpoint |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Alternative traces-only OTLP endpoint |
| `ORT_DYLIB_PATH` | Path to ONNX Runtime shared library (macOS) |
| `DYLD_LIBRARY_PATH` | Library path for ONNX Runtime (macOS) |
| `LD_LIBRARY_PATH` | Library path for ONNX Runtime (Linux) |
| `AXON_DEVICE`, `AXON_GPU_DEVICE_IDS`, `AXON_GPU_MEM_LIMIT_MB`, `AXON_TRT_FP16`, `AXON_TRT_CACHE_DIR` | GPU settings (see above) |
| `AXON_API_KEY` | API key (see Security) |
| `AXON_HOST` | Bind address |
| `OMP_NUM_THREADS`, `OMP_WAIT_POLICY` | Docker entrypoint defaults `1` / `PASSIVE`; override with `-e` |

---

## Full Example

```bash
axon-server \
  --model-repository=/opt/models \
  --model-control-mode=poll \
  --repository-poll-secs=10 \
  --http-port=8000 \
  --grpc-port=8001 \
  --metrics-port=8002 \
  --inference-timeout-ms=15000 \
  --num-threads=4 \
  --concurrency-per-model=8 \
  --log-level=debug \
  --log-dir=/var/log/axon
```

---

## Docker Example

```bash
docker run \
  -v ./models:/models \
  -v ./logs:/tmp/logs/axon \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317 \
  mustafa12/axon:latest \
  --model-repository=/models \
  --model-control-mode=poll \
  --log-level=debug
```

## Docker GPU Example

```bash
docker run --gpus all \
  -v ./models:/models \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -e AXON_API_KEY=change-me \
  mustafa12/axon:latest-gpu \
  --model-repository=/models \
  --device=cuda \
  --gpu-device-ids=0
```
