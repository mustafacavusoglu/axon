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
- `poll` — Periodically scan repository for new/updated models

```bash
axon-server --model-control-mode=poll
```

### `--repository-poll-secs <SECONDS>`
**Default:** `30`

How often to scan the model repository for changes (only with `--model-control-mode=poll`).

---

## Network

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

Number of inference worker threads (rayon thread pool). Set to 0 to auto-detect.

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
  mustdo12/axon-server:latest \
  --model-repository=/models \
  --model-control-mode=poll \
  --log-level=debug
```
