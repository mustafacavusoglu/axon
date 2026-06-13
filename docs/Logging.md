# Logging

Axon uses a three-layer logging architecture designed to keep stdout clean while providing full observability through files and OpenTelemetry.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 tracing subscriber                   │
├────────────────┬────────────────┬───────────────────┤
│    stdout      │     file       │      OTEL         │
│   (filtered)   │  (daily rot.)  │    (traces)       │
│                │                │                   │
│  Non-blocking  │  Non-blocking  │   Batch export    │
│  Compact fmt   │  JSON format   │   gRPC/OTLP      │
│                │                │                   │
│  Only:         │  Everything:   │  Everything:      │
│  • Model load  │  • Inference   │  • Spans          │
│  • Health      │  • Errors      │  • Events         │
│  • Shutdown    │  • Warnings    │  • If configured  │
│                │  • Debug/Trace │                   │
└────────────────┴────────────────┴───────────────────┘
```

---

## Configuration

### CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--log-level` | `info` | Minimum log level for file output |
| `--log-dir` | `/tmp/logs/axon` | Directory for JSON log files |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint (enables OTEL layer) |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Alternative OTLP traces endpoint |

---

## Stdout Output

Stdout is filtered to show only essential operational events:

1. **Startup**: Model loading progress
2. **Startup Table**: Endpoints + model status (ASCII table)
3. **Health Changes**: When model state changes
4. **Shutdown**: Graceful shutdown notice

Example stdout output:
```
  loading models from /models

  ╔══════════════════════════════════════════════════════════════╗
  ║                    axon-server v0.3.5                       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Endpoints                                                  ║
  ║    HTTP     http://0.0.0.0:8000                             ║
  ║    gRPC     0.0.0.0:8001                                    ║
  ║    Metrics  http://0.0.0.0:8002/metrics                     ║
  ║    Logs     /tmp/logs/axon                                  ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  Models                                                     ║
  ║    Name                     Ver  Platform     Status         ║
  ║    ─────────────────────────────────────────────────────     ║
  ║    roberta_tokenizer        1    script       READY          ║
  ║    roberta_sentiment        1    onnxruntime  READY          ║
  ║    roberta_decoder          1    script       READY          ║
  ║    roberta_pipeline         1    ensemble     READY          ║
  ╚══════════════════════════════════════════════════════════════╝

  shutdown signal received, draining...
  axon-server stopped
```

---

## File Logging

Log files are written to `--log-dir` with daily rotation.

**File naming**: `axon-server.json.YYYY-MM-DD`

**Format**: JSON (one object per line), compatible with log aggregators (Loki, ELK, Datadog).

### Example Log Entry
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "target": "axon_server::http_server",
  "span": {"name": "run_inference"},
  "fields": {
    "model": "roberta_pipeline",
    "latency_ms": "12.34",
    "total_ms": "15.67",
    "message": "inference completed"
  }
}
```

### Log Levels

| Level | File Content |
|-------|-------------|
| `error` | Only errors |
| `warn` | Errors + warnings (timeouts, bad requests) |
| `info` | + Inference completions, model load/unload |
| `debug` | + Request received, circuit breaker decisions |
| `trace` | + Internal engine details |

---

## OpenTelemetry Integration

When `OTEL_EXPORTER_OTLP_ENDPOINT` is set, Axon exports traces via gRPC OTLP.

```bash
# Example: export to Jaeger
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  axon-server --model-repository=/models

# Example: export to Grafana Tempo
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317 \
  axon-server --model-repository=/models
```

Traces include:
- Service name: `axon-server`
- Inference spans with model name, duration
- Model loading spans
- Error details

---

## Inference Tracing

Every inference request is logged with timing information:

| Field | Description |
|-------|-------------|
| `model` | Model name |
| `latency_ms` | Pure inference time (model execution only) |
| `total_ms` | Total request time (queue + inference + serialization) |

The difference `total_ms - latency_ms` represents overhead (queue waiting, input parsing, output serialization).

---

## Docker Logging

In Docker, stdout goes to container logs (`docker logs`):
```bash
docker logs axon-server
```

File logs are inside the container at `/tmp/logs/axon/`. Mount a volume to persist:
```bash
docker run -v ./logs:/tmp/logs/axon \
  mustdo12/axon-server:latest --model-repository=/models
```

---

## Grafana Loki Integration

JSON file logs are directly compatible with Loki's `json` parser:

```yaml
# promtail config
scrape_configs:
  - job_name: axon
    static_configs:
      - targets: [localhost]
        labels:
          job: axon-server
          __path__: /tmp/logs/axon/*.json*
    pipeline_stages:
      - json:
          expressions:
            level: level
            model: fields.model
            latency: fields.latency_ms
```

---

## Performance Impact

All log writers use `tracing-appender`'s `non_blocking()` wrapper:
- Log calls return immediately (write to bounded channel)
- Background thread handles actual I/O
- Inference latency is not affected by disk/network I/O
- If the channel fills up (under extreme load), events are dropped rather than blocking
