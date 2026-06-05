#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="$SCRIPT_DIR/local_models/model_repository"
SOCKET="/tmp/inference.sock"

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $ENGINE_PID $CP_PID 2>/dev/null || true
    wait 2>/dev/null || true
    rm -f "$SOCKET"
    echo "Done."
}
trap cleanup EXIT INT TERM

echo "=== Building Rust engine ==="
cd "$SCRIPT_DIR/inference-engine"
cargo build 2>/dev/null

echo "=== Building Go control plane ==="
cd "$SCRIPT_DIR/control-plane"
go build -o /tmp/control-plane ./cmd/server 2>/dev/null

echo ""
echo "=== Starting inference engine ==="
rm -f "$SOCKET"
cd "$SCRIPT_DIR/inference-engine"
SOCKET_PATH="$SOCKET" NUM_THREADS=4 RUST_LOG=info ./target/debug/inference-engine &
ENGINE_PID=$!
sleep 2

if [ ! -S "$SOCKET" ]; then
    echo "ERROR: Engine failed to start"
    exit 1
fi
echo "Engine ready (PID=$ENGINE_PID)"

echo ""
echo "=== Starting control plane ==="
MODEL_REPO_PATH="$MODEL_REPO" INFERENCE_SOCKET="$SOCKET" HTTP_PORT=8080 GRPC_PORT=8001 /tmp/control-plane &
CP_PID=$!
sleep 4
echo "Control plane ready (PID=$CP_PID)"

echo ""
echo "══════════════════════════════════════════════"
echo "  Server is running!"
echo "  HTTP:  http://localhost:8080"
echo "  gRPC:  localhost:8001"
echo "══════════════════════════════════════════════"
echo ""
echo "Commands to test in another terminal:"
echo ""
echo "  # Health check"
echo "  curl http://localhost:8080/v2/health/live | jq"
echo ""
echo "  # List models"
echo "  curl http://localhost:8080/v2/models | jq"
echo ""
echo "  # Model metadata"
echo "  curl http://localhost:8080/v2/models/lgbm_breast_cancer | jq"
echo ""
echo "  # Run inference"
echo "  curl -s -X POST http://localhost:8080/v2/models/lgbm_breast_cancer/infer \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"inputs\":[{\"name\":\"input\",\"shape\":[30],\"datatype\":\"FP32\",\"data\":["
printf "1.0"
for i in $(seq 2 30); do printf ", 1.0"; done
echo "]}]}' | jq"
echo ""
echo "Press Ctrl+C to stop"
echo ""

wait
