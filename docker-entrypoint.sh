#!/bin/sh
set -e

export OMP_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE

# WSL2 workaround: copy bind-mounted models to local fs
# WSL2's 9p filesystem is ~100x slower for random reads than native ext4.
# ONNX Runtime's CreateSession does heavy random I/O during graph parsing,
# which appears as a hang on bind-mounted volumes.
MODEL_SRC="${MODEL_SRC:-/models-in}"
MODEL_DST="${MODEL_DST:-/models}"

if [ -d "$MODEL_SRC" ] && [ "$(ls -A "$MODEL_SRC" 2>/dev/null)" ]; then
    echo "axon: copying models from $MODEL_SRC to $MODEL_DST (WSL2/9p workaround)..."
    cp -r "$MODEL_SRC"/* "$MODEL_DST"/
    echo "axon: models copied, starting server..."
    echo "=== /models contents ==="
    find /models -type f -o -type d | sort
    echo "========================="
fi

exec axon-server "$@"
