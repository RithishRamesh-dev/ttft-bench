#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://ttft-server:30000}"
MODEL="${MODEL:-moonshotai/Kimi-K2.5}"
OUTPUT_DIR="${OUTPUT_DIR:-/results}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"

mkdir -p "${OUTPUT_DIR}"

# Wait for server readiness
ELAPSED=0
echo "Waiting for server at ${BASE_URL}..."
while ! curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if (( ELAPSED >= SERVER_WAIT_TIMEOUT )); then
        echo "ERROR: server not ready after ${SERVER_WAIT_TIMEOUT}s"
        exit 1
    fi
    echo "  ... waiting (${ELAPSED}s / ${SERVER_WAIT_TIMEOUT}s)"
done
echo "Server is ready."

exec /app/run_benchmark.sh \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --output-dir "${OUTPUT_DIR}" \
    ${BENCHMARK_ARGS:-} \
    "$@"
