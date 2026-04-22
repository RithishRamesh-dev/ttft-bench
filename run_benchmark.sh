#!/usr/bin/env bash
#
# Orchestrator: collect system profile, optionally start sglang, run TTFT benchmark.
#
# Usage:
#   # Against an already-running server:
#   uv sync
#   ./run_benchmark.sh --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5
#
#   # Let the script start the server, benchmark, then tear it down:
#   uv sync --extra server
#   ./run_benchmark.sh --start-server --model moonshotai/Kimi-K2.5
#
#   # Full sweep with custom params:
#   ./run_benchmark.sh --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5 \
#       --prompt-lengths 32,128,512,2048,8192 --concurrency-levels 1,2,4,8,16 --trials 20
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ensure_uv() {
    if command -v uv &>/dev/null; then
        return
    fi

    echo "ERROR: uv is required but was not found."
    echo "Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
}

sync_benchmark_environment() {
    if [[ "${SKIP_UV_SYNC:-}" == "1" ]]; then
        echo "  Skipping uv sync (SKIP_UV_SYNC=1)"
        return
    fi
    echo ""
    echo "Syncing uv environment (benchmark deps)..."
    uv sync --project "${SCRIPT_DIR}"
    echo ""
}

uv_project_python() {
    if [[ "${SKIP_UV_SYNC:-}" == "1" ]]; then
        python "$@"
    else
        uv run --project "${SCRIPT_DIR}" python "$@"
    fi
}

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_URL="${BASE_URL:-http://localhost:30000}"
MODEL="${MODEL:-moonshotai/Kimi-K2.5}"
PROMPT_LENGTHS="32,128,512,2048"
CONCURRENCY_LEVELS="1,2,4"
TRIALS=5
MAX_TOKENS=16
API="completion"
START_SERVER=false
OUTPUT_DIR=""
EXTRA_BENCH_ARGS=()

# ── Arg parsing ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url)       BASE_URL="$2"; shift 2 ;;
        --model)          MODEL="$2"; shift 2 ;;
        --prompt-lengths) PROMPT_LENGTHS="$2"; shift 2 ;;
        --concurrency-levels) CONCURRENCY_LEVELS="$2"; shift 2 ;;
        --trials)         TRIALS="$2"; shift 2 ;;
        --max-tokens)     MAX_TOKENS="$2"; shift 2 ;;
        --api)            API="$2"; shift 2 ;;
        --start-server)   START_SERVER=true; shift ;;
        --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --base-url URL          Server URL (default: http://localhost:30000)"
            echo "  --model MODEL           Model name (default: moonshotai/Kimi-K2.5)"
            echo "  --prompt-lengths L      Comma-sep prompt lengths (default: 32,128,512,2048)"
            echo "  --concurrency-levels C  Comma-sep concurrency (default: 1,2,4)"
            echo "  --trials N              Trials per combo (default: 5)"
            echo "  --max-tokens N          Max output tokens (default: 16)"
            echo "  --api TYPE              completion or chat (default: completion)"
            echo "  --start-server          Launch sglang server before benchmarking"
            echo "  --output-dir DIR        Directory for output files (default: ./ttft-results-TIMESTAMP)"
            exit 0
            ;;
        *) EXTRA_BENCH_ARGS+=("$1"); shift ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
HOSTNAME="$(hostname)"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./ttft-results-${HOSTNAME}-${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  TTFT Benchmark Orchestrator"
echo "════════════════════════════════════════════════════════════════"
echo "  Host:       ${HOSTNAME}"
echo "  Timestamp:  ${TIMESTAMP}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Model:      ${MODEL}"
echo "  Server:     ${BASE_URL}"
echo "════════════════════════════════════════════════════════════════"

ensure_uv
sync_benchmark_environment

# ── Phase 1: System profile ──────────────────────────────────────────────────
echo ""
echo "Phase 1: Collecting system profile..."
uv_project_python "${SCRIPT_DIR}/system_profile.py" -o "${OUTPUT_DIR}/system_profile.json"

# ── Phase 2: Optionally start server ─────────────────────────────────────────
SERVER_PID=""
if [[ "$START_SERVER" == "true" ]]; then
    echo ""
    echo "Phase 2: Starting sglang server..."
    MODEL="${MODEL}" PORT="${PORT:-30000}" bash "${SCRIPT_DIR}/serve_kimi_k2.sh" &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    echo "  Waiting for server to be ready..."
    MAX_WAIT=600
    ELAPSED=0
    while ! curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; do
        sleep 5
        ELAPSED=$((ELAPSED + 5))
        if (( ELAPSED >= MAX_WAIT )); then
            echo "ERROR: server did not become ready within ${MAX_WAIT}s"
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
        echo "    ... waiting (${ELAPSED}s / ${MAX_WAIT}s)"
    done
    echo "  Server is ready."
else
    echo ""
    echo "Phase 2: Checking server at ${BASE_URL}..."
    if ! curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then
        echo "ERROR: no server reachable at ${BASE_URL}/v1/models"
        echo "  Start a server first or use --start-server"
        exit 1
    fi
    echo "  Server is reachable."
fi

# ── Phase 3: Run benchmark ───────────────────────────────────────────────────
echo ""
echo "Phase 3: Running TTFT benchmark..."
uv_project_python "${SCRIPT_DIR}/ttft_bench.py" \
    --base-url "${BASE_URL}" \
    --model "${MODEL}" \
    --api "${API}" \
    --prompt-lengths "${PROMPT_LENGTHS}" \
    --concurrency-levels "${CONCURRENCY_LEVELS}" \
    --trials "${TRIALS}" \
    --max-tokens "${MAX_TOKENS}" \
    -o "${OUTPUT_DIR}/ttft_results.json" \
    "${EXTRA_BENCH_ARGS[@]}"

# ── Phase 4: Generate comparison-ready summary ───────────────────────────────
echo ""
echo "Phase 4: Generating summary..."
uv_project_python - <<'PYEOF'
import json, sys
from pathlib import Path

out_dir = sys.argv[1] if len(sys.argv) > 1 else "."
profile_path = Path(out_dir) / "system_profile.json"
results_path = Path(out_dir) / "ttft_results.json"
summary_path = Path(out_dir) / "summary.json"

profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
results = json.loads(results_path.read_text()) if results_path.exists() else {}

cpu = profile.get("cpu", {})
gpu_info = profile.get("gpu", {})
gpus = gpu_info.get("gpus", [])

summary = {
    "hostname": profile.get("hostname", "unknown"),
    "timestamp": profile.get("timestamp", ""),
    "cpu_model": cpu.get("model_name", ""),
    "cpu_governor": cpu.get("scaling_governors", []),
    "cpu_cores": cpu.get("physical_cores"),
    "turbo_boost": cpu.get("turbo_boost", ""),
    "numa_nodes": profile.get("numa", {}).get("node_count", 0),
    "ram_gb": round(profile.get("memory", {}).get("memtotal_kb", 0) / 1048576, 1),
    "gpu_count": gpu_info.get("count", 0),
    "gpu_name": gpus[0].get("name", "") if gpus else "",
    "gpu_persistence": gpus[0].get("persistence_mode", "") if gpus else "",
    "kernel": profile.get("kernel", {}).get("release", ""),
    "ttft_results": results.get("results", []),
}

summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"  Summary written to {summary_path}")
PYEOF
"${OUTPUT_DIR}"

# ── Cleanup ──────────────────────────────────────────────────────────────────
if [[ -n "$SERVER_PID" ]]; then
    echo ""
    echo "Stopping server (PID ${SERVER_PID})..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Done. All output in: ${OUTPUT_DIR}/"
echo ""
echo "  Files:"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "  To compare across machines, copy the summary.json files"
echo "  and run:"
echo "    uv run python compare_results.py machine_a/summary.json machine_b/summary.json"
echo "════════════════════════════════════════════════════════════════"
