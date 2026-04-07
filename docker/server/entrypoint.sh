#!/usr/bin/env bash
set -euo pipefail

# ── CUDA 13 library path setup ─────────────────────────────────────────────
# Replicate the LD_LIBRARY_PATH logic from serve_kimi_k2.sh to ensure
# CUDA 13 NVRTC and torch libs are found before any system CUDA 12 libs.
SITE_PACKAGES=$(/app/.venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")

_CUDA13_LIBS=()
for _libdir in "${SITE_PACKAGES}/torch/lib" "${SITE_PACKAGES}/nvidia/cu13/lib"; do
    if [[ -d "${_libdir}" ]]; then
        _CUDA13_LIBS+=("${_libdir}")
    fi
done
if (( ${#_CUDA13_LIBS[@]} > 0 )); then
    _JOINED="$(IFS=:; echo "${_CUDA13_LIBS[*]}")"
    export LD_LIBRARY_PATH="${_JOINED}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"

# ── CUDA 12 runtime libs ──────────────────────────────────────────────────
# sgl_kernel's sm100 binaries are linked against CUDA 12 versioned symbols.
# The actual CUDA 12 runtime libs (libcublas-12-8 etc.) are installed via
# apt in the Dockerfile alongside the CUDA 13 base. Add their path.
if [[ -d /usr/local/cuda-12.8/lib64 ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda-12.8/lib64"
fi

# ── GPU persistence mode (best-effort, requires privileged) ────────────────
nvidia-smi -pm 1 2>/dev/null || true

# ── Configuration via environment variables ────────────────────────────────
MODEL="${MODEL:-moonshotai/Kimi-K2.5}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION="${MEM_FRACTION:-0.95}"

# Auto-detect GPU count and adjust TP if needed
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || echo 0)
if (( GPU_COUNT > 0 && GPU_COUNT < TP )); then
    echo "WARNING: found ${GPU_COUNT} GPUs but TP=${TP}. Adjusting TP to ${GPU_COUNT}."
    TP="${GPU_COUNT}"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  sglang server"
echo "════════════════════════════════════════════════════════════════"
echo "  Model:            ${MODEL}"
echo "  Tensor parallel:  ${TP}"
echo "  Endpoint:         http://${HOST}:${PORT}"
echo "  GPU mem fraction: ${MEM_FRACTION}"
echo "  LD_LIBRARY_PATH:  ${LD_LIBRARY_PATH:-<unset>}"
echo "════════════════════════════════════════════════════════════════"

exec /app/.venv/bin/python3 -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tp "${TP}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --trust-remote-code \
    --watchdog-timeout 1200 \
    ${EXTRA_ARGS:-} \
    "$@"
