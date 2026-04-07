#!/usr/bin/env bash
#
# Launch sglang server for Kimi K2.5 on B300 NVL8 (8-GPU tensor parallelism).
#
# The Python environment for both the benchmark client and the server is managed
# with uv from this directory's `pyproject.toml`. The server extra pins the
# CUDA 13 / Blackwell-compatible stack (torch, sglang, sglang-kernel, NVRTC).
#
# Usage:
#   ./serve_kimi_k2.sh                    # sync uv env + launch server
#   ./serve_kimi_k2.sh --port 31000       # custom port
#   MODEL=custom/model ./serve_kimi_k2.sh # override model
#
# Environment variables:
#   MODEL          - HuggingFace model path (default: moonshotai/Kimi-K2.5)
#   TP             - Tensor parallel degree (default: 8 for NVL8)
#   PORT           - Server port (default: 30000)
#   HOST           - Bind address (default: 0.0.0.0)
#   MEM_FRACTION   - GPU memory fraction (default: 0.95)
#   EXTRA_ARGS     - Additional sglang arguments
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-moonshotai/Kimi-K2.5}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"
MEM_FRACTION="${MEM_FRACTION:-0.95}"

ensure_uv() {
    if command -v uv &>/dev/null; then
        return
    fi

    echo "ERROR: uv is required but was not found."
    echo "Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
}

is_blackwell_gpu() {
    nvidia-smi -L 2>/dev/null | grep -qiE 'B[0-9]{3}|Blackwell'
}

uv_project_run() {
    uv run --project "${SCRIPT_DIR}" "$@"
}

uv_project_python() {
    uv_project_run python "$@"
}

sync_server_environment() {
    echo ""
    echo "  Syncing uv environment (server extra)..."
    uv sync --project "${SCRIPT_DIR}" --extra server
    echo ""
}

torch_cuda_version() {
    uv_project_python - <<'PY'
try:
    import torch
    cuda_version = torch.version.cuda or ""
except Exception:
    cuda_version = ""
print(cuda_version)
PY
}

echo "════════════════════════════════════════════════════════════════"
echo "  sglang server — Kimi K2.5"
echo "════════════════════════════════════════════════════════════════"
echo "  Model:            ${MODEL}"
echo "  Tensor parallel:  ${TP}"
echo "  Endpoint:         http://${HOST}:${PORT}"
echo "  GPU mem fraction: ${MEM_FRACTION}"
echo "════════════════════════════════════════════════════════════════"

# ── Preflight checks ────────────────────────────────────────────────────────
ensure_uv
sync_server_environment

if ! uv_project_python -c "import sglang" 2>/dev/null; then
    echo "ERROR: sglang is unavailable in the uv environment"
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if (( GPU_COUNT < TP )); then
    echo "WARNING: found ${GPU_COUNT} GPUs but TP=${TP}. Adjusting TP to ${GPU_COUNT}."
    TP="${GPU_COUNT}"
fi

# Detect Blackwell and check that PyTorch is built for CUDA ≥ 13
if is_blackwell_gpu; then
    TORCH_CUDA="$(torch_cuda_version)"
    TORCH_CUDA_MAJOR="${TORCH_CUDA%%.*}"
    if [[ -z "$TORCH_CUDA_MAJOR" ]] || (( TORCH_CUDA_MAJOR < 13 )); then
        echo ""
        echo "  ╔══════════════════════════════════════════════════════════════╗"
        echo "  ║  WARNING: Blackwell GPU detected but PyTorch reports        ║"
        echo "  ║  CUDA ${TORCH_CUDA} — needs cu130 builds.                  ║"
        echo "  ║                                                              ║"
        echo "  ║  The Kimi K2.5 vision tower calls erfinv_() during init,    ║"
        echo "  ║  triggering NVRTC JIT compilation. An older NVRTC runtime   ║"
        echo "  ║  may not recognize Blackwell (sm_100/sm_120), causing:      ║"
        echo "  ║    nvrtc: error: invalid value for --gpu-architecture       ║"
        echo "  ║                                                              ║"
        echo "  ║  Fix: ensure uv synced the server extra from pyproject.toml ║"
        echo "  ║  and that the CUDA 13 indexes resolved correctly.           ║"
        echo "  ║                                                              ║"
        echo "  ║  The uv environment should include PyTorch cu130 and        ║"
        echo "  ║  sglang-kernel from the CUDA 13 wheel indexes.             ║"
        echo "  ╚══════════════════════════════════════════════════════════════╝"
        echo ""
        exit 1
    fi
fi

# Enable persistence mode on all GPUs to reduce first-call latency
if command -v nvidia-smi &>/dev/null; then
    echo "  Enabling GPU persistence mode..."
    sudo nvidia-smi -pm 1 2>/dev/null || true
fi

# Prefer the CUDA 13 runtime shipped by the PyTorch/cu130 stack and avoid the
# CUDA 12 NVRTC namespace package, which can shadow libnvrtc.so.13 at import time.
mapfile -t _SITE_PKG_ROOTS < <(uv_project_python - <<'PY'
import site

paths = []
try:
    paths.extend(site.getsitepackages())
except Exception:
    pass

user_site = site.getusersitepackages()
if user_site:
    paths.append(user_site)

for path in dict.fromkeys(paths):
    print(path)
PY
)
_SGL_CUDA13_LIBS=()
for _pkg_root in "${_SITE_PKG_ROOTS[@]}"; do
    _SGL_CUDA13_LIBS+=(
        "${_pkg_root}/torch/lib"
        "${_pkg_root}/nvidia/cu13/lib"
    )
done
_SGL_EXISTING_LIBS=()
for _libdir in "${_SGL_CUDA13_LIBS[@]}"; do
    if [[ -d "${_libdir}" ]]; then
        _SGL_EXISTING_LIBS+=("${_libdir}")
    fi
done
if (( ${#_SGL_EXISTING_LIBS[@]} > 0 )); then
    _SGL_JOINED_LIBS="$(IFS=:; echo "${_SGL_EXISTING_LIBS[*]}")"
    export LD_LIBRARY_PATH="${_SGL_JOINED_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Triton bundles ptxas from CUDA 12.x which doesn't recognize Blackwell
# (sm_103a). Point it at the system CUDA 13 ptxas instead.
if [[ -z "${TRITON_PTXAS_PATH:-}" ]] && command -v ptxas &>/dev/null; then
    export TRITON_PTXAS_PATH="$(command -v ptxas)"
fi

exec uv run --project "${SCRIPT_DIR}" sglang serve \
    --model-path "${MODEL}" \
    --tp "${TP}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION}" \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    --trust-remote-code \
    ${EXTRA_ARGS:-} \
    "$@"
