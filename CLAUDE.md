# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTFT (Time-To-First-Token) benchmarking toolkit for LLM inference servers. Measures TTFT across varying prompt lengths and concurrency levels to isolate host-level performance differences (CPU governor, NUMA, kernel tunables) between machines running the same GPU/model. Includes a launcher for sglang serving Kimi-K2.5 on Blackwell (B300 NVL8) GPUs with CUDA 13.

## Setup & Commands

```bash
# Install dependencies (requires uv)
uv sync                    # benchmark-only deps
uv sync --extra server     # + sglang/CUDA 13 server stack

# Run benchmark against an existing server
./run_benchmark.sh --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5

# Run benchmark with auto-started sglang server
./run_benchmark.sh --start-server --model moonshotai/Kimi-K2.5

# Run individual Python tools
uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5
uv run python system_profile.py -o profile.json
uv run python compare_results.py run_a/summary.json run_b/summary.json
```

There are no tests or linting configured in this project.

## Architecture

All source files are at the repo root (no package structure):

- **`run_benchmark.sh`** — Top-level orchestrator. Syncs uv env, collects system profile (phase 1), optionally starts sglang (phase 2), runs the TTFT benchmark (phase 3), then generates a comparison-ready `summary.json` (phase 4). Output lands in `ttft-results-<hostname>-<timestamp>/`.
- **`ttft_bench.py`** — Async benchmark client using aiohttp. Supports both `/v1/completions` and `/v1/chat/completions` streaming endpoints. Sweeps a matrix of (prompt_length x concurrency x trials), measures wall-clock time to first SSE data chunk containing a token. Key dataclasses: `TTFTSample` (single measurement) and `TTFTResult` (aggregated stats per combo).
- **`system_profile.py`** — Collects host-level config that affects TTFT: CPU governors/frequencies, NUMA topology, memory/THP, kernel tunables, GPU state (persistence mode, P-state, NVLink, PCIe), disk bandwidth (via fio). Outputs JSON and a human-readable summary with TTFT-relevant warnings.
- **`compare_results.py`** — Side-by-side comparison of `summary.json` files from multiple machines. Highlights system config differences and auto-generates diagnosis hints (governor mismatch, persistence mode, NUMA, etc.).
- **`serve_kimi_k2.sh`** — sglang server launcher for Kimi-K2.5 with 8-GPU TP. Handles Blackwell-specific concerns: CUDA 13 NVRTC detection, LD_LIBRARY_PATH for cu130 libs, Triton ptxas override. Configurable via env vars (MODEL, TP, PORT, HOST, MEM_FRACTION, EXTRA_ARGS).

## Key Design Details

- The benchmark generates deterministic filler prompts (~1.3 tokens/word heuristic) for consistency across runs — exact token count doesn't matter, only A/B consistency.
- `warmup()` in `ttft_bench.py` uses synchronous `requests` (not aiohttp) to send 3 throwaway requests before the async benchmark loop.
- The server launcher sets `LD_LIBRARY_PATH` dynamically from the uv virtualenv's site-packages to ensure CUDA 13 NVRTC is found before any system CUDA 12 libs.
- `pyproject.toml` uses explicit uv index sources for CUDA 13 wheels (pytorch-cu130, sglang-cu130) — these only apply to the `server` optional dependency group.
