# `ttft-bench`

TTFT benchmarking utilities for inference servers, plus a helper to launch `sglang` for `moonshotai/Kimi-K2.5`.

## What This Includes

- `run_benchmark.sh`: collect system profile, optionally start a server, run the benchmark, and write a summary bundle
- `ttft_bench.py`: low-level TTFT benchmark client
- `system_profile.py`: collect host and GPU configuration that can affect TTFT
- `compare_results.py`: compare benchmark summaries across machines
- `serve_kimi_k2.sh`: launch `sglang` with the CUDA 13 / Blackwell-compatible stack

## Prerequisites

- Linux
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- `curl`
- `nvidia-smi` for GPU profiling and server startup

For `--start-server` or `serve_kimi_k2.sh`, you also need:

- NVIDIA GPUs
- CUDA 13-compatible runtime/tooling for Blackwell-class GPUs
- access to download model and Python wheels

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you do not already have Python 3.12 available locally, install it with `uv`:

```bash
uv python install 3.12
```

## Dependency Setup

The project uses `uv` with a local project environment.

## Using `uv` In This Project

`uv` reads `pyproject.toml` and `.python-version`, creates a local `.venv`, and runs commands inside that environment.

Typical `uv` commands for this directory:

```bash
# Create/update the local virtualenv with benchmark dependencies
uv sync

# Include the optional server stack too
uv sync --extra server

# Run commands inside the project environment without activating it
uv run python ttft_bench.py --help

# Optional: activate the environment manually if you prefer
source .venv/bin/activate
```

In short:

- use `uv sync` to install dependencies
- use `uv sync --extra server` when you also need `sglang` and CUDA 13 server packages
- use `uv run ...` for Python commands so they always use the project's environment

Install dependencies:

```bash
uv sync
```

## Common Workflows

### 1. Benchmark an existing server

```bash
uv sync
./run_benchmark.sh --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5
```

### 2. Start a local server and benchmark it

```bash
uv sync --extra server
./run_benchmark.sh --start-server --model moonshotai/Kimi-K2.5
```

### 3. Start only the local server

```bash
uv sync --extra server
./serve_kimi_k2.sh
```

## Direct Python Entry Points

If you want to run the Python tools directly, use `uv run`:

```bash
uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5
uv run python system_profile.py -o profile.json
uv run python compare_results.py run_a/summary.json run_b/summary.json
```

## Output Files

Each `run_benchmark.sh` run creates an output directory like:

```text
ttft-results-<hostname>-<timestamp>/
```

It contains:

- `system_profile.json`: hardware and OS profile
- `ttft_results.json`: full benchmark results
- `summary.json`: compact comparison-ready summary

## Comparing Machines

After collecting results on multiple hosts:

```bash
uv sync
uv run python compare_results.py machine_a/summary.json machine_b/summary.json
```

You can also pass output directories directly if they contain `summary.json`:

```bash
uv run python compare_results.py run_a/ run_b/ run_c/
```

## Notes

- `run_benchmark.sh` always syncs the benchmark environment before running.
- `serve_kimi_k2.sh` always syncs the `server` extra before launching.
- On Blackwell GPUs, the launcher expects the CUDA 13 wheels declared in `pyproject.toml`.
- `serve_kimi_k2.sh` enables GPU persistence mode when possible.
