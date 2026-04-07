#!/usr/bin/env python3
"""
TTFT (Time-To-First-Token) benchmarking tool for LLM inference servers.

Sends requests to an OpenAI-compatible /v1/completions or /v1/chat/completions
endpoint with streaming enabled, and precisely measures the time from request
dispatch to the first SSE chunk containing a token.

Designed to isolate TTFT differences across machines that share the same GPU
hardware and model but differ in host-level config (CPU governor, NUMA, kernel,
provider firmware, etc.).

Test matrix dimensions:
  - Prompt length: sweep from short (32 tokens) to long (8192+)
  - Concurrency: 1 → N simultaneous requests (prefill contention)
  - Cold vs warm: optional server restart between rounds
  - Repeated trials: statistical confidence

Usage:
    uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5
    uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5 \
        --prompt-lengths 32,128,512,2048,8192 --concurrency-levels 1,4,16 --trials 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import string
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp


@dataclass
class TTFTSample:
    trial: int
    prompt_length: int
    concurrency: int
    ttft_s: float
    total_latency_s: float
    output_tokens: int
    error: Optional[str] = None
    timestamp: float = 0.0


@dataclass
class TTFTResult:
    prompt_length: int
    concurrency: int
    trials: int
    samples: List[TTFTSample] = field(default_factory=list)

    @property
    def successful(self) -> List[TTFTSample]:
        return [s for s in self.samples if s.error is None]

    def ttft_values(self) -> List[float]:
        return [s.ttft_s for s in self.successful]

    def summary(self) -> Dict[str, Any]:
        vals = self.ttft_values()
        if not vals:
            return {"prompt_length": self.prompt_length, "concurrency": self.concurrency,
                    "error": "all requests failed"}
        vals_ms = [v * 1000 for v in vals]
        return {
            "prompt_length": self.prompt_length,
            "concurrency": self.concurrency,
            "trials": self.trials,
            "successful": len(vals),
            "ttft_ms": {
                "mean": round(statistics.mean(vals_ms), 2),
                "median": round(statistics.median(vals_ms), 2),
                "p90": round(sorted(vals_ms)[int(len(vals_ms) * 0.9)], 2),
                "p99": round(sorted(vals_ms)[min(int(len(vals_ms) * 0.99), len(vals_ms) - 1)], 2),
                "min": round(min(vals_ms), 2),
                "max": round(max(vals_ms), 2),
                "stdev": round(statistics.stdev(vals_ms), 2) if len(vals_ms) > 1 else 0,
            },
        }


def generate_prompt_text(target_tokens: int) -> str:
    """Generate a deterministic filler prompt of approximately target_tokens tokens.

    Uses a repeating pattern of common English words (roughly 1 token per word
    for most tokenizers). Not perfect, but consistent across runs which is what
    matters for A/B comparison.
    """
    words = (
        "The quick brown fox jumps over the lazy dog while the sun sets behind "
        "the mountain range casting long shadows across the valley below where "
        "a river winds its way through dense forests and open meadows filled "
        "with wildflowers swaying gently in the evening breeze as birds return "
        "to their nests and the first stars begin to appear in the darkening sky "
    )
    # ~1.3 tokens per word on average for most tokenizers
    target_words = int(target_tokens / 1.3)
    repeated = (words + " ") * ((target_words // len(words.split())) + 1)
    return " ".join(repeated.split()[:target_words])


async def measure_ttft_completion(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    extra_body: Optional[Dict] = None,
) -> Tuple[float, float, int, Optional[str]]:
    """Send a streaming completion request and return (ttft, total_latency, output_tokens, error)."""
    url = f"{base_url}/v1/completions"
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if extra_body:
        body.update(extra_body)

    ttft = 0.0
    total_tokens = 0
    start = time.perf_counter()

    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                return 0, 0, 0, f"HTTP {resp.status}: {text[:500]}"

            first_token_seen = False
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                text_piece = choices[0].get("text", "")
                if text_piece and not first_token_seen:
                    ttft = time.perf_counter() - start
                    first_token_seen = True
                if text_piece:
                    total_tokens += 1

    except asyncio.TimeoutError:
        return 0, 0, 0, "timeout"
    except Exception as exc:
        return 0, 0, 0, str(exc)

    total_latency = time.perf_counter() - start
    if not first_token_seen:
        return 0, total_latency, 0, "no tokens received"
    return ttft, total_latency, total_tokens, None


async def measure_ttft_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    extra_body: Optional[Dict] = None,
) -> Tuple[float, float, int, Optional[str]]:
    """Send a streaming chat completion request and return (ttft, total_latency, output_tokens, error)."""
    url = f"{base_url}/v1/chat/completions"
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    if extra_body:
        body.update(extra_body)

    ttft = 0.0
    total_tokens = 0
    start = time.perf_counter()

    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                text = await resp.text()
                return 0, 0, 0, f"HTTP {resp.status}: {text[:500]}"

            first_token_seen = False
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text_piece = delta.get("content", "")
                if text_piece and not first_token_seen:
                    ttft = time.perf_counter() - start
                    first_token_seen = True
                if text_piece:
                    total_tokens += 1

    except asyncio.TimeoutError:
        return 0, 0, 0, "timeout"
    except Exception as exc:
        return 0, 0, 0, str(exc)

    total_latency = time.perf_counter() - start
    if not first_token_seen:
        return 0, total_latency, 0, "no tokens received"
    return ttft, total_latency, total_tokens, None


async def run_concurrent_ttft(
    base_url: str,
    model: str,
    prompt_text: str,
    concurrency: int,
    max_tokens: int,
    trial: int,
    prompt_length: int,
    api: str,
    extra_body: Optional[Dict] = None,
) -> List[TTFTSample]:
    """Fire `concurrency` simultaneous requests and collect TTFT for each."""
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        measure_fn = measure_ttft_chat if api == "chat" else measure_ttft_completion

        tasks = [
            measure_fn(session, base_url, model, prompt_text, max_tokens, extra_body)
            for _ in range(concurrency)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    samples = []
    for r in results:
        if isinstance(r, Exception):
            samples.append(TTFTSample(
                trial=trial, prompt_length=prompt_length, concurrency=concurrency,
                ttft_s=0, total_latency_s=0, output_tokens=0,
                error=str(r), timestamp=time.time(),
            ))
        else:
            ttft, total_lat, out_tok, err = r
            samples.append(TTFTSample(
                trial=trial, prompt_length=prompt_length, concurrency=concurrency,
                ttft_s=ttft, total_latency_s=total_lat, output_tokens=out_tok,
                error=err, timestamp=time.time(),
            ))
    return samples


def warmup(base_url: str, model: str, api: str) -> bool:
    """Send a few synchronous requests to warm up the server."""
    import requests as req

    print("  Warming up server...", end=" ", flush=True)
    url = f"{base_url}/v1/chat/completions" if api == "chat" else f"{base_url}/v1/completions"
    for i in range(3):
        try:
            if api == "chat":
                body = {"model": model, "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 4, "temperature": 0}
            else:
                body = {"model": model, "prompt": "Hi", "max_tokens": 4, "temperature": 0}
            r = req.post(url, json=body, timeout=120)
            if r.status_code != 200:
                print(f"\n  Warmup request {i+1} returned {r.status_code}: {r.text[:200]}")
                return False
        except Exception as exc:
            print(f"\n  Warmup request {i+1} failed: {exc}")
            return False
    print("done")
    return True


def check_server(base_url: str) -> bool:
    import requests as req
    try:
        r = req.get(f"{base_url}/v1/models", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def print_results_table(results: List[TTFTResult]) -> None:
    print(f"\n{'═' * 90}")
    print(f"  TTFT Benchmark Results")
    print(f"{'═' * 90}")
    header = f"  {'Prompt Len':>10}  {'Concur':>6}  {'Trials':>6}  {'Mean':>8}  {'Median':>8}  {'P90':>8}  {'P99':>8}  {'StdDev':>8}"
    print(header)
    print(f"  {'':>10}  {'':>6}  {'':>6}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}")
    print(f"  {'─' * 78}")

    for r in results:
        s = r.summary()
        if "error" in s:
            print(f"  {s['prompt_length']:>10}  {s['concurrency']:>6}  {'—':>6}  {'FAILED':>8}")
            continue
        t = s["ttft_ms"]
        print(
            f"  {s['prompt_length']:>10}  {s['concurrency']:>6}  {s['successful']:>6}"
            f"  {t['mean']:>8.1f}  {t['median']:>8.1f}  {t['p90']:>8.1f}"
            f"  {t['p99']:>8.1f}  {t['stdev']:>8.1f}"
        )
    print(f"{'═' * 90}\n")


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]
    concurrency_levels = [int(x) for x in args.concurrency_levels.split(",")]

    if not check_server(args.base_url):
        print(f"ERROR: server at {args.base_url} is not reachable")
        sys.exit(1)

    if args.warmup:
        if not warmup(args.base_url, args.model, args.api):
            print("WARNING: warmup failed, continuing anyway")

    all_results: List[TTFTResult] = []

    total_combos = len(prompt_lengths) * len(concurrency_levels)
    combo = 0
    for pl in prompt_lengths:
        prompt_text = generate_prompt_text(pl)
        for conc in concurrency_levels:
            combo += 1
            result = TTFTResult(prompt_length=pl, concurrency=conc, trials=args.trials)
            print(f"\n  [{combo}/{total_combos}] prompt_len={pl} concurrency={conc} trials={args.trials}")

            for t in range(args.trials):
                if args.inter_trial_delay > 0 and t > 0:
                    await asyncio.sleep(args.inter_trial_delay)

                samples = await run_concurrent_ttft(
                    base_url=args.base_url,
                    model=args.model,
                    prompt_text=prompt_text,
                    concurrency=conc,
                    max_tokens=args.max_tokens,
                    trial=t,
                    prompt_length=pl,
                    api=args.api,
                    extra_body=json.loads(args.extra_body) if args.extra_body else None,
                )
                result.samples.extend(samples)

                ok = sum(1 for s in samples if s.error is None)
                ttfts = [s.ttft_s * 1000 for s in samples if s.error is None]
                if ttfts:
                    print(f"    trial {t}: {ok}/{conc} ok, "
                          f"TTFT mean={statistics.mean(ttfts):.1f}ms "
                          f"median={statistics.median(ttfts):.1f}ms")
                else:
                    errs = [s.error for s in samples if s.error]
                    print(f"    trial {t}: all failed — {errs[0] if errs else '?'}")

            all_results.append(result)

    print_results_table(all_results)

    output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "model": args.model,
            "api": args.api,
            "max_tokens": args.max_tokens,
            "warmup": args.warmup,
            "inter_trial_delay_s": args.inter_trial_delay,
        },
        "results": [r.summary() for r in all_results],
        "raw_samples": [asdict(s) for r in all_results for s in r.samples],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TTFT benchmark for LLM inference servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5

  # Full sweep
  uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5 \\
      --prompt-lengths 32,128,512,2048,8192 --concurrency-levels 1,2,4,8,16 \\
      --trials 20 -o results.json

  # Chat API with extra body params
  uv run python ttft_bench.py --base-url http://localhost:30000 --model moonshotai/Kimi-K2.5 \\
      --api chat --extra-body '{"top_p": 0.9}'
""",
    )
    parser.add_argument("--base-url", required=True, help="Server base URL (e.g. http://localhost:30000)")
    parser.add_argument("--model", required=True, help="Model name as registered on the server")
    parser.add_argument("--api", choices=["completion", "chat"], default="completion",
                        help="Which API endpoint to use (default: completion)")
    parser.add_argument("--prompt-lengths", default="32,128,512,2048",
                        help="Comma-separated prompt lengths in approximate tokens (default: 32,128,512,2048)")
    parser.add_argument("--concurrency-levels", default="1,2,4",
                        help="Comma-separated concurrency levels (default: 1,2,4)")
    parser.add_argument("--trials", type=int, default=5,
                        help="Number of trials per (prompt_length, concurrency) combo (default: 5)")
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="Max tokens to generate per request — keep small to isolate TTFT (default: 16)")
    parser.add_argument("--warmup", action="store_true", default=True,
                        help="Send warmup requests before benchmarking (default: True)")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--inter-trial-delay", type=float, default=0.5,
                        help="Seconds to wait between trials (default: 0.5)")
    parser.add_argument("--extra-body", default=None,
                        help="JSON string of extra fields to include in the request body")
    parser.add_argument("-o", "--output", help="Write full JSON results to file")
    args = parser.parse_args()

    output = asyncio.run(run_benchmark(args))

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(output["results"], indent=2))


if __name__ == "__main__":
    main()
