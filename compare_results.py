#!/usr/bin/env python3
"""
Compare TTFT benchmark results across machines.

Reads summary.json (or ttft_results.json + system_profile.json) from multiple
benchmark runs and produces a side-by-side comparison highlighting where TTFT
diverges and which system-level factors likely explain the difference.

Usage:
    uv run python compare_results.py run_a/summary.json run_b/summary.json
    uv run python compare_results.py run_a/ run_b/ run_c/   # directories with summary.json inside
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_summary(path: str) -> Dict[str, Any]:
    p = Path(path)
    if p.is_dir():
        p = p / "summary.json"
    return json.loads(p.read_text())


def fmt_ms(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}"


def compare(summaries: List[Dict[str, Any]]) -> None:
    names = [s.get("hostname", f"machine_{i}") for i, s in enumerate(summaries)]
    max_name = max(len(n) for n in names)

    print(f"\n{'═' * 100}")
    print(f"  TTFT Cross-Machine Comparison  ({len(summaries)} machines)")
    print(f"{'═' * 100}")

    # ── System config comparison ─────────────────────────────────────────
    print(f"\n  System Configuration:")
    print(f"  {'':>{max_name}}  ", end="")
    for n in names:
        print(f"  {n:>20}", end="")
    print()
    print(f"  {'─' * (max_name + 22 * len(names))}")

    fields = [
        ("CPU", "cpu_model"),
        ("Cores", "cpu_cores"),
        ("Governor", "cpu_governor"),
        ("Turbo", "turbo_boost"),
        ("NUMA nodes", "numa_nodes"),
        ("RAM (GB)", "ram_gb"),
        ("GPU", "gpu_name"),
        ("GPU count", "gpu_count"),
        ("Persistence", "gpu_persistence"),
        ("Kernel", "kernel"),
    ]

    for label, key in fields:
        vals = []
        for s in summaries:
            v = s.get(key, "—")
            if isinstance(v, list):
                v = ",".join(str(x) for x in v)
            vals.append(str(v))

        differs = len(set(vals)) > 1
        marker = " ←" if differs else ""
        print(f"  {label:>{max_name}}  ", end="")
        for v in vals:
            display = v[:20] if len(v) > 20 else v
            print(f"  {display:>20}", end="")
        if differs:
            print(f"  ← DIFFERS", end="")
        print()

    # ── TTFT comparison ──────────────────────────────────────────────────
    all_combos: Dict[Tuple[int, int], Dict[str, Dict]] = {}
    for i, s in enumerate(summaries):
        for r in s.get("ttft_results", []):
            key = (r["prompt_length"], r["concurrency"])
            if key not in all_combos:
                all_combos[key] = {}
            all_combos[key][names[i]] = r

    if not all_combos:
        print("\n  No TTFT results to compare.")
        return

    print(f"\n  TTFT Results (ms):")
    print(f"  {'Prompt':>8} {'Conc':>5}", end="")
    for n in names:
        print(f"  {n + ' mean':>20} {n + ' p90':>15}", end="")
    print(f"  {'Delta%':>10}")
    print(f"  {'─' * (13 + 35 * len(names) + 10)}")

    for (pl, conc) in sorted(all_combos.keys()):
        combo = all_combos[(pl, conc)]
        print(f"  {pl:>8} {conc:>5}", end="")

        means = []
        for n in names:
            r = combo.get(n, {})
            ttft = r.get("ttft_ms", {})
            mean = ttft.get("mean")
            p90 = ttft.get("p90")
            means.append(mean)
            print(f"  {fmt_ms(mean):>20} {fmt_ms(p90):>15}", end="")

        valid = [m for m in means if m is not None and m > 0]
        if len(valid) >= 2:
            delta_pct = ((max(valid) - min(valid)) / min(valid)) * 100
            flag = " !!!" if delta_pct > 50 else " *" if delta_pct > 20 else ""
            print(f"  {delta_pct:>8.1f}%{flag}", end="")
        print()

    # ── Diagnosis ────────────────────────────────────────────────────────
    print(f"\n  Diagnosis Hints:")
    print(f"  {'─' * 80}")

    governor_vals = [str(s.get("cpu_governor", "")) for s in summaries]
    if len(set(governor_vals)) > 1:
        print(f"  • CPU governor differs: {dict(zip(names, governor_vals))}")
        print(f"    → Non-'performance' governor adds variable latency to the prefill CPU path")
        print(f"      Fix: sudo cpupower frequency-set -g performance")

    turbo_vals = [s.get("turbo_boost", "") for s in summaries]
    if len(set(turbo_vals)) > 1:
        print(f"  • Turbo boost differs: {dict(zip(names, turbo_vals))}")
        print(f"    → Turbo off limits single-thread freq, slowing tokenization + scheduling")

    numa_vals = [s.get("numa_nodes", 0) for s in summaries]
    if len(set(numa_vals)) > 1:
        print(f"  • NUMA topology differs: {dict(zip(names, numa_vals))}")
        print(f"    → Cross-NUMA memory access adds latency during model init and KV-cache setup")

    persist_vals = [s.get("gpu_persistence", "") for s in summaries]
    if len(set(persist_vals)) > 1:
        print(f"  • GPU persistence mode differs: {dict(zip(names, persist_vals))}")
        print(f"    → Without persistence, first CUDA call pays driver init (~100-500ms)")

    kernel_vals = [s.get("kernel", "") for s in summaries]
    if len(set(kernel_vals)) > 1:
        print(f"  • Kernel version differs: {dict(zip(names, kernel_vals))}")
        print(f"    → Different schedulers, NUMA balancing defaults, cgroup behavior")

    # Check for consistently high TTFT on one machine
    for (pl, conc) in sorted(all_combos.keys()):
        combo = all_combos[(pl, conc)]
        machine_means = {}
        for n in names:
            r = combo.get(n, {})
            mean = r.get("ttft_ms", {}).get("mean")
            if mean is not None:
                machine_means[n] = mean
        if len(machine_means) >= 2:
            slowest = max(machine_means, key=machine_means.get)
            fastest = min(machine_means, key=machine_means.get)
            ratio = machine_means[slowest] / machine_means[fastest] if machine_means[fastest] > 0 else 0
            if ratio > 1.5:
                print(f"\n  • At prompt_len={pl}, concurrency={conc}: "
                      f"{slowest} is {ratio:.1f}x slower than {fastest}")
                if conc == 1:
                    print(f"    → Single-request TTFT gap suggests CPU/init overhead, not GPU contention")
                else:
                    print(f"    → Under concurrency, this could also reflect scheduling or memory bandwidth")

    print(f"\n{'═' * 100}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TTFT results across machines")
    parser.add_argument("paths", nargs="+", help="Paths to summary.json files or directories")
    parser.add_argument("-o", "--output", help="Write comparison as JSON to file")
    args = parser.parse_args()

    summaries = [load_summary(p) for p in args.paths]
    compare(summaries)

    if args.output:
        Path(args.output).write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"Raw data written to {args.output}")


if __name__ == "__main__":
    main()
