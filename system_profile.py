#!/usr/bin/env python3
"""
Collect system-level configuration that affects TTFT / initialization latency.

Captures CPU frequency governors, NUMA topology, memory, kernel tunables,
GPU info, PCIe topology, and IRQ affinity — the kind of differences that
explain why the *same* model on the *same* GPU SKU shows different TTFT
across hosting providers while throughput stays identical.

Usage:
    uv run python system_profile.py                 # print JSON to stdout
    uv run python system_profile.py -o profile.json # save to file
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run(cmd: str, timeout: int = 10) -> Optional[str]:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def _glob_read(pattern: str) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for p in sorted(Path("/").glob(pattern.lstrip("/"))):
        try:
            results[str(p)] = p.read_text().strip()
        except Exception:
            pass
    return results


# ── CPU ──────────────────────────────────────────────────────────────────────

def collect_cpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    info["model_name"] = _read("/proc/cpuinfo") or ""
    match = re.search(r"model name\s*:\s*(.+)", info["model_name"])
    info["model_name"] = match.group(1) if match else "unknown"

    info["physical_cores"] = os.cpu_count()
    info["online_cpus"] = _read("/sys/devices/system/cpu/online")

    governors = _glob_read("sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    unique = set(governors.values())
    info["scaling_governors"] = sorted(unique) if unique else ["unknown (no cpufreq)"]
    info["scaling_governors_unanimous"] = len(unique) <= 1

    freqs: Dict[str, Any] = {}
    for tag in ("scaling_cur_freq", "scaling_min_freq", "scaling_max_freq", "cpuinfo_max_freq"):
        raw = _glob_read(f"sys/devices/system/cpu/cpu*/cpufreq/{tag}")
        vals = [int(v) for v in raw.values() if v.isdigit()]
        if vals:
            freqs[tag + "_khz"] = {"min": min(vals), "max": max(vals), "median": sorted(vals)[len(vals) // 2]}
    info["frequencies"] = freqs

    driver = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
    info["scaling_driver"] = driver or "unknown"

    boost = _read("/sys/devices/system/cpu/cpufreq/boost")
    if boost is None:
        boost = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
        if boost is not None:
            boost = "off" if boost == "1" else "on"
    else:
        boost = "on" if boost == "1" else "off"
    info["turbo_boost"] = boost or "unknown"

    return info


# ── NUMA ─────────────────────────────────────────────────────────────────────

def collect_numa() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    numa_out = _run("numactl --hardware 2>/dev/null || lscpu | grep -i numa")
    info["raw"] = numa_out or "unavailable"

    node_dirs = sorted(Path("/sys/devices/system/node").glob("node*")) if Path("/sys/devices/system/node").exists() else []
    info["node_count"] = len(node_dirs)
    nodes = {}
    for nd in node_dirs:
        name = nd.name
        cpulist = _read(str(nd / "cpulist"))
        meminfo = _read(str(nd / "meminfo"))
        total_kb = None
        if meminfo:
            m = re.search(r"MemTotal:\s+(\d+)", meminfo)
            total_kb = int(m.group(1)) if m else None
        nodes[name] = {"cpulist": cpulist, "memtotal_kb": total_kb}
    info["nodes"] = nodes

    policy = _read("/proc/sys/vm/zone_reclaim_mode")
    info["zone_reclaim_mode"] = policy

    return info


# ── Memory ───────────────────────────────────────────────────────────────────

def collect_memory() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    meminfo = _read("/proc/meminfo") or ""
    for key in ("MemTotal", "MemFree", "MemAvailable", "SwapTotal", "HugePages_Total", "Hugepagesize"):
        m = re.search(rf"{key}:\s+(\d+)", meminfo)
        if m:
            info[key.lower() + "_kb"] = int(m.group(1))

    thp = _read("/sys/kernel/mm/transparent_hugepage/enabled")
    info["transparent_hugepages"] = thp or "unknown"

    return info


# ── Kernel ───────────────────────────────────────────────────────────────────

def collect_kernel() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["release"] = platform.release()
    info["version"] = platform.version()

    tunables = {
        "vm.swappiness": "/proc/sys/vm/swappiness",
        "vm.dirty_ratio": "/proc/sys/vm/dirty_ratio",
        "vm.dirty_background_ratio": "/proc/sys/vm/dirty_background_ratio",
        "kernel.sched_autogroup_enabled": "/proc/sys/kernel/sched_autogroup_enabled",
        "kernel.numa_balancing": "/proc/sys/kernel/numa_balancing",
        "kernel.perf_event_paranoid": "/proc/sys/kernel/perf_event_paranoid",
        "net.core.somaxconn": "/proc/sys/net/core/somaxconn",
    }
    info["sysctl"] = {}
    for name, path in tunables.items():
        val = _read(path)
        if val is not None:
            info["sysctl"][name] = val

    info["cmdline"] = _read("/proc/cmdline")

    return info


# ── GPU ──────────────────────────────────────────────────────────────────────

def collect_gpu() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    smi = _run(
        "nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,clocks.current.sm,"
        "clocks.max.sm,persistence_mode,power.limit,temperature.gpu,driver_version,"
        "pstate,mig.mode.current "
        "--format=csv,noheader,nounits",
        timeout=15,
    )
    if smi:
        gpus = []
        for line in smi.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 12:
                gpus.append({
                    "index": parts[0],
                    "name": parts[1],
                    "pci_bus_id": parts[2],
                    "memory_mib": parts[3],
                    "clock_sm_mhz": parts[4],
                    "clock_max_sm_mhz": parts[5],
                    "persistence_mode": parts[6],
                    "power_limit_w": parts[7],
                    "temp_c": parts[8],
                    "driver_version": parts[9],
                    "pstate": parts[10],
                    "mig_mode": parts[11],
                })
        info["gpus"] = gpus
        info["count"] = len(gpus)
    else:
        info["gpus"] = []
        info["count"] = 0
        info["note"] = "nvidia-smi not available"

    topo = _run("nvidia-smi topo -m", timeout=15)
    info["topology_matrix"] = topo

    nvlink = _run("nvidia-smi nvlink --status", timeout=15)
    info["nvlink_status"] = nvlink[:2000] if nvlink else None

    cuda_version = _run("nvcc --version 2>/dev/null | grep 'release'")
    info["cuda_version"] = cuda_version

    return info


# ── PCIe ─────────────────────────────────────────────────────────────────────

def collect_pcie() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    lspci = _run("lspci -vvv 2>/dev/null | grep -A5 -i 'nvidia\\|3d controller'")
    info["nvidia_pcie_details"] = lspci[:3000] if lspci else None

    iommu = _read("/sys/kernel/iommu_groups") is not None
    info["iommu_present"] = iommu

    return info


# ── Process / cgroup ─────────────────────────────────────────────────────────

def collect_process_env() -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    for var in ("CUDA_VISIBLE_DEVICES", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
                "NCCL_SOCKET_IFNAME", "OMP_NUM_THREADS", "LD_PRELOAD",
                "MALLOC_ARENA_MAX", "CUDA_DEVICE_MAX_CONNECTIONS"):
        val = os.environ.get(var)
        if val is not None:
            info[var] = val

    cgroup_cpu = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    cgroup_mem = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cgroup_cpu:
        info["cgroup_cpu_quota_us"] = cgroup_cpu
    if cgroup_mem:
        info["cgroup_mem_limit_bytes"] = cgroup_mem

    return info


# ── Disk ─────────────────────────────────────────────────────────────────────

def collect_disk() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    df_out = _run("df -h / /tmp 2>/dev/null")
    info["df"] = df_out

    fio_test = _run(
        "fio --name=seqread --rw=read --bs=1M --size=256M --numjobs=1 "
        "--time_based --runtime=3 --group_reporting --filename=/tmp/.ttft_fio_probe "
        "--output-format=json 2>/dev/null"
    )
    if fio_test:
        try:
            fio_json = json.loads(fio_test)
            job = fio_json["jobs"][0]["read"]
            info["seq_read_bw_mibps"] = round(job["bw"] / 1024, 1)
            info["seq_read_iops"] = job["iops"]
        except Exception:
            pass
    _run("rm -f /tmp/.ttft_fio_probe")

    return info


# ── Assemble ─────────────────────────────────────────────────────────────────

def collect_all() -> Dict[str, Any]:
    hostname = platform.node()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": hostname,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": collect_cpu(),
        "numa": collect_numa(),
        "memory": collect_memory(),
        "kernel": collect_kernel(),
        "gpu": collect_gpu(),
        "pcie": collect_pcie(),
        "process_env": collect_process_env(),
        "disk": collect_disk(),
    }


def print_summary(profile: Dict[str, Any]) -> None:
    """Human-readable summary of the most TTFT-relevant settings."""
    print(f"\n{'═' * 70}")
    print(f"  System Profile Summary — {profile['hostname']}")
    print(f"{'═' * 70}")

    cpu = profile.get("cpu", {})
    print(f"\n  CPU: {cpu.get('model_name', '?')}")
    print(f"  Cores: {cpu.get('physical_cores', '?')}  Online: {cpu.get('online_cpus', '?')}")
    govs = cpu.get("scaling_governors", [])
    unanimous = cpu.get("scaling_governors_unanimous", False)
    gov_str = ", ".join(govs) + (" (all cores)" if unanimous else " (MIXED — check per-core)")
    print(f"  Scaling governor: {gov_str}")
    print(f"  Scaling driver: {cpu.get('scaling_driver', '?')}")
    print(f"  Turbo boost: {cpu.get('turbo_boost', '?')}")

    freqs = cpu.get("frequencies", {})
    if "scaling_cur_freq_khz" in freqs:
        cur = freqs["scaling_cur_freq_khz"]
        print(f"  Current freq: {cur['min']/1000:.0f}–{cur['max']/1000:.0f} MHz (median {cur['median']/1000:.0f})")
    if "cpuinfo_max_freq_khz" in freqs:
        mx = freqs["cpuinfo_max_freq_khz"]
        print(f"  Max capable freq: {mx['max']/1000:.0f} MHz")

    numa = profile.get("numa", {})
    print(f"\n  NUMA nodes: {numa.get('node_count', '?')}")
    print(f"  Zone reclaim: {numa.get('zone_reclaim_mode', '?')}")

    mem = profile.get("memory", {})
    total_gb = mem.get("memtotal_kb", 0) / 1048576
    print(f"\n  RAM: {total_gb:.0f} GB")
    print(f"  THP: {mem.get('transparent_hugepages', '?')}")
    if mem.get("hugepages_total_kb"):
        print(f"  HugePages total: {mem['hugepages_total_kb']} kB")

    kern = profile.get("kernel", {})
    print(f"\n  Kernel: {kern.get('release', '?')}")
    sysctl = kern.get("sysctl", {})
    if sysctl.get("kernel.numa_balancing"):
        print(f"  NUMA balancing: {sysctl['kernel.numa_balancing']}")

    gpu = profile.get("gpu", {})
    if gpu.get("gpus"):
        g0 = gpu["gpus"][0]
        print(f"\n  GPU: {gpu['count']}x {g0.get('name', '?')}")
        print(f"  Driver: {g0.get('driver_version', '?')}  CUDA: {gpu.get('cuda_version', '?')}")
        print(f"  Persistence mode: {g0.get('persistence_mode', '?')}")
        print(f"  Power limit: {g0.get('power_limit_w', '?')} W")
        print(f"  P-state: {g0.get('pstate', '?')}")

    disk = profile.get("disk", {})
    if disk.get("seq_read_bw_mibps"):
        print(f"\n  Disk seq read: {disk['seq_read_bw_mibps']} MiB/s")

    warnings: List[str] = []
    if govs and govs != ["performance"]:
        warnings.append("CPU governor is NOT 'performance' — expect higher/variable TTFT")
    if not cpu.get("scaling_governors_unanimous"):
        warnings.append("Mixed CPU governors across cores")
    if cpu.get("turbo_boost") == "off":
        warnings.append("Turbo boost is OFF")
    if mem.get("transparent_hugepages", "").startswith("[never]"):
        warnings.append("Transparent HugePages disabled")
    if sysctl.get("kernel.numa_balancing") == "1" and numa.get("node_count", 0) > 1:
        warnings.append("NUMA balancing is ON with multiple nodes — can cause jitter")
    if gpu.get("gpus"):
        for g in gpu["gpus"]:
            if g.get("persistence_mode", "").lower() != "enabled":
                warnings.append(f"GPU {g['index']} persistence mode is OFF — adds latency to first CUDA call")
                break

    if warnings:
        print(f"\n  ⚠  TTFT-relevant warnings:")
        for w in warnings:
            print(f"     • {w}")

    print(f"\n{'═' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect system profile for TTFT analysis")
    parser.add_argument("-o", "--output", help="Write JSON to file instead of stdout")
    parser.add_argument("--quiet", action="store_true", help="Suppress human-readable summary")
    args = parser.parse_args()

    profile = collect_all()

    if not args.quiet:
        print_summary(profile)

    blob = json.dumps(profile, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(blob + "\n")
        print(f"Profile written to {args.output}")
    else:
        if args.quiet:
            print(blob)


if __name__ == "__main__":
    main()
