#!/usr/bin/env python3
"""Low-rate process, system-memory, and NVIDIA GPU sampler for FPS diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time


GROUPS = {
    "gazebo": ("gz sim",),
    "bridge": ("ros_gz_bridge", "parameter_bridge"),
    "smash_fcs": ("drone_sim/smash_fcs",),
    "scope_viewer": ("drone_sim/smash_scope_viewer",),
    "turret_command": ("drone_sim/turret_command",),
    "robot_state_publisher": ("robot_state_publisher",),
}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] if lo == hi else xs[lo] * (hi - p) + xs[hi] * (p - lo)


def stats(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 3) if values else None,
        "p95": round(percentile(values, 0.95), 3) if values else None,
        "max": round(max(values), 3) if values else None,
    }


def memory_used_mb() -> float | None:
    values = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                key, value = line.split(":", 1)
                values[key] = float(value.strip().split()[0])
        return (values["MemTotal"] - values["MemAvailable"]) / 1024.0
    except (OSError, KeyError, ValueError):
        return None


def process_cpu() -> dict[str, float]:
    out = subprocess.run(
        ["ps", "-eo", "pcpu=,args="], check=False, capture_output=True, text=True
    ).stdout
    grouped = {name: 0.0 for name in GROUPS}
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        try:
            cpu = float(parts[0])
        except ValueError:
            continue
        args = parts[1]
        for name, needles in GROUPS.items():
            if all(needle in args for needle in needles):
                grouped[name] += cpu
    return grouped


def gpu() -> tuple[float | None, float | None]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()[0]
        util, mem = (float(v.strip()) for v in out.split(",")[:2])
        return util, mem
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    samples = []
    started = time.monotonic()
    while time.monotonic() - started < args.duration:
        proc = process_cpu()
        gpu_util, gpu_mem = gpu()
        samples.append({
            "elapsed_s": round(time.monotonic() - started, 3),
            "process_cpu_pct": proc,
            "memory_used_mb": memory_used_mb(),
            "gpu_util_pct": gpu_util,
            "gpu_memory_mb": gpu_mem,
        })
        time.sleep(max(0.1, args.interval))
    summary = {
        "samples": len(samples),
        "process_cpu_pct": {
            name: stats([s["process_cpu_pct"][name] for s in samples]) for name in GROUPS
        },
        "memory_used_mb": stats([s["memory_used_mb"] for s in samples if s["memory_used_mb"] is not None]),
        "gpu_util_pct": stats([s["gpu_util_pct"] for s in samples if s["gpu_util_pct"] is not None]),
        "gpu_memory_mb": stats([s["gpu_memory_mb"] for s in samples if s["gpu_memory_mb"] is not None]),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples_raw": samples}, f, indent=2)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
