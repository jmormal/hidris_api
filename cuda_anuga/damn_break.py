#!/usr/bin/env python3
"""
ANUGA Dam-Break Benchmark: CPU vs MPI vs GPU
=============================================

Usage:
    python dam_break_benchmark.py                          # all three
    python dam_break_benchmark.py --run gpu                # GPU only
    python dam_break_benchmark.py --run mpi,gpu            # skip CPU
    python dam_break_benchmark.py --run cpu,gpu --size 100 # skip MPI
    python dam_break_benchmark.py --run mpi,gpu --size 400 --np 4

Options:
    --run   Comma-separated list of: cpu, mpi, gpu (default: cpu,mpi,gpu)
    --size  Mesh NxN (default 200 → 80k triangles)
    --np    MPI ranks (default 8)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="ANUGA dam-break benchmark")
    p.add_argument(
        "--size", type=int, default=200, help="Mesh NxN (default 200 → 80k triangles)"
    )
    p.add_argument("--finaltime", type=float, default=60.0)
    p.add_argument("--yieldstep", type=float, default=10.0)
    p.add_argument("--np", type=int, default=8, help="Number of MPI ranks (default 8)")
    p.add_argument(
        "--run",
        type=str,
        default="cpu,mpi,gpu",
        help="Comma-separated configs to run: cpu,mpi,gpu (default: cpu,mpi,gpu)",
    )
    p.add_argument("--worker", choices=["cpu", "mpi", "gpu"], help=argparse.SUPPRESS)
    return p.parse_args()


# ─── Terrain ────────────────────────────────────────────────


def topography(x, y):
    z = -x / 20.0
    z += 0.05 * np.sin(y * np.pi / 2500.0)
    z += 0.02 * np.sin(x * np.pi / 500.0) * np.sin(y * np.pi / 500.0)
    return z


def dam_stage(x, y):
    return np.where(x < 2000.0, 0.5, -99.0)


# ─── Worker ─────────────────────────────────────────────────


def run_worker(mode, N, finaltime, yieldstep):
    import anuga

    is_mpi = mode == "mpi"

    if is_mpi:
        from anuga import distribute, myid, numprocs, finalize, barrier

        rank, nprocs = myid, numprocs
    else:
        rank, nprocs = 0, 1

    if rank == 0:
        domain = anuga.rectangular_cross_domain(N, N, len1=10000.0, len2=5000.0)
        domain.set_flow_algorithm("DE0")
        domain.set_name(f"dam_{mode}")
        domain.set_datadir("/tmp")
        domain.set_store(False)
        domain.set_quantity("elevation", topography)
        domain.set_quantity("friction", 0.03)
        domain.set_quantity("stage", dam_stage)
        Br = anuga.Reflective_boundary(domain)
        domain.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    else:
        domain = None

    if is_mpi:
        domain = distribute(domain)

    if mode == "gpu":
        domain.set_multiprocessor_mode(2)

    ntri = domain.number_of_elements

    if rank == 0:
        label = {
            "cpu": "CPU (1 core)",
            "mpi": f"MPI ({nprocs} cores)",
            "gpu": "GPU (mode=2)",
        }[mode]
        print(f"\n{'─' * 60}")
        print(f"  {label}  |  {N}x{N}  |  {ntri:,} tri{'/rank' if is_mpi else ''}")
        print(f"{'─' * 60}")

    if is_mpi:
        barrier()

    t_start = time.time()
    for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        if rank == 0:
            elapsed = time.time() - t_start
            depths = (
                domain.get_quantity("stage").centroid_values
                - domain.get_quantity("elevation").centroid_values
            )
            print(
                f"  t={domain.get_time():7.1f}s | wall={elapsed:6.1f}s | "
                f"max_depth={np.max(depths):5.2f}m | "
                f"wet={np.sum(depths > 0.001):,}"
            )

    wall_time = time.time() - t_start

    if rank == 0:
        result = {
            "mode": mode,
            "wall_time": wall_time,
            "triangles": ntri,
            "nprocs": nprocs,
        }
        with open(f"/tmp/bench_{mode}.json", "w") as f:
            json.dump(result, f)
        print(f"  → {wall_time:.2f}s total")

    if is_mpi:
        finalize()


# ─── Orchestrator ───────────────────────────────────────────


def run_benchmark(args):
    configs = [c.strip() for c in args.run.split(",") if c.strip()]
    valid = {"cpu", "mpi", "gpu"}
    for c in configs:
        if c not in valid:
            print(f"Unknown config '{c}'. Choose from: cpu, mpi, gpu")
            sys.exit(1)

    script = os.path.abspath(__file__)
    base_cmd = [
        sys.executable,
        script,
        "--size",
        str(args.size),
        "--finaltime",
        str(args.finaltime),
        "--yieldstep",
        str(args.yieldstep),
        "--np",
        str(args.np),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    N = args.size
    total_tri = 2 * N * N

    print("=" * 60)
    print(f"  ANUGA Dam-Break Benchmark")
    print(f"  Mesh: {N}x{N} = {total_tri:,} triangles")
    print(f"  Simulation: {args.finaltime:.0f}s, yieldstep {args.yieldstep:.0f}s")
    print(f"  Running: {', '.join(configs)}")
    print("=" * 60)

    results = {}

    if "cpu" in configs:
        print("\n▶ Running: CPU (1 core) ...")
        r = subprocess.run(base_cmd + ["--worker", "cpu"], env=env)
        if r.returncode == 0 and os.path.exists("/tmp/bench_cpu.json"):
            with open("/tmp/bench_cpu.json") as f:
                results["cpu"] = json.load(f)

    if "mpi" in configs:
        print(f"\n▶ Running: MPI ({args.np} cores) ...")
        mpi_cmd = (
            [
                "mpiexec",
                "--allow-run-as-root",
                "--oversubscribe",
                "-np",
                str(args.np),
            ]
            + base_cmd
            + ["--worker", "mpi"]
        )
        r = subprocess.run(mpi_cmd, env=env)
        if r.returncode == 0 and os.path.exists("/tmp/bench_mpi.json"):
            with open("/tmp/bench_mpi.json") as f:
                results["mpi"] = json.load(f)

    if "gpu" in configs:
        print("\n▶ Running: GPU ...")
        r = subprocess.run(base_cmd + ["--worker", "gpu"], env=env)
        if r.returncode == 0 and os.path.exists("/tmp/bench_gpu.json"):
            with open("/tmp/bench_gpu.json") as f:
                results["gpu"] = json.load(f)

    # ── Summary table ──
    if not results:
        print("\nNo successful runs.")
        return

    # Use the fastest available as baseline, prefer cpu if present
    baseline_key = "cpu" if "cpu" in results else list(results.keys())[0]
    baseline_time = results[baseline_key]["wall_time"]

    print()
    print("=" * 60)
    print("  RESULTS")
    print("─" * 60)

    rows = [
        ("CPU (1 core)", "cpu"),
        (f"MPI ({args.np} cores)", "mpi"),
        ("GPU (mode=2)", "gpu"),
    ]

    fmt = "  {:<20s} {:>8s}   {:>8s}   {:>12s}"
    print(fmt.format("Config", "Time", "Speedup", "Triangles"))
    print(fmt.format("─" * 20, "─" * 8, "─" * 8, "─" * 12))

    for label, key in rows:
        if key not in configs:
            continue
        r = results.get(key)
        if r:
            t = r["wall_time"]
            tri = r.get("triangles", 0)
            nprocs = r.get("nprocs", 1)
            tri_label = f"{tri:,}"
            if key == "mpi":
                tri_label += f" ({tri * nprocs:,} tot)"
            sp = f"{baseline_time / t:.1f}x" if baseline_time else "—"
            ref = " (base)" if key == baseline_key else ""
            print(fmt.format(label, f"{t:.2f}s", sp + ref, tri_label))
        else:
            print(fmt.format(label, "FAILED", "—", "—"))

    print("─" * 60)

    if total_tri < 200000:
        print(f"\n  Tip: GPU excels at >200k triangles.")
        print(
            f"       Try: python {os.path.basename(script)} --run {args.run} --size 400"
        )

    print("=" * 60)


# ─── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args.worker, args.size, args.finaltime, args.yieldstep)
    else:
        run_benchmark(args)
