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
    --size  Mesh NxN (default 200 -> 80k triangles)
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
        "--size", type=int, default=200, help="Mesh NxN (default 200 -> 80k triangles)"
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
    elev = topography(x, y)
    return np.where(x < 2000.0, elev + 0.5, elev)


# ─── SWW reader (for merged MPI output) ────────────────────


def _read_sww_final_volume(sww_path):
    """Read a SWW (NetCDF) file and return the total volume at the final timestep."""
    from netCDF4 import Dataset

    ds = Dataset(sww_path, "r")
    stage = ds.variables["stage"]
    elev = ds.variables["elevation"]
    elev_vals = elev[:] if elev.ndim == 1 else elev[-1, :]
    depths = stage[-1, :] - elev_vals

    if "areas" in ds.variables:
        areas = ds.variables["areas"][:]
    else:
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        vols = ds.variables["volumes"][:]
        x0, x1, x2 = x[vols[:, 0]], x[vols[:, 1]], x[vols[:, 2]]
        y0, y1, y2 = y[vols[:, 0]], y[vols[:, 1]], y[vols[:, 2]]
        areas = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    vol = float(np.sum(depths * areas))
    t = float(ds.variables["time"][-1])
    ntri = stage.shape[1]
    ds.close()
    return {"final_volume": vol, "final_time": t, "triangles": ntri}


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
        domain.set_datadir("../output")
        domain.set_store(True)
        domain.set_quantity("elevation", topography)
        domain.set_quantity("friction", 0.03)
        domain.set_quantity("stage", dam_stage)

        Br = anuga.Reflective_boundary(domain)
        # Apply boundary condition directly to the domain (only valid for 1 core before distribution)
        # If using MPI, this will be handled after distribution
        if not is_mpi:
            domain.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})
    else:
        domain = None

    if is_mpi:
        domain = distribute(domain)
        barrier()
        # Set boundaries on the DISTRIBUTED domain (all ranks)
        Br = anuga.Reflective_boundary(domain)
        domain.set_boundary({"left": Br, "right": Br, "top": Br, "bottom": Br})

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
    snapshots = []

    for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        stage_C = domain.get_quantity("stage").centroid_values
        elev_C = domain.get_quantity("elevation").centroid_values
        xmom_C = domain.get_quantity("xmomentum").centroid_values
        ymom_C = domain.get_quantity("ymomentum").centroid_values
        depths = stage_C - elev_C

        snap = {
            "time": float(domain.get_time()),
            "max_depth": float(np.max(depths)),
            "min_depth": float(np.min(depths)),
            "mean_depth": float(np.mean(depths)),
            "total_volume": float(np.sum(depths * domain.areas)),
            "wet_cells": int(np.sum(depths > 0.001)),
            "max_xmom": float(np.max(np.abs(xmom_C))),
            "max_ymom": float(np.max(np.abs(ymom_C))),
            "max_speed": float(
                np.max(
                    np.sqrt(xmom_C**2 + ymom_C**2)
                    / np.maximum(depths, 1e-6)
                    * (depths > 0.001)
                )
            ),
        }
        snapshots.append(snap)

        if rank == 0:
            elapsed = time.time() - t_start
            print(
                f"  t={snap['time']:7.1f}s | wall={elapsed:6.1f}s | "
                f"max_depth={snap['max_depth']:5.2f}m | "
                f"wet={snap['wet_cells']:,} | "
                f"vol={snap['total_volume']:.2e}"
            )

    wall_time = time.time() - t_start

    # ── MPI: merge partitioned SWW files ──
    if is_mpi:
        barrier()
        domain.sww_merge()
        if rank == 0:
            print("  Merged partitioned SWW files.")

    if rank == 0:
        result = {
            "mode": mode,
            "wall_time": wall_time,
            "triangles": ntri,
            "nprocs": nprocs,
            "snapshots": snapshots,
        }
        with open(f"/tmp/bench_{mode}.json", "w") as f:
            json.dump(result, f)
        print(f"  -> {wall_time:.2f}s total")

    if is_mpi:
        finalize()


# ─── Results comparison ────────────────────────────────────


def compare_results(results, output_dir="../output"):
    """Read SWW files and compare final total volume across modes."""

    modes = list(results.keys())
    if len(modes) < 2:
        print("  (need at least 2 successful runs to compare)")
        return

    print()
    print("=" * 72)
    print("  RESULTS COMPARISON  (final total volume from SWW files)")
    print("─" * 72)

    sww_data = {}
    for mode in modes:
        sww_path = os.path.join(output_dir, f"dam_{mode}.sww")
        try:
            info = _read_sww_final_volume(sww_path)
            sww_data[mode] = info
            print(
                f"  {mode.upper():>4s}: vol = {info['final_volume']:>16.2f}  "
                f"({info['triangles']:,} tri, t={info['final_time']:.1f}s)"
            )
        except Exception as exc:
            print(f"  {mode.upper():>4s}: ⚠ could not read {sww_path}: {exc}")

    if len(sww_data) < 2:
        print("\n  (need at least 2 readable SWW files to compare)")
        print("=" * 72)
        return

    print()
    ref_key = list(sww_data.keys())[0]
    ref_vol = sww_data[ref_key]["final_volume"]
    all_pass = True

    for other_key in list(sww_data.keys())[1:]:
        other_vol = sww_data[other_key]["final_volume"]
        denom = max(abs(ref_vol), abs(other_vol), 1e-12)
        rel_diff = abs(ref_vol - other_vol) / denom
        tol = 0.01  # 1%
        ok = rel_diff <= tol
        mark = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(
            f"  {mark} {ref_key.upper()} vs {other_key.upper()}: "
            f"Δ = {rel_diff * 100:.4f}%  (tol {tol * 100:.0f}%)"
        )

    print()
    if all_pass:
        print("  ✓ ALL COMPARISONS PASSED")
    else:
        print("  ✗ SOME COMPARISONS EXCEEDED TOLERANCE")
    print("=" * 72)


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
                "mpirun",
                "--allow-run-as-root",
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

    # ── Compare results across modes ──
    compare_results(results)


# ─── Main ───────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args.worker, args.size, args.finaltime, args.yieldstep)
    else:
        run_benchmark(args)

