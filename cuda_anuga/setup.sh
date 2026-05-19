#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  ANUGA GPU (OpenMP target offloading) — complete Docker setup
#
#  Tested on: NVIDIA RTX 5060 Ti, Driver 595.71, CUDA 13.2
#  Based on:  https://anuga.readthedocs.io/en/develop/parallel/
#             use_gpu_offloading.html
#
#  Host requirements:
#    - Docker
#    - NVIDIA Container Toolkit (nvidia-ctk)
#    - NVIDIA GPU (tested with cc89 arch flag, works on newer
#      GPUs via PTX JIT)
#
#  Usage:
#    chmod +x setup_anuga_gpu.sh
#    ./setup_anuga_gpu.sh              # build + smoke test
#    ./setup_anuga_gpu.sh build        # build only
#    ./setup_anuga_gpu.sh test         # smoke test
#    ./setup_anuga_gpu.sh dam-break    # run dam-break example
#    ./setup_anuga_gpu.sh shell        # interactive shell
#
#  GPU architecture:
#    Default is cc89 (Ada Lovelace). Override with GPU_ARCH:
#      GPU_ARCH=cc90 ./setup_anuga_gpu.sh build
#    Use "ccnative" to auto-detect (requires GPU visible at
#    build time).
# ============================================================

IMAGE_NAME="anuga-gpu"
CONTAINER_NAME="anuga-gpu-run"
GPU_ARCH="${GPU_ARCH:-cc89}"

setup_build() {
mkdir -p anuga-gpu-docker

cat > anuga-gpu-docker/Dockerfile <<DOCKERFILE
# -----------------------------------------------------------
#  NVIDIA HPC SDK 24.7 + ANUGA develop branch
#  OpenMP target offloading to NVIDIA GPUs
# -----------------------------------------------------------
FROM nvcr.io/nvidia/nvhpc:24.7-devel-cuda_multi-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ── System packages (incl. MPI — required for GPU ext) ─────
RUN apt-get update && apt-get install -y --no-install-recommends \\
        python3 python3-dev python3-pip python3-venv \\
        git wget curl build-essential gfortran \\
        libnetcdf-dev libhdf5-dev gdal-bin libgdal-dev \\
        swig pkg-config ninja-build \\
        openmpi-bin libopenmpi-dev \\
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \\
    update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3    1

# ── Python packages ────────────────────────────────────────
#  numpy>=2.0 is required (provides numpy-config for meson)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip install --no-cache-dir \\
        "numpy>=2.0" scipy matplotlib netCDF4 \\
        cython meshpy pytest pybind11 \\
        meson meson-python ninja \\
        mpi4py

# ── Clone ANUGA develop branch ─────────────────────────────
RUN git clone --branch develop --depth 1 \\
    https://github.com/anuga-community/anuga_core.git /opt/anuga_core

WORKDIR /opt/anuga_core

# ── Build with NVIDIA compiler + GPU offloading ────────────
#  - CC=nvc / CXX=nvc++  : NVIDIA HPC compiler (OpenMP offload)
#  - gpu_offload=true     : enable GPU extension build
#  - gpu_arch=${GPU_ARCH} : target GPU architecture
#  - --no-build-isolation : use globally installed numpy/meson
RUN CC=nvc CXX=nvc++ pip install --no-cache-dir --no-build-isolation -e . \\
        -Csetup-args=-Dgpu_offload=true \\
        -Csetup-args=-Dgpu_arch=${GPU_ARCH}

# ── Verify the GPU extension was built ─────────────────────
RUN python -c "from anuga.shallow_water import sw_domain_gpu_ext; \\
    print('sw_domain_gpu_ext: OK')"

ENV OMP_NUM_THREADS=1
CMD ["/bin/bash"]
DOCKERFILE

# ── Smoke test script ──────────────────────────────────────
cat > anuga-gpu-docker/test_gpu.py <<'TESTSCRIPT'
#!/usr/bin/env python3
"""ANUGA GPU smoke test."""
import sys, time
import numpy as np

print("=" * 60)
print("  ANUGA GPU smoke test")
print("=" * 60)

# 1 — Import
print("\n[1/3] Import ANUGA")
import anuga
print(f"      version: {getattr(anuga, '__version__', '?')}")

# 2 — GPU extension
print("\n[2/3] GPU extension")
from anuga.shallow_water import sw_domain_gpu_ext
print(f"      gpu_available: {sw_domain_gpu_ext.gpu_available()}")

# 3 — GPU evolve
print("\n[3/3] GPU evolve (mode=2)")
d = anuga.rectangular_cross_domain(100, 100, len1=2000., len2=1000.)
d.set_flow_algorithm('DE0')
d.set_name('smoke'); d.set_datadir('/tmp'); d.set_store(False)
d.set_quantity('elevation', lambda x, y: -x / 10.)
d.set_quantity('friction', 0.01)
d.set_quantity('stage', -0.2)
Br = anuga.Reflective_boundary(d)
d.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
d.set_multiprocessor_mode(2)

t0 = time.time()
for t in d.evolve(yieldstep=10., finaltime=30.):
    d.print_timestepping_statistics()
print(f"      Wall time: {time.time() - t0:.2f}s")
print("      PASS")

print("\n" + "=" * 60)
print("  All checks passed!")
print("=" * 60)
TESTSCRIPT

# ── Dam-break example ──────────────────────────────────────
cat > anuga-gpu-docker/dam_break_gpu.py <<'EXAMPLE'
#!/usr/bin/env python3
"""
Dam-break flood — CPU vs GPU comparison
========================================

A wall of water released over a sloping valley with terrain.
Runs once on CPU (mode=1) then on GPU (mode=2) and compares.

Usage:
    python dam_break_gpu.py                # 200x200 mesh (~80k tri)
    python dam_break_gpu.py --size 400     # 320k tri (GPU shines)
    python dam_break_gpu.py --size 50      # quick test
    python dam_break_gpu.py --cpu-only     # skip GPU run
"""
import argparse, time
import numpy as np
import anuga

parser = argparse.ArgumentParser()
parser.add_argument("--size", type=int, default=200,
                    help="Mesh NxN (default 200 = ~80k triangles)")
parser.add_argument("--finaltime", type=float, default=60.)
parser.add_argument("--yieldstep", type=float, default=10.)
parser.add_argument("--cpu-only", action="store_true")
args = parser.parse_args()
N = args.size

def topography(x, y):
    z = -x / 20.0
    z += 0.05 * np.sin(y * np.pi / 2500.)
    z += 0.02 * np.sin(x * np.pi / 500.) * np.sin(y * np.pi / 500.)
    return z

def dam_stage(x, y):
    return np.where(x < 2000., 0.5, -99.)

def run(name, mode):
    label = "GPU (mode=2)" if mode == 2 else "CPU (mode=1)"
    print(f"\n{'─' * 55}")
    print(f"  {label}  |  {N}x{N} = {2*N*N:,} triangles")
    print(f"{'─' * 55}")

    d = anuga.rectangular_cross_domain(N, N, len1=10000., len2=5000.)
    d.set_flow_algorithm('DE0')
    d.set_name(name); d.set_datadir('/tmp'); d.set_store(False)
    d.set_quantity('elevation', topography)
    d.set_quantity('friction', 0.03)
    d.set_quantity('stage', dam_stage)
    Br = anuga.Reflective_boundary(d)
    d.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

    if mode == 2:
        d.set_multiprocessor_mode(2)

    t0 = time.time()
    for t in d.evolve(yieldstep=args.yieldstep, finaltime=args.finaltime):
        el = time.time() - t0
        depths = (d.get_quantity('stage').centroid_values -
                  d.get_quantity('elevation').centroid_values)
        print(f"  t={d.get_time():7.1f}s | wall={el:6.1f}s | "
              f"max_depth={np.max(depths):5.2f}m | "
              f"wet={np.sum(depths > 0.001):,}")
    return time.time() - t0, d.number_of_elements

print("=" * 55)
print(f"  ANUGA Dam-Break  |  {N}x{N} mesh  |  {2*N*N:,} triangles")
print("=" * 55)

cpu_t, ntri = run('dam_cpu', 1)

if not args.cpu_only:
    gpu_t, _ = run('dam_gpu', 2)

    print(f"\n{'=' * 55}")
    print(f"  RESULTS ({ntri:,} triangles, {args.finaltime:.0f}s simulated)")
    print(f"{'─' * 55}")
    print(f"  CPU (mode=1):  {cpu_t:6.2f}s")
    print(f"  GPU (mode=2):  {gpu_t:6.2f}s")
    print(f"  Speedup:       {cpu_t / gpu_t:6.2f}x")
    if ntri < 200000:
        print(f"\n  Tip: GPU excels at >200k triangles.")
        print(f"       Try: python dam_break_gpu.py --size 400")
    print(f"{'=' * 55}")
else:
    print(f"\n  CPU: {cpu_t:.2f}s  (GPU skipped)")
EXAMPLE

echo "  Build context ready."
}

do_build() {
    setup_build
    echo ""
    echo "Building '${IMAGE_NAME}' (GPU_ARCH=${GPU_ARCH}) ..."
    echo "(NVIDIA HPC SDK is ~15 GB — first pull takes a while)"
    echo ""
    docker build -t "${IMAGE_NAME}" anuga-gpu-docker/
    echo ""
    echo "Image '${IMAGE_NAME}' built successfully."
}

do_test() {
    echo ""
    echo "Running ANUGA GPU smoke test ..."
    echo ""
    # Ensure test script exists
    [ -f "anuga-gpu-docker/test_gpu.py" ] || setup_build
    docker run --rm --gpus all \
        --name "${CONTAINER_NAME}" \
        -v "$(pwd)/anuga-gpu-docker/test_gpu.py:/examples/test_gpu.py:ro" \
        -v "$(pwd)/anuga-gpu-docker/dam_break_gpu.py:/examples/dam_break_gpu.py:ro" \
        -e OMP_NUM_THREADS=1 \
        "${IMAGE_NAME}" \
        python /examples/test_gpu.py
}

do_dambreak() {
    echo ""
    echo "Running dam-break GPU example ..."
    echo ""
    [ -f "anuga-gpu-docker/dam_break_gpu.py" ] || setup_build
    docker run --rm --gpus all \
        --name "${CONTAINER_NAME}-dam" \
        -v "$(pwd)/anuga-gpu-docker/dam_break_gpu.py:/examples/dam_break_gpu.py:ro" \
        -e OMP_NUM_THREADS=1 \
        "${IMAGE_NAME}" \
        python /examples/dam_break_gpu.py "$@"
}

do_shell() {
    [ -f "anuga-gpu-docker/test_gpu.py" ] || setup_build
    echo "Starting container with GPU access ..."
    echo ""
    echo "  Examples:"
    echo "    python /examples/test_gpu.py                      # smoke test"
    echo "    python /examples/dam_break_gpu.py                  # CPU vs GPU"
    echo "    python /examples/dam_break_gpu.py --size 400       # bigger mesh"
    echo ""
    docker run --rm -it --gpus all \
        --name "${CONTAINER_NAME}-sh" \
        -v "$(pwd)/anuga-gpu-docker/test_gpu.py:/examples/test_gpu.py:ro" \
        -v "$(pwd)/anuga-gpu-docker/dam_break_gpu.py:/examples/dam_break_gpu.py:ro" \
        -e OMP_NUM_THREADS=1 \
        "${IMAGE_NAME}" /bin/bash
}

case "${1:-all}" in
    build)     do_build ;;
    test)      do_test ;;
    dam-break) shift; do_dambreak "$@" ;;
    shell)     do_shell ;;
    all)       do_build && do_test ;;
    *)
        echo "Usage: $0 {build|test|dam-break|shell|all}"
        echo ""
        echo "  build      Build the Docker image"
        echo "  test       Run GPU smoke test"
        echo "  dam-break  Run dam-break CPU vs GPU benchmark"
        echo "  shell      Interactive shell with GPU"
        echo "  all        Build + test (default)"
        echo ""
        echo "Environment:"
        echo "  GPU_ARCH=cc89  GPU compute capability (default: cc89)"
        echo "                 Options: cc60 cc70 cc75 cc80 cc86 cc89 cc90"
        exit 1
        ;;
esac
