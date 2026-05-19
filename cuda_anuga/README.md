# ANUGA CUDA Development Environment

## Prerequisites

- [Devbox](https://www.jetify.com/devbox) installed (`curl -fsSL https://get.jetify.com/devbox | bash`)
- An NVIDIA GPU with recent drivers (550+)
- The ANUGA source repo that contains the CUDA module (`sw_domain_cuda.py`)

## Quick start

```bash
# Enter the devbox shell (downloads Nix packages on first run)
devbox shell

# Install Python dependencies
devbox run setup

# Activate the venv (needed each time you enter the shell)
source .venv/bin/activate

# Run the test
devbox run test

# Profile with Nsight Systems (optional)
devbox run profile
```

## CUDA version

The `devbox.json` pulls the latest CUDA toolkit from nixpkgs. If you need
a specific version (e.g. CUDA 11.8), replace:

```json
"cudaPackages.cudatoolkit@latest"
```

with:

```json
"cudaPackages_11_8.cudatoolkit@latest"
```

And in `setup.sh`, change `cupy-cuda12x` to `cupy-cuda11x`.

## What's included

| Package     | Purpose                                    |
|-------------|--------------------------------------------|
| python312   | Python interpreter                         |
| cudatoolkit | nvcc compiler, CUDA runtime, NVTX, nsight  |
| gcc12       | C compiler for ANUGA's C extensions        |
| openmpi     | Parallel mesh partitioning (optional)      |
| gdal        | Geospatial data handling                   |
| netcdf/hdf5 | Output file formats                        |

## Important note

The standard `pip install anuga` does **not** include `sw_domain_cuda`.
You need the specific fork/branch that contains the GPU kernels.
Edit the repo URL in `setup.sh` and uncomment the git clone block.
# Hidris
