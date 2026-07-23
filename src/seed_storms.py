"""
seed_storms.py — load a stack of single-band GeoTIFF frames into the storms
catalog as one compressed (T, H, W) float32 cube.

Each TIFF = one timestep. Cell value = mm of rain ACCUMULATED during that
interval (units="mm_per_step"). The worker converts to m/s at solve time using
timestep_s, so nothing is converted here — we store the raw depths.

Usage:
    python seed_storms.py \
        --name "Valencia DANA 2024" \
        --event-date 2024-10-29 \
        --timestep 600 \
        --glob "./storms/valencia_2024/*.tif"

Notes:
  - Frames are sorted by filename, so name them zero-padded and chronological
    (frame_000.tif, frame_001.tif, ...). Pass --sort-numeric to sort by the
    first integer found in each filename instead.
  - cell_size_m is read from the first TIFF's transform (assumes a projected
    CRS in metres; if your TIFFs are in degrees, reproject them first or pass
    --cell-size explicitly).
  - nodata is read from the rasters and zeroed out in the cube.
"""

import argparse
import glob
import re
import sys
from datetime import date

import numpy as np
import rasterio

# Reuse the storms helpers. Adjust the import to your worker's layout.
import db                      # the module that has get_conn (your db.py)
import db_storms as ds


def _sort_key_numeric(path):
    m = re.search(r"(\d+)", path.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else path


def load_cube(paths, sort_numeric=False):
    paths = sorted(paths, key=_sort_key_numeric if sort_numeric else None)
    if not paths:
        raise SystemExit("No TIFF files matched.")

    frames = []
    cell_size_m = None
    nodata = None
    shape = None

    for p in paths:
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)  # band 1
            if shape is None:
                shape = arr.shape
                # |a| is x pixel size; |e| is y pixel size. Assume square cells.
                cell_size_m = float(abs(src.transform.a))
                nodata = src.nodata
            elif arr.shape != shape:
                raise SystemExit(
                    f"Frame {p} shape {arr.shape} != first frame {shape}. "
                    "All frames must share the same grid."
                )
            frames.append(arr)

    cube = np.stack(frames, axis=0)  # (T, H, W)

    # Normalise nodata / non-finite to 0 rain
    if nodata is not None:
        cube[cube == nodata] = 0.0
    cube[~np.isfinite(cube)] = 0.0
    cube[cube < 0] = 0.0  # negative rain is nonsense

    return cube, cell_size_m, nodata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--description", default=None)
    ap.add_argument("--event-date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--timestep", type=int, required=True,
                    help="seconds per frame: 600 (10-min) or 3600 (hourly)")
    ap.add_argument("--glob", required=True, help="glob for the TIFF frames")
    ap.add_argument("--sort-numeric", action="store_true",
                    help="sort frames by the first integer in the filename")
    ap.add_argument("--cell-size", type=float, default=None,
                    help="override cell size in metres (else read from TIFF)")
    ap.add_argument("--units", default="mm_per_step",
                    choices=["mm_per_step", "mm_hr"])
    args = ap.parse_args()

    paths = glob.glob(args.glob)
    cube, cell_size_m, nodata = load_cube(paths, args.sort_numeric)
    if args.cell_size:
        cell_size_m = args.cell_size

    ev = date.fromisoformat(args.event_date) if args.event_date else None

    print(f"Loaded {cube.shape[0]} frames, grid {cube.shape[1]}x{cube.shape[2]}, "
          f"cell {cell_size_m} m, nodata={nodata}, units={args.units}")
    print(f"Raw cube size: {cube.nbytes / 1e6:.1f} MB; compressing…")

    ds.init_storms(db.get_conn)
    pid = ds.insert_storm(
        db.get_conn,
        name=args.name,
        description=args.description,
        event_date=ev,
        timestep_s=args.timestep,
        cube=cube,
        cell_size_m=cell_size_m,
        units=args.units,
        nodata=nodata,
    )
    print(f"Inserted storm {pid}")


if __name__ == "__main__":
    sys.exit(main())
