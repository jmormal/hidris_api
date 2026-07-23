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
import contextlib
import glob
import re
import sys
from datetime import date

import numpy as np
import rasterio

# Reuse the storms helpers.
import db


def _sort_key_numeric(path):
    m = re.search(r"(\d+)", path.rsplit("/", 1)[-1])
    return int(m.group(1)) if m else path


def load_cube(paths, sort_numeric=False):
    paths = sorted(paths, key=_sort_key_numeric if sort_numeric else None)
    if not paths:
        raise SystemExit("No TIFF files matched.")

    with contextlib.ExitStack() as stack:
        readers = [stack.enter_context(rasterio.open(p)) for p in paths]
        try:
            return db.build_cube_from_rasters(readers)
        except ValueError as e:
            raise SystemExit(str(e))


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
    cube, cell_size_m, nodata, center_lng, center_lat = load_cube(paths, args.sort_numeric)
    if args.cell_size:
        cell_size_m = args.cell_size

    ev = date.fromisoformat(args.event_date) if args.event_date else None

    print(f"Loaded {cube.shape[0]} frames, grid {cube.shape[1]}x{cube.shape[2]}, "
          f"cell {cell_size_m} m, nodata={nodata}, units={args.units}, "
          f"center=({center_lng}, {center_lat})")
    print(f"Raw cube size: {cube.nbytes / 1e6:.1f} MB; compressing…")

    db.init_storms()
    pid = db.insert_storm(
        name=args.name,
        description=args.description,
        event_date=ev,
        timestep_s=args.timestep,
        cube=cube,
        cell_size_m=cell_size_m,
        units=args.units,
        nodata=nodata,
        center_lng=center_lng,
        center_lat=center_lat,
    )
    print(f"Inserted storm {pid}")


if __name__ == "__main__":
    sys.exit(main())
