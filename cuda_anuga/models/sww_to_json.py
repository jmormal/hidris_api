#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWW → GeoJSON converter for Leaflet flood viewer.
===================================================
Reads ANUGA .sww (NetCDF4) output, computes derived quantities,
groups triangles into value classes, merges geometries per class,
and writes a single JSON bundle for the Leaflet viewer.

Usage:
    python sww_to_geojson.py resultado.sww -o flood_data.json
    python sww_to_geojson.py resultado.sww -o flood_data.json --nclasses 8 --scale log
    python sww_to_geojson.py resultado.sww -o flood_data.json --step-stride 5
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path

try:
    from netCDF4 import Dataset
except ImportError:
    sys.exit("netCDF4 required — pip install netCDF4")

try:
    from shapely.geometry import Polygon, mapping
    from shapely.ops import unary_union
    from shapely import make_valid
except ImportError:
    sys.exit("shapely required — pip install shapely")

try:
    from pyproj import Transformer
except ImportError:
    sys.exit("pyproj required — pip install pyproj")


# ─── helpers ─────────────────────────────────────────────


def make_breaks(vmin, vmax, nclasses, scale="linear"):
    """Return class break values."""
    if scale == "log":
        if vmin <= 0:
            vmin = 1e-3
        return np.geomspace(vmin, vmax, nclasses + 1)
    return np.linspace(vmin, vmax, nclasses + 1)


def classify(values, breaks):
    """Assign each value to a class index (0..nclasses-1). -1 = below min."""
    idx = np.digitize(values, breaks) - 1
    idx = np.clip(idx, -1, len(breaks) - 2)
    return idx


def merge_triangles_by_class(triangles_xy, class_indices, nclasses):
    """
    For each class, dissolve triangles into (multi)polygons.
    Returns list[class_idx] → shapely geometry | None
    """
    result = [None] * nclasses
    for ci in range(nclasses):
        mask = class_indices == ci
        if not np.any(mask):
            continue
        polys = []
        idxs = np.where(mask)[0]
        for i in idxs:
            tri = triangles_xy[i]
            try:
                p = Polygon(tri)
                if not p.is_valid:
                    p = make_valid(p)
                if not p.is_empty:
                    polys.append(p)
            except Exception:
                continue
        if polys:
            merged = unary_union(polys)
            if not merged.is_valid:
                merged = make_valid(merged)
            result[ci] = merged
    return result


# ─── main pipeline ───────────────────────────────────────


def process_sww(
    sww_path: str,
    output_path: str,
    epsg_source: int = 25830,
    nclasses: int = 6,
    scale: str = "linear",
    depth_min: float = 0.05,
    step_stride: int = 1,
):
    t_start = time.time()
    print(f"Reading {sww_path} …")
    ds = Dataset(sww_path, "r")

    # ── coordinates & triangles ──
    x = np.array(ds.variables["x"][:])
    y = np.array(ds.variables["y"][:])

    # Apply geo_reference offset if present
    xll = float(getattr(ds, "xllcorner", 0))
    yll = float(getattr(ds, "yllcorner", 0))
    x += xll
    y += yll

    volumes = np.array(ds.variables["volumes"][:], dtype=int)  # (ntri, 3)
    ntri = volumes.shape[0]

    # ── time axis ──
    times = np.array(ds.variables["time"][:])
    step_indices = list(range(0, len(times), step_stride))
    print(
        f"  {len(times)} timesteps, using every {step_stride} → {len(step_indices)} frames"
    )
    print(f"  {ntri} triangles, {len(x)} nodes")

    # ── read quantities ──
    stage_all = np.array(ds.variables["stage"][:])  # (ntime, nnodes) or (ntime, ntri)
    elev_all = np.array(ds.variables["elevation"][:])  # could be 1D (static) or 2D
    xmom_all = np.array(ds.variables["xmomentum"][:])
    ymom_all = np.array(ds.variables["ymomentum"][:])
    ds.close()

    # Elevation may be static (1D)
    static_elev = elev_all.ndim == 1

    # ── build triangle coords in source CRS ──
    # Each triangle: [[x0,y0],[x1,y1],[x2,y2],[x0,y0]]
    tri_coords = np.zeros((ntri, 4, 2))
    for k in range(3):
        tri_coords[:, k, 0] = x[volumes[:, k]]
        tri_coords[:, k, 1] = y[volumes[:, k]]
    tri_coords[:, 3, :] = tri_coords[:, 0, :]  # close ring

    # ── centroid values helper ──
    def centroid_values(arr_2d, ti):
        """Average vertex values to get centroid value per triangle."""
        if arr_2d.ndim == 1:
            row = arr_2d
        else:
            row = arr_2d[ti]
        if row.shape[0] == ntri:
            return row.copy()
        # node-based → centroid average
        return (row[volumes[:, 0]] + row[volumes[:, 1]] + row[volumes[:, 2]]) / 3.0

    # ── coordinate transformer ──
    transformer = Transformer.from_crs(
        f"EPSG:{epsg_source}", "EPSG:4326", always_xy=True
    )

    # Transform all triangle coords at once
    flat_x = tri_coords[:, :, 0].ravel()
    flat_y = tri_coords[:, :, 1].ravel()
    lng, lat = transformer.transform(flat_x, flat_y)
    tri_coords_wgs = np.column_stack([lng, lat]).reshape(ntri, 4, 2)

    # ── define properties to export ──
    properties_config = {
        "depth": {
            "label": "Water Depth (m)",
            "cmap": ["#f7fbff", "#6baed6", "#2171b5", "#08306b"],
            "range": [0, 5.0],
        },
        "speed": {
            "label": "Flow Speed (m/s)",
            "cmap": ["#ffffb2", "#fd8d3c", "#e31a1c", "#800026"],
            "range": [0, 5.0],
        },
        "momentum": {
            "label": "Momentum Magnitude (m²/s)",
            "cmap": ["#f7fcf5", "#74c476", "#238b45", "#00441b"],
            "range": [0, 10.0],
        },
        "hazard": {
            "label": "Flood Hazard (h × v)",
            "cmap": ["#ffffd4", "#fe9929", "#d95f0e", "#993404"],
            "range": [0, 7.0],
        },
    }

    # ── first pass: determine actual max values ──
    print("  Scanning for global max values …")
    global_max = {"depth": 0, "speed": 0, "momentum": 0, "hazard": 0}
    for si in step_indices:
        elev = (
            centroid_values(elev_all, si)
            if not static_elev
            else centroid_values(elev_all, 0)
        )
        stage = centroid_values(stage_all, si)
        xm = centroid_values(xmom_all, si)
        ym = centroid_values(ymom_all, si)

        h = np.maximum(stage - elev, 0)
        mask = h > depth_min
        v = np.zeros_like(h)
        if np.any(mask):
            v[mask] = np.sqrt(xm[mask] ** 2 + ym[mask] ** 2) / h[mask]
        mom = np.sqrt(xm**2 + ym**2)
        haz = h * v

        global_max["depth"] = max(global_max["depth"], np.max(h))
        global_max["speed"] = max(global_max["speed"], np.max(v))
        global_max["momentum"] = max(global_max["momentum"], np.max(mom))
        global_max["hazard"] = max(global_max["hazard"], np.max(haz))

    # Update ranges with actual data
    for prop in properties_config:
        gm = global_max[prop]
        if gm > 0:
            properties_config[prop]["range"][1] = float(np.ceil(gm * 10) / 10)

    print(
        f"  Global max → depth={global_max['depth']:.2f}m, speed={global_max['speed']:.2f}m/s, "
        f"momentum={global_max['momentum']:.2f}, hazard={global_max['hazard']:.2f}"
    )

    # ── compute breaks per property ──
    breaks_dict = {}
    for prop, cfg in properties_config.items():
        breaks_dict[prop] = make_breaks(
            cfg["range"][0], cfg["range"][1], nclasses, scale
        )

    # ── process each timestep ──
    frames = []
    for frame_idx, si in enumerate(step_indices):
        t_sec = float(times[si])
        print(
            f"  Frame {frame_idx + 1}/{len(step_indices)}  t={t_sec / 60:.1f} min …",
            end="\r",
        )

        elev = (
            centroid_values(elev_all, si)
            if not static_elev
            else centroid_values(elev_all, 0)
        )
        stage = centroid_values(stage_all, si)
        xm = centroid_values(xmom_all, si)
        ym = centroid_values(ymom_all, si)

        h = np.maximum(stage - elev, 0)
        mask = h > depth_min
        v = np.zeros_like(h)
        if np.any(mask):
            v[mask] = np.sqrt(xm[mask] ** 2 + ym[mask] ** 2) / h[mask]
        mom = np.sqrt(xm**2 + ym**2)
        haz = h * v

        prop_values = {"depth": h, "speed": v, "momentum": mom, "hazard": haz}

        frame_data = {
            "time_s": t_sec,
            "time_min": round(t_sec / 60, 1),
            "properties": {},
        }

        for prop, vals in prop_values.items():
            breaks = breaks_dict[prop]
            ci = classify(vals, breaks)

            # Only process triangles above minimum threshold
            active = ci >= 0
            if not np.any(active):
                frame_data["properties"][prop] = []
                continue

            active_ci = ci[active]
            active_tris = tri_coords_wgs[active]

            merged_geoms = merge_triangles_by_class(active_tris, active_ci, nclasses)

            classes = []
            for k, geom in enumerate(merged_geoms):
                if geom is None or geom.is_empty:
                    continue
                classes.append(
                    {
                        "class": k,
                        "min": round(float(breaks[k]), 3),
                        "max": round(float(breaks[k + 1]), 3),
                        "geometry": mapping(geom),
                    }
                )
            frame_data["properties"][prop] = classes

        frames.append(frame_data)

    print()

    # ── interpolation colours for legend ──
    def interp_colors(cmap_list, n):
        """Linearly interpolate hex color list to n colors."""
        from colorsys import rgb_to_hls, hls_to_rgb

        def hex2rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

        def rgb2hex(r, g, b):
            return "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )

        src = [hex2rgb(c) for c in cmap_list]
        result = []
        for i in range(n):
            t = i / max(n - 1, 1) * (len(src) - 1)
            lo = int(t)
            hi = min(lo + 1, len(src) - 1)
            frac = t - lo
            r = src[lo][0] + (src[hi][0] - src[lo][0]) * frac
            g = src[lo][1] + (src[hi][1] - src[lo][1]) * frac
            b = src[lo][2] + (src[hi][2] - src[lo][2]) * frac
            result.append(rgb2hex(r, g, b))
        return result

    # ── build output ──
    legend = {}
    for prop, cfg in properties_config.items():
        brk = breaks_dict[prop]
        colors = interp_colors(cfg["cmap"], nclasses)
        legend[prop] = {
            "label": cfg["label"],
            "classes": [
                {
                    "min": round(float(brk[i]), 3),
                    "max": round(float(brk[i + 1]), 3),
                    "color": colors[i],
                }
                for i in range(nclasses)
            ],
        }

    output = {
        "meta": {
            "epsg_source": epsg_source,
            "nclasses": nclasses,
            "scale": scale,
            "ntriangles": int(ntri),
            "nframes": len(frames),
            "properties": list(properties_config.keys()),
        },
        "legend": legend,
        "frames": frames,
    }

    print(f"Writing {output_path} …")
    with open(output_path, "w") as f:
        json.dump(output, f)

    size_mb = Path(output_path).stat().st_size / 1e6
    elapsed = time.time() - t_start
    print(f"Done — {size_mb:.1f} MB, {elapsed:.1f}s")


# ─── CLI ─────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert ANUGA .sww → GeoJSON bundle")
    p.add_argument("sww", help="Path to .sww file")
    p.add_argument("-o", "--output", default="flood_data.json", help="Output JSON path")
    p.add_argument("--epsg", type=int, default=25830, help="Source EPSG code")
    p.add_argument("--nclasses", type=int, default=6, help="Number of value classes")
    p.add_argument(
        "--scale", choices=["linear", "log"], default="linear", help="Class break scale"
    )
    p.add_argument(
        "--depth-min", type=float, default=0.05, help="Min depth threshold (m)"
    )
    p.add_argument("--step-stride", type=int, default=1, help="Use every Nth timestep")
    args = p.parse_args()

    process_sww(
        args.sww,
        args.output,
        epsg_source=args.epsg,
        nclasses=args.nclasses,
        scale=args.scale,
        depth_min=args.depth_min,
        step_stride=args.step_stride,
    )
