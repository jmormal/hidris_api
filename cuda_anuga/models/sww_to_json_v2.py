#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SWW → Compact JSON for Leaflet flood viewer (v2.1 — fixed encoding).
=====================================================================
Stores mesh once, per-frame only compact class-index strings.

Encoding:  '.' = inactive,  'A' = class 0,  'B' = class 1, … 'Z' = class 25
RLE uses digits 0-9 for counts — unambiguous since classes are letters/dot.
  e.g. "A350B12C1" → 'A'×350, 'B'×12, 'C'×1

Mesh stored flat for compactness:
  vertices: [lng0, lat0, lng1, lat1, ...]
  triangles: [v0, v1, v2, v0, v1, v2, ...]

Usage:
    python sww_to_geojson_v2.py result.sww -o flood_data.json
    python sww_to_geojson_v2.py result.sww -o flood_data.json.gz
    python sww_to_geojson_v2.py result.sww -o flood_data.json --nclasses 8 --scale log
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    sys.exit("netCDF4 required — pip install netCDF4")

try:
    from pyproj import Transformer
except ImportError:
    sys.exit("pyproj required — pip install pyproj")


INACTIVE_CHAR = "."


def class_to_char(ci: int) -> str:
    return INACTIVE_CHAR if ci < 0 else chr(ord("A") + ci)


def encode_classes(indices: np.ndarray) -> str:
    return "".join(class_to_char(int(i)) for i in indices)


def rle_encode(s: str) -> str:
    if not s:
        return ""
    parts = []
    prev = s[0]
    count = 1
    for ch in s[1:]:
        if ch == prev:
            count += 1
        else:
            parts.append(f"{prev}{count}")
            prev = ch
            count = 1
    parts.append(f"{prev}{count}")
    encoded = "".join(parts)
    return encoded if len(encoded) < len(s) else s


def make_breaks(vmin, vmax, n, scale):
    if scale == "log":
        return np.geomspace(max(vmin, 1e-3), vmax, n + 1)
    return np.linspace(vmin, vmax, n + 1)


def classify(values, breaks):
    return np.clip(np.digitize(values, breaks) - 1, -1, len(breaks) - 2)


def interp_colors(cmap, n):
    def h2r(h):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16)/255 for i in (0,2,4)]
    def r2h(r,g,b):
        return "#{:02x}{:02x}{:02x}".format(int(r*255),int(g*255),int(b*255))
    src = [h2r(c) for c in cmap]
    out = []
    for i in range(n):
        t = i / max(n-1,1) * (len(src)-1)
        lo, hi = int(t), min(int(t)+1, len(src)-1)
        f = t - lo
        out.append(r2h(*(src[lo][j]+(src[hi][j]-src[lo][j])*f for j in range(3))))
    return out


PROPERTIES = {
    "depth":    {"label": "Water Depth (m)",          "cmap": ["#f7fbff","#6baed6","#2171b5","#08306b"], "range": [0,5]},
    "speed":    {"label": "Flow Speed (m/s)",         "cmap": ["#ffffb2","#fd8d3c","#e31a1c","#800026"], "range": [0,5]},
    "momentum": {"label": "Momentum Magnitude (m²/s)","cmap": ["#f7fcf5","#74c476","#238b45","#00441b"], "range": [0,10]},
    "hazard":   {"label": "Flood Hazard (h × v)",     "cmap": ["#ffffd4","#fe9929","#d95f0e","#993404"], "range": [0,7]},
}


def process_sww(sww_path, output_path, epsg_source=25830, nclasses=6,
                scale="linear", depth_min=0.05, step_stride=1,
                coord_precision=5, skip_unchanged=True, use_rle=True):

    if nclasses > 26:
        sys.exit("Max 26 classes (A–Z)")

    t0 = time.time()
    print(f"Reading {sww_path} …")
    ds = Dataset(sww_path, "r")

    x = np.array(ds.variables["x"][:]) + float(getattr(ds, "xllcorner", 0))
    y = np.array(ds.variables["y"][:]) + float(getattr(ds, "yllcorner", 0))
    vols = np.array(ds.variables["volumes"][:], dtype=int)
    ntri, nnodes = vols.shape[0], len(x)

    times = np.array(ds.variables["time"][:])
    steps = list(range(0, len(times), step_stride))
    print(f"  {len(times)} timesteps → stride {step_stride} → {len(steps)} frames")
    print(f"  {ntri:,} triangles, {nnodes:,} nodes")

    stage = np.array(ds.variables["stage"][:])
    elev  = np.array(ds.variables["elevation"][:])
    xmom  = np.array(ds.variables["xmomentum"][:])
    ymom  = np.array(ds.variables["ymomentum"][:])
    ds.close()

    static_elev = elev.ndim == 1

    def cv(arr, ti):
        row = arr if arr.ndim == 1 else arr[ti]
        if row.shape[0] == ntri:
            return row.copy()
        return (row[vols[:,0]] + row[vols[:,1]] + row[vols[:,2]]) / 3.0

    print("  Transforming coordinates …")
    tr = Transformer.from_crs(f"EPSG:{epsg_source}", "EPSG:4326", always_xy=True)
    lng, lat = tr.transform(x, y)
    lng = np.round(lng, coord_precision)
    lat = np.round(lat, coord_precision)

    verts_flat = []
    for i in range(nnodes):
        verts_flat.append(float(lng[i]))
        verts_flat.append(float(lat[i]))

    print("  Scanning global maxima …")
    gmax = {k: 0.0 for k in PROPERTIES}
    for si in steps:
        e = cv(elev, 0 if static_elev else si)
        s = cv(stage, si)
        xm, ym_ = cv(xmom, si), cv(ymom, si)
        h = np.maximum(s - e, 0)
        v = np.zeros_like(h)
        m = h > depth_min
        if m.any(): v[m] = np.sqrt(xm[m]**2 + ym_[m]**2) / h[m]
        gmax["depth"]    = max(gmax["depth"],    h.max())
        gmax["speed"]    = max(gmax["speed"],    v.max())
        gmax["momentum"] = max(gmax["momentum"], np.sqrt(xm**2+ym_**2).max())
        gmax["hazard"]   = max(gmax["hazard"],   (h*v).max())

    for k in PROPERTIES:
        if gmax[k] > 0:
            PROPERTIES[k]["range"][1] = float(np.ceil(gmax[k]*10)/10)
    print(f"  Max → depth={gmax['depth']:.2f}  speed={gmax['speed']:.2f}  "
          f"momentum={gmax['momentum']:.2f}  hazard={gmax['hazard']:.2f}")

    breaks = {k: make_breaks(c["range"][0], c["range"][1], nclasses, scale) for k,c in PROPERTIES.items()}

    legend = {}
    for k, cfg in PROPERTIES.items():
        brk = breaks[k]
        colors = interp_colors(cfg["cmap"], nclasses)
        legend[k] = {"label": cfg["label"], "classes": [
            {"min": round(float(brk[i]),3), "max": round(float(brk[i+1]),3), "color": colors[i]}
            for i in range(nclasses)
        ]}

    print("  Encoding frames …")
    frames, prev_enc, skipped = [], None, 0
    for fi, si in enumerate(steps):
        ts = float(times[si])
        if fi % 10 == 0: print(f"    [{fi+1}/{len(steps)}] t={ts/60:.1f} min")
        e = cv(elev, 0 if static_elev else si)
        s_ = cv(stage, si)
        xm, ym_ = cv(xmom, si), cv(ymom, si)
        h = np.maximum(s_ - e, 0)
        act = h > depth_min
        v = np.zeros_like(h)
        if act.any(): v[act] = np.sqrt(xm[act]**2+ym_[act]**2)/h[act]
        raw = {"depth": h, "speed": v, "momentum": np.sqrt(xm**2+ym_**2), "hazard": h*v}
        enc = {}
        for k in PROPERTIES:
            ci = classify(raw[k], breaks[k])
            ci[~act] = -1
            s2 = encode_classes(ci)
            enc[k] = rle_encode(s2) if use_rle else s2
        if skip_unchanged and prev_enc and not any(enc[k]!=prev_enc.get(k,"") for k in enc):
            skipped += 1; continue
        frame = {"t": round(ts/60,1)}
        frame.update(enc)
        frames.append(frame)
        prev_enc = enc

    if skipped: print(f"  Skipped {skipped} unchanged frames")

    output = {
        "version": 2,
        "meta": {"epsg_source": epsg_source, "nclasses": nclasses, "scale": scale,
                 "ntriangles": ntri, "nnodes": nnodes, "nframes": len(frames),
                 "properties": list(PROPERTIES.keys()), "rle": use_rle},
        "legend": legend,
        "mesh": {"vertices": verts_flat, "triangles": vols.ravel().tolist()},
        "frames": frames,
    }

    raw = json.dumps(output, separators=(",",":"))
    p = Path(output_path)
    if p.suffix == ".gz" or output_path.endswith(".json.gz"):
        with gzip.open(output_path, "wt", compresslevel=6) as f: f.write(raw)
    else:
        with open(output_path, "w") as f: f.write(raw)

    print(f"\nDone — {time.time()-t0:.1f}s")
    print(f"  Raw JSON: {len(raw)/1e6:.1f} MB  |  On disk: {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SWW → compact flood JSON (v2.1)")
    p.add_argument("sww")
    p.add_argument("-o", "--output", default="flood_data.json")
    p.add_argument("--epsg", type=int, default=25830)
    p.add_argument("--nclasses", type=int, default=6)
    p.add_argument("--scale", choices=["linear","log"], default="linear")
    p.add_argument("--depth-min", type=float, default=0.05)
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--coord-precision", type=int, default=5)
    p.add_argument("--no-rle", action="store_true")
    p.add_argument("--keep-unchanged", action="store_true")
    args = p.parse_args()
    process_sww(args.sww, args.output, args.epsg, args.nclasses, args.scale,
                args.depth_min, args.step_stride, args.coord_precision,
                not args.keep_unchanged, not args.no_rle)
