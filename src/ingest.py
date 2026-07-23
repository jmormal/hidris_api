import os
import re
import glob
import subprocess
import datetime as dt

# Reuses get_conn() and init_db_storms() from your db module.
# from db import get_conn, init_db_storms

from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import ThreadedConnectionPool

import io
import numpy as np

_pool = ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    port=os.getenv("DB_PORT", "5432"),
)


@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


# ── config.txt parsing ──────────────────────────────────────


def _parse_config(path: str) -> tuple[dt.datetime, int]:
    """
    Read a storm folder's config.txt:
        date=2026-06-30T00:00:00Z
        step_min=10
    Returns (start_datetime_utc, step_minutes).
    """
    date_str = None
    step_min = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key, val = key.strip(), val.strip()
            if key == "date":
                date_str = val
            elif key == "step_min":
                step_min = int(val)
    if date_str is None or step_min is None:
        raise ValueError(f"{path}: missing date= or step_min=")
    # tolerate trailing Z
    start = dt.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    return start, step_min


def _natural_key(path: str):
    """Sort filenames so frame2 < frame10 (numeric-aware)."""
    name = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


# ── raster2pgsql wrapper ────────────────────────────────────


def _raster_to_sql(tif_path: str, table: str, srid: int, tile_size: str) -> str:
    """
    Run raster2pgsql and return its SQL text. -a appends, but here we always
    create a fresh temp table per file with -d (drop+create) so the output is
    self-contained and we control the final insert ourselves.
    """
    out_path = os.path.join(os.path.dirname(tif_path), "a" + os.path.basename(tif_path))
    cmd1 = ["gdalwarp", "-t_srs", "EPSG:4326", tif_path, out_path]

    result = subprocess.run(cmd1, capture_output=True, text=True, check=True)
    cmd = [
        "raster2pgsql",
        "-s",
        str(srid),
        "-t",
        tile_size,
        # NOTE: do NOT use -M here. It appends VACUUM ANALYZE, which cannot
        # run inside a transaction block (we load everything in one txn).
        # We ANALYZE once after all storms instead.
        "-d",  # drop & recreate the temp table
        out_path,
        table,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout


# ── main ingestion ──────────────────────────────────────────


def ingest_storms(
    storm_root: str = "./storm",
    measure_type: str = "rain_depth",  # or "rain_rate"
    srid: int = 4326,
    tile_size: str = "256x256",
):
    """
    Walk storm_root/<id>/, read config.txt, load each TIFF into PostGIS with
    recorded_at = date + index*step_min. Folder name is the storm id.
    Idempotent per (storm_id, measure_type): clears prior frames first.

    Requires raster2pgsql on PATH (GDAL/PostGIS client tools) and the schema
    created by init_db_storms().
    """
    if subprocess.run(["which", "raster2pgsql"], capture_output=True).returncode != 0:
        raise RuntimeError("raster2pgsql not found on PATH")

    storm_dirs = sorted(
        d for d in glob.glob(os.path.join(storm_root, "*")) if os.path.isdir(d)
    )

    for sdir in storm_dirs:
        storm_id_str = os.path.basename(sdir.rstrip("/"))
        if not storm_id_str.isdigit():
            print(f"SKIP {storm_id_str} (folder name not numeric)")
            continue
        storm_id = int(storm_id_str)

        cfg = os.path.join(sdir, "config.txt")
        if not os.path.isfile(cfg):
            print(f"SKIP {storm_id} (no config.txt)")
            continue

        start, step_min = _parse_config(cfg)

        tifs = sorted(
            glob.glob(os.path.join(sdir, "*.tif"))
            + glob.glob(os.path.join(sdir, "*.tiff")),
            key=_natural_key,
        )
        if not tifs:
            print(f"SKIP {storm_id} (no tif files)")
            continue

        print(
            f">>> storm {storm_id}: {len(tifs)} frames, "
            f"start={start.isoformat()} step={step_min}min"
        )

        with get_conn() as conn:
            with conn.cursor() as cur:
                # catalog upsert
                cur.execute(
                    """
                    INSERT INTO storm_catalog (id, name, year)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (storm_id, f"storm-{storm_id}", start.year),
                )
                # idempotent re-ingest for this storm+measure
                cur.execute(
                    """
                    DELETE FROM storm_raster_data
                    WHERE storm_id = %s AND measure_type = %s;
                    """,
                    (storm_id, measure_type),
                )

                tmp_table = f"tmp_ingest_{storm_id}_{os.getpid()}"
                for idx, tif in enumerate(tifs):
                    recorded_at = start + dt.timedelta(minutes=idx * step_min)

                    # convert this tif into the temp table
                    sql_text = _raster_to_sql(
                        tif, f"public.{tmp_table}", srid, tile_size
                    )
                    cur.execute(sql_text)

                    # move into the real table, tagging storm/time/measure
                    cur.execute(
                        f"""
                        INSERT INTO storm_raster_data
                              (storm_id, recorded_at, measure_type,
                               interval_min, rast)
                        SELECT %s, %s, %s, %s, rast
                        FROM public.{tmp_table};
                        DROP TABLE public.{tmp_table};
                        """,
                        (storm_id, recorded_at, measure_type, step_min),
                    )

                    print(
                        f"    [{idx:3d}] {recorded_at.isoformat()}  "
                        f"<- {os.path.basename(tif)}"
                    )
            conn.commit()

    # one ANALYZE for the whole table after bulk load (replaces per-file -M).
    # autocommit so it isn't wrapped in a transaction.
    with get_conn() as conn:
        old_autocommit = conn.autocommit
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("ANALYZE storm_raster_data;")
        finally:
            conn.autocommit = old_autocommit

    # summary
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT storm_id, measure_type, count(*),
                       min(recorded_at), max(recorded_at)
                FROM storm_raster_data
                GROUP BY storm_id, measure_type
                ORDER BY storm_id, measure_type;
                """
            )
            print("\nFrames per storm/measure:")
            for row in cur.fetchall():
                print(
                    f"  storm {row[0]} [{row[1]}]: {row[2]} frames {row[3]} -> {row[4]}"
                )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ingest storm TIFFs into PostGIS")
    p.add_argument("storm_root", nargs="?", default="../storms")
    p.add_argument(
        "--measure-type", default="rain_depth", choices=["rain_depth", "rain_rate"]
    )
    p.add_argument("--srid", type=int, default=4326)
    p.add_argument("--tile-size", default="256x256")
    args = p.parse_args()

    # init_db_storms()   # ensure schema exists first
    ingest_storms(
        storm_root=args.storm_root,
        measure_type=args.measure_type,
        srid=args.srid,
        tile_size=args.tile_size,
    )
