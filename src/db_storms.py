"""
db_storms.py — storms catalog (shared reference data).

Append these functions to your existing src/db.py (they reuse get_conn from
there). Shown as a separate file for clarity; copy the functions in.

Storage decision: the compressed (T,H,W) float32 cube lives in Postgres BYTEA
(`data_grid`), matching how solutions are stored. Summary/metadata columns live
alongside so the picker list never touches the multi-hundred-MB blob.
"""

import io
import gzip
import numpy as np
from psycopg2.extras import RealDictCursor, Json

# from .db import get_conn   # in practice these live IN db.py; import not needed


def init_storms(get_conn):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS storms (
                    id           SERIAL PRIMARY KEY,
                    public_id    UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
                    name         VARCHAR(255) NOT NULL,
                    description  TEXT,
                    event_date   DATE,
                    timestep_s   INTEGER NOT NULL,
                    n_frames     INTEGER NOT NULL,
                    grid_rows    INTEGER NOT NULL,
                    grid_cols    INTEGER NOT NULL,
                    cell_size_m  REAL NOT NULL,
                    units        VARCHAR(16) NOT NULL DEFAULT 'mm_per_step',
                    nodata       REAL,
                    total_depth_mm       REAL,
                    peak_intensity_mm_hr REAL,
                    data_grid    BYTEA NOT NULL,
                    created_at   TIMESTAMPTZ DEFAULT now()
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_storms_event_date "
                "ON storms (event_date);"
            )
        conn.commit()


def list_storms(get_conn):
    """Summary rows for the picker — never selects the heavy data_grid."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT public_id, name, description, event_date,
                       timestep_s, n_frames, grid_rows, grid_cols,
                       cell_size_m, units, total_depth_mm, peak_intensity_mm_hr,
                       created_at
                FROM storms
                ORDER BY event_date DESC NULLS LAST, name;
            """)
            return cur.fetchall()


def get_storm_meta(get_conn, public_id: str):
    """Metadata only (no cube) — used to build the placement feature."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT public_id, name, description, event_date,
                       timestep_s, n_frames, grid_rows, grid_cols,
                       cell_size_m, units, nodata,
                       total_depth_mm, peak_intensity_mm_hr
                FROM storms
                WHERE public_id = %s;
            """,
                (public_id,),
            )
            return cur.fetchone()


def get_storm_cube(get_conn, public_id: str):
    """Worker-side. Returns (cube ndarray (T,H,W) float32, meta dict) or (None, None)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT timestep_s, n_frames, grid_rows, grid_cols,
                       cell_size_m, units, nodata, data_grid
                FROM storms
                WHERE public_id = %s;
            """,
                (public_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None, None
            raw = bytes(row.pop("data_grid"))
            cube = _decompress_cube(raw)
            return cube, row


# ── (de)compression helpers ──────────────────────────────────────────────


def compress_cube(cube: np.ndarray) -> bytes:
    """(T,H,W) float32 -> gzipped .npy bytes. Rainfall is mostly zeros, so this
    compresses hard."""
    cube = np.ascontiguousarray(cube, dtype=np.float32)
    buf = io.BytesIO()
    np.save(buf, cube)
    return gzip.compress(buf.getvalue(), compresslevel=6)


def _decompress_cube(raw: bytes) -> np.ndarray:
    buf = io.BytesIO(gzip.decompress(raw))
    return np.load(buf)


def insert_storm(
    get_conn,
    *,
    name,
    description,
    event_date,
    timestep_s,
    cube: np.ndarray,
    cell_size_m,
    units="mm_per_step",
    nodata=None,
):
    """Seed-script helper. Computes summary stats and stores the compressed cube."""
    n_frames, rows, cols = cube.shape

    valid = cube.copy()
    if nodata is not None:
        valid[valid == nodata] = 0.0
    valid[~np.isfinite(valid)] = 0.0

    # depth conversion for summary stats
    if units == "mm_per_step":
        # total depth at the wettest cell = sum over time of per-step depth
        total_depth_mm = float(valid.sum(axis=0).max())
        peak_intensity_mm_hr = float(valid.max() / (timestep_s / 3600.0))
    else:  # mm_hr
        total_depth_mm = float((valid * (timestep_s / 3600.0)).sum(axis=0).max())
        peak_intensity_mm_hr = float(valid.max())

    blob = compress_cube(cube)
    print(f"compressed blob: {len(blob) / 1e6:.1f} MB")
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO storms
                    (name, description, event_date, timestep_s, n_frames,
                     grid_rows, grid_cols, cell_size_m, units, nodata,
                     total_depth_mm, peak_intensity_mm_hr, data_grid)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING public_id;
            """,
                (
                    name,
                    description,
                    event_date,
                    timestep_s,
                    n_frames,
                    rows,
                    cols,
                    cell_size_m,
                    units,
                    nodata,
                    round(total_depth_mm, 2),
                    round(peak_intensity_mm_hr, 2),
                    __import__("psycopg2").Binary(blob),
                ),
            )
            pid = cur.fetchone()["public_id"]
        conn.commit()
        return pid
