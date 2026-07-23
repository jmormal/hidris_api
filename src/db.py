import os
import json
import gzip
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
    broken = False
    try:
        yield conn
    except psycopg2.OperationalError:
        # Connection itself is dead (e.g. "SSL connection has been closed
        # unexpectedly") — discard it instead of returning it to the pool,
        # otherwise every subsequent request keeps drawing the same broken
        # connection and fails identically.
        broken = True
        raise
    finally:
        _pool.putconn(conn, close=broken)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id           SERIAL PRIMARY KEY,
                    public_id    UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
                    user_id      UUID NOT NULL,
                    instance_name        VARCHAR(255),
                    instance_description TEXT,
                    created_at   TIMESTAMPTZ DEFAULT now(),
                    updated_at   TIMESTAMPTZ DEFAULT now(),
                    is_solved    BOOLEAN NOT NULL DEFAULT FALSE,
                    instance     JSONB,
                    solution     BYTEA
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_simulations_user_id "
                "ON simulations (user_id);"
            )
        conn.commit()


def list_instances(user_id: str):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT public_id, instance_name, instance_description,
                       is_solved, created_at, updated_at
                FROM simulations
                WHERE user_id = %s
                ORDER BY updated_at DESC;
                """,
                (user_id,),
            )
            return cur.fetchall()


def create_instance(
    user_id: str, name: str, description: str | None, instance: dict | None = None
):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO simulations
                    (user_id, instance_name, instance_description, instance)
                VALUES (%s, %s, %s, %s)
                RETURNING public_id, instance_name, instance_description,
                          is_solved, created_at, updated_at;
                """,
                (
                    user_id,
                    name,
                    description,
                    Json(instance) if instance is not None else None,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return row


def get_instance(user_id: str, public_id: str):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT public_id, instance_name, instance_description,
                       is_solved, created_at, updated_at, instance
                FROM simulations
                WHERE public_id = %s AND user_id = %s;
                """,
                (public_id, user_id),
            )
            return cur.fetchone()


def update_instance(
    user_id: str,
    public_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    instance: dict | None = None,
):
    """
    Any change to the setup invalidates the solution: clears the stored
    solution bytes and sets is_solved = false, atomically. Only provided
    fields change (COALESCE keeps the rest).
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE simulations
                SET instance_name = COALESCE(%s, instance_name),
                    instance_description = COALESCE(%s, instance_description),
                    instance = COALESCE(%s, instance),
                    updated_at = now()
                WHERE public_id = %s AND user_id = %s
                RETURNING public_id, instance_name, instance_description,
                          is_solved, created_at, updated_at;
                """,
                # is_solved = TRUE,
                # solution = NULL,
                (
                    name,
                    description,
                    Json(instance) if instance is not None else None,
                    public_id,
                    user_id,
                ),
            )
            row = cur.fetchone()
        conn.commit()
        return row


def delete_instance(user_id: str, public_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM simulations WHERE public_id = %s AND user_id = %s;",
                (public_id, user_id),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted


def get_solution_bytes(user_id: str, public_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT solution, is_solved
                FROM simulations
                WHERE public_id = %s AND user_id = %s;
                """,
                (public_id, user_id),
            )
            row = cur.fetchone()
            if row is None:
                return None, None
            solution, is_solved = row
            return (bytes(solution) if solution is not None else None), is_solved


def save_solution(public_id: str, dataset: dict):
    """
    Worker-side. Gzips a result dict and stores it, flipping is_solved.
    Use this when you have a Python dict. If you already have gzipped bytes
    (the GPU worker does), call save_solution_bytes instead to avoid
    double-compressing.
    """
    raw = gzip.compress(json.dumps(dataset).encode("utf-8"))
    save_solution_bytes(public_id, raw)


def save_solution_bytes(public_id: str, gz: bytes):
    """
    Worker-side. Stores already-gzipped solution bytes and flips is_solved.
    Keyed by public_id only — the worker acts on behalf of the owner.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE simulations
                SET solution = %s, is_solved = TRUE, updated_at = now()
                WHERE public_id = %s;
                """,
                (psycopg2.Binary(gz), public_id),
            )
        conn.commit()


# ── Historical storms (compressed rain cube, shared reference data) ──
#
# A storm is a (T, H, W) float32 grid of rainfall per timestep, gzip-
# compressed and stored whole in `data_grid`. No user_id — storms are a
# shared catalog, not per-user data (unlike `simulations`).


def init_storms():
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
                    -- Geographic center of the source rasters (WGS84), used to
                    -- default-place the storm where it actually occurred.
                    -- Nullable: older rows, or a source CRS that couldn't be
                    -- reprojected, just fall back to the map's default view.
                    center_lng   DOUBLE PRECISION,
                    center_lat   DOUBLE PRECISION,
                    total_depth_mm       REAL,
                    peak_intensity_mm_hr REAL,
                    -- Large object OID, not inline BYTEA: psycopg2 sends bytea
                    -- parameters as hex-escaped SQL text, which roughly doubles
                    -- their size and can blow past Postgres's 1GB single-value/
                    -- allocation ceiling for a big multi-frame cube. Large
                    -- objects are streamed instead.
                    data_grid_oid OID NOT NULL,
                    -- Precomputed accumulated-rain grid (H,W), gzip'd — small
                    -- enough to stay inline. Computed once at upload time so
                    -- GET /preview never has to decompress the full (T,H,W)
                    -- cube just to sum it.
                    preview_grid BYTEA,
                    preview_min  REAL,
                    preview_max  REAL,
                    created_at   TIMESTAMPTZ DEFAULT now()
                );
            """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_storms_event_date "
                "ON storms (event_date);"
            )
            # Migration for tables created before center_lng/lat, the
            # large-object storage switch, and the preview cache existed.
            # Idempotent.
            cur.execute("ALTER TABLE storms ADD COLUMN IF NOT EXISTS center_lng DOUBLE PRECISION;")
            cur.execute("ALTER TABLE storms ADD COLUMN IF NOT EXISTS center_lat DOUBLE PRECISION;")
            cur.execute("ALTER TABLE storms ADD COLUMN IF NOT EXISTS data_grid_oid OID;")
            cur.execute("ALTER TABLE storms ADD COLUMN IF NOT EXISTS preview_grid BYTEA;")
            cur.execute("ALTER TABLE storms ADD COLUMN IF NOT EXISTS preview_min REAL;")
            cur.execute("ALTER TABLE storms ADD COLUMN IF NOT EXISTS preview_max REAL;")
            cur.execute("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'storms' AND column_name = 'data_grid'
                    ) THEN
                        ALTER TABLE storms ALTER COLUMN data_grid DROP NOT NULL;
                    END IF;
                END $$;
            """)
        conn.commit()


def list_storms():
    """Summary rows for the picker — never selects the heavy data_grid."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT public_id, name, description, event_date,
                       timestep_s, n_frames, grid_rows, grid_cols,
                       cell_size_m, units, center_lng, center_lat,
                       total_depth_mm, peak_intensity_mm_hr,
                       created_at
                FROM storms
                ORDER BY event_date DESC NULLS LAST, name;
            """)
            return cur.fetchall()


def get_storm_meta(public_id: str):
    """Metadata only (no cube) — used to build the placement feature."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT public_id, name, description, event_date,
                       timestep_s, n_frames, grid_rows, grid_cols,
                       cell_size_m, units, nodata, center_lng, center_lat,
                       total_depth_mm, peak_intensity_mm_hr
                FROM storms
                WHERE public_id = %s;
            """,
                (public_id,),
            )
            return cur.fetchone()


def get_storm_cube(public_id: str):
    """Worker-side. Returns (cube ndarray (T,H,W) float32, meta dict) or (None, None)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT timestep_s, n_frames, grid_rows, grid_cols,
                       cell_size_m, units, nodata, data_grid_oid, data_grid
                FROM storms
                WHERE public_id = %s;
            """,
                (public_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None, None
            oid = row.pop("data_grid_oid")
            legacy_blob = row.pop("data_grid", None)
            if oid is None:
                # Pre-migration row: the cube is still in the old inline
                # bytea column (only possible for storms inserted before the
                # large-object switch).
                if legacy_blob is None:
                    return None, None
                cube = _decompress_cube(bytes(legacy_blob))
                return cube, row
            lo = conn.lobject(oid, "rb")
            try:
                raw = lo.read()
            finally:
                lo.close()
            cube = _decompress_cube(raw)
            return cube, row


def get_storm_preview_grid(public_id: str):
    """
    Accumulated-rain grid (H,W) for the map bitmap. Reads the small
    precomputed `preview_grid` — never touches the (T,H,W) cube/large object
    — so this stays fast regardless of how many frames the storm has.

    Returns (grid, rows, cols, min, max) or (None, None, None, None, None).
    Falls back to computing it from the full cube for the rare pre-migration
    row that predates the preview cache.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT grid_rows, grid_cols, preview_grid, preview_min, preview_max
                FROM storms WHERE public_id = %s;
            """,
                (public_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None, None, None, None, None
            if row["preview_grid"] is not None:
                grid = _decompress_cube(bytes(row["preview_grid"]))
                return grid, row["grid_rows"], row["grid_cols"], row["preview_min"], row["preview_max"]

    # Fallback: no cached preview yet (row predates this feature) — compute
    # it once from the full cube. Slow, but self-heals: recompute_preview
    # below persists the result so this only happens the first time.
    cube, meta = get_storm_cube(public_id)
    if cube is None:
        return None, None, None, None, None
    grid = _accumulate(cube, meta["units"], meta["timestep_s"])
    _save_preview_grid(public_id, grid)
    return grid, meta["grid_rows"], meta["grid_cols"], float(grid.min()), float(grid.max())


def _accumulate(cube: np.ndarray, units: str, timestep_s) -> np.ndarray:
    """Per-cell accumulated depth (mm) across every frame."""
    if units == "mm_hr":
        accum = (cube * (float(timestep_s) / 3600.0)).sum(axis=0)
    else:
        accum = cube.sum(axis=0)
    return np.nan_to_num(accum, nan=0.0, posinf=0.0, neginf=0.0)


def _save_preview_grid(public_id: str, grid: np.ndarray):
    blob = compress_cube(grid)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE storms
                SET preview_grid = %s, preview_min = %s, preview_max = %s
                WHERE public_id = %s;
            """,
                (psycopg2.Binary(blob), float(grid.min()), float(grid.max()), public_id),
            )
        conn.commit()


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


def build_cube_from_rasters(readers: list):
    """
    Stack a list of already-opened, same-shaped rasterio datasets (in the
    desired frame order) into a (T,H,W) float32 cube, normalising nodata /
    non-finite / negative values to 0. Returns (cube, cell_size_m, nodata,
    center_lng, center_lat) — the last two are the first raster's own centre,
    reprojected to WGS84 (None if that fails), so a newly-added storm can
    default-place itself where it actually occurred instead of on the map's
    generic default view.

    Shared by seed_storms.py (opens from disk paths) and the /api/storms
    upload endpoint (opens from uploaded bytes via rasterio.MemoryFile) so
    the normalisation logic can't drift between the two entry points.
    """
    if not readers:
        raise ValueError("No rasters provided")

    frames = []
    cell_size_m = None
    nodata = None
    shape = None
    center_lng = None
    center_lat = None

    for src in readers:
        arr = src.read(1).astype(np.float32)  # band 1
        if shape is None:
            shape = arr.shape
            cell_size_m = float(abs(src.transform.a))
            nodata = src.nodata
            center_lng, center_lat = _raster_center_wgs84(src)
        elif arr.shape != shape:
            raise ValueError(
                f"Frame shape {arr.shape} != first frame shape {shape}. "
                "All frames must share the same grid."
            )
        frames.append(arr)

    cube = np.stack(frames, axis=0)  # (T, H, W)

    if nodata is not None:
        cube[cube == nodata] = 0.0
    cube[~np.isfinite(cube)] = 0.0
    cube[cube < 0] = 0.0  # negative rain is nonsense

    return cube, cell_size_m, nodata, center_lng, center_lat


def _raster_center_wgs84(src):
    """(lon, lat) of a rasterio dataset's own centre pixel, in EPSG:4326."""
    try:
        import rasterio.warp

        rows, cols = src.height, src.width
        cx, cy = src.transform * (cols / 2.0, rows / 2.0)
        if src.crs is None:
            return None, None
        lngs, lats = rasterio.warp.transform(src.crs, "EPSG:4326", [cx], [cy])
        return float(lngs[0]), float(lats[0])
    except Exception:
        # Best-effort — a missing/unparseable CRS shouldn't fail the upload,
        # it just means the storm falls back to a generic default placement.
        return None, None


def insert_storm(
    *,
    name,
    description,
    event_date,
    timestep_s,
    cube: np.ndarray,
    cell_size_m,
    units="mm_per_step",
    nodata=None,
    center_lng=None,
    center_lat=None,
):
    """Computes summary stats and stores the compressed cube."""
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

    # Precompute the accumulated-rain preview grid once here, so GET /preview
    # is just a small-blob read instead of decompressing the whole cube.
    preview_grid = _accumulate(valid, units, timestep_s)
    preview_blob = compress_cube(preview_grid)

    blob = compress_cube(cube)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Large object: streamed in chunks, not embedded as hex-escaped
            # SQL text, so it isn't subject to Postgres's 1GB single-value/
            # allocation limit the way a big bytea query parameter would be.
            lo = conn.lobject(0, "wb")
            try:
                lo.write(blob)
                oid = lo.oid
            finally:
                lo.close()

            cur.execute(
                """
                INSERT INTO storms
                    (name, description, event_date, timestep_s, n_frames,
                     grid_rows, grid_cols, cell_size_m, units, nodata,
                     center_lng, center_lat,
                     total_depth_mm, peak_intensity_mm_hr, data_grid_oid,
                     preview_grid, preview_min, preview_max)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
                    center_lng,
                    center_lat,
                    round(total_depth_mm, 2),
                    round(peak_intensity_mm_hr, 2),
                    oid,
                    psycopg2.Binary(preview_blob),
                    float(preview_grid.min()),
                    float(preview_grid.max()),
                ),
            )
            pid = cur.fetchone()["public_id"]
        conn.commit()
        return pid


# ── add to your existing db.py (reuses get_conn from that module) ──


def init_db_storms():
    """
    Create the storm raster schema (catalog + raster data) and its indexes.
    Idempotent — safe to call on every startup, like init_db().

    Requires the postgis and postgis_raster extensions; we create them here
    too so a fresh database is self-contained.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster;")

            # storm catalog — id comes from the folder name (provided, not serial)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS storm_catalog (
                    id          INT PRIMARY KEY,
                    name        VARCHAR(100),
                    year        INT,
                    description TEXT,
                    created_at  TIMESTAMPTZ DEFAULT now()
                );
            """)

            # raster frames. measure_type defaults to rain_depth; interval_min
            # is the minutes each frame spans (needed when accumulating rates).
            cur.execute("""
                CREATE TABLE IF NOT EXISTS storm_raster_data (
                    id            BIGSERIAL PRIMARY KEY,
                    storm_id      INT NOT NULL
                                  REFERENCES storm_catalog(id) ON DELETE CASCADE,
                    recorded_at   TIMESTAMPTZ NOT NULL,
                    measure_type  VARCHAR(16) NOT NULL DEFAULT 'rain_depth',
                    interval_min  INT NOT NULL DEFAULT 10,
                    rast          RASTER NOT NULL
                );
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_storm_raster_spatial
                    ON storm_raster_data USING gist (ST_ConvexHull(rast));
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_storm_raster_lookup
                    ON storm_raster_data (storm_id, measure_type, recorded_at);
            """)
        conn.commit()


def load_storm_functions(sql_path: str = "sql/02_functions.sql"):
    """
    Load (or reload) the Martin tile/value functions from the .sql file.
    Kept separate from init_db_storms() because the PL/pgSQL bodies are long
    and change independently of the schema. CREATE OR REPLACE makes this
    safe to re-run.
    """
    with open(sql_path, "r", encoding="utf-8") as f:
        sql = f.read()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


if __name__ == "__main__":
    init_db()  # your existing simulations table
    init_db_storms()  # storm catalog + raster
