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
    try:
        yield conn
    finally:
        _pool.putconn(conn)


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
