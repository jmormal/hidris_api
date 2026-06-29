import os
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

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
    """Idempotent: safe to run on every startup."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS simulations (
                    id SERIAL PRIMARY KEY,
                    public_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
                    user_id UUID NOT NULL,
                    instance_name VARCHAR(255),
                    instance_description TEXT,
                    created_at TIMESTAMPTZ DEFAULT now(),
                    is_solved BOOLEAN NOT NULL DEFAULT FALSE,
                    instance BYTEA,
                    solution BYTEA
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_simulations_user_id "
                "ON simulations (user_id);"
            )
        conn.commit()


def get_instances(user_id: str):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, user_id, public_id, instance_name, instance_description,
                       is_solved, created_at
                FROM simulations
                WHERE user_id = %s
                ORDER BY created_at DESC;
                """,
                (user_id,),
            )
            return cur.fetchall()


def create_instance(user_id: str, name: str, description: str | None = None):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO simulations (user_id, instance_name, instance_description)
                VALUES (%s, %s, %s)
                RETURNING id, user_id, public_id, instance_name, instance_description,
                          is_solved, created_at;
                """,
                (user_id, name, description),
            )
            row = cur.fetchone()
        conn.commit()
        return row


if __name__ == "__main__":
    init_db()
