import logging
import os
from contextlib import contextmanager

import psycopg
from psycopg import pq, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from settings import database_search_path, database_url

logger = logging.getLogger(__name__)

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS evaluations(
      evaluation_id TEXT PRIMARY KEY,
      input_digest TEXT NOT NULL,
      request_json BYTEA NOT NULL,
      verifier_json TEXT NOT NULL,
      response_json BYTEA NOT NULL,
      status TEXT NOT NULL,
      created_at BIGINT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions(
      dossier_id TEXT NOT NULL,
      fingerprint TEXT NOT NULL,
      proposal_json BYTEA NOT NULL,
      PRIMARY KEY(dossier_id, fingerprint)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS receipts(
      evaluation_id TEXT NOT NULL,
      dossier_id TEXT NOT NULL,
      call_id TEXT NOT NULL,
      receipt_json BYTEA NOT NULL,
      outcome_json BYTEA NOT NULL,
      PRIMARY KEY(evaluation_id, dossier_id),
      FOREIGN KEY(evaluation_id) REFERENCES evaluations(evaluation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS commits(
      evaluation_id TEXT PRIMARY KEY,
      request_digest TEXT NOT NULL,
      response_json BYTEA NOT NULL,
      FOREIGN KEY(evaluation_id) REFERENCES evaluations(evaluation_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS evaluations_input_digest_idx ON evaluations(input_digest)",
    "CREATE INDEX IF NOT EXISTS decisions_fingerprint_idx ON decisions(fingerprint)",
    "CREATE INDEX IF NOT EXISTS receipts_evaluation_idx ON receipts(evaluation_id)",
)

_pool: ConnectionPool | None = None


def _validate_search_path(path: str) -> str:
    if not path.replace("_", "").isalnum():
        raise RuntimeError("Invalid DATABASE_SEARCH_PATH")
    return path


def _connect_kwargs() -> dict:
    return {
        "autocommit": True,
        "row_factory": dict_row,
        "connect_timeout": int(os.getenv("DATABASE_CONNECT_TIMEOUT", "15")),
    }


def configure(url: str | None = None, search_path: str | None = None) -> ConnectionPool:
    global _pool
    close_pool()
    dsn = url or database_url()
    path = search_path if search_path is not None else database_search_path()
    if path:
        _validate_search_path(path)
    try:
        if path:
            with psycopg.connect(dsn, **_connect_kwargs()) as bootstrap:
                bootstrap.execute(f"CREATE SCHEMA IF NOT EXISTS {path}")

        def configure_connection(con):
            if path:
                con.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(path)))

        _pool = ConnectionPool(
            dsn,
            min_size=1,
            max_size=5,
            open=True,
            kwargs=_connect_kwargs(),
            configure=configure_connection,
            check=ConnectionPool.check_connection,
        )
        with _pool.connection() as con:
            con.execute("SELECT 1")
        return _pool
    except Exception as exc:
        close_pool()
        logger.exception("Database initialization failed")
        raise RuntimeError(f"Database initialization failed: {type(exc).__name__}") from exc


def get_pool() -> ConnectionPool:
    if _pool is None:
        configure()
    assert _pool is not None
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            logger.exception("Error closing database pool")
        _pool = None


def init_schema(con=None) -> None:
    if con is None:
        with db() as owned:
            init_schema(owned)
            return
    for statement in SCHEMA_STATEMENTS:
        con.execute(statement)


@contextmanager
def db():
    with get_pool().connection() as con:
        try:
            yield con
        finally:
            if con.info.transaction_status != pq.TransactionStatus.IDLE:
                try:
                    con.execute("ROLLBACK")
                except Exception:
                    pass


def lock_evaluation(con, eval_id: str) -> None:
    con.execute("BEGIN")
    con.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (eval_id,))
