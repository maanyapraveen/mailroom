import json
import threading
import time

import pytest

from tests.conftest import postgres_configured

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not postgres_configured(), reason="BLOCKED: DATABASE_URL is not configured"),
]


@pytest.fixture
def pg():
    from db import close_pool, configure, db, init_schema

    configure()
    init_schema()
    with db() as con:
        con.execute("TRUNCATE receipts, commits, evaluations, decisions CASCADE")
    yield
    with db() as con:
        con.execute("TRUNCATE receipts, commits, evaluations, decisions CASCADE")
    close_pool()


def test_connection_and_schema(pg):
    from db import SCHEMA_STATEMENTS, db, init_schema

    with db() as con:
        row = con.execute("SELECT current_database() AS db, 1 AS ok").fetchone()
        assert row["ok"] == 1
        assert row["db"]
        tables = {
            item["table_name"]
            for item in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
            ).fetchall()
        }
        assert {"evaluations", "decisions", "receipts", "commits"} <= tables
        init_schema(con)
        assert "CREATE TABLE IF NOT EXISTS evaluations" in SCHEMA_STATEMENTS[0]


def test_create_retrieve_evaluation_and_proposals(pg):
    from db import db

    with db() as con:
        con.execute(
            "INSERT INTO evaluations VALUES (%s,%s,%s,%s,%s,%s,%s)",
            ("e1", "a" * 64, b"{}", "{}", b"{}", "awaiting_receipts", int(time.time())),
        )
        con.execute(
            "INSERT INTO decisions VALUES (%s,%s,%s)",
            ("d1", "f1", json.dumps({"action": "no_action"}).encode()),
        )
        evaluation = con.execute("SELECT * FROM evaluations WHERE evaluation_id=%s", ("e1",)).fetchone()
        decision = con.execute("SELECT * FROM decisions WHERE dossier_id=%s", ("d1",)).fetchone()
        assert evaluation["status"] == "awaiting_receipts"
        assert json.loads(bytes(decision["proposal_json"]))["action"] == "no_action"


def test_persist_receipts_and_commits(pg):
    from db import db

    with db() as con:
        con.execute(
            "INSERT INTO evaluations VALUES (%s,%s,%s,%s,%s,%s,%s)",
            ("e2", "b" * 64, b"{}", "{}", b"{}", "awaiting_receipts", int(time.time())),
        )
        con.execute("INSERT INTO commits VALUES (%s,%s,%s)", ("e2", "digest", b"{}"))
        con.execute("INSERT INTO receipts VALUES (%s,%s,%s,%s,%s)", ("e2", "d1", "c1", b"{}", b"{}"))
        assert con.execute("SELECT count(*) AS n FROM receipts").fetchone()["n"] == 1
        assert con.execute("SELECT request_digest FROM commits").fetchone()["request_digest"] == "digest"


def test_commit_and_rollback_transaction(pg):
    from db import db

    with db() as con:
        con.execute("BEGIN")
        con.execute(
            "INSERT INTO evaluations VALUES (%s,%s,%s,%s,%s,%s,%s)",
            ("e-roll", "c" * 64, b"{}", "{}", b"{}", "awaiting_receipts", int(time.time())),
        )
        con.execute("ROLLBACK")
        assert con.execute("SELECT count(*) AS n FROM evaluations WHERE evaluation_id=%s", ("e-roll",)).fetchone()["n"] == 0

    with db() as con:
        con.execute("BEGIN")
        con.execute(
            "INSERT INTO evaluations VALUES (%s,%s,%s,%s,%s,%s,%s)",
            ("e-commit", "d" * 64, b"{}", "{}", b"{}", "awaiting_receipts", int(time.time())),
        )
        con.execute("COMMIT")
        assert con.execute("SELECT count(*) AS n FROM evaluations WHERE evaluation_id=%s", ("e-commit",)).fetchone()["n"] == 1


def test_restart_persistence(pg):
    from db import close_pool, configure, db

    with db() as con:
        con.execute(
            "INSERT INTO evaluations VALUES (%s,%s,%s,%s,%s,%s,%s)",
            ("e-restart", "e" * 64, b"req", '{"algorithm":"Ed25519"}', b"resp", "awaiting_receipts", int(time.time())),
        )
    close_pool()
    configure()
    with db() as con:
        row = con.execute("SELECT * FROM evaluations WHERE evaluation_id=%s", ("e-restart",)).fetchone()
        assert bytes(row["request_json"]) == b"req"
        assert row["verifier_json"] == '{"algorithm":"Ed25519"}'


def test_duplicate_concurrent_inserts(pg):
    from db import db
    from psycopg.errors import UniqueViolation

    errors = []

    def insert():
        try:
            with db() as con:
                con.execute("BEGIN")
                con.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ("e-dupe",))
                existing = con.execute("SELECT 1 FROM evaluations WHERE evaluation_id=%s", ("e-dupe",)).fetchone()
                if existing:
                    con.execute("ROLLBACK")
                    return
                con.execute(
                    "INSERT INTO evaluations VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    ("e-dupe", "f" * 64, b"{}", "{}", b"{}", "awaiting_receipts", int(time.time())),
                )
                con.execute("COMMIT")
        except UniqueViolation:
            errors.append("unique")
        except Exception as exc:
            errors.append(type(exc).__name__)

    threads = [threading.Thread(target=insert) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with db() as con:
        assert con.execute("SELECT count(*) AS n FROM evaluations WHERE evaluation_id=%s", ("e-dupe",)).fetchone()["n"] == 1
    assert not errors or set(errors) <= {"unique"}
