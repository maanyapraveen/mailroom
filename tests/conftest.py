import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MAILROOM_SKIP_DOTENV"] = "1"

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ["DATABASE_SEARCH_PATH"] = "mailroom_agent_ci"
os.environ.setdefault("MODEL_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
os.environ.setdefault("MODEL_NAME", "gemini-2.5-flash")

from tests.helpers import informational_decision


def postgres_configured() -> bool:
    return bool((os.getenv("DATABASE_URL") or "").strip())


def gemini_configured() -> bool:
    return (
        bool((os.getenv("MODEL_API_KEY") or "").strip())
        and os.getenv("MODEL_URL", "").rstrip("/")
        == "https://generativelanguage.googleapis.com/v1beta/openai"
        and os.getenv("MODEL_NAME", "").strip() == "gemini-2.5-flash"
    )


@pytest.fixture
def mock_model(monkeypatch):
    async def fake_model(dossier):
        return informational_decision(dossier)

    async def fake_batch(dossiers):
        return {item["dossierId"]: informational_decision(item) for item in dossiers}

    monkeypatch.setattr("main.call_model", fake_model)
    monkeypatch.setattr("main.call_model_batch", fake_batch)
    monkeypatch.setattr("model_client.call_model", fake_model)
    monkeypatch.setattr("model_client.call_model_batch", fake_batch)
    return fake_model


@contextmanager
def _test_client():
    if not postgres_configured():
        pytest.skip("BLOCKED: DATABASE_URL is not configured")
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app, follow_redirects=False) as test_client:
        from db import db

        with db() as con:
            con.execute("TRUNCATE receipts, commits, evaluations, decisions CASCADE")
        yield test_client


@pytest.fixture
def client(mock_model):
    with _test_client() as test_client:
        yield test_client


@pytest.fixture
def raw_client():
    with _test_client() as test_client:
        yield test_client
