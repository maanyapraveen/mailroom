import asyncio
import json

import httpx

from model_client import request_model


def test_model_retries_rate_limit(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "https://example.test/openai/v1")
    monkeypatch.setenv("MODEL_NAME", "dummy-model")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-key")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "2")
    states = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        states["n"] += 1
        if states["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate"}})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def wrapper(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", wrapper)
    result = asyncio.run(request_model([{"role": "user", "content": "hi"}], 1000))
    assert result == {"ok": True}
    assert states["n"] == 2


def test_model_timeout_exhausted(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "https://example.test/openai/v1")
    monkeypatch.setenv("MODEL_NAME", "dummy-model")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-key")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def wrapper(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", wrapper)
    try:
        asyncio.run(request_model([{"role": "user", "content": "hi"}], 1000, attempts=2))
        raise AssertionError("expected timeout")
    except httpx.TimeoutException:
        pass


def test_missing_model_config_error(monkeypatch):
    monkeypatch.delenv("MODEL_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    try:
        asyncio.run(request_model([], 10, attempts=1))
        raise AssertionError("expected configuration error")
    except RuntimeError as exc:
        assert "Model configuration missing" in str(exc)
