import asyncio
import json

import httpx

from model_client import call_model_batch, model_prompt, request_model
from tests.helpers import dossier


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


def test_model_prompt_keeps_decision_context_and_omits_internal_metadata():
    item = dossier(partition="stable_core")
    item["sources"][0]["sourceId"] = "internal-source-id"

    message = model_prompt(item)
    prompt_data = json.loads(message["content"].split("Dossier:\n", 1)[1])

    assert prompt_data == {
        "dossierId": item["dossierId"],
        "receivedAt": item["receivedAt"],
        "mailbox": item["mailbox"],
        "objective": item["objective"],
        "sources": [
            {
                "kind": "email",
                "provenance": "untrusted_mailbox",
                "title": "Customer message",
                "lines": [{"lineId": "l1", "text": "Please advise on order ORD-1 status."}],
            }
        ],
    }
    assert "internal-source-id" not in message["content"]
    assert "stable_core" not in message["content"]


def test_model_batch_splits_on_payload_too_large(monkeypatch):
    items = [dossier(dossier_id=f"dossier-{index}", partition="stable_core") for index in range(3)]
    batch_sizes = []

    async def fake_request_model(messages, _max_output_chars):
        user_content = messages[1]["content"]
        input_dossiers = json.loads(user_content.split("Dossiers:\n", 1)[1])
        batch_sizes.append(len(input_dossiers))
        if len(input_dossiers) > 1:
            request = httpx.Request("POST", "https://example.test/chat/completions")
            response = httpx.Response(413, request=request)
            raise httpx.HTTPStatusError("Payload too large", request=request, response=response)
        return {
            "decisions": [
                {
                    "dossierId": input_dossiers[0]["dossierId"],
                    "action": "no_action",
                    "target": None,
                    "payload": {"reasonCode": "INFORMATIONAL", "referenceId": input_dossiers[0]["dossierId"]},
                    "evidence": [input_dossiers[0]["sources"][0]["lines"][0]["lineId"]],
                }
            ]
        }

    monkeypatch.setattr("model_client.request_model", fake_request_model)

    decisions = asyncio.run(call_model_batch(items))

    assert batch_sizes == [3, 1, 2, 1, 1]
    assert set(decisions) == {item["dossierId"] for item in items}
    assert all(decisions[item["dossierId"]]["evidence"] == ["l1"] for item in items)
