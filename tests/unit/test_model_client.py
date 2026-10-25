import asyncio
import json
import logging

import httpx
import pytest

from model_client import ModelPayloadTooLarge, call_model_batch, model_prompt, request_model
from tests.helpers import dossier


def _mock_transport(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def wrapper(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", wrapper)


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

    _mock_transport(monkeypatch, handler)
    result = asyncio.run(request_model([{"role": "user", "content": "hi"}], 1000))
    assert result == {"ok": True}
    assert states["n"] == 2


def test_model_payload_too_large_is_not_retried(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "https://example.test/openai/v1")
    monkeypatch.setenv("MODEL_NAME", "dummy-model")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-key")
    states = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        states["n"] += 1
        return httpx.Response(413, request=request)

    _mock_transport(monkeypatch, handler)
    try:
        asyncio.run(request_model([{"role": "user", "content": "too large"}], 1000))
    except ModelPayloadTooLarge as exc:
        assert str(exc) == "Model input exceeds provider request limit"
    else:
        raise AssertionError("expected a safe payload-size error")

    assert states["n"] == 1


def test_model_timeout_exhausted(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "https://example.test/openai/v1")
    monkeypatch.setenv("MODEL_NAME", "dummy-model")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-key")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timeout")

    _mock_transport(monkeypatch, handler)
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


def test_gemini_endpoint_model_and_environment_key_are_used_without_logging(monkeypatch, caplog):
    secret = "test-gemini-key-do-not-log"
    monkeypatch.setenv("MODEL_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("MODEL_API_KEY", secret)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    _mock_transport(monkeypatch, handler)
    caplog.set_level(logging.WARNING)

    assert asyncio.run(request_model([{"role": "user", "content": "ping"}], 1000)) == {"ok": True}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert request.headers["Authorization"] == f"Bearer {secret}"
    payload = json.loads(request.content)
    assert payload["model"] == "gemini-2.5-flash"
    assert payload["response_format"] == {"type": "json_object"}
    assert secret not in caplog.text


def test_gemini_provider_error_is_safe_and_not_retried(monkeypatch, caplog):
    secret = "test-gemini-key-do-not-log"
    monkeypatch.setenv("MODEL_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("MODEL_API_KEY", secret)
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            401,
            request=request,
            json={"error": {"message": secret}},
        )

    _mock_transport(monkeypatch, handler)

    with pytest.raises(httpx.HTTPStatusError) as error:
        asyncio.run(request_model([{"role": "user", "content": "ping"}], 1000))

    assert error.value.response.status_code == 401
    assert len(requests) == 1
    assert secret not in str(error.value)
    assert secret not in caplog.text


def test_gemini_malformed_response_fails_safely(monkeypatch):
    secret = "test-gemini-key-do-not-log"
    monkeypatch.setenv("MODEL_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("MODEL_API_KEY", secret)

    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{invalid-json"}}]},
        )

    _mock_transport(monkeypatch, handler)

    with pytest.raises(RuntimeError, match="Model returned malformed output") as error:
        asyncio.run(request_model([{"role": "user", "content": "ping"}], 1000, attempts=1))

    assert len(requests) == 1
    assert "{invalid-json" not in str(error.value)
    assert secret not in str(error.value)


def test_gemini_retries_a_malformed_response_with_bounded_attempts(monkeypatch):
    monkeypatch.setenv("MODEL_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-key")
    responses = iter(
        (
            httpx.Response(200, json={"choices": [{"message": {"content": "{bad-json"}}]}),
            httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}}]}),
        )
    )
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return next(responses)

    _mock_transport(monkeypatch, handler)

    result = asyncio.run(request_model([{"role": "user", "content": "ping"}], 1000, attempts=2))

    assert result == {"ok": True}
    assert len(requests) == 2


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
            raise ModelPayloadTooLarge("Model input exceeds provider request limit")
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


def test_model_batch_stops_after_single_dossier_payload_too_large(monkeypatch):
    item = dossier(dossier_id="oversized-dossier", partition="stable_core")
    attempts = []

    async def fake_request_model(messages, _max_output_chars):
        input_dossiers = json.loads(messages[1]["content"].split("Dossiers:\n", 1)[1])
        attempts.append(input_dossiers)
        raise ModelPayloadTooLarge("Model input exceeds provider request limit")

    monkeypatch.setattr("model_client.request_model", fake_request_model)

    try:
        asyncio.run(call_model_batch([item]))
    except ModelPayloadTooLarge:
        pass
    else:
        raise AssertionError("expected a payload-size error for a single dossier")

    assert 1 < len(attempts) <= 64
    assert all(len(batch) == 1 for batch in attempts)


def test_model_single_dossier_is_split_without_losing_text_or_evidence(monkeypatch):
    item = dossier(dossier_id="long-dossier", partition="fresh_audit")
    long_text = "Order update with complete decision context. " * 40
    item["sources"][0]["lines"] = [{"lineId": "long-line", "text": long_text}]
    split_texts = []

    async def fake_request_model(messages, _max_output_chars):
        input_dossier = json.loads(messages[1]["content"].split("Dossiers:\n", 1)[1])[0]
        lines = input_dossier["sources"][0]["lines"]
        text = "".join(line["text"] for line in lines)
        if len(text) > 80:
            raise ModelPayloadTooLarge("Model input exceeds provider request limit")
        split_texts.append(text)
        return {
            "decisions": [
                {
                    "dossierId": input_dossier["dossierId"],
                    "action": "no_action",
                    "target": None,
                    "payload": {"reasonCode": "INFORMATIONAL", "referenceId": input_dossier["dossierId"]},
                    "evidence": [lines[0]["lineId"]],
                }
            ]
        }

    monkeypatch.setattr("model_client.request_model", fake_request_model)

    decision = asyncio.run(call_model_batch([item]))[item["dossierId"]]

    assert "".join(split_texts) == long_text
    assert decision["action"] == "no_action"
    assert decision["evidence"] == ["long-line"]


def test_model_split_rejects_inconsistent_chunk_decisions(monkeypatch):
    item = dossier(dossier_id="conflicting-dossier", partition="stable_core")
    item["sources"][0]["lines"] = [
        {"lineId": "line-1", "text": "First independent context."},
        {"lineId": "line-2", "text": "Second independent context."},
    ]
    actions = iter(("no_action", "request_confirmation"))
    initial_request = True

    async def fake_request_model(messages, _max_output_chars):
        nonlocal initial_request
        input_dossier = json.loads(messages[1]["content"].split("Dossiers:\n", 1)[1])[0]
        if initial_request:
            initial_request = False
            raise ModelPayloadTooLarge("Model input exceeds provider request limit")
        line_id = input_dossier["sources"][0]["lines"][0]["lineId"]
        action = next(actions)
        payload = (
            {"reasonCode": "INFORMATIONAL", "referenceId": input_dossier["dossierId"]}
            if action == "no_action"
            else {
                "claimedSender": "sender@example.com",
                "questionCode": "VERIFY_REQUEST",
                "referenceId": input_dossier["dossierId"],
            }
        )
        target = None if action == "no_action" else {"kind": "approval_queue", "id": "orders"}
        return {
            "decisions": [
                {
                    "dossierId": input_dossier["dossierId"],
                    "action": action,
                    "target": target,
                    "payload": payload,
                    "evidence": [line_id],
                }
            ]
        }

    monkeypatch.setattr("model_client.request_model", fake_request_model)

    from model_client import ModelOutputError

    try:
        asyncio.run(call_model_batch([item]))
    except ModelOutputError as exc:
        assert "inconsistent decisions" in str(exc)
    else:
        raise AssertionError("inconsistent chunk decisions must not produce a proposal")
