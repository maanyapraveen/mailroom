import asyncio

import pytest

from tests.conftest import gemini_configured
from tests.helpers import dossier
from validation import BadRequest, validate_proposal

pytestmark = [
    pytest.mark.gemini,
    pytest.mark.skipif(not gemini_configured(), reason="BLOCKED: Gemini credentials are not configured"),
]


def test_real_gemini_json_object_and_dossier_schema():
    from model_client import call_model, request_model

    ping = asyncio.run(
        request_model(
            [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": '{"ping":true}'},
            ],
            1000,
        )
    )
    assert isinstance(ping, dict)

    item = dossier(
        objective="This is an FYI status copy. No action required.",
        sources=[
            {
                "sourceId": "email-1",
                "kind": "email",
                "provenance": "untrusted_mailbox",
                "title": "FYI",
                "lines": [
                    {"lineId": "l1", "text": "FYI: order ORD-9 was already completed yesterday."},
                    {"lineId": "l2", "text": "No action required."},
                ],
            }
        ],
    )
    result = asyncio.run(call_model(item))
    assert set(result) == {"action", "target", "payload", "evidence"}
    allowed = {
        "create_draft",
        "update_internal_record",
        "send_approved_notice",
        "request_confirmation",
        "quarantine_item",
        "no_action",
    }
    try:
        validate_proposal(item, result, allowed)
    except BadRequest as exc:
        pytest.fail(f"Gemini returned schema-invalid proposal: {result!r} ({exc})")


def test_no_secret_leak_in_model_errors(monkeypatch):
    from model_client import request_model

    monkeypatch.setenv("MODEL_API_KEY", "test_api_key_SHOULD_NOT_APPEAR")
    monkeypatch.setenv("MODEL_URL", "https://127.0.0.1:1/openai/v1")
    monkeypatch.setenv("MODEL_NAME", "dummy")
    monkeypatch.setenv("MODEL_TIMEOUT_SECONDS", "1")
    try:
        asyncio.run(request_model([{"role": "user", "content": "x"}], 100, attempts=1))
    except Exception as exc:
        assert "test_api_key_SHOULD_NOT_APPEAR" not in str(exc)
