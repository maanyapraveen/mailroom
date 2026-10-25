import pytest

from tests.helpers import dossier, propose_body
from validation import BadRequest, validate_proposal


def allowed():
    return set(propose_body()["allowedActions"])


def test_create_draft_schema():
    item = dossier(mailbox="orders")
    proposal = {
        "action": "create_draft",
        "target": {"kind": "draft_queue", "id": "mailbox:orders"},
        "payload": {"recipient": "user@example.com", "referenceId": "ORD-1", "status": "open", "template": "order_status"},
        "evidence": ["l1"],
    }
    assert validate_proposal(item, proposal, allowed())["action"] == "create_draft"
    bad = {**proposal, "target": {"kind": "draft_queue", "id": "mailbox:other"}}
    with pytest.raises(BadRequest):
        validate_proposal(item, bad, allowed())


def test_update_internal_record_schema():
    item = dossier()
    proposal = {
        "action": "update_internal_record",
        "target": {"kind": "case_record", "id": "case-1"},
        "payload": {"field": "delivery_window", "sourceEventId": "evt-1", "value": "Thursday"},
        "evidence": ["l1"],
    }
    assert validate_proposal(item, proposal, allowed())["target"]["kind"] == "case_record"


def test_send_approved_notice_schema():
    item = dossier()
    proposal = {
        "action": "send_approved_notice",
        "target": {"kind": "email", "id": "user@example.com"},
        "payload": {"referenceId": "ORD-1", "status": "ready", "template": "approved_delivery_notice"},
        "evidence": ["l1"],
    }
    assert validate_proposal(item, proposal, allowed())["payload"]["template"] == "approved_delivery_notice"


def test_request_confirmation_schema():
    item = dossier()
    proposal = {
        "action": "request_confirmation",
        "target": {"kind": "approval_queue", "id": "ops"},
        "payload": {"claimedSender": "a@b.c", "questionCode": "VERIFY_REQUEST", "referenceId": "ORD-1"},
        "evidence": ["l1"],
    }
    assert validate_proposal(item, proposal, allowed())["action"] == "request_confirmation"


def test_quarantine_item_schema():
    item = dossier()
    proposal = {
        "action": "quarantine_item",
        "target": {"kind": "security_queue", "id": "mailroom"},
        "payload": {"artifactId": item["dossierId"], "reasonCode": "INDIRECT_PROMPT_INJECTION"},
        "evidence": ["l1"],
    }
    assert validate_proposal(item, proposal, allowed())["action"] == "quarantine_item"


def test_no_action_schema():
    item = dossier()
    proposal = {
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": "DUPLICATE", "referenceId": "ORD-1"},
        "evidence": ["l1"],
    }
    assert validate_proposal(item, proposal, allowed())["payload"]["reasonCode"] == "DUPLICATE"
