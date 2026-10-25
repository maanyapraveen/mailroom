import json

import pytest

from canonical import canonical, sha256
from tests.helpers import dossier, propose_body, source
from validation import BadRequest, validate_commit_envelope, validate_propose, validate_proposal


def test_valid_propose_request():
    validate_propose(propose_body())


def test_duplicate_dossier_ids():
    item = dossier(dossier_id="same")
    with pytest.raises(BadRequest):
        validate_propose(propose_body(dossiers=[item, {**item}]))


def test_missing_required_fields():
    body = propose_body()
    del body["corpus"]
    with pytest.raises(BadRequest):
        validate_propose(body)


def test_invalid_operation_on_commit_envelope():
    with pytest.raises(BadRequest):
        validate_commit_envelope(
            {
                "profile": "ga5-mailroom-action-gate/v2",
                "operation": "propose",
                "evaluationId": "e1",
                "inputDigest": "a" * 64,
                "receipts": [],
            }
        )


def test_unknown_and_duplicate_evidence_ids():
    item = dossier()
    allowed = set(propose_body()["allowedActions"])
    with pytest.raises(BadRequest, match="Unknown evidence"):
        validate_proposal(
            item,
            {
                "action": "no_action",
                "target": None,
                "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "r"},
                "evidence": ["missing"],
            },
            allowed,
        )
    with pytest.raises(BadRequest, match="Invalid evidence"):
        validate_proposal(
            item,
            {
                "action": "no_action",
                "target": None,
                "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "r"},
                "evidence": ["l1", "l1"],
            },
            allowed,
        )


def test_invalid_action_schema_extra_payload_field():
    item = dossier()
    allowed = set(propose_body()["allowedActions"])
    with pytest.raises(BadRequest):
        validate_proposal(
            item,
            {
                "action": "no_action",
                "target": None,
                "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "r", "extra": "nope"},
                "evidence": ["l1"],
            },
            allowed,
        )


def test_input_digest_is_canonical():
    dossiers = [dossier(dossier_id="d1"), dossier(dossier_id="d2")]
    assert sha256(dossiers) == sha256(json.loads(canonical(dossiers)))
    reordered_keys = json.loads(canonical(dossiers[0]))
    assert sha256(dossiers[0]) == sha256(reordered_keys)


def test_array_order_changes_digest():
    first = [dossier(dossier_id="a"), dossier(dossier_id="b")]
    second = [first[1], first[0]]
    assert sha256(first) != sha256(second)


def test_empty_source_lines_rejected():
    item = dossier(sources=[source("s1", "email", "untrusted_mailbox", "t", [])])
    with pytest.raises(BadRequest):
        validate_propose(propose_body(dossiers=[item]))
