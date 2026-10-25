import base64

import pytest

from canonical import PROFILE, proposal_digest
from receipts import validate_receipts, verify_signature
from tests.helpers import make_verifier, sign_receipt
from validation import BadRequest


def _proposal():
    return {
        "dossierId": "d1",
        "callId": "mr-abc",
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": "d1"},
        "evidence": ["l1"],
    }


def _public_raw(verifier):
    return base64.urlsafe_b64decode(verifier["publicKeyJwk"]["x"] + "==")


def test_valid_accepted_and_rejected_receipts():
    key, verifier = make_verifier()
    proposal = _proposal()
    accepted = sign_receipt(key, "e1", "a" * 64, proposal, accepted=True)
    rejected = sign_receipt(key, "e1", "a" * 64, {**proposal, "dossierId": "d2"}, accepted=False)
    expected = {"d1": proposal, "d2": {**proposal, "dossierId": "d2"}}
    body = {"evaluationId": "e1", "inputDigest": "a" * 64, "receipts": [accepted, rejected]}
    validated = validate_receipts(body, expected, _public_raw(verifier))
    assert [item[0]["accepted"] for item in validated] == [True, False]


def test_invalid_and_missing_signature():
    key, verifier = make_verifier()
    proposal = _proposal()
    receipt = sign_receipt(key, "e1", "a" * 64, proposal)
    receipt["receiptSignature"] = base64.b64encode(b"\x00" * 64).decode()
    with pytest.raises(BadRequest, match="Invalid receipt signature"):
        validate_receipts(
            {"evaluationId": "e1", "inputDigest": "a" * 64, "receipts": [receipt]},
            {"d1": proposal},
            _public_raw(verifier),
        )
    missing = sign_receipt(key, "e1", "a" * 64, proposal)
    del missing["receiptSignature"]
    with pytest.raises(BadRequest):
        validate_receipts(
            {"evaluationId": "e1", "inputDigest": "a" * 64, "receipts": [missing]},
            {"d1": proposal},
            _public_raw(verifier),
        )


def test_duplicate_receipt_rejected():
    key, verifier = make_verifier()
    proposal = _proposal()
    receipt = sign_receipt(key, "e1", "a" * 64, proposal)
    with pytest.raises(BadRequest, match="Duplicate"):
        validate_receipts(
            {"evaluationId": "e1", "inputDigest": "a" * 64, "receipts": [receipt, receipt]},
            {"d1": proposal},
            _public_raw(verifier),
        )


def test_changed_call_id_action_and_digest():
    key, verifier = make_verifier()
    proposal = _proposal()
    public = _public_raw(verifier)
    for overrides in (
        {"callId": "changed"},
        {"action": "quarantine_item"},
        {"proposalDigest": "b" * 64},
    ):
        receipt = sign_receipt(key, "e1", "a" * 64, proposal, **overrides)
        with pytest.raises(BadRequest, match="Receipt scope mismatch"):
            validate_receipts(
                {"evaluationId": "e1", "inputDigest": "a" * 64, "receipts": [receipt]},
                {"d1": proposal},
                public,
            )


def test_receipt_from_another_evaluation_fails_signature():
    key, verifier = make_verifier()
    proposal = _proposal()
    receipt = sign_receipt(key, "other-eval", "a" * 64, proposal)
    with pytest.raises(BadRequest, match="Invalid receipt signature"):
        validate_receipts(
            {"evaluationId": "e1", "inputDigest": "a" * 64, "receipts": [receipt]},
            {"d1": proposal},
            _public_raw(verifier),
        )


def test_verify_signature_helper_and_digest_stability():
    key, verifier = make_verifier()
    proposal = _proposal()
    digest = proposal_digest(proposal)
    assert digest == proposal_digest(proposal)
    receipt = sign_receipt(key, "e1", "a" * 64, proposal)
    from receipts import receipt_signed_bytes

    assert verify_signature(
        _public_raw(verifier),
        receipt_signed_bytes("e1", "a" * 64, receipt),
        receipt["receiptSignature"],
    )
    assert PROFILE.startswith("ga5-")
