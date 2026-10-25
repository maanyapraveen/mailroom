import base64
import copy
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from canonical import ACTIONS, PROFILE, canonical, proposal_digest
from receipts import receipt_signed_bytes

ALLOWED_ACTIONS = sorted(ACTIONS)


def new_id(prefix: str = "id") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def make_verifier():
    key = Ed25519PrivateKey.generate()
    raw = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    x = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    verifier = {"algorithm": "Ed25519", "publicKeyJwk": {"kty": "OKP", "crv": "Ed25519", "x": x}}
    return key, verifier


def source(source_id, kind, provenance, title, lines):
    return {
        "sourceId": source_id,
        "kind": kind,
        "provenance": provenance,
        "title": title,
        "lines": [{"lineId": line_id, "text": text} for line_id, text in lines],
    }


def dossier(
    dossier_id=None,
    partition="fresh_audit",
    mailbox="orders",
    received_at="2026-10-25T12:00:00Z",
    objective="Handle inbound mail",
    sources=None,
):
    if sources is None:
        sources = [
            source(
                "email-1",
                "email",
                "untrusted_mailbox",
                "Customer message",
                [("l1", "Please advise on order ORD-1 status.")],
            )
        ]
    return {
        "dossierId": dossier_id or new_id("dos"),
        "partition": partition,
        "receivedAt": received_at,
        "mailbox": mailbox,
        "objective": objective,
        "sources": sources,
    }


def propose_body(dossiers=None, evaluation_id=None, verifier=None, **overrides):
    if dossiers is None:
        dossiers = [dossier()]
    if verifier is None:
        _key, verifier = make_verifier()
    stable = sum(1 for item in dossiers if item["partition"] == "stable_core")
    fresh = len(dossiers) - stable
    body = {
        "profile": PROFILE,
        "operation": "propose",
        "evaluationId": evaluation_id or new_id("eval"),
        "receiptVerifier": verifier,
        "corpus": {
            "coreId": "core-1",
            "auditId": "audit-1",
            "stableCount": stable,
            "freshCount": fresh,
        },
        "allowedActions": ALLOWED_ACTIONS,
        "dossiers": dossiers,
    }
    body.update(overrides)
    return body


def informational_decision(item):
    line_id = item["sources"][0]["lines"][0]["lineId"]
    return {
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": item["dossierId"]},
        "evidence": [line_id],
    }


def sign_receipt(private_key, evaluation_id, input_digest, proposal, accepted=True, receipt_id=None, **overrides):
    receipt = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "accepted": accepted,
        "proposalDigest": proposal_digest(proposal),
        "receiptId": receipt_id or new_id("rcp"),
    }
    receipt.update(overrides)
    signed = receipt_signed_bytes(evaluation_id, input_digest, receipt)
    receipt["receiptSignature"] = base64.b64encode(private_key.sign(signed)).decode()
    return receipt


def commit_body(evaluation_id, input_digest, receipts):
    return {
        "profile": PROFILE,
        "operation": "commit",
        "evaluationId": evaluation_id,
        "inputDigest": input_digest,
        "receipts": receipts,
    }


def clone(obj):
    return copy.deepcopy(obj)
