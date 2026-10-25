import base64

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from canonical import PROFILE, canonical, proposal_digest
from validation import SHA256_RE, BadRequest, require_keys


def verify_signature(public_key_raw: bytes, signed_bytes: bytes, signature_b64: str) -> bool:
    try:
        sig = base64.b64decode(signature_b64, validate=True)
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(sig, signed_bytes)
        return True
    except Exception:
        return False


def receipt_signed_bytes(evaluation_id: str, input_digest: str, receipt: dict) -> bytes:
    signed = {
        "profile": PROFILE,
        "evaluationId": evaluation_id,
        "inputDigest": input_digest,
        "receipt": {k: receipt[k] for k in ["dossierId", "callId", "action", "accepted", "proposalDigest", "receiptId"]},
    }
    return canonical(signed)


def validate_receipts(body, expected, public_raw: bytes):
    seen = set()
    seen_receipt_ids = set()
    seen_signatures = set()
    validated = []
    for receipt in body["receipts"]:
        require_keys(
            receipt,
            ["dossierId", "callId", "action", "accepted", "proposalDigest", "receiptId", "receiptSignature"],
        )
        did = receipt["dossierId"]
        if did in seen or did not in expected:
            raise BadRequest("Duplicate or unknown receipt")
        seen.add(did)
        proposal = expected[did]
        digest = proposal_digest(proposal)
        if receipt["callId"] != proposal["callId"] or receipt["action"] != proposal["action"] or receipt["proposalDigest"] != digest:
            raise BadRequest("Receipt scope mismatch")
        if not isinstance(receipt["accepted"], bool) or not isinstance(receipt["receiptId"], str) or not receipt["receiptId"]:
            raise BadRequest("Invalid receipt fields")
        if (
            receipt["receiptId"] in seen_receipt_ids
            or not isinstance(receipt["receiptSignature"], str)
            or receipt["receiptSignature"] in seen_signatures
        ):
            raise BadRequest("Duplicate receipt")
        if not SHA256_RE.fullmatch(receipt["proposalDigest"]):
            raise BadRequest("Invalid proposalDigest")
        seen_receipt_ids.add(receipt["receiptId"])
        seen_signatures.add(receipt["receiptSignature"])
        signed = receipt_signed_bytes(body["evaluationId"], body["inputDigest"], receipt)
        if not verify_signature(public_raw, signed, receipt["receiptSignature"]):
            raise BadRequest("Invalid receipt signature")
        validated.append((receipt, proposal))
    if len(validated) != len(expected):
        raise BadRequest("Receipt count mismatch")
    return validated
