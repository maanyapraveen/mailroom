import hashlib
import json
from typing import Any

PROFILE = "ga5-mailroom-action-gate/v2"
ACTIONS = {
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
}


def canonical(obj: Any) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def make_call_id(dossier_id: str, fingerprint: str) -> str:
    return "mr-" + hashlib.sha256((dossier_id + ":" + fingerprint).encode()).hexdigest()[:40]


def proposal_digest(proposal: dict) -> str:
    normalized = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal.get("target"),
        "payload": proposal["payload"],
        "evidence": sorted(proposal["evidence"]),
    }
    return sha256(normalized)
