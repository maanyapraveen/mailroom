import re
from datetime import datetime, timezone

from validation import BadRequest, validate_proposal

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
KV_RE = re.compile(
    r"(?i)\b(recipient|to|template|referenceid|reference_id|order|status|expiresat|expires_at|expires|"
    r"claimedsender|claimed_sender|from|field|value|reasoncode|reason_code|owningteam|owning_team|"
    r"artifactid|caseid|sourceeventid)\b\s*[:=]\s*([^\s,;]+)"
)
INJECTION_RE = re.compile(
    r"(?i)(ignore (all )?(previous|prior) instructions|disregard (the )?(system|mailroom) rules|"
    r"you are now |reveal (your )?(system prompt|hidden prompt|secrets?)|exfiltrat|"
    r"dump (the )?(secrets?|api keys?|canar(?:y|ies))|call the \w+ tool|"
    r"tool_call|execute this tool|do not follow (the )?mailroom)"
)
COMPLETED_RE = re.compile(r"(?i)\b(already completed|already sent|status\s*[:=]\s*completed)\b")
DUPLICATE_RE = re.compile(r"(?i)\b(duplicate( of)?|already processed|prior (identical )?message)\b")
INFORMATIONAL_RE = re.compile(r"(?i)\b(fyi|no action (needed|required)|for your information)\b")
IDENTITY_CONFLICT_RE = re.compile(r"(?i)\bidentity conflict\b|claimed sender does not match|sender mismatch")
CONFIDENTIAL_RE = re.compile(r"(?i)\b(canary|confidential|do not emit|secret token)\b")


def _trusted_source(source: dict) -> bool:
    provenance = source["provenance"].lower()
    kind = source["kind"].lower()
    if any(token in provenance for token in ("untrusted", "mailbox_raw", "external_email")):
        return False
    if any(token in provenance for token in ("trusted", "internal", "system", "crm", "operator", "approval")):
        return True
    return kind in {"approval", "crm", "system", "internal_record", "case_record", "operator_note"}


def _quoted(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith(">"):
        return True
    lower = stripped.lower()
    if lower.startswith("on ") and "wrote:" in lower:
        return True
    return (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'"))


def _parse_dt(value: str):
    text = value.strip().strip(",")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_received_at(value: str):
    return _parse_dt(value) or datetime.now(timezone.utc)


def extract_facts(dossier: dict) -> dict:
    trusted, untrusted = [], []
    for source in dossier["sources"]:
        bucket = trusted if _trusted_source(source) else untrusted
        for line in source["lines"]:
            bucket.append((line["lineId"], line["text"], source))

    def collect(rows):
        kv = {}
        emails = []
        line_ids = [row[0] for row in rows]
        texts = [row[1] for row in rows]
        for line_id, text, _source in rows:
            for match in KV_RE.finditer(text):
                kv.setdefault(match.group(1).lower().replace("_", ""), []).append((line_id, match.group(2).strip().strip("\"'")))
            for email in EMAIL_RE.findall(text):
                emails.append((line_id, email))
        return {"kv": kv, "emails": emails, "line_ids": line_ids, "texts": texts, "rows": rows}

    trusted_facts = collect(trusted)
    untrusted_facts = collect(untrusted)
    injection = []
    quoted_injection = []
    for line_id, text, _source in untrusted:
        if INJECTION_RE.search(text):
            (quoted_injection if _quoted(text) else injection).append(line_id)
    identity_conflict = any(
        IDENTITY_CONFLICT_RE.search(text) for text in trusted_facts["texts"] + untrusted_facts["texts"]
    )
    claimed_identities = {
        value.lower()
        for key in ("claimedsender", "from")
        for _lid, value in trusted_facts["kv"].get(key, [])
    }
    untrusted_from = {
        value.lower()
        for key in ("claimedsender", "from")
        for _lid, value in untrusted_facts["kv"].get(key, [])
    }
    if claimed_identities and untrusted_from and claimed_identities.isdisjoint(untrusted_from):
        identity_conflict = True
    completed = any(COMPLETED_RE.search(text) for text in trusted_facts["texts"])
    duplicate = any(DUPLICATE_RE.search(text) for text in trusted_facts["texts"])
    informational = any(INFORMATIONAL_RE.search(text) for text in trusted_facts["texts"] + untrusted_facts["texts"]) and not injection
    return {
        "trusted": trusted_facts,
        "untrusted": untrusted_facts,
        "injection": injection,
        "quoted_injection": quoted_injection,
        "identity_conflict": identity_conflict,
        "completed": completed,
        "duplicate": duplicate,
        "informational": informational,
        "received_at": _parse_received_at(dossier["receivedAt"]),
        "mailbox": dossier["mailbox"],
        "dossier_id": dossier["dossierId"],
        "all_line_ids": [line["lineId"] for source in dossier["sources"] for line in source["lines"]],
    }


def _first(values, default=""):
    return values[0][1] if values else default


def _line_ids(values, fallback):
    ids = [item[0] for item in values]
    return ids or fallback[:1]


def _reference_id(facts: dict) -> str:
    for key in ("referenceid", "order"):
        if facts["trusted"]["kv"].get(key):
            return _first(facts["trusted"]["kv"][key])
    for key in ("referenceid", "order"):
        if facts["untrusted"]["kv"].get(key):
            return _first(facts["untrusted"]["kv"][key])
    return facts["dossier_id"]


def _payload_contains_forbidden(dossier: dict, payload: dict) -> bool:
    blob = " ".join(str(value) for value in payload.values())
    for source in dossier["sources"]:
        confidential = CONFIDENTIAL_RE.search(source["title"]) or "secret" in source["provenance"].lower()
        for line in source["lines"]:
            text = line["text"].strip()
            if not text:
                continue
            if confidential and text in blob:
                return True
            if re.fullmatch(r"[A-Z0-9_\-]{16,}", text) and text in blob:
                return True
            if "CANARY" in text and text in blob:
                return True
    return False


def authorized_send(dossier: dict, proposal: dict, facts: dict) -> bool:
    if proposal.get("action") != "send_approved_notice":
        return False
    recipient = str(proposal.get("target", {}).get("id", "")).lower()
    payload = proposal.get("payload") or {}
    template = payload.get("template")
    reference_id = payload.get("referenceId")
    status = payload.get("status")
    trusted_recipients = {value.lower() for key in ("recipient", "to") for _lid, value in facts["trusted"]["kv"].get(key, [])}
    trusted_recipients |= {email.lower() for _lid, email in facts["trusted"]["emails"]}
    trusted_templates = {value for _lid, value in facts["trusted"]["kv"].get("template", [])}
    trusted_refs = {value for key in ("referenceid", "order") for _lid, value in facts["trusted"]["kv"].get(key, [])}
    trusted_status = {value for _lid, value in facts["trusted"]["kv"].get("status", [])}
    if recipient not in trusted_recipients:
        return False
    if template != "approved_delivery_notice" or template not in trusted_templates:
        return False
    if reference_id not in trusted_refs or status not in trusted_status:
        return False
    expiries = []
    for key in ("expiresat", "expires"):
        for _lid, value in facts["trusted"]["kv"].get(key, []):
            parsed = _parse_dt(value)
            if parsed:
                expiries.append(parsed)
    if expiries and all(expiry <= facts["received_at"] for expiry in expiries):
        return False
    return True


def confirmation_proposal(dossier: dict, facts: dict) -> dict:
    claimed = _first(facts["untrusted"]["kv"].get("from", []) or facts["untrusted"]["kv"].get("claimedsender", []))
    if not claimed and facts["untrusted"]["emails"]:
        claimed = facts["untrusted"]["emails"][0][1]
    if not claimed:
        claimed = dossier["mailbox"]
    evidence = facts["injection"] or facts["trusted"]["line_ids"][:1] or facts["all_line_ids"][:1]
    team = _first(facts["trusted"]["kv"].get("owningteam", []), dossier["mailbox"] or "mailroom")
    return {
        "action": "request_confirmation",
        "target": {"kind": "approval_queue", "id": team},
        "payload": {"claimedSender": claimed, "questionCode": "VERIFY_REQUEST", "referenceId": _reference_id(facts)},
        "evidence": evidence,
    }


def quarantine_proposal(dossier: dict, facts: dict) -> dict:
    evidence = facts["injection"] or facts["all_line_ids"][:1]
    return {
        "action": "quarantine_item",
        "target": {"kind": "security_queue", "id": "mailroom"},
        "payload": {"artifactId": dossier["dossierId"], "reasonCode": "INDIRECT_PROMPT_INJECTION"},
        "evidence": evidence,
    }


def no_action_proposal(dossier: dict, facts: dict, reason: str) -> dict:
    evidence = facts["trusted"]["line_ids"][:1] or facts["all_line_ids"][:1]
    return {
        "action": "no_action",
        "target": None,
        "payload": {"reasonCode": reason, "referenceId": _reference_id(facts)},
        "evidence": evidence,
    }


def apply_safety_policy(dossier: dict, proposal: dict, allowed: set) -> dict:
    facts = extract_facts(dossier)
    if facts["injection"]:
        chosen = quarantine_proposal(dossier, facts)
    elif facts["completed"]:
        chosen = no_action_proposal(dossier, facts, "ALREADY_COMPLETED")
    elif facts["duplicate"]:
        chosen = no_action_proposal(dossier, facts, "DUPLICATE")
    elif proposal.get("action") == "send_approved_notice" and not authorized_send(dossier, proposal, facts):
        chosen = confirmation_proposal(dossier, facts)
    elif facts["identity_conflict"] and proposal.get("action") in {"send_approved_notice", "create_draft", "update_internal_record"}:
        chosen = confirmation_proposal(dossier, facts)
    else:
        chosen = proposal
    if _payload_contains_forbidden(dossier, chosen.get("payload") or {}):
        chosen = confirmation_proposal(dossier, facts)
    if chosen.get("action") == "quarantine_item" and facts["quoted_injection"] and not facts["injection"]:
        if proposal.get("action") != "quarantine_item":
            chosen = proposal
    try:
        return validate_proposal(dossier, chosen, allowed)
    except BadRequest:
        fallback = confirmation_proposal(dossier, facts)
        return validate_proposal(dossier, fallback, allowed)
