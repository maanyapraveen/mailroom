import base64
import re

from canonical import ACTIONS, PROFILE

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BadRequest(Exception):
    pass


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BadRequest("Duplicate JSON object key")
        result[key] = value
    return result


def require_keys(obj, required, optional=()):
    if not isinstance(obj, dict) or not set(required).issubset(obj) or set(obj) - set(required) - set(optional):
        raise BadRequest("Malformed schema")


def require_string(value, field, *, nonempty=True):
    if not isinstance(value, str) or (nonempty and not value):
        raise BadRequest("Invalid " + field)


def exact_object(obj, keys):
    return isinstance(obj, dict) and set(obj) == set(keys)


def validate_propose(body):
    require_keys(body, ["profile", "operation", "evaluationId", "receiptVerifier", "corpus", "allowedActions", "dossiers"])
    if body["profile"] != PROFILE or body["operation"] != "propose":
        raise BadRequest("Invalid profile or operation")
    require_string(body["evaluationId"], "evaluationId")
    rv = body["receiptVerifier"]
    require_keys(rv, ["algorithm", "publicKeyJwk"])
    if rv["algorithm"] != "Ed25519":
        raise BadRequest("Unsupported receipt verifier")
    jwk = rv["publicKeyJwk"]
    require_keys(jwk, ["kty", "crv", "x"])
    if jwk["kty"] != "OKP" or jwk["crv"] != "Ed25519":
        raise BadRequest("Invalid public key JWK")
    require_string(jwk["x"], "public key")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", jwk["x"]):
        raise BadRequest("Invalid public key encoding")
    try:
        raw = base64.urlsafe_b64decode(jwk["x"] + "=" * (-len(jwk["x"]) % 4))
        if len(raw) != 32:
            raise ValueError()
    except (ValueError, TypeError):
        raise BadRequest("Invalid public key encoding")
    corpus = body["corpus"]
    require_keys(corpus, ["coreId", "auditId", "stableCount", "freshCount"])
    require_string(corpus["coreId"], "corpus coreId")
    require_string(corpus["auditId"], "corpus auditId")
    if (
        type(corpus["stableCount"]) is not int
        or type(corpus["freshCount"]) is not int
        or corpus["stableCount"] < 0
        or corpus["freshCount"] < 0
        or corpus["stableCount"] + corpus["freshCount"] != len(body["dossiers"])
    ):
        raise BadRequest("Invalid corpus counts")
    if (
        not isinstance(body["allowedActions"], list)
        or set(body["allowedActions"]) != ACTIONS
        or len(body["allowedActions"]) != len(ACTIONS)
    ):
        raise BadRequest("Invalid allowedActions")
    if not isinstance(body["dossiers"], list) or not body["dossiers"]:
        raise BadRequest("dossiers must be a non-empty array")
    seen = set()
    stable_count = 0
    fresh_count = 0
    for d in body["dossiers"]:
        require_keys(d, ["dossierId", "partition", "receivedAt", "mailbox", "objective", "sources"])
        if not isinstance(d["dossierId"], str) or not d["dossierId"] or d["dossierId"] in seen:
            raise BadRequest("Invalid or duplicate dossierId")
        seen.add(d["dossierId"])
        if d["partition"] not in ("stable_core", "fresh_audit") or not isinstance(d["sources"], list) or not d["sources"]:
            raise BadRequest("Invalid dossier")
        if d["partition"] == "stable_core":
            stable_count += 1
        else:
            fresh_count += 1
        for key in ("receivedAt", "mailbox", "objective"):
            require_string(d[key], "dossier " + key, nonempty=(key != "objective"))
        line_ids = set()
        for s in d["sources"]:
            require_keys(s, ["sourceId", "kind", "provenance", "title", "lines"])
            for key in ("sourceId", "kind", "provenance", "title"):
                require_string(s[key], "source " + key)
            if not isinstance(s["lines"], list) or not s["lines"]:
                raise BadRequest("Invalid source lines")
            for line in s["lines"]:
                require_keys(line, ["lineId", "text"])
                if not isinstance(line["lineId"], str) or not line["lineId"] or line["lineId"] in line_ids:
                    raise BadRequest("Invalid or duplicate lineId")
                require_string(line["text"], "line text", nonempty=False)
                line_ids.add(line["lineId"])
    if stable_count != corpus["stableCount"] or fresh_count != corpus["freshCount"]:
        raise BadRequest("Corpus partition counts do not match dossiers")


def validate_commit_envelope(body):
    require_keys(body, ["profile", "operation", "evaluationId", "inputDigest", "receipts"])
    if body["profile"] != PROFILE or body["operation"] != "commit":
        raise BadRequest("Invalid commit schema")
    require_string(body["evaluationId"], "evaluationId")
    if not isinstance(body["inputDigest"], str) or not SHA256_RE.fullmatch(body["inputDigest"]):
        raise BadRequest("Invalid inputDigest")
    if not isinstance(body["receipts"], list):
        raise BadRequest("Invalid commit schema")


def validate_proposal(dossier, p, allowed):
    if not exact_object(p, ["action", "target", "payload", "evidence"]):
        raise BadRequest("Invalid model proposal schema")
    a, t, pl, ev = p["action"], p["target"], p["payload"], p["evidence"]
    if a not in ACTIONS or a not in allowed:
        raise BadRequest("Action not allowed")
    if not isinstance(ev, list) or not ev or any(not isinstance(x, str) for x in ev) or len(ev) != len(set(ev)):
        raise BadRequest("Invalid evidence")
    known = {line["lineId"] for s in dossier["sources"] for line in s["lines"]}
    if set(ev) - known:
        raise BadRequest("Unknown evidence lineId")
    if a == "create_draft":
        keys = ["recipient", "referenceId", "status", "template"]
        if (
            not exact_object(t, ["kind", "id"])
            or t["kind"] != "draft_queue"
            or t["id"] != "mailbox:" + dossier["mailbox"]
            or not exact_object(pl, keys)
            or pl["template"] != "order_status"
        ):
            raise BadRequest("Invalid create_draft schema")
        for key in ("recipient", "referenceId", "status"):
            require_string(pl[key], "create_draft " + key)
    elif a == "update_internal_record":
        if (
            not exact_object(t, ["kind", "id"])
            or t["kind"] != "case_record"
            or not isinstance(t["id"], str)
            or not t["id"]
            or not exact_object(pl, ["field", "sourceEventId", "value"])
            or pl["field"] != "delivery_window"
        ):
            raise BadRequest("Invalid update_internal_record schema")
        for key in ("sourceEventId", "value"):
            require_string(pl[key], "update_internal_record " + key)
    elif a == "send_approved_notice":
        if (
            not exact_object(t, ["kind", "id"])
            or t["kind"] != "email"
            or not isinstance(t["id"], str)
            or not t["id"]
            or not exact_object(pl, ["referenceId", "status", "template"])
            or pl["template"] != "approved_delivery_notice"
        ):
            raise BadRequest("Invalid send_approved_notice schema")
        require_string(pl["referenceId"], "send_approved_notice referenceId")
        require_string(pl["status"], "send_approved_notice status")
    elif a == "request_confirmation":
        if (
            not exact_object(t, ["kind", "id"])
            or t["kind"] != "approval_queue"
            or not isinstance(t["id"], str)
            or not t["id"]
            or not exact_object(pl, ["claimedSender", "questionCode", "referenceId"])
            or pl["questionCode"] != "VERIFY_REQUEST"
        ):
            raise BadRequest("Invalid request_confirmation schema")
        require_string(pl["claimedSender"], "request_confirmation claimedSender")
        require_string(pl["referenceId"], "request_confirmation referenceId")
    elif a == "quarantine_item":
        if (
            not exact_object(t, ["kind", "id"])
            or t != {"kind": "security_queue", "id": "mailroom"}
            or not exact_object(pl, ["artifactId", "reasonCode"])
            or pl["reasonCode"] != "INDIRECT_PROMPT_INJECTION"
        ):
            raise BadRequest("Invalid quarantine_item schema")
        require_string(pl["artifactId"], "quarantine_item artifactId")
    elif a == "no_action":
        if (
            t is not None
            or not exact_object(pl, ["reasonCode", "referenceId"])
            or pl["reasonCode"] not in {"ALREADY_COMPLETED", "DUPLICATE", "INFORMATIONAL"}
        ):
            raise BadRequest("Invalid no_action schema")
        require_string(pl["referenceId"], "no_action referenceId")
    return {"action": a, "target": t, "payload": pl, "evidence": ev}
