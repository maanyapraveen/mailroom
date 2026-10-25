import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg import Error as PsycopgError
from psycopg.errors import UniqueViolation
import httpx

from canonical import PROFILE, canonical, make_call_id, sha256
from db import close_pool, configure, db, init_schema, lock_evaluation
from model_client import ModelOutputError, ModelPayloadTooLarge, call_model, call_model_batch
from policy import apply_safety_policy
from receipts import validate_receipts
from settings import MAX_BODY, MAX_RESPONSE
from validation import BadRequest, reject_duplicate_keys, validate_commit_envelope, validate_propose, validate_proposal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure()
    init_schema()
    yield
    close_pool()


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    redirect_slashes=False,
    lifespan=lifespan,
)


def error(status: int, detail: str):
    return JSONResponse({"error": detail}, status_code=status, headers={"Content-Type": "application/json"})


def _public_key_raw(verifier_json: str) -> bytes:
    import base64

    jwk = json.loads(verifier_json)["publicKeyJwk"]
    return base64.urlsafe_b64decode(jwk["x"] + "=" * (-len(jwk["x"]) % 4))


def _json_content_type(request: Request) -> bool:
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    return content_type in ("", "application/json")


async def _build_proposals(body: dict) -> list[dict]:
    proposals: list[dict | None] = []
    uncached = []
    for index, dossier in enumerate(body["dossiers"]):
        fingerprint = sha256(dossier)
        with db() as con:
            cached = con.execute(
                "SELECT proposal_json FROM decisions WHERE dossier_id=%s AND fingerprint=%s",
                (dossier["dossierId"], fingerprint),
            ).fetchone()
        if cached:
            proposal = json.loads(bytes(cached["proposal_json"]))
            if proposal["action"] not in body["allowedActions"]:
                raise BadRequest("Cached action is not allowed")
            proposals.append(proposal)
        else:
            uncached.append((index, dossier, fingerprint))
            proposals.append(None)

    stable_uncached = [item for item in uncached if item[1]["partition"] == "stable_core"]
    fresh_uncached = [item for item in uncached if item[1]["partition"] == "fresh_audit"]
    batch_decisions = {}
    if stable_uncached:
        batch_decisions = await call_model_batch([dossier for _, dossier, _ in stable_uncached])
    fresh_decisions = await _gather_fresh(fresh_uncached)
    decisions_by_index = {
        index: batch_decisions[dossier["dossierId"]] for index, dossier, _ in stable_uncached
    }
    decisions_by_index.update({item[0]: decision for item, decision in zip(fresh_uncached, fresh_decisions)})

    for index, dossier, fingerprint in uncached:
        raw_decision = decisions_by_index[index]
        try:
            decision = validate_proposal(dossier, raw_decision, body["allowedActions"])
        except BadRequest as exc:
            raise RuntimeError("Model returned an invalid proposal") from exc
        decision = apply_safety_policy(dossier, decision, set(body["allowedActions"]))
        proposal = {
            "dossierId": dossier["dossierId"],
            "callId": make_call_id(dossier["dossierId"], fingerprint),
            "action": decision["action"],
            "target": decision["target"],
            "payload": decision["payload"],
            "evidence": decision["evidence"],
        }
        with db() as con:
            con.execute(
                "INSERT INTO decisions(dossier_id,fingerprint,proposal_json) VALUES(%s,%s,%s) ON CONFLICT DO NOTHING",
                (dossier["dossierId"], fingerprint, canonical(proposal)),
            )
            row = con.execute(
                "SELECT proposal_json FROM decisions WHERE dossier_id=%s AND fingerprint=%s",
                (dossier["dossierId"], fingerprint),
            ).fetchone()
            proposal = json.loads(bytes(row["proposal_json"]))
        proposals[index] = proposal
    return proposals  # type: ignore[return-value]


async def _gather_fresh(fresh_uncached):
    return await asyncio.gather(*(call_model(dossier) for _, dossier, _ in fresh_uncached))


def _store_evaluation(eval_id: str, input_digest: str, request_bytes: bytes, verifier, response: dict):
    with db() as con:
        lock_evaluation(con, eval_id)
        existing = con.execute("SELECT * FROM evaluations WHERE evaluation_id=%s", (eval_id,)).fetchone()
        if existing:
            if bytes(existing["request_json"]) != request_bytes:
                con.execute("ROLLBACK")
                return error(409, "evaluationId reused with changed content")
            con.execute("ROLLBACK")
            return JSONResponse(json.loads(bytes(existing["response_json"])), headers={"Content-Type": "application/json"})
        try:
            con.execute(
                "INSERT INTO evaluations(evaluation_id,input_digest,request_json,verifier_json,response_json,status,created_at) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (
                    eval_id,
                    input_digest,
                    request_bytes,
                    json.dumps(verifier, separators=(",", ":")),
                    canonical(response),
                    "awaiting_receipts",
                    int(time.time()),
                ),
            )
            con.execute("COMMIT")
        except UniqueViolation:
            con.execute("ROLLBACK")
            existing = con.execute("SELECT * FROM evaluations WHERE evaluation_id=%s", (eval_id,)).fetchone()
            if existing and bytes(existing["request_json"]) == request_bytes:
                return JSONResponse(
                    json.loads(bytes(existing["response_json"])),
                    headers={"Content-Type": "application/json"},
                )
            return error(409, "evaluationId reused with changed content")
    return JSONResponse(response, headers={"Content-Type": "application/json"})


@app.get("/healthz")
def healthz():
    with db() as con:
        con.execute("SELECT 1")
    return {"status": "ok"}


@app.post("/")
async def endpoint(request: Request):
    try:
        if not _json_content_type(request):
            return error(415, "Unsupported media type")
        chunks = []
        body_size = 0
        async for chunk in request.stream():
            body_size += len(chunk)
            if body_size > MAX_BODY:
                return error(413, "Request body too large")
            chunks.append(chunk)
        raw_body = b"".join(chunks)
        try:
            body = json.loads(raw_body, object_pairs_hook=reject_duplicate_keys)
        except Exception:
            return error(400, "Malformed JSON")
        if not isinstance(body, dict):
            return error(400, "JSON envelope must be an object")
        op = body.get("operation")
        if op == "propose":
            validate_propose(body)
            eval_id = body["evaluationId"]
            input_digest = sha256(body["dossiers"])
            request_bytes = canonical(body)
            with db() as con:
                old = con.execute("SELECT * FROM evaluations WHERE evaluation_id=%s", (eval_id,)).fetchone()
                if old:
                    if bytes(old["request_json"]) != request_bytes:
                        return error(409, "evaluationId reused with changed content")
                    return JSONResponse(
                        json.loads(bytes(old["response_json"])),
                        headers={"Content-Type": "application/json"},
                    )
            proposals = await _build_proposals(body)
            response = {
                "profile": PROFILE,
                "evaluationId": eval_id,
                "status": "awaiting_receipts",
                "inputDigest": input_digest,
                "proposals": proposals,
            }
            if len(canonical(response)) > MAX_RESPONSE:
                return error(413, "Response body too large")
            return _store_evaluation(eval_id, input_digest, request_bytes, body["receiptVerifier"], response)
        if op == "commit":
            validate_commit_envelope(body)
            eval_id = body["evaluationId"]
            with db() as con:
                lock_evaluation(con, eval_id)
                ev = con.execute("SELECT * FROM evaluations WHERE evaluation_id=%s", (eval_id,)).fetchone()
                if not ev:
                    con.execute("ROLLBACK")
                    return error(409, "Unknown evaluation")
                prior = con.execute("SELECT * FROM commits WHERE evaluation_id=%s", (eval_id,)).fetchone()
                commit_digest = sha256(body)
                if prior:
                    con.execute("ROLLBACK")
                    if prior["request_digest"] != commit_digest:
                        return error(409, "Commit replay changed content")
                    return JSONResponse(
                        json.loads(bytes(prior["response_json"])),
                        headers={"Content-Type": "application/json"},
                    )
                if body["inputDigest"] != ev["input_digest"]:
                    con.execute("ROLLBACK")
                    return error(409, "inputDigest mismatch")
                saved = json.loads(bytes(ev["response_json"]))
                expected = {p["dossierId"]: p for p in saved["proposals"]}
                public_raw = _public_key_raw(ev["verifier_json"])
                try:
                    validated = validate_receipts(body, expected, public_raw)
                except BadRequest:
                    con.execute("ROLLBACK")
                    raise
                outcomes = []
                for receipt, _proposal in validated:
                    outcomes.append(
                        {
                            "dossierId": receipt["dossierId"],
                            "callId": receipt["callId"],
                            "action": receipt["action"],
                            "proposalDigest": receipt["proposalDigest"],
                            "receiptId": receipt["receiptId"],
                            "status": "executed" if receipt["accepted"] else "rejected",
                        }
                    )
                response = {
                    "profile": PROFILE,
                    "evaluationId": eval_id,
                    "status": "completed",
                    "inputDigest": body["inputDigest"],
                    "outcomes": outcomes,
                }
                if len(canonical(response)) > MAX_RESPONSE:
                    con.execute("ROLLBACK")
                    return error(413, "Response body too large")
                try:
                    con.execute("INSERT INTO commits VALUES(%s,%s,%s)", (eval_id, commit_digest, canonical(response)))
                    for (receipt, _proposal), outcome in zip(validated, outcomes):
                        con.execute(
                            "INSERT INTO receipts VALUES(%s,%s,%s,%s,%s)",
                            (eval_id, receipt["dossierId"], receipt["callId"], canonical(receipt), canonical(outcome)),
                        )
                    con.execute("UPDATE evaluations SET status='completed' WHERE evaluation_id=%s", (eval_id,))
                    con.execute("COMMIT")
                except UniqueViolation:
                    con.execute("ROLLBACK")
                    prior = con.execute("SELECT * FROM commits WHERE evaluation_id=%s", (eval_id,)).fetchone()
                    if prior and prior["request_digest"] == commit_digest:
                        return JSONResponse(
                            json.loads(bytes(prior["response_json"])),
                            headers={"Content-Type": "application/json"},
                        )
                    return error(409, "Commit replay changed content")
                return JSONResponse(response, headers={"Content-Type": "application/json"})
        return error(400, "Invalid operation")
    except BadRequest as exc:
        return error(422, str(exc))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return error(422, "Malformed schema")
    except PsycopgError:
        logger.exception("Database failure")
        return error(503, "Storage unavailable")
    except ModelOutputError:
        logger.warning("Model returned inconsistent or malformed output")
        return error(502, "Model returned invalid output")
    except ModelPayloadTooLarge:
        logger.warning("Model rejected request because its payload was too large")
        return error(413, "Dossier exceeds model provider request limit")
    except httpx.TimeoutException:
        logger.warning("Model timeout")
        return error(504, "Model service timeout")
    except httpx.HTTPError:
        return error(502, "Model service unavailable")
    except RuntimeError as exc:
        message = str(exc)
        if "Model configuration missing" in message or "Database" in message:
            return error(503, message)
        if "malformed" in message.lower() or "invalid proposal" in message.lower():
            return error(502, "Model returned invalid output")
        return error(503, "Upstream service unavailable")
    except Exception:
        logger.exception("Unhandled mailroom request failure")
        return error(500, "Internal server error")
