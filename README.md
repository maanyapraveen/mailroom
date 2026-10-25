# Mailroom Agent

Implements the `ga5-mailroom-action-gate/v2` propose/commit HTTP contract.
The service uses PostgreSQL for durable state and an OpenAI-compatible chat
completions API for proposals; the supplied `.env.example` targets GroqCloud.

## Configuration

Copy `.env.example` to `.env` and set the credentials locally. `.env` is
excluded from Git and Docker build contexts; never commit credentials.

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | Yes | PostgreSQL connection URL. Use the provider's TLS-enabled URL. |
| `MODEL_URL` | Yes | OpenAI-compatible API base URL, or full `/chat/completions` URL. |
| `MODEL_NAME` | Yes | Model identifier accepted by the configured provider. |
| `MODEL_API_KEY` | Yes | API key for the model provider. |
| `DATABASE_SEARCH_PATH` | No | PostgreSQL schema to create and use. |
| `DATABASE_CONNECT_TIMEOUT` | No | PostgreSQL connection timeout in seconds (default `15`). |
| `MODEL_TIMEOUT_SECONDS` | No | Model request timeout in seconds (default `25`, capped at `40`). |
| `MAX_MODEL_CONCURRENCY` | No | Concurrent model requests (default `8`, limited to `1`–`16`). |
| `PORT` | No | HTTP listen port (default `10000` in Docker). |

The checked-in `.env.example` contains placeholders only. For local development,
install dependencies and run the server:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Edit .env with your PostgreSQL URL and model API key.
uvicorn main:app --host 127.0.0.1 --port 8000
```

`GET /healthz` checks application/database readiness. `POST /` accepts JSON
propose and commit envelopes; unsupported content types return `415`, malformed
or invalid envelopes return `400`/`422`, and upstream/database failures return
safe service errors. The OpenAPI documentation endpoints are disabled.

## Tests

Run the full suite with `python -m pytest -q` after installing
`requirements-dev.txt`. Unit tests run without external credentials. PostgreSQL
and GroqCloud integration tests require `DATABASE_URL` and
`MODEL_URL`/`MODEL_NAME`/`MODEL_API_KEY`; without those settings, pytest marks
the corresponding tests as skipped.

Integration tests use the `mailroom_agent_ci` schema and truncate its application
tables. Use a dedicated test database/schema: do not point the test suite at
production data.

## Docker

Build and start the service with credentials supplied at runtime, not baked into
the image:

```powershell
docker build -t mailroom-agent .
docker run --rm --env-file .env -p 10000:10000 mailroom-agent
```

The container runs as a non-root user and starts one Uvicorn worker. Configure
the service's port and secrets in the deployment platform. Keep PostgreSQL
backups and access controls enabled, terminate public traffic with HTTPS, and
apply appropriate request and network limits. The service does not include a
local database or a public deployment URL.

## Implemented safeguards

- Request/schema, duplicate-key, corpus-count, and body-size validation before
  model calls.
- Canonical JSON and SHA-256 digests.
- Frozen target/payload and line-ID validation for model proposals.
- Stable content-fingerprint decision cache and stable call IDs.
- PostgreSQL persistence for evaluations, proposals, receipts, and commit
  replays.
- Per-evaluation Ed25519 receipt signature and scope verification before
  outcomes are stored.
- Changed-content evaluation conflicts and exact replay responses.
