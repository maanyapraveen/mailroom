import os

try:
    if os.getenv("MAILROOM_SKIP_DOTENV") != "1":
        from dotenv import load_dotenv

        load_dotenv()
except ImportError:
    pass

MAX_BODY = 512 * 1024
MAX_RESPONSE = 512 * 1024
MAX_MODEL_CONCURRENCY = min(max(int(os.getenv("MAX_MODEL_CONCURRENCY", "8")), 1), 16)


def database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("Database configuration missing: set DATABASE_URL")
    if url.startswith("sqlite"):
        raise RuntimeError("SQLite is not supported; set DATABASE_URL to PostgreSQL")
    if not url.startswith(("postgres://", "postgresql://")):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL URL")
    return url


def database_search_path() -> str | None:
    value = (os.getenv("DATABASE_SEARCH_PATH") or "").strip()
    return value or None


def model_url() -> str:
    return (os.getenv("MODEL_URL") or "").rstrip("/")


def model_name() -> str:
    return (os.getenv("MODEL_NAME") or "").strip()


def model_api_key() -> str:
    return (os.getenv("MODEL_API_KEY") or "").strip()


def model_timeout() -> float:
    return min(max(float(os.getenv("MODEL_TIMEOUT_SECONDS", "25")), 1), 40)


def require_model_config() -> tuple[str, str, str]:
    url, name, key = model_url(), model_name(), model_api_key()
    if not url or not name or not key:
        raise RuntimeError("Model configuration missing: set MODEL_URL, MODEL_NAME, MODEL_API_KEY")
    return url, name, key
