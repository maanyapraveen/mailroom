import asyncio
import json

import httpx

from settings import model_timeout, require_model_config

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-2.5-flash"


async def run_smoke_test(endpoint: str, model: str, api_key: str) -> int:
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": 'Return only this JSON object: {"smoke_test":"ok"}',
            }
        ],
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=model_timeout()) as client:
            response = await client.post(
                endpoint,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json=payload,
            )
    except httpx.TimeoutException:
        print("Gemini API result: request timed out.")
        return 1
    except httpx.RequestError:
        print("Gemini API result: network request failed.")
        return 1

    print(f"Gemini API result: HTTP {response.status_code}.")
    if not response.is_success:
        print("Gemini smoke test failed: provider returned an error.")
        return 1

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        result = json.loads(content)
    except (KeyError, TypeError, ValueError, IndexError):
        print("Gemini smoke test failed: response was malformed.")
        return 1

    if not isinstance(result, dict):
        print("Gemini smoke test failed: response was not a JSON object.")
        return 1

    print("Gemini smoke test passed: valid JSON response received.")
    return 0


def main() -> int:
    try:
        url, model, api_key = require_model_config()
    except RuntimeError:
        print("Gemini smoke test failed: model configuration is incomplete.")
        return 1

    if url.rstrip("/") != GEMINI_URL or model != GEMINI_MODEL:
        print("Gemini smoke test failed: configure the requested Gemini URL and model.")
        return 1

    endpoint = f"{GEMINI_URL}/chat/completions"
    return asyncio.run(run_smoke_test(endpoint, model, api_key))


if __name__ == "__main__":
    raise SystemExit(main())
