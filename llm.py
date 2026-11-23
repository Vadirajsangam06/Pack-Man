# llm.py
"""Gemini API wrapper for M1 orchestration.

Key features required by the contract:
* Global ``asyncio.Semaphore`` limiting concurrent calls (default 3, configurable via ``LLM_MAX_CONCURRENCY``).
* ``GEMINI_API_KEY`` read from the environment – missing key raises ``RuntimeError`` on real calls.
* Exponential back‑off with jitter for HTTP 429 and 5xx errors (max 5 attempts).
* ``call_json`` – request JSON output, validate against a supplied ``pydantic`` model, retry on malformed/invalid JSON up to 2 times, feeding the validation error back into the prompt.
* ``call_text`` – plain‑text variant.
* Model name configurable via ``GEMINI_MODEL``.
* Every call logs an event via :func:`db.log_event` for UI visibility.
"""

import asyncio
import json
import os
import random
import time
from typing import Any, Dict, Type, Optional

from pydantic import BaseModel, ValidationError

# Local import – ``db`` provides ``log_event``.
import db

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "3"))

# Global semaphore limiting parallel Gemini calls (lazy to avoid loop binding issues).
_SEMAPHORE: Optional[asyncio.Semaphore] = None

def get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        _SEMAPHORE = asyncio.Semaphore(_MAX_CONCURRENCY)
    return _SEMAPHORE

# ---------------------------------------------------------------------------
# Low‑level request – placeholder that can be monkey‑patched in tests.
# ---------------------------------------------------------------------------
async def _gemini_request(payload: Dict[str, Any]) -> Any:
    """Send *payload* to Gemini and return the decoded response.

    The real implementation would perform an HTTP POST with authentication.
    For unit tests this function is patched to simulate network behaviour.
    """
    key = payload.get("key") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY environment variable is required for llm.py")
    await asyncio.sleep(0.05)
    return {"echo": payload}

# ---------------------------------------------------------------------------
# Helper for exponential back‑off with jitter.
# ---------------------------------------------------------------------------
def _backoff(attempt: int) -> float:
    base = 0.5 * (2 ** (attempt - 1))  # 0.5, 1, 2, 4, 8 … seconds
    jitter = random.uniform(-0.1, 0.1)
    return max(0.0, base + jitter)

# ---------------------------------------------------------------------------
# Public API – JSON variant.
# ---------------------------------------------------------------------------
async def call_json(
    prompt: str,
    schema_model: Type[BaseModel],
    system: Optional[str] = None,
    max_attempts: int = 5,
) -> BaseModel:
    """Call Gemini expecting a JSON response that conforms to *schema_model*.

    Retries:
    * Network‑level (429 / 5xx) – up to *max_attempts* with back‑off.
    * Validation/malformed JSON errors – up to 2 additional attempts; the validation error
      is appended to the prompt for the next try.
    """
    attempt = 0
    validation_retries = 0
    sem = get_semaphore()
    current_prompt = prompt

    while attempt < max_attempts:
        attempt += 1
        async with sem:
            payload = {
                "model": _GEMINI_MODEL,
                "prompt": current_prompt,
                "system": system,
                "key": os.getenv("GEMINI_API_KEY") or _GEMINI_API_KEY,
                "response_format": "json",
            }
            try:
                raw = await _gemini_request(payload)
                if isinstance(raw, str):
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError as je:
                        if validation_retries < 2:
                            validation_retries += 1
                            current_prompt = f"{prompt}\n\nPrevious error: Invalid JSON ({je})\nPlease respond with valid JSON."
                            continue
                        else:
                            raise ValueError(f"Invalid JSON after retries: {je}")
                else:
                    data = raw

                # Validate with Pydantic
                try:
                    if hasattr(schema_model, "model_validate"):
                        result = schema_model.model_validate(data)
                    else:
                        result = schema_model.parse_obj(data)
                except ValidationError as ve:
                    if validation_retries < 2:
                        validation_retries += 1
                        current_prompt = f"{prompt}\n\nPrevious validation error: {ve}\nPlease respond with valid JSON."
                        continue
                    else:
                        raise

                # Log success and return.
                db.log_event(task_id=0, agent="llm", type="call_json_success", payload={"attempt": attempt, "model": _GEMINI_MODEL})
                return result
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in (429,) or (status and 500 <= status < 600):
                    if attempt >= max_attempts:
                        db.log_event(task_id=0, agent="llm", type="call_json_failure", payload={"attempt": attempt, "error": str(exc)})
                        raise
                    back = _backoff(attempt)
                    db.log_event(task_id=0, agent="llm", type="call_json_backoff", payload={"attempt": attempt, "backoff_s": back, "error": str(exc)})
                    await asyncio.sleep(back)
                    continue
                # If it's a validation or json error that hasn't exhausted retries, it already continued.
                # Otherwise, it's fatal.
                db.log_event(task_id=0, agent="llm", type="call_json_failure", payload={"attempt": attempt, "error": str(exc)})
                raise
    raise RuntimeError("call_json exhausted all attempts")

# ---------------------------------------------------------------------------
# Public API – plain‑text variant.
# ---------------------------------------------------------------------------
async def call_text(
    prompt: str,
    system: Optional[str] = None,
    max_attempts: int = 5,
) -> str:
    """Call Gemini and return raw text.

    Mirrors ``call_json`` but does not attempt JSON validation.
    """
    attempt = 0
    sem = get_semaphore()
    while attempt < max_attempts:
        attempt += 1
        async with sem:
            payload = {
                "model": _GEMINI_MODEL,
                "prompt": prompt,
                "system": system,
                "key": os.getenv("GEMINI_API_KEY") or _GEMINI_API_KEY,
            }
            try:
                raw = await _gemini_request(payload)
                text = raw if isinstance(raw, str) else json.dumps(raw)
                db.log_event(task_id=0, agent="llm", type="call_text_success", payload={"attempt": attempt, "model": _GEMINI_MODEL})
                return text
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in (429,) or (status and 500 <= status < 600):
                    if attempt >= max_attempts:
                        db.log_event(task_id=0, agent="llm", type="call_text_failure", payload={"attempt": attempt, "error": str(exc)})
                        raise
                    back = _backoff(attempt)
                    db.log_event(task_id=0, agent="llm", type="call_text_backoff", payload={"attempt": attempt, "backoff_s": back, "error": str(exc)})
                    await asyncio.sleep(back)
                    continue
                db.log_event(task_id=0, agent="llm", type="call_text_failure", payload={"attempt": attempt, "error": str(exc)})
                raise
    raise RuntimeError("call_text exhausted all attempts")

# ---------------------------------------------------------------------------
# Synchronous convenience wrappers for legacy callers.
# ---------------------------------------------------------------------------
def call_json_sync(prompt: str, schema_model: Type[BaseModel], system: Optional[str] = None) -> BaseModel:
    """Run :func:`call_json` in a blocking event loop."""
    return asyncio.run(call_json(prompt, schema_model, system))

def call_text_sync(prompt: str, system: Optional[str] = None) -> str:
    """Run :func:`call_text` in a blocking event loop."""
    return asyncio.run(call_text(prompt, system))
