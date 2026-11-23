# llm.py
"""Gemini API wrapper for M1 orchestrator.

The wrapper provides:
* A global ``asyncio.Semaphore`` limiting concurrent Gemini calls (default 10).
* Exponential back‑off on HTTP 429 responses.
* Validation of the LLM JSON output against a supplied ``pydantic`` model.
* Automatic retry when the returned payload is not valid JSON.

The actual network call is abstracted into ``_gemini_request`` – in production it
uses the Gemini REST endpoint, but for the current stub it simply returns a fake
JSON payload so the rest of the system can run tests without external access.
"""

import asyncio
import json
import os
import time
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

# Global semaphore limiting parallel calls (as required by the contract).
SEMAPHORE = asyncio.Semaphore(10)

# Base URL for Gemini – read from the environment for flexibility.
_GEMINI_BASE_URL = os.getenv("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1/models")
_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Internal low‑level request – replace with real HTTP call in production.
# ---------------------------------------------------------------------------
async def _gemini_request(payload: dict) -> dict:
    """Send *payload* to Gemini and return the decoded JSON response.

    This stub returns a deterministic JSON object that matches the expected
    schema of the caller. Real implementation would perform an HTTP POST with
    proper authentication and streaming handling.
    """
    # Simulate network latency.
    await asyncio.sleep(0.1)
    # Return a minimal echo payload – callers are expected to validate the
    # structure against their own ``BaseModel``.
    return {"echo": payload}

# ---------------------------------------------------------------------------
# Public helper used by other agents.
# ---------------------------------------------------------------------------
async def call_gemini(
    prompt: str,
    model: str = "gemini-1.5-flash",
    response_model: Type[BaseModel] | None = None,
    max_retries: int = 3,
) -> Any:
    """Call Gemini with *prompt* and optionally validate the JSON response.

    Parameters
    ----------
    prompt: str
        The user‑facing prompt to send to the model.
    model: str, optional
        Model identifier – defaults to the flash model.
    response_model: pydantic.BaseModel subclass, optional
        If provided, the raw JSON returned by Gemini will be parsed with
        ``response_model.parse_obj``. Validation errors trigger a retry.
    max_retries: int, optional
        Number of attempts before giving up (default 3).
    """
    async with SEMAPHORE:
        attempt = 0
        backoff = 0.5
        while True:
            attempt += 1
            payload = {
                "model": model,
                "prompt": prompt,
                "key": _API_KEY,
            }
            try:
                raw_resp = await _gemini_request(payload)
                # The stub returns a dict; in real life we'd decode a JSON string.
                if response_model is not None:
                    # Validate the structure using pydantic.
                    return response_model.parse_obj(raw_resp)
                return raw_resp
            except ValidationError as ve:
                if attempt >= max_retries:
                    raise
                # malformed JSON – retry after back‑off.
                await asyncio.sleep(backoff)
                backoff *= 2
            except Exception as exc:  # pragma: no cover – generic fallback.
                # Simulate handling of 429 and other transient errors.
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

# Convenience sync wrapper for legacy callers.
def call_gemini_sync(prompt: str, model: str = "gemini-1.5-flash", response_model: Type[BaseModel] | None = None) -> Any:
    """Run :func:`call_gemini` in an event loop for synchronous code paths."""
    return asyncio.run(call_gemini(prompt, model, response_model))
