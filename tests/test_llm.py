import sys, os, json, asyncio
from unittest import mock

import pytest
from pydantic import BaseModel

# Ensure project root is on sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm import call_json, call_text, _gemini_request
import db

@pytest.fixture
def anyio_backend():
    return "asyncio"

# Silence db.log_event during tests
@pytest.fixture(autouse=True)
def no_db_events(monkeypatch):
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)
    yield

# Simple schema for testing JSON validation
class DummySchema(BaseModel):
    echo: dict

# Helper to simulate _gemini_request behavior
async def fake_gemini_success(payload):
    return json.dumps({"echo": payload})

async def fake_gemini_malformed(payload):
    return "{invalid json"

async def fake_gemini_429_then_success(payload, call_counter=[0]):
    call_counter[0] += 1
    if call_counter[0] == 1:
        exc = Exception("rate limited")
        setattr(exc, "status_code", 429)
        raise exc
    return json.dumps({"echo": payload})

@pytest.mark.anyio
async def test_call_json_success(monkeypatch):
    monkeypatch.setattr('llm._gemini_request', fake_gemini_success)
    result = await call_json("test", DummySchema)
    assert result.echo["model"] == os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

@pytest.mark.anyio
async def test_call_json_malformed_retry(monkeypatch):
    calls = [fake_gemini_malformed, fake_gemini_success]
    async def side_effect(payload):
        fn = calls.pop(0)
        return await fn(payload)
    monkeypatch.setattr('llm._gemini_request', side_effect)
    result = await call_json("test", DummySchema)
    assert isinstance(result, DummySchema)

@pytest.mark.anyio
async def test_call_json_429_backoff(monkeypatch):
    monkeypatch.setattr('llm._gemini_request', fake_gemini_429_then_success)
    result = await call_json("test", DummySchema)
    assert isinstance(result, DummySchema)

@pytest.mark.anyio
async def test_semaphore_limits(monkeypatch):
    concurrent = 0
    max_concurrent = 0
    async def slow_success(payload):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.2)
        concurrent -= 1
        return json.dumps({"echo": payload})
    monkeypatch.setattr('llm._gemini_request', slow_success)
    tasks = [call_json(f"prompt {i}", DummySchema) for i in range(10)]
    await asyncio.gather(*tasks)
    assert max_concurrent <= int(os.getenv("LLM_MAX_CONCURRENCY", "3"))

@pytest.mark.anyio
async def test_real_api_smoke(monkeypatch):
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set – skipping real API smoke test")
    result = await call_text("Hello", system=None)
    assert isinstance(result, str)
