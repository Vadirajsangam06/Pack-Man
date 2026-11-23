"""tests/test_verifier.py - Unit tests for M3 deterministic and LLM verification."""

import sys, os
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Claim, VerificationResult
from agents.verifier import verify, verify_claim, normalize_text, fuzzy_match_snippet, ClaimSupportDecision
from fixtures.m3_fixtures import (
    MATCHING_SNIPPET_CLAIM,
    SNIPPET_ABSENT_CLAIM,
    PAYWALLED_CLAIM,
    MOCK_PAGES,
)
import llm
import db

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
def silence_db(monkeypatch):
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)
    yield

def mock_fetch(url: str):
    """Mock fetcher returning preset content from MOCK_PAGES."""
    return MOCK_PAGES.get(url)

@pytest.mark.anyio
async def test_verify_matching_snippet(monkeypatch):
    """Test verification when snippet is present on page and LLM supports it."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return ClaimSupportDecision(supports=True, reason="Snippet explicitly states pricing of ₹2500 per month.")

    monkeypatch.setattr(llm, "call_json", mock_call_json)

    results = await verify([MATCHING_SNIPPET_CLAIM], fetch_fn=mock_fetch)
    assert len(results) == 1
    res = results[0]
    assert res.claim_id == 0
    assert res.status == "verified"
    assert "₹2500" in res.details

@pytest.mark.anyio
async def test_verify_snippet_not_found(monkeypatch):
    """Test verification when snippet is absent from the fetched page."""
    results = await verify([SNIPPET_ABSENT_CLAIM], fetch_fn=mock_fetch)
    assert len(results) == 1
    res = results[0]
    assert res.status == "failed"
    assert "snippet_not_found" in res.details

@pytest.mark.anyio
async def test_verify_unverifiable_paywall(monkeypatch):
    """Test verification when page is paywalled or unfetchable."""
    def paywall_fetch(url):
        return None  # simulates HTTP 403 / connection failure

    results = await verify([PAYWALLED_CLAIM], fetch_fn=paywall_fetch)
    assert len(results) == 1
    res = results[0]
    assert res.status == "unverifiable"
    assert "unverifiable" in res.details

@pytest.mark.anyio
async def test_verify_snippet_matches_but_unsupported_by_llm(monkeypatch):
    """Test snippet present on page, but LLM determines it contradicts/doesn't support claim."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return ClaimSupportDecision(supports=False, reason="Snippet mentions standard tier, but claim refers to enterprise tier.")

    monkeypatch.setattr(llm, "call_json", mock_call_json)

    results = await verify([MATCHING_SNIPPET_CLAIM], fetch_fn=mock_fetch)
    assert len(results) == 1
    res = results[0]
    assert res.status == "failed"
    assert "unsupported" in res.details

def test_normalization_and_fuzzy_matching():
    """Test text normalization and fuzzy matching resilience."""
    snippet = "Pricing starts at   ₹2500\nper month."
    page = "Welcome! pricing starts at ₹2500 per month for all clinics."
    assert fuzzy_match_snippet(snippet, page) is True

    absent_snippet = "Completely different text."
    assert fuzzy_match_snippet(absent_snippet, page) is False

@pytest.mark.anyio
async def test_real_api_smoke_verifier():
    """Optional smoke test hitting real Gemini API, skipped if GEMINI_API_KEY is unset."""
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set – skipping verifier real API smoke test")

    claim = Claim(
        vendor="HealthPlix",
        attribute="appointment_reminders",
        value="WhatsApp reminders supported",
        source_url="https://healthplix.com",
        snippet="HealthPlix automates patient reminders directly via WhatsApp and SMS."
    )
    def simple_fetch(url):
        return "HealthPlix automates patient reminders directly via WhatsApp and SMS."

    results = await verify([claim], fetch_fn=simple_fetch)
    assert len(results) == 1
    assert results[0].status == "verified"
