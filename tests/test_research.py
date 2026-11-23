"""tests/test_research.py - Unit tests for agents/research.py (M2/M3 research module)."""

from __future__ import annotations

import os
import sys
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Claim
from tools.fetch import FetchError
from agents.research import (
    build_search_queries,
    discover_vendors_from_text,
    extract_claims_from_page,
    run_research,
    run_research_a,
    run_research_b,
)
import db


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def silence_db(monkeypatch):
    """Silence DB event logging in unit tests."""
    monkeypatch.setattr(db, "log_event", lambda *args, **kwargs: None)


# ===========================================================================
# 1. Search Query Strategy Tests
# ===========================================================================

def test_query_formatting_never_starts_with_bare_noun():
    """Verify search queries use category-first format and never start with a bare noun like 'patient'."""
    queries = build_search_queries("compare patient appointment reminder software for Indian hospitals")
    assert len(queries) >= 3

    for q in queries:
        words = q.strip().split()
        first_word = words[0].lower()
        # Must not start with bare noun 'patient'
        assert first_word != "patient", f"Query '{q}' starts with bare noun 'patient'!"
        # Must contain category terms
        assert any(term in q.lower() for term in ["appointment", "hospital", "software", "reminder"])


def test_vendor_focus_query_formatting():
    """Verify queries generated with vendor_focus prioritize the specific vendor and category."""
    queries = build_search_queries("patient reminders", vendor_focus="HealthPlix")
    for q in queries:
        assert "healthplix" in q.lower()
        assert "patient" != q.strip().split()[0].lower()


# ===========================================================================
# 2. Vendor Discovery & Validation Tests
# ===========================================================================

def test_vendor_discovery_from_directory_text():
    """Verify identifying vendor names from directory/listicle snippets."""
    sample_directory_snippet = (
        "Top 10 Hospital Software in India: SourceForge ranks HealthPlix, Qikwell, and SparkTG "
        "as leading automated appointment reminder solutions with WhatsApp integration."
    )
    vendors = discover_vendors_from_text(sample_directory_snippet)
    assert "HealthPlix" in vendors
    assert "Qikwell" in vendors
    assert "SparkTG" in vendors


def test_vendor_name_must_appear_in_fetched_page():
    """Verify rule: only accept a vendor name that appears in a fetched page."""
    page_text = (
        "SparkTG provides enterprise cloud telephony and automated IVR appointment reminders "
        "for multispecialty hospitals across India with TRAI DLT approved headers."
    )
    url = "https://sparktg.com/healthcare"

    # 1. Vendor present in text -> accepted and claims extracted
    claims_valid = extract_claims_from_page("SparkTG", page_text, url)
    assert len(claims_valid) > 0
    assert all(c.vendor == "SparkTG" for c in claims_valid)

    # 2. Vendor absent from text -> rejected immediately, empty list returned
    claims_rejected = extract_claims_from_page("FakeVendorNotOnPage", page_text, url)
    assert claims_rejected == []


# ===========================================================================
# 3. Verbatim Snippet & Claim Extraction Tests
# ===========================================================================

def test_claims_contain_exact_verbatim_snippets():
    """Verify that every extracted claim has an exact verbatim snippet matching the source page."""
    page_text = (
        "HealthPlix offers an official WhatsApp Business API integration for automated appointment reminders. "
        "All patient transactional notifications comply with TRAI DLT registration guidelines. "
        "Subscription starts at ₹2500 per month for standard OPD clinics."
    )
    url = "https://healthplix.com/solutions"

    claims = extract_claims_from_page("HealthPlix", page_text, url)
    assert len(claims) >= 3

    attrs = {c.attribute: c for c in claims}
    assert "whatsapp_api" in attrs
    assert "dlt_sms" in attrs
    assert "price_per_month" in attrs

    for c in claims:
        assert isinstance(c, Claim)
        # Crucial for verifier: snippet must be a literal substring of page_text
        assert c.snippet in page_text, f"Snippet '{c.snippet}' is not a verbatim substring of source page!"


# ===========================================================================
# 4. FetchError Handling & Offline Unit Pipeline
# ===========================================================================

@pytest.mark.anyio
async def test_fetch_error_skips_cleanly_never_invents_text():
    """Verify rule: FetchError means the page is unusable: skip it, never invent text."""
    async def mock_search(q: str, max_results: int = 6):
        return [
            {"title": "Broken Paywall Page", "url": "https://paywall.com/reminders", "snippet": "HealthPlix details"},
            {"title": "Valid Source", "url": "https://valid.com/solutions", "snippet": "Qikwell reminder features"},
        ]

    async def mock_fetch(url: str):
        if "paywall" in url:
            raise FetchError("403 Forbidden: Subscription paywall", status_code=403)
        return (
            "Qikwell automated appointment reminders support WhatsApp Business API. "
            "Hospital integration with ABDM and ABHA health accounts is fully certified."
        )

    claims = await run_research(
        task="appointment reminder software hospital India",
        search_fn=mock_search,
        fetch_fn=mock_fetch,
    )

    # The paywall page should have been skipped without crashing or hallucinating claims
    assert len(claims) > 0
    # All claims should only come from the successfully fetched valid page
    assert all(c.vendor == "Qikwell" for c in claims)
    assert all("valid.com" in str(c.source_url) for c in claims)


@pytest.mark.anyio
async def test_vendor_focus_targeted_retry_run():
    """Verify targeted re-research using vendor_focus."""
    async def mock_search(q: str, max_results: int = 6):
        assert "mishtel" in q.lower()
        return [
            {"title": "Mishtel Reminders", "url": "https://mishtel.com/reminders", "snippet": "Mishtel features"}
        ]

    async def mock_fetch(url: str):
        return (
            "Mishtel Patient Appointment Reminder Software for Hospitals. "
            "Supports two-way confirmation and automated transactional SMS with TRAI DLT approved headers."
        )

    claims = await run_research(
        task="re-research mishtel appointment reminders",
        vendor_focus="Mishtel",
        search_fn=mock_search,
        fetch_fn=mock_fetch,
    )

    assert len(claims) >= 1
    assert all(c.vendor == "Mishtel" for c in claims)
    assert any(c.attribute == "dlt_sms" for c in claims)


@pytest.mark.anyio
async def test_research_tracks_features_and_pricing():
    """Verify track separation between Research A (features) and Research B (pricing)."""
    page_text = (
        "HealthPlix reminder suite supports WhatsApp Business API and EMR integration with Apollo. "
        "Starting plan is ₹3000 per month."
    )
    async def mock_search(q, max_results=6):
        return [{"title": "HealthPlix", "url": "https://healthplix.com/pricing", "snippet": "HealthPlix"}]

    async def mock_fetch(url):
        return page_text

    claims_a = await run_research_a(
        "hospital reminder",
        search_fn=mock_search,
        fetch_fn=mock_fetch,
    )
    # Research A: includes features, excludes price_per_month
    attrs_a = {c.attribute for c in claims_a}
    assert "whatsapp_api" in attrs_a or "emr_integration" in attrs_a
    assert "price_per_month" not in attrs_a

    claims_b = await run_research_b(
        "hospital reminder",
        search_fn=mock_search,
        fetch_fn=mock_fetch,
    )
    # Research B: includes pricing, excludes operational features
    attrs_b = {c.attribute for c in claims_b}
    assert "price_per_month" in attrs_b
    assert "emr_integration" not in attrs_b
