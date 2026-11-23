"""tests/test_analysis.py - Unit tests for M3 analysis, conflict detection, and India scoring."""

import sys, os
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Claim, Analysis, Conflict, VerificationResult
from agents.analysis import (
    analyze,
    analyze_async,
    detect_conflicts_deterministic,
    adjudicate_conflicts_with_llm,
    build_feature_matrix,
    score_india_criteria,
    generate_retry_list,
    calculate_metrics,
    AdjudicationResult,
    NeedsRetryItem,
)
from fixtures.m3_fixtures import (
    MATCHING_SNIPPET_CLAIM,
    SNIPPET_ABSENT_CLAIM,
    CONFLICTING_CLAIM_A,
    CONFLICTING_CLAIM_B,
    PAYWALLED_CLAIM,
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

def test_detect_conflicts_deterministic():
    """Verify deterministic detection of conflicting values for same vendor and attribute."""
    claims = [CONFLICTING_CLAIM_A, CONFLICTING_CLAIM_B, MATCHING_SNIPPET_CLAIM]
    conflicts, agreements = detect_conflicts_deterministic(claims)

    assert len(conflicts) == 1
    conf = conflicts[0]
    assert conf.vendor == "mPower Health"
    assert conf.attribute == "price_per_month"
    assert "₹2500" in conf.values
    assert "₹5000" in conf.values
    assert agreements == 1

def test_zero_conflicts_reporting():
    """Verify that when claims agree, report explicitly outputs '0 conflicts, N agreements'."""
    claims = [
        Claim(vendor="HealthPlix", attribute="pricing", value="₹2500", source_url="https://example.com/1", snippet="price is 2500"),
        Claim(vendor="HealthPlix", attribute="pricing", value="₹2500", source_url="https://example.com/2", snippet="price is 2500"),
    ]
    conflicts, agreements = detect_conflicts_deterministic(claims)
    assert len(conflicts) == 0
    assert agreements == 2

    res = analyze(claims)
    assert "0 conflicts, 2 agreements" in res.summary

def test_feature_matrix_unknown_cells():
    """Verify that unrecorded attributes are explicitly assigned 'unknown'."""
    claims = [
        Claim(vendor="VendorA", attribute="whatsapp", value="supported", source_url="https://a.com", snippet="a"),
        Claim(vendor="VendorB", attribute="pricing", value="₹4000", source_url="https://b.com", snippet="b"),
    ]
    matrix = build_feature_matrix(claims)

    assert matrix["VendorA"]["whatsapp"] == "supported"
    assert matrix["VendorA"]["pricing"] == "unknown"
    assert matrix["VendorB"]["whatsapp"] == "unknown"
    assert matrix["VendorB"]["pricing"] == "₹4000"

def test_score_india_criteria():
    """Verify scoring against the 7 Indian healthcare criteria."""
    claims = [
        Claim(vendor="HealthRemind", attribute="whatsapp_api", value="supported", source_url="https://h.com", snippet="whatsapp"),
        Claim(vendor="HealthRemind", attribute="dlt_sms", value="TRAI compliant", source_url="https://h.com", snippet="dlt"),
        Claim(vendor="HealthRemind", attribute="inr_pricing", value="₹3000/mo", source_url="https://h.com", snippet="pricing"),
    ]
    scorecard = score_india_criteria(claims)

    assert "HealthRemind" in scorecard
    sc = scorecard["HealthRemind"]
    assert sc["scores"]["whatsapp_api"] == "Supported"
    assert sc["scores"]["dlt_sms"] == "Supported"
    assert sc["scores"]["inr_pricing"] == "Supported"
    assert sc["scores"]["regional_languages"] == "unknown"
    assert "3/7" in sc["total_score"]

def test_generate_retry_list():
    """Verify retry item generation matches (vendor, attribute, reason) format."""
    claims = [MATCHING_SNIPPET_CLAIM, SNIPPET_ABSENT_CLAIM]
    verifications = [
        VerificationResult(claim_id=0, status="verified", details="ok"),
        VerificationResult(claim_id=1, status="failed", details="snippet_not_found"),
    ]
    retries = generate_retry_list(claims, verifications)

    assert len(retries) == 1
    assert isinstance(retries[0], NeedsRetryItem)
    assert retries[0].vendor == SNIPPET_ABSENT_CLAIM.vendor
    assert retries[0].attribute == SNIPPET_ABSENT_CLAIM.attribute
    assert "snippet_not_found" in retries[0].reason

def test_calculate_metrics():
    """Verify metrics object generation."""
    claims = [MATCHING_SNIPPET_CLAIM, SNIPPET_ABSENT_CLAIM]
    conflicts = [Conflict(vendor="V", attribute="A", values=["1", "2"], sources=["https://a.com", "https://b.com"])]
    verifications = [
        VerificationResult(claim_id=0, status="verified"),
        VerificationResult(claim_id=1, status="failed"),
    ]
    metrics = calculate_metrics(claims, conflicts, verifications)

    assert metrics.total_claims == 2
    assert metrics.verified_percentage == 50.0
    assert metrics.conflicts_found == 1
    assert metrics.citations_per_claim == 1.0

@pytest.mark.anyio
async def test_adjudicate_conflicts_with_llm(monkeypatch):
    """Test LLM adjudication of detected conflicts."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return AdjudicationResult(
            resolved_value="₹2500 for Standard tier, ₹5000 for Enterprise tier",
            explanation="The two values represent different subscription tiers."
        )

    monkeypatch.setattr(llm, "call_json", mock_call_json)

    claims = [CONFLICTING_CLAIM_A, CONFLICTING_CLAIM_B]
    conflicts, _ = detect_conflicts_deterministic(claims)
    adjudicated = await adjudicate_conflicts_with_llm(conflicts, claims)

    assert len(adjudicated) == 1
    assert "Standard tier" in adjudicated[0]["resolved_value"]

@pytest.mark.anyio
async def test_analyze_async_full_pipeline(monkeypatch):
    """Test full async analysis pipeline produces valid Analysis."""
    async def mock_call_json(prompt, schema_model, system=None, max_attempts=5):
        return AdjudicationResult(resolved_value="₹2500", explanation="Standard tier confirmed.")

    monkeypatch.setattr(llm, "call_json", mock_call_json)

    claims = [CONFLICTING_CLAIM_A, CONFLICTING_CLAIM_B, MATCHING_SNIPPET_CLAIM]
    res = await analyze_async(claims)

    assert isinstance(res, Analysis)
    assert len(res.conflicts) == 1
    assert "India Healthcare Compliance Scores" in res.summary
