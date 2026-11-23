"""tests/test_m3_stubs.py - Unit tests for M3 stubs and fixtures."""

import sys, os
import pytest

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import Claim, Analysis, VerificationResult
from agents.verifier import verify
from agents.analysis import analyze
from fixtures.m3_fixtures import (
    MATCHING_SNIPPET_CLAIM,
    SNIPPET_ABSENT_CLAIM,
    CONFLICTING_CLAIM_A,
    CONFLICTING_CLAIM_B,
    PAYWALLED_CLAIM,
)

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.mark.anyio
async def test_verifier_stub():
    """Verify that verify() returns valid list[VerificationResult]."""
    claims = [MATCHING_SNIPPET_CLAIM, SNIPPET_ABSENT_CLAIM, PAYWALLED_CLAIM]
    def dummy_fetch(url):
        return "Sample page content with Pricing starts at ₹2500 per month for small to medium healthcare practices."
    results = await verify(claims, fetch_fn=dummy_fetch)
    assert isinstance(results, list)
    assert len(results) == 3
    for i, res in enumerate(results):
        assert isinstance(res, VerificationResult)
        assert res.claim_id == i
        assert res.status in ("verified", "unverifiable", "failed")

def test_analysis_stub():
    """Verify that analyze() stub returns valid Analysis object."""
    claims = [MATCHING_SNIPPET_CLAIM, CONFLICTING_CLAIM_A, CONFLICTING_CLAIM_B]
    result = analyze(claims)
    assert isinstance(result, Analysis)
    assert result.agreements == 3
    assert isinstance(result.conflicts, list)
    assert "3 claims" in result.summary
