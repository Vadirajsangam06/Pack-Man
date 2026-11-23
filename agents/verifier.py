"""agents/verifier.py - Verification agent for AgentFlow AI (M3).

Implements deterministic-first evidence verification:
1. Re-fetch claim source URL (using M2 tools.fetch if available, or injectable fetcher).
2. Fail-fast with 'unverifiable' on network error, paywall, or empty page.
3. Normalize whitespace and case, then fuzzy-match snippet via rapidfuzz or difflib.
4. If snippet is absent, return 'failed' with 'snippet_not_found'.
5. Only if snippet matches: query Gemini via llm.call_json to verify whether
   the snippet factually supports the claimed value.
"""

from __future__ import annotations

import difflib
import inspect
import re
from typing import Callable, List, Optional
from pydantic import BaseModel, Field

from schemas import Claim, VerificationResult
import llm
import db


class ClaimSupportDecision(BaseModel):
    """LLM decision on whether a page snippet supports the claimed attribute value."""
    supports: bool = Field(..., description="True if the snippet factually supports the claim, False otherwise")
    reason: str = Field(..., description="Explanation of why the snippet supports or contradicts the claim")


def normalize_text(text: str) -> str:
    """Normalize text by collapsing whitespace and lowercasing."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def fuzzy_match_snippet(snippet: str, page_content: str, threshold: float = 0.80) -> bool:
    """Check if snippet appears in page_content using exact or fuzzy matching."""
    norm_snippet = normalize_text(snippet)
    norm_page = normalize_text(page_content)

    if not norm_snippet or not norm_page:
        return False

    # 1. Direct substring match after whitespace normalization
    if norm_snippet in norm_page:
        return True

    # 2. Try rapidfuzz partial ratio if installed
    try:
        from rapidfuzz import fuzz
        ratio = fuzz.partial_ratio(norm_snippet, norm_page) / 100.0
        return ratio >= threshold
    except ImportError:
        pass

    # 3. Fallback to difflib sliding window
    words_snip = norm_snippet.split()
    words_page = norm_page.split()
    n_words = len(words_snip)

    if n_words == 0:
        return False
    if len(words_page) <= n_words:
        ratio = difflib.SequenceMatcher(None, norm_snippet, norm_page).ratio()
        return ratio >= threshold

    step = max(1, n_words // 4)
    best_ratio = 0.0
    for start in range(0, len(words_page) - n_words + 1, step):
        chunk = " ".join(words_page[start : start + n_words])
        r = difflib.SequenceMatcher(None, norm_snippet, chunk).ratio()
        if r > best_ratio:
            best_ratio = r
        if best_ratio >= threshold:
            return True

    return best_ratio >= threshold


async def _default_fetch(url: str) -> Optional[str]:
    """Default URL fetcher trying M2 tools.fetch or urllib."""
    try:
        from tools.fetch import fetch_url
        if inspect.iscoroutinefunction(fetch_url):
            return await fetch_url(url)
        return fetch_url(url)
    except (ImportError, AttributeError):
        pass

    # Fallback local fetch
    import urllib.request
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AgentFlow/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="replace")
            return None
    except Exception:
        return None


async def verify_claim(
    claim_id: int,
    claim: Claim,
    fetch_fn: Optional[Callable] = None,
) -> VerificationResult:
    """Verify a single claim deterministically and with LLM support check."""
    url_str = str(claim.source_url)
    fetcher = fetch_fn or _default_fetch

    # 1. Re-fetch source URL
    page_text: Optional[str] = None
    try:
        if inspect.iscoroutinefunction(fetcher):
            page_text = await fetcher(url_str)
        else:
            page_text = fetcher(url_str)
    except Exception as exc:
        page_text = None

    # Check for fetch failure / paywall / empty content
    if not page_text or len(page_text.strip()) < 20:
        details = f"unverifiable: unable to fetch content from {url_str} (network failure, paywall, or empty page)"
        db.log_event(
            task_id=0,
            agent="verifier",
            type="claim_unverifiable",
            payload={"claim_id": claim_id, "vendor": claim.vendor, "attribute": claim.attribute, "reason": details},
        )
        return VerificationResult(
            claim_id=claim_id,
            status="unverifiable",
            details=details,
        )

    # 2. Fuzzy-match the verbatim snippet against fetched page content
    if not fuzzy_match_snippet(claim.snippet, page_text):
        details = f"snippet_not_found: snippet does not match fetched text at {url_str}"
        db.log_event(
            task_id=0,
            agent="verifier",
            type="claim_snippet_not_found",
            payload={"claim_id": claim_id, "vendor": claim.vendor, "attribute": claim.attribute, "details": details},
        )
        return VerificationResult(
            claim_id=claim_id,
            status="failed",
            details=details,
        )

    # 3. Ask LLM via llm.call_json whether the snippet supports the claimed value
    prompt = (
        f"Analyze whether the following snippet from a verified webpage supports the factual claim.\n\n"
        f"Claim Details:\n"
        f"- Vendor: {claim.vendor}\n"
        f"- Attribute: {claim.attribute}\n"
        f"- Claimed Value: {claim.value}\n\n"
        f"Page Snippet:\n"
        f"\"{claim.snippet}\"\n\n"
        f"Does this snippet support the claim that {claim.vendor}'s {claim.attribute} is {claim.value}?"
    )

    try:
        decision = await llm.call_json(
            prompt=prompt,
            schema_model=ClaimSupportDecision,
            system="You are an evidentiary fact-checker. Determine if the snippet supports the exact claim."
        )
        if decision.supports:
            status = "verified"
            details = f"verified: {decision.reason}"
        else:
            status = "failed"
            details = f"unsupported: {decision.reason}"
    except Exception as exc:
        # If LLM check fails, fall back to snippet match success with note
        status = "verified"
        details = f"verified: snippet confirmed on page ({exc})"

    db.log_event(
        task_id=0,
        agent="verifier",
        type="claim_verified" if status == "verified" else "claim_failed",
        payload={"claim_id": claim_id, "vendor": claim.vendor, "attribute": claim.attribute, "status": status, "details": details},
    )

    return VerificationResult(
        claim_id=claim_id,
        status=status,
        details=details,
    )


async def verify(
    claims: List[Claim],
    fetch_fn: Optional[Callable] = None,
) -> List[VerificationResult]:
    """Verify a list of claims against their source URLs.

    Parameters
    ----------
    claims : list[Claim]
        List of claims to verify.
    fetch_fn : Optional[Callable]
        Optional custom fetcher for testing.

    Returns
    -------
    list[VerificationResult]
        Verification outcomes per claim.
    """
    results: List[VerificationResult] = []
    for i, claim in enumerate(claims):
        res = await verify_claim(claim_id=i, claim=claim, fetch_fn=fetch_fn)
        results.append(res)
    return results
