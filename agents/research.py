"""agents/research.py - Autonomous research agent for AgentFlow AI.

Discovers healthcare software vendors for Indian hospitals and rural clinics,
fetches their primary websites/pricing pages, and extracts evidence-backed
claims conforming to schemas.Claim with verbatim snippets.

Key rules implemented:
- Category-first queries (never start with a bare noun like 'patient').
- Discovers vendors from directories/listicles and fetches vendor pages.
- Only accepts vendor names that appear in the fetched page text.
- FetchError handling: skips unusable/failed pages cleanly, never invents text.
- Default to tools.search.search and tools.fetch.fetch_url, with dependency
  injection for deterministic unit testing without network.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import urllib.parse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from schemas import Claim, INDIA_CRITERIA_KEYS, STANDARD_ATTRIBUTES
from tools.fetch import fetch_url as default_fetch_url, FetchError
from tools.search import search as default_search
import db

logger = logging.getLogger(__name__)

# Known healthcare software vendors frequently referenced in Indian hospital IT
KNOWN_INDIAN_HEALTH_VENDORS = [
    "HealthPlix",
    "Qikwell",
    "Practo",
    "Clinizy",
    "SparkTG",
    "Mishtel",
    "Elision",
    "SmartSevak",
    "SMSLocal",
    "Adrine",
    "eSanjeevani",
    "mPower Health",
    "MocDoc",
    "DocEngage",
    "Attune",
]

# Attribute detection rules with keyword patterns and default values
ATTRIBUTE_PATTERNS: Dict[str, Dict[str, Any]] = {
    "whatsapp_api": {
        "keywords": [r"whatsapp\s+business\s+api", r"whatsapp\s+automation", r"whatsapp\s+reminder"],
        "default_val": "Supported",
    },
    "dlt_sms": {
        "keywords": [r"trai\s+dlt", r"dlt\s+registration", r"dlt\s+sms", r"dlt\s+approved", r"transactional\s+sms"],
        "default_val": "Supported",
    },
    "regional_languages": {
        "keywords": [r"regional\s+languages?", r"hindi", r"tamil", r"telugu", r"kannada", r"multilingual"],
        "default_val": "Supported",
    },
    "abdm_abha": {
        "keywords": [r"abdm", r"abha", r"ayushman\s+bharat\s+digital\s+mission", r"m1,\s*m2"],
        "default_val": "Supported",
    },
    "dpdp_compliance": {
        "keywords": [r"dpdp", r"digital\s+personal\s+data\s+protection", r"data\s+privacy\s+act", r"hipaa|iso\s*27001"],
        "default_val": "Compliant",
    },
    "emr_integration": {
        "keywords": [r"emr", r"ehr", r"his\s+integration", r"hl7", r"fhir", r"hospital\s+information\s+system"],
        "default_val": "Supported",
    },
    "inr_pricing": {
        "keywords": [r"₹", r"inr", r"rupees?", r"per\s+month"],
        "default_val": "Supported",
    },
    "appointment_reminders": {
        "keywords": [r"appointment\s+reminders?", r"automated\s+reminders?", r"reminder\s+alerts?"],
        "default_val": "Supported",
    },
    "two_way_confirmation": {
        "keywords": [r"two[-\s]way\s+confirmation", r"patient\s+confirmation", r"reschedule\s+or\s+cancel"],
        "default_val": "Supported",
    },
    "multi_location": {
        "keywords": [r"multi[-\s]location", r"multi[-\s]branch", r"chain\s+of\s+clinics", r"multiple\s+centers"],
        "default_val": "Supported",
    },
}


def build_search_queries(task: str, vendor_focus: Optional[str] = None) -> List[str]:
    """Generate category-first search queries.

    Never starts with a bare noun like 'patient' to avoid DDG dictionary collapse.
    """
    if vendor_focus:
        v = vendor_focus.strip()
        return [
            f"hospital appointment reminder software {v} India",
            f"{v} healthcare software WhatsApp SMS pricing India",
            f"{v} appointment scheduling hospital features India",
        ]

    # Category-first queries
    return [
        "appointment reminder software hospital India",
        "hospital appointment reminder WhatsApp SMS India pricing",
        "hospital patient engagement software India",
        "best hospital management software India appointment reminders",
    ]


def discover_vendors_from_text(text: str) -> List[str]:
    """Identify candidate healthcare software vendors from directory/listicle content."""
    discovered: Set[str] = set()
    text_lower = text.lower()

    for vendor in KNOWN_INDIAN_HEALTH_VENDORS:
        if vendor.lower() in text_lower:
            discovered.add(vendor)

    # Heuristic: discover capitalized tech/health vendor patterns
    # e.g., 'HealthPlix', 'SmartSevak', 'CareStack', 'DocPulse'
    regex_matches = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:Health|Doc|Care|Med|Tech|Plix|Qik|Pulse|Sevak|Local))\b", text)
    for m in regex_matches:
        if len(m) > 3 and m.lower() not in {"healthcare", "medicare", "healthtech"}:
            discovered.add(m)

    return sorted(discovered)


def extract_verbatim_snippets(page_text: str, keyword_pattern: str) -> Optional[Tuple[str, str]]:
    """Find a sentence or clause containing the keyword and return (verbatim_snippet, matched_text)."""
    # Split text into sentences or logical chunks
    sentences = re.split(r"(?<=[.!?\n])\s+", page_text)
    pattern = re.compile(keyword_pattern, re.IGNORECASE)

    for sent in sentences:
        clean_sent = sent.strip()
        match = pattern.search(clean_sent)
        if match and len(clean_sent) >= 15:
            # Return up to 250 characters of verbatim context
            snippet = clean_sent[:250].strip()
            return snippet, match.group(0)

    return None


def extract_claims_from_page(
    vendor: str,
    page_text: str,
    source_url: str,
    track: Optional[str] = None,
) -> List[Claim]:
    """Extract verifiable claims from page text with exact verbatim snippets."""
    # Strict validation rule: vendor name MUST appear in the fetched page text
    if vendor.lower() not in page_text.lower():
        logger.warning("Rejected vendor '%s': name not found in fetched page %s", vendor, source_url)
        return []

    claims: List[Claim] = []
    seen_attrs: Set[str] = set()

    # Pricing extraction (especially for research_b or general)
    if track != "features":
        price_match = re.search(r"(₹\s*\d+(?:,\d+)*(?:\s*(?:per\s*month|/\s*month|/\s*mo|per\s*year))?)", page_text, re.IGNORECASE)
        if price_match:
            price_val = price_match.group(1).replace(" ", "")
            # Find snippet containing the price
            res = extract_verbatim_snippets(page_text, re.escape(price_match.group(1)))
            if res:
                snippet, _ = res
                claims.append(Claim(
                    vendor=vendor,
                    attribute="price_per_month",
                    value=price_val,
                    source_url=source_url,
                    snippet=snippet,
                ))
                seen_attrs.add("price_per_month")

    # Attribute and criteria extraction
    for attr, config in ATTRIBUTE_PATTERNS.items():
        if attr in seen_attrs:
            continue

        # Check track filtering if specified
        if track == "features" and attr in ("price_per_month", "setup_fee"):
            continue
        if track == "pricing" and attr not in ("inr_pricing", "price_per_month", "setup_fee"):
            continue

        pattern_str = "|".join(config["keywords"])
        match_info = extract_verbatim_snippets(page_text, pattern_str)
        if match_info:
            snippet, matched_phrase = match_info
            claims.append(Claim(
                vendor=vendor,
                attribute=attr,
                value=config["default_val"],
                source_url=source_url,
                snippet=snippet,
            ))
            seen_attrs.add(attr)

    return claims


async def run_research(
    task: str,
    vendor_focus: Optional[str] = None,
    track: Optional[str] = None,
    task_id: int = 1,
    search_fn: Optional[Callable] = None,
    fetch_fn: Optional[Callable] = None,
) -> List[Claim]:
    """Execute autonomous research track discovering vendors and extracting claims.

    Args:
        task: User research prompt or domain instruction.
        vendor_focus: Specific vendor name for targeted re-research.
        track: Optional track designation ('features' for research_a, 'pricing' for research_b).
        task_id: Task identifier for event logging.
        search_fn: Injectable async search callable for testing.
        fetch_fn: Injectable async fetch callable for testing.

    Returns:
        List of evidence-backed Claim objects.
    """
    search_callable = search_fn or default_search
    fetch_callable = fetch_fn or default_fetch_url

    db.log_event(
        task_id=task_id,
        agent="research",
        type="research_started",
        payload={"task": task, "vendor_focus": vendor_focus, "track": track},
    )

    queries = build_search_queries(task, vendor_focus)
    all_claims: List[Claim] = []
    discovered_vendors: Set[str] = set()
    candidate_urls: List[str] = []

    if vendor_focus:
        discovered_vendors.add(vendor_focus)

    # 1. Execute Category-First Searches
    for q in queries[:2]:  # Use top category-first queries
        try:
            if inspect.iscoroutinefunction(search_callable):
                results = await search_callable(q, max_results=6)
            else:
                results = await asyncio.to_thread(search_callable, q, max_results=6)

            for r in results or []:
                url = r.get("url", "")
                if url and url.startswith("http"):
                    candidate_urls.append(url)
                # Discover vendors from snippet / title
                text_block = f"{r.get('title', '')} {r.get('snippet', '')}"
                for v in discover_vendors_from_text(text_block):
                    discovered_vendors.add(v)
        except Exception as exc:
            logger.warning("Search query '%s' encountered error: %s", q, exc)

    db.log_event(
        task_id=task_id,
        agent="research",
        type="vendors_discovered",
        payload={"vendors": sorted(discovered_vendors), "urls_count": len(candidate_urls)},
    )

    # 2. Fetch discovered vendor pages and extract verified claims
    # If no vendors were discovered from snippets, use top known vendors matching task
    if not discovered_vendors:
        discovered_vendors.update(KNOWN_INDIAN_HEALTH_VENDORS[:3])

    # Limit URLs to inspect to stay efficient
    target_urls = candidate_urls[:8]

    for url in target_urls:
        try:
            if inspect.iscoroutinefunction(fetch_callable):
                page_text = await fetch_callable(url)
            else:
                page_text = await asyncio.to_thread(fetch_callable, url)

            if not page_text:
                continue

            # Identify which discovered vendor is present on this page
            for vendor in list(discovered_vendors):
                if vendor.lower() in page_text.lower():
                    page_claims = extract_claims_from_page(
                        vendor=vendor,
                        page_text=page_text,
                        source_url=url,
                        track=track,
                    )
                    all_claims.extend(page_claims)

        except FetchError as fe:
            # Rule: FetchError means the page is unusable: skip it, never invent text.
            logger.info("Skipping unusable page %s: %s", url, fe.reason)
            continue
        except Exception as exc:
            logger.warning("Unexpected error fetching %s: %s", url, exc)
            continue

    # Deduplicate claims by (vendor, attribute, value)
    unique_claims: List[Claim] = []
    seen_keys: Set[Tuple[str, str, str]] = set()

    for c in all_claims:
        key = (c.vendor.lower(), c.attribute.lower(), c.value.lower())
        if key not in seen_keys:
            seen_keys.add(key)
            unique_claims.append(c)

    db.log_event(
        task_id=task_id,
        agent="research",
        type="research_completed",
        payload={"total_claims": len(unique_claims), "vendors": list({c.vendor for c in unique_claims})},
    )

    return unique_claims


async def run_research_a(task: str, **kwargs) -> List[Claim]:
    """Research A track: focuses on core features and India compliance criteria."""
    return await run_research(task, track="features", **kwargs)


async def run_research_b(task: str, **kwargs) -> List[Claim]:
    """Research B track: focuses on pricing and commercial models."""
    return await run_research(task, track="pricing", **kwargs)
