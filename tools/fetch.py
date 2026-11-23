"""tools/fetch.py - Async web page fetcher with text extraction and error typing."""

from __future__ import annotations

import asyncio
import re
import time
import urllib.parse
from typing import Dict, Optional

import httpx
import trafilatura

from tools.cache import get_cached, set_cached

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36 AgentFlow/1.0"
)

# Per-domain rate limiting state
_DOMAIN_LOCKS: Dict[str, asyncio.Lock] = {}
_DOMAIN_LAST_FETCH: Dict[str, float] = {}
DEFAULT_RATE_LIMIT_DELAY = 0.5  # seconds minimum between requests to the same domain

# Patterns indicating paywall / login wall
PAYWALL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"paywall",
        r"subscribe to read",
        r"subscription required",
        r"premium members only",
        r"log in to continue reading",
        r"you have reached your limit of free articles",
        r"create a free account to read",
        r"sign in to read",
        r"access to this page is restricted",
    ]
]

# Patterns indicating JavaScript-required shells
JS_ONLY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"you need to enable javascript",
        r"please enable javascript to run this app",
        r"javascript is required to view this site",
        r"enable javascript and refresh the page",
    ]
]


class FetchError(Exception):
    """Typed exception raised when page fetching or text extraction fails."""

    def __init__(self, reason: str, status_code: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


async def _rate_limit_domain(domain: str, delay: float = DEFAULT_RATE_LIMIT_DELAY) -> None:
    """Enforce polite minimum delay between requests to the same domain."""
    if domain not in _DOMAIN_LOCKS:
        _DOMAIN_LOCKS[domain] = asyncio.Lock()

    async with _DOMAIN_LOCKS[domain]:
        now = time.time()
        last = _DOMAIN_LAST_FETCH.get(domain, 0.0)
        elapsed = now - last
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        _DOMAIN_LAST_FETCH[domain] = time.time()


async def fetch_url(
    url: str,
    timeout: float = 10.0,
    use_cache: bool = True,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Fetch URL and extract main clean text using httpx and trafilatura.

    Args:
        url: Target web page URL.
        timeout: Network timeout in seconds (default 10s).
        use_cache: Whether to check/store in SQLite cache (default True).
        client: Optional httpx.AsyncClient for testing/dependency injection.

    Returns:
        Extracted main text content.

    Raises:
        FetchError: If timeout, non-200 status, paywall, JS-only, or empty extraction occurs.
    """
    if not url or not url.startswith(("http://", "https://")):
        raise FetchError(f"Invalid URL format: '{url}'")

    # 1. Check cache
    if use_cache:
        cached_content = get_cached(url)
        if cached_content is not None:
            return cached_content

    # 2. Rate limiting
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    await _rate_limit_domain(domain)

    # 3. HTTP Request
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    html: str = ""

    try:
        if client is not None:
            resp = await client.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            status_code = resp.status_code
            html = resp.text
        else:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True) as local_client:
                resp = await local_client.get(url, timeout=timeout)
                status_code = resp.status_code
                html = resp.text
    except httpx.TimeoutException as exc:
        raise FetchError(f"Timeout fetching {url}: {exc}") from exc
    except httpx.RequestError as exc:
        raise FetchError(f"Network request error fetching {url}: {exc}") from exc
    except Exception as exc:
        raise FetchError(f"Unexpected error fetching {url}: {exc}") from exc

    # 4. Check HTTP status code
    if status_code in (401, 403):
        raise FetchError(f"Paywall or authentication required (HTTP {status_code}) at {url}", status_code=status_code)
    if status_code != 200:
        raise FetchError(f"HTTP {status_code} fetching {url}", status_code=status_code)

    # 5. Check paywall indicators in raw HTML
    for pattern in PAYWALL_PATTERNS:
        if pattern.search(html):
            raise FetchError(f"Paywall or login wall detected at {url}", status_code=200)

    # 6. Check JavaScript-only indicators
    for pattern in JS_ONLY_PATTERNS:
        if pattern.search(html):
            raise FetchError(f"JavaScript-only page or dynamic script required at {url}", status_code=200)

    # 7. Extract main text using trafilatura
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    if not extracted or not extracted.strip():
        # Check if the page was a minimal JS shell like <div id="root"></div>
        if re.search(r'<div\s+id=["\'](?:root|app|__next)["\']\s*>\s*</div>', html, re.IGNORECASE):
            raise FetchError(f"JavaScript-only single page application (SPA) shell at {url}")
        raise FetchError(f"Empty text extracted from {url}")

    clean_text = extracted.strip()

    # 8. Save to cache
    if use_cache:
        set_cached(url, clean_text)

    return clean_text
