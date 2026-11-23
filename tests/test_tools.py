"""tests/test_tools.py - Unit tests and smoke test for tools (fetch, cache, search)."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
import pytest
import httpx

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.cache import get_cached, set_cached, delete_cached, clear_all_cache
from tools.fetch import fetch_url, FetchError
from tools.search import search
from duckduckgo_search.exceptions import RatelimitException


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean cache before and after each test."""
    clear_all_cache()
    yield
    clear_all_cache()


def is_online(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.0) -> bool:
    """Check if outbound network connectivity is available."""
    try:
        socket.setdefaulttimeout(timeout)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


# ===========================================================================
# 1. Cache Unit Tests
# ===========================================================================

def test_cache_set_and_get():
    """Verify storing and retrieving items in SQLite cache."""
    assert get_cached("test_url") is None
    set_cached("test_url", "Cached content 123")
    assert get_cached("test_url") == "Cached content 123"


def test_cache_miss():
    """Verify cache miss returns None."""
    assert get_cached("non_existent_key") is None


def test_cache_ttl_expiry():
    """Verify expired TTL causes cache miss and deletion."""
    set_cached("expiring_key", "value", ttl=0.1)
    assert get_cached("expiring_key") == "value"
    time.sleep(0.15)
    assert get_cached("expiring_key") is None


# ===========================================================================
# 2. Fetch Unit Tests (Deterministic via mock client)
# ===========================================================================

class MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self.handler = handler

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self.handler(request)


@pytest.mark.anyio
async def test_fetch_url_success():
    """Verify successful page fetch and clean text extraction."""
    html_content = """
    <html>
      <head><title>HealthPlix Features</title></head>
      <body>
        <article>
          <h1>HealthPlix Reminder Suite</h1>
          <p>Official WhatsApp Business API integration with automated appointment reminders.</p>
          <p>Pre-approved TRAI DLT headers for patient transactional alerts across India.</p>
        </article>
      </body>
    </html>
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html_content, request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    url = "https://healthplix.com/features"
    text = await fetch_url(url, client=client)

    assert "WhatsApp Business API" in text
    assert "TRAI DLT" in text
    # Verify cached
    assert get_cached(url) == text


@pytest.mark.anyio
async def test_fetch_url_cache_hit():
    """Verify cached URL is returned without invoking network transport."""
    url = "https://example.com/cached"
    set_cached(url, "Pre-cached text content")

    # Client that would fail if called
    async def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("Network should not be called on cache hit")

    client = httpx.AsyncClient(transport=MockTransport(handler))
    text = await fetch_url(url, client=client, use_cache=True)
    assert text == "Pre-cached text content"


@pytest.mark.anyio
async def test_fetch_url_timeout():
    """Verify timeout raises typed FetchError."""
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out", request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    with pytest.raises(FetchError) as exc_info:
        await fetch_url("https://slowsite.com/data", client=client, use_cache=False)
    assert "Timeout" in str(exc_info.value.reason)


@pytest.mark.anyio
async def test_fetch_url_non_200():
    """Verify HTTP non-200 status raises typed FetchError with status code."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found", request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    with pytest.raises(FetchError) as exc_info:
        await fetch_url("https://example.com/404", client=client, use_cache=False)
    assert "HTTP 404" in exc_info.value.reason
    assert exc_info.value.status_code == 404


@pytest.mark.anyio
async def test_fetch_url_paywall_status():
    """Verify HTTP 403 / 401 raises typed FetchError indicating paywall/auth."""
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="Forbidden Access", request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    with pytest.raises(FetchError) as exc_info:
        await fetch_url("https://paywall.com/article", client=client, use_cache=False)
    assert "Paywall" in exc_info.value.reason
    assert exc_info.value.status_code == 403


@pytest.mark.anyio
async def test_fetch_url_paywall_html_body():
    """Verify paywall wording in HTML body raises typed FetchError."""
    html = """
    <html>
      <body>
        <p>Subscribe to read the full article and clinical data.</p>
        <p>Log in to continue reading premium healthcare analysis.</p>
      </body>
    </html>
    """
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    with pytest.raises(FetchError) as exc_info:
        await fetch_url("https://medicaljournal.com/study", client=client, use_cache=False)
    assert "Paywall or login wall" in exc_info.value.reason


@pytest.mark.anyio
async def test_fetch_url_js_only_page():
    """Verify JavaScript-only shell raises typed FetchError."""
    html = """
    <html>
      <body>
        <div id="root"></div>
        <p>Please enable JavaScript to run this app.</p>
      </body>
    </html>
    """
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    with pytest.raises(FetchError) as exc_info:
        await fetch_url("https://spasite.com/app", client=client, use_cache=False)
    assert "JavaScript" in exc_info.value.reason


@pytest.mark.anyio
async def test_fetch_url_empty_extraction():
    """Verify empty extraction raises typed FetchError."""
    html = "<html><body></body></html>"
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, request=request)

    client = httpx.AsyncClient(transport=MockTransport(handler))
    with pytest.raises(FetchError) as exc_info:
        await fetch_url("https://empty.com", client=client, use_cache=False)
    assert "Empty text" in exc_info.value.reason


# ===========================================================================
# 3. Search Unit Tests (Deterministic via backend_fn injection & cache)
# ===========================================================================

@pytest.mark.anyio
async def test_search_caching():
    """Verify search queries are cached in SQLite."""
    mock_calls = 0

    def mock_backend(q: str, max_r: int):
        nonlocal mock_calls
        mock_calls += 1
        return [{"title": "HealthPlix", "url": "https://healthplix.com", "snippet": "Reminders"}]

    results1 = await search("patient reminder", max_results=5, backend_fn=mock_backend)
    assert len(results1) == 1
    assert mock_calls == 1

    # Second call should be served from cache
    results2 = await search("patient reminder", max_results=5, backend_fn=mock_backend)
    assert len(results2) == 1
    assert mock_calls == 1  # No second invocation


@pytest.mark.anyio
async def test_search_rate_limit_fallback(monkeypatch):
    """Verify rate-limit triggers fallback mechanism."""
    attempts = []

    def mock_sync_search(query: str, max_results: int, backend: str):
        attempts.append(backend)
        if backend == "auto":
            raise RatelimitException("Rate limit 429")
        return [{"title": "Fallback Result", "url": "https://fallback.com", "snippet": "Found via fallback"}]

    import sys
    monkeypatch.setattr(sys.modules["tools.search"], "_sync_ddgs_search", mock_sync_search)

    results = await search("telemedicine india", max_results=3, use_cache=False)
    assert len(results) == 1
    assert results[0]["title"] == "Fallback Result"
    assert "auto" in attempts
    assert "lite" in attempts


@pytest.mark.anyio
async def test_search_empty_query():
    """Verify searching for an empty string returns empty list immediately."""
    assert await search("") == []
    assert await search("   ") == []


# ===========================================================================
# 4. Outbound Network Smoke Test (Skips cleanly when offline)
# ===========================================================================

@pytest.mark.anyio
@pytest.mark.skipif(not is_online(), reason="Network smoke test requires outbound internet connectivity")
async def test_network_smoke_fetch_real_url():
    """One live network smoke test that skips cleanly when offline."""
    url = "https://example.com"
    content = await fetch_url(url, timeout=5, use_cache=False)
    assert isinstance(content, str)
    assert len(content) > 10
    assert "example" in content.lower()
