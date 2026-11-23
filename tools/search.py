"""tools/search.py - Async web search using duckduckgo-search with caching and fallback."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException, RatelimitException

from tools.cache import get_cached, set_cached

logger = logging.getLogger(__name__)


def _sync_ddgs_search(query: str, max_results: int, backend: str = "auto") -> List[Dict[str, str]]:
    """Execute synchronous DuckDuckGo search in worker thread."""
    with DDGS(timeout=10) as ddgs:
        raw_results = ddgs.text(keywords=query, max_results=max_results, backend=backend)
        normalized = []
        for r in raw_results or []:
            normalized.append({
                "title": r.get("title", "").strip(),
                "url": r.get("href") or r.get("url", "").strip(),
                "snippet": r.get("body") or r.get("snippet", "").strip(),
            })
        return normalized


async def search(
    query: str,
    max_results: int = 8,
    use_cache: bool = True,
    backend_fn: Optional[Callable[[str, int], Any]] = None,
) -> List[Dict[str, str]]:
    """Async search returning list of {title, url, snippet}.

    Features:
    - Cached via SQLite cache table (instant offline repeat queries).
    - Rate-limit handling with exponential backoff and backend fallback ('auto' -> 'lite' -> 'html').
    - Injectable backend_fn for deterministic unit testing without network.

    Args:
        query: Search query string.
        max_results: Maximum number of search results to return (default 8).
        use_cache: Whether to use SQLite caching (default True).
        backend_fn: Optional custom callable (sync or async) for testing.

    Returns:
        List of dicts with 'title', 'url', and 'snippet'.
    """
    if not query or not query.strip():
        return []

    clean_query = query.strip()
    cache_key = f"search:{clean_query.lower()}:{max_results}"

    # 1. Check cache
    if use_cache:
        cached_json = get_cached(cache_key)
        if cached_json is not None:
            try:
                return json.loads(cached_json)
            except Exception:
                pass

    # 2. If test callable provided, use it
    if backend_fn is not None:
        if asyncio.iscoroutinefunction(backend_fn):
            results = await backend_fn(clean_query, max_results)
        else:
            results = backend_fn(clean_query, max_results)

        if use_cache:
            set_cached(cache_key, json.dumps(results))
        return results

    # 3. DuckDuckGo search with rate-limit retry and backend fallback
    backends = ["auto", "lite", "html"]
    last_exc: Optional[Exception] = None

    for attempt, backend in enumerate(backends):
        try:
            results = await asyncio.to_thread(_sync_ddgs_search, clean_query, max_results, backend)
            if results:
                if use_cache:
                    set_cached(cache_key, json.dumps(results))
                return results
        except (RatelimitException, DuckDuckGoSearchException, Exception) as exc:
            last_exc = exc
            logger.warning(
                "Search attempt %d failed for query '%s' with backend '%s': %s",
                attempt + 1,
                clean_query,
                backend,
                exc,
            )
            # Short backoff before next backend fallback
            if attempt < len(backends) - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))

    # If all backends exhausted, return empty list or log
    logger.error("All search backends exhausted for query '%s': %s", clean_query, last_exc)
    return []
