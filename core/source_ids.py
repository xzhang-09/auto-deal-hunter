"""Source-agnostic stable id for a deal URL.

Deduplication and estimate-pairing key on a *stable* id rather than the full URL, because a
marketplace can change a URL's slug or query string between RSS pulls while the underlying
product is unchanged. Each supported source contributes a small extractor here; a new
marketplace is added by registering one function, not by editing the domain models -- which
stay free of any site-specific URL knowledge. When no source recognizes the URL, the full URL
is its own id, so distinct URLs remain distinct.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

# DealNews product pages end in "/<numeric-id>.html". That id is stable across slug edits
# (e.g. a "200-W" -> "2000-W" typo fix) and query params (e.g. ?iref=rss).
_DEALNEWS_ID = re.compile(r"/(\d+)\.html")


def _dealnews_id(url: str) -> Optional[str]:
    match = _DEALNEWS_ID.search(url or "")
    return match.group(1) if match else None


# Ordered extractors; the first to return a non-empty id wins. A new source registers here.
_EXTRACTORS: List[Callable[[str], Optional[str]]] = [_dealnews_id]


def register_source(extractor: Callable[[str], Optional[str]]) -> None:
    """Register a URL->id extractor for another marketplace. It should return the stable id for
    URLs it recognizes and ``None`` otherwise, so extractors can be tried in turn."""
    _EXTRACTORS.append(extractor)


def deal_id(url: str) -> str:
    """Stable product id for a deal URL, robust to slug/query changes. Falls back to the full
    URL when no registered source recognizes it (each distinct URL then being its own id)."""
    for extract in _EXTRACTORS:
        found = extract(url)
        if found:
            return found
    return url
