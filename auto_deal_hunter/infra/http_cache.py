"""Tiny on-disk read-through cache for scraped deal pages.

``ScrapedDeal`` fetches every DealNews product page on every run. Across back-to-back
runs (and the duplicate URLs that RSS feeds overlap on) that is a lot of repeated,
rate-limit-sensitive GETs. ``read``/``write`` store raw page bytes in a local SQLite file
with a TTL so a repeated scan is fast and gentle on the source. Everything is best-effort:
any cache error returns a miss and the caller falls back to a live request.

The cache is disabled (always misses, never writes) when ``DEALHUNTER_HTTP_CACHE`` is set
to ``0``/``off``/``false``, which keeps it out of the way in tests.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

from auto_deal_hunter.infra.paths import DATA_DIR

DEFAULT_TTL_SECONDS = 6 * 60 * 60  # deal prices move slowly; 6h is a safe default
_CACHE_PATH = DATA_DIR / "http_cache.sqlite"


def _enabled() -> bool:
    return os.getenv("DEALHUNTER_HTTP_CACHE", "1").lower() not in {"0", "off", "false"}


def _connect() -> sqlite3.Connection:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, fetched_at REAL NOT NULL, body BLOB NOT NULL)"
    )
    return conn


@contextmanager
def _connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def read(url: str, ttl: float = DEFAULT_TTL_SECONDS) -> Optional[bytes]:
    """Return cached page bytes for ``url`` when present and fresher than ``ttl``, else None."""
    if not _enabled():
        return None
    try:
        with _connection() as conn:
            row = conn.execute("SELECT fetched_at, body FROM pages WHERE url = ?", (url,)).fetchone()
    except sqlite3.Error as exc:
        logging.debug("http_cache read failed: %s", exc)
        return None
    if not row:
        return None
    fetched_at, body = row
    if time.time() - fetched_at > ttl:
        return None
    return body


def write(url: str, body: bytes) -> None:
    """Store page bytes for ``url`` with the current timestamp. Best-effort; errors are ignored."""
    if not _enabled():
        return
    try:
        with _connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO pages (url, fetched_at, body) VALUES (?, ?, ?)",
                (url, time.time(), body),
            )
    except sqlite3.Error as exc:
        logging.debug("http_cache write failed: %s", exc)
