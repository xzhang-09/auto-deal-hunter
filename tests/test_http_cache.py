import importlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class HttpCacheTests(unittest.TestCase):
    def _fresh_module(self, tmpdir):
        # Reimport with the cache path pointed at a temp file so tests never touch data/.
        import auto_deal_hunter.infra.http_cache as http_cache

        importlib.reload(http_cache)
        http_cache._CACHE_PATH = Path(tmpdir) / "cache.sqlite"
        return http_cache

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "1"}):
            http_cache = self._fresh_module(tmpdir)
            http_cache.write("https://x.test/1.html", b"hello")
            self.assertEqual(http_cache.read("https://x.test/1.html"), b"hello")

    def test_expired_entry_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "1"}):
            http_cache = self._fresh_module(tmpdir)
            http_cache.write("https://x.test/1.html", b"hello")
            self.assertIsNone(http_cache.read("https://x.test/1.html", ttl=-1))

    def test_missing_url_is_none(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "1"}):
            http_cache = self._fresh_module(tmpdir)
            self.assertIsNone(http_cache.read("https://x.test/absent.html"))

    def test_disabled_cache_never_reads_or_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "0"}):
            http_cache = self._fresh_module(tmpdir)
            http_cache.write("https://x.test/1.html", b"hello")
            self.assertIsNone(http_cache.read("https://x.test/1.html"))

    def test_write_closes_connection(self):
        class TrackingConnection:
            def __init__(self, path):
                self._conn = sqlite3.connect(path)
                self._conn.execute(
                    "CREATE TABLE pages (url TEXT PRIMARY KEY, fetched_at REAL NOT NULL, body BLOB NOT NULL)"
                )
                self.closed = False

            def __enter__(self):
                self._conn.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                return self._conn.__exit__(exc_type, exc, tb)

            def execute(self, *args, **kwargs):
                return self._conn.execute(*args, **kwargs)

            def commit(self):
                self._conn.commit()

            def close(self):
                self.closed = True
                self._conn.close()

        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(os.environ, {"DEALHUNTER_HTTP_CACHE": "1"}):
            http_cache = self._fresh_module(tmpdir)
            connection = TrackingConnection(Path(tmpdir) / "cache.sqlite")

            with patch.object(http_cache, "_connect", return_value=connection):
                http_cache.write("https://x.test/1.html", b"hello")

            self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
