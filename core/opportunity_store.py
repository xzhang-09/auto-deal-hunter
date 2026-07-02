import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from core.source_ids import deal_id
from domain.deal import Opportunity

_NEW_SCHEMA = """
    CREATE TABLE IF NOT EXISTS opportunities (
        dedup_id TEXT PRIMARY KEY,
        url TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
"""


class OpportunityStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(_NEW_SCHEMA)
            # A fresh DB now has the dedup_id schema. An older DB still has the url-primary-key
            # table (the CREATE above no-ops on it), so migrate it onto the stable deal_id key.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(opportunities)")}
            if "dedup_id" not in columns:
                self._migrate_to_dedup_id(conn, columns)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_created_at ON opportunities(created_at)"
            )

    def _migrate_to_dedup_id(self, conn: sqlite3.Connection, columns: set[str]) -> None:
        """Rebuild a legacy url-primary-keyed table onto a deal_id primary key.

        The old table keyed on the full URL, so the same product under a changed slug/query
        could occupy two rows; keying on the stable ``deal_id`` collapses those. SQLite can't
        change a primary key in place, so copy into a fresh table, deduping by deal_id and
        keeping the most recently confirmed row for each id."""
        # Tables predating updated_at have only created_at; synthesize updated_at from it so the
        # "keep the freshest" comparison below always has a value.
        select_updated = "updated_at" if "updated_at" in columns else "created_at AS updated_at"
        rows = conn.execute(
            f"SELECT url, payload_json, created_at, {select_updated} FROM opportunities"
        ).fetchall()

        # Collapse by deal_id, keeping the freshest row (max updated_at, then created_at).
        # Timestamps are UTC 'YYYY-MM-DD HH:MM:SS' text, so lexical comparison is chronological.
        latest: dict[str, tuple] = {}
        for url, payload_json, created_at, updated_at in rows:
            did = deal_id(url)
            incoming = (updated_at or "", created_at or "")
            current = latest.get(did)
            if current is None or incoming >= (current[3] or "", current[2] or ""):
                latest[did] = (url, payload_json, created_at, updated_at)

        conn.execute("ALTER TABLE opportunities RENAME TO opportunities_legacy")
        conn.execute(_NEW_SCHEMA)
        conn.executemany(
            """
            INSERT INTO opportunities (dedup_id, url, payload_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (did, url, payload_json, created_at, updated_at)
                for did, (url, payload_json, created_at, updated_at) in latest.items()
            ],
        )
        conn.execute("DROP TABLE opportunities_legacy")

    def append(self, opportunity: Opportunity) -> None:
        # Upsert keyed on the stable deal_id (not the raw URL): a re-scrape refreshes the stored
        # url/price/list/estimate (DealNews edits listings and can change a slug) and bumps
        # updated_at so the deal reads as freshly confirmed, while created_at stays as first-seen.
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO opportunities (dedup_id, url, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT(dedup_id) DO UPDATE SET
                    url = excluded.url,
                    payload_json = excluded.payload_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (deal_id(opportunity.deal.url), opportunity.deal.url, json.dumps(opportunity.model_dump())),
            )

    def prune_stale(self, max_age_hours: float) -> int:
        """Delete opportunities not confirmed within max_age_hours, returning the count
        removed. A non-positive max_age_hours disables expiry (no-op). Comparison runs in
        SQLite (UTC) against updated_at, falling back to created_at for legacy rows."""
        if max_age_hours <= 0:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM opportunities
                WHERE COALESCE(updated_at, created_at) < datetime('now', ?)
                """,
                (f"-{max_age_hours} hours",),
            )
            return cursor.rowcount

    def list_opportunities(self) -> list[Opportunity]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM opportunities ORDER BY created_at"
            ).fetchall()
        return [Opportunity(**json.loads(row[0])) for row in rows]

    def migrate_from_json(self, legacy_path: str | Path) -> None:
        """One-time seed from a legacy memory.json. Insert-or-ignore, NOT upsert: this runs
        on every startup, so it must never clobber rows already refreshed by a live scrape
        with the (older, often list_price-less) legacy snapshot."""
        path = Path(legacy_path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO opportunities (dedup_id, url, payload_json) VALUES (?, ?, ?)",
                [
                    (
                        deal_id(item["deal"]["url"]),
                        item["deal"]["url"],
                        json.dumps(Opportunity(**item).model_dump()),
                    )
                    for item in data
                ],
            )
