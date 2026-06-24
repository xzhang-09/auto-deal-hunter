import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from models.deals import Opportunity


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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS opportunities (
                    url TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_opportunities_created_at ON opportunities(created_at)"
            )

    def append(self, opportunity: Opportunity) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO opportunities (url, payload_json)
                VALUES (?, ?)
                """,
                (opportunity.deal.url, json.dumps(opportunity.model_dump())),
            )

    def replace_all(self, opportunities: list[Opportunity]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM opportunities")
            conn.executemany(
                """
                INSERT INTO opportunities (url, payload_json)
                VALUES (?, ?)
                """,
                [(opp.deal.url, json.dumps(opp.model_dump())) for opp in opportunities],
            )

    def update(self, opportunity: Opportunity) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE opportunities
                SET payload_json = ?
                WHERE url = ?
                """,
                (json.dumps(opportunity.model_dump()), opportunity.deal.url),
            )

    def list_opportunities(self) -> list[Opportunity]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM opportunities ORDER BY created_at"
            ).fetchall()
        return [Opportunity(**json.loads(row[0])) for row in rows]

    def migrate_from_json(self, legacy_path: str | Path) -> None:
        path = Path(legacy_path)
        if not path.exists():
            return
        data = json.loads(path.read_text())
        for item in data:
            self.append(Opportunity(**item))
