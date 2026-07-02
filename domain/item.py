from typing import Optional
from pydantic import BaseModel


class Item(BaseModel):
    title: str
    category: str
    price: float
    summary: str
    quantity: int = 1  # pack size, so the price can be normalized to per-unit at query time
    variant: Optional[str] = None  # price-affecting spec (e.g. "512GB"), recorded for retrieval

    @classmethod
    def from_mcauley_row(cls, row: dict) -> Optional["Item"]:
        """Build an Item from a raw_meta_Electronics row of McAuley-Lab/Amazon-Reviews-2023."""
        price_str = row.get("price")
        if not price_str or price_str == "None":
            return None
        try:
            price = float(price_str)
        except (TypeError, ValueError):
            return None
        title = (row.get("title") or "").strip()
        if not title:
            return None
        description = row.get("description") or []
        summary = title if not description else f"{title}\n" + " ".join(description)
        category = row.get("main_category") or "Unknown"
        return cls(title=title, category=category, price=price, summary=summary)
