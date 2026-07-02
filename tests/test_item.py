import unittest

from domain.item import Item


def _row(price, title="3-Prong AC Power Cord", category="All Electronics", description=None):
    return {"title": title, "main_category": category, "price": price, "description": description}


class ItemFromMcauleyRowTests(unittest.TestCase):
    def test_valid_row_builds_item(self):
        item = Item.from_mcauley_row(_row("19.99"))
        self.assertIsNotNone(item)
        self.assertEqual(item.price, 19.99)
        self.assertEqual(item.category, "All Electronics")

    def test_missing_or_unparseable_price_is_dropped(self):
        self.assertIsNone(Item.from_mcauley_row(_row("None")))
        self.assertIsNone(Item.from_mcauley_row(_row(None)))
        self.assertIsNone(Item.from_mcauley_row(_row("not-a-number")))

    def test_missing_title_is_dropped(self):
        self.assertIsNone(Item.from_mcauley_row(_row("19.99", title="  ")))


if __name__ == "__main__":
    unittest.main()
