import unittest

from bs4 import BeautifulSoup

from models.deals import extract_callout_prices, extract_list_price


class CalloutPriceTests(unittest.TestCase):
    def _soup(self, html):
        return BeautifulSoup(html, "html.parser")

    def test_reads_deal_and_struck_through_list_price(self):
        soup = self._soup(
            '<div class="callout-group"><span class="callout">$150 w/ Prime '
            '<span class="callout-comparison">$250</span></span></div>'
        )
        self.assertEqual(extract_callout_prices(soup), (150.0, 250.0))

    def test_list_price_is_none_without_comparison(self):
        soup = self._soup('<div class="callout-group"><span class="callout">$42</span></div>')
        self.assertEqual(extract_callout_prices(soup), (42.0, None))

    def test_returns_none_pair_without_widget(self):
        soup = self._soup("<div>no price widget here</div>")
        self.assertEqual(extract_callout_prices(soup), (None, None))


class ExtractListPriceTests(unittest.TestCase):
    def test_reconstructs_list_price_from_under_list_callout(self):
        # DealNews states the saving, not the list price: 400 + 150 = 550.
        self.assertEqual(
            extract_list_price("It's $150 under list price.", deal_price=400.0), 550.0
        )

    def test_handles_off_regular_phrasing(self):
        self.assertEqual(
            extract_list_price("Save $50 off regular price", deal_price=199.0), 249.0
        )

    def test_explicit_list_price_takes_precedence_over_relative(self):
        self.assertEqual(
            extract_list_price("List price: $899. $50 under list price", deal_price=700.0),
            899.0,
        )

    def test_ignores_off_without_list_keyword(self):
        # "$10 off your order" is a promo, not a list-price delta.
        self.assertIsNone(
            extract_list_price("Get $10 off your first order", deal_price=80.0)
        )

    def test_relative_callout_needs_a_deal_price_baseline(self):
        self.assertIsNone(extract_list_price("$150 under list price", deal_price=None))


if __name__ == "__main__":
    unittest.main()
