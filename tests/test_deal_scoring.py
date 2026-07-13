import unittest

from auto_deal_hunter.core.scoring import best_opportunity, rank_opportunities
from auto_deal_hunter.domain.deal import Deal, Opportunity


def _opp(price, estimate, list_price=None, url="https://x.test/1.html", quantity=1):
    return Opportunity(
        deal=Deal(
            product_description="d", price=price, list_price=list_price, url=url, quantity=quantity
        ),
        estimate=estimate,
    )


class DealScoringTests(unittest.TestCase):
    def test_ranks_by_capped_discount_not_raw_estimate(self):
        # A has a bigger raw estimate but its value is capped at list price; B wins on
        # capped discount.
        a = _opp(price=100, estimate=500, list_price=120, url="https://x.test/1.html")  # capped: 20
        b = _opp(price=100, estimate=160, list_price=None, url="https://x.test/2.html")  # 60
        ranked = rank_opportunities([a, b])
        self.assertEqual(ranked[0].deal.url, "https://x.test/2.html")

    def test_best_opportunity_picks_max_discount(self):
        a = _opp(price=100, estimate=130, url="https://x.test/1.html")  # 30
        b = _opp(price=100, estimate=180, url="https://x.test/2.html")  # 80
        self.assertEqual(best_opportunity([a, b]).deal.url, "https://x.test/2.html")

    def test_ranks_by_total_savings_across_pack(self):
        # Single item saves $5 total; a 36-pack saves $1/unit but $36 total, so the pack wins.
        single = _opp(price=20, estimate=25, url="https://x.test/1.html")  # per-unit 5, total 5
        pack = _opp(price=0.5, estimate=1.5, url="https://x.test/2.html", quantity=36)  # 1, total 36
        self.assertEqual(best_opportunity([single, pack]).deal.url, "https://x.test/2.html")

    def test_best_opportunity_returns_none_when_no_real_bargain(self):
        a = _opp(price=100, estimate=100)  # discount 0, not > min_discount default 0
        self.assertIsNone(best_opportunity([a]))

    def test_best_opportunity_empty(self):
        self.assertIsNone(best_opportunity([]))


if __name__ == "__main__":
    unittest.main()
