import unittest

from auto_deal_hunter.core.source_ids import deal_id


class SourceIdsTests(unittest.TestCase):
    def test_dealnews_id_is_stable_across_slug_and_query(self):
        a = deal_id("https://www.dealnews.com/products/Foo/Old-Slug/123456.html?iref=rss")
        b = deal_id("https://www.dealnews.com/products/Foo/New-Slug/123456.html")
        self.assertEqual(a, "123456")
        self.assertEqual(a, b)

    def test_unknown_url_falls_back_to_full_url(self):
        url = "https://example.test/some/page-without-id"
        self.assertEqual(deal_id(url), url)


if __name__ == "__main__":
    unittest.main()
