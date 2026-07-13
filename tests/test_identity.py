import unittest

from auto_deal_hunter.core.identity_policy import (
    display_description,
    is_priceable,
    per_unit_fields,
    per_unit_note,
    resolve,
)
from auto_deal_hunter.domain.identity import ItemKind, ProductIdentity
from auto_deal_hunter.ingest.identity import extract_identity_rule


class ExtractIdentityRuleTests(unittest.TestCase):
    def test_plain_single_item_has_no_signal(self):
        self.assertIsNone(extract_identity_rule("Anker 65W USB-C Wall Charger"))

    def test_model_numbers_are_not_quantities(self):
        # "4K" and "65W" must not register as a 4-pack / 65-count.
        self.assertIsNone(extract_identity_rule("Roku 4K Streaming Stick 65W rated"))

    def test_multipack_hyphen(self):
        ident = extract_identity_rule("AmazonBasics AAA Batteries 36-Pack")
        self.assertEqual(ident.kind, ItemKind.MULTIPACK)
        self.assertEqual(ident.quantity, 36)

    def test_multipack_pack_of(self):
        ident = extract_identity_rule("HDMI Cable, Pack of 3, 6.6ft")
        self.assertEqual(ident.kind, ItemKind.MULTIPACK)
        self.assertEqual(ident.quantity, 3)

    def test_multipack_set_of(self):
        ident = extract_identity_rule("Smart Bulbs, Set of 2")
        self.assertEqual(ident.kind, ItemKind.MULTIPACK)
        self.assertEqual(ident.quantity, 2)

    def test_multipack_abbreviations_and_pieces(self):
        # Short forms (pk/ct), spaced or glued, plus pcs/pieces.
        for text, qty in [
            ("HDMI Cable 4pk", 4),
            ("AAA Batteries 24 ct", 24),
            ("BIC pens 36ct Assorted", 36),
            ("Nylon Cable Zip Ties 1000 Pcs", 1000),
            ("USB Cable, 5 Pieces", 5),
        ]:
            ident = extract_identity_rule(text)
            self.assertIsNotNone(ident, text)
            self.assertEqual(ident.kind, ItemKind.MULTIPACK, text)
            self.assertEqual(ident.quantity, qty, text)

    def test_part_numbers_are_not_quantities(self):
        # Model and part-number fragments must not be treated as pack sizes.
        for text in [
            "PK Power 3-Prong AC Power Cord Lead for HP 8121-0840\nPK Power cord",
            "UpBright 19V AC/DC Adapter Compatible with HP Pavilion ST-C-090-190004",
            "New CT-90325 Remote Compatible with Toshiba TV 40E200UM",
            "PK Power AC Power Cord Outlet Socket Cable for Dell P2214H P2014H",
            "Canon CT-100 printer",
        ]:
            self.assertIsNone(extract_identity_rule(text), text)

    def test_subscription(self):
        ident = extract_identity_rule("NYTimes Digital Subscription, 1 year")
        self.assertEqual(ident.kind, ItemKind.SUBSCRIPTION)

    def test_subscription_per_month(self):
        ident = extract_identity_rule("Cloud storage plan $5 per month")
        self.assertEqual(ident.kind, ItemKind.SUBSCRIPTION)

    def test_bundle(self):
        ident = extract_identity_rule("PS5 Console + 2 Games Bundle")
        self.assertEqual(ident.kind, ItemKind.BUNDLE)

    def test_single_with_storage_variant(self):
        ident = extract_identity_rule("Samsung 990 Pro 2TB NVMe SSD")
        self.assertEqual(ident.kind, ItemKind.SINGLE)
        self.assertEqual(ident.variant, "2TB")

    def test_single_with_screen_variant(self):
        ident = extract_identity_rule('LG UltraGear 27-inch Gaming Monitor')
        self.assertEqual(ident.kind, ItemKind.SINGLE)
        self.assertEqual(ident.variant, "27-inch")

    def test_subscription_takes_precedence_over_variant(self):
        ident = extract_identity_rule("128GB cloud subscription per month")
        self.assertEqual(ident.kind, ItemKind.SUBSCRIPTION)

    def test_subscription_slash_month(self):
        ident = extract_identity_rule("Lyca Mobile Unlimited Plan for $12.50/month")
        self.assertEqual(ident.kind, ItemKind.SUBSCRIPTION)

    def _is_bundle(self, text):
        ident = extract_identity_rule(text)
        return ident is not None and ident.kind == ItemKind.BUNDLE

    def test_free_shipping_is_not_a_bundle(self):
        # Regression: "+ free shipping" is in almost every DealNews title and must not flag.
        # (The TV here is a priceable single item; it just carries a screen-size variant.)
        self.assertFalse(self._is_bundle('Insignia 50" 4K Smart TV for $160 + free shipping'))

    def test_with_free_shipping_is_not_a_bundle(self):
        self.assertFalse(self._is_bundle("Bose SoundLink Speaker for $52 w/ free shipping"))

    def test_combo_in_product_name_is_not_a_bundle(self):
        # "Combo" is part of the product name here, not a heterogeneous bundle.
        self.assertFalse(self._is_bundle("Roborock Q7 Combo Robot Vacuum & Mop"))

    def test_aggregator_up_to_percent_off(self):
        ident = extract_identity_rule("Best Buy 4th of July TV Deals: Up to 45% off + free shipping")
        self.assertEqual(ident.kind, ItemKind.AGGREGATOR)

    def test_aggregator_sale_heading(self):
        ident = extract_identity_rule("Newegg 4th of July Sale: Up to 70% off + free shipping")
        self.assertEqual(ident.kind, ItemKind.AGGREGATOR)

    def test_aggregator_promo_code(self):
        ident = extract_identity_rule("Yale Home Promo Code: 10% off sitewide + free shipping")
        self.assertEqual(ident.kind, ItemKind.AGGREGATOR)

    def test_aggregator_deals_from(self):
        ident = extract_identity_rule("Best Buy 4th of July Printer Deals for From $60 + free shipping")
        self.assertEqual(ident.kind, ItemKind.AGGREGATOR)

    def test_aggregator_top_n_deals(self):
        ident = extract_identity_rule("Best Buy 4th of July Top 100 Deals: Up to 50% off")
        self.assertEqual(ident.kind, ItemKind.AGGREGATOR)

    def test_single_product_with_percent_off_is_not_aggregator(self):
        # A single product can be "50% off"; only a *range* ("up to N% off") signals a roundup.
        ident = extract_identity_rule("Bose SoundLink Speaker 50% off, now $52 + free shipping")
        self.assertNotEqual(getattr(ident, "kind", None), ItemKind.AGGREGATOR)

    def test_aggregator_is_not_priceable(self):
        self.assertFalse(
            is_priceable(ProductIdentity(kind=ItemKind.AGGREGATOR, confidence=0.9))
        )


class ResolvePolicyTests(unittest.TestCase):
    def test_none_is_priceable(self):
        self.assertEqual(resolve(None)[0], "price")
        self.assertTrue(is_priceable(None))

    def test_single_is_priceable(self):
        self.assertTrue(is_priceable(ProductIdentity(kind=ItemKind.SINGLE)))

    def test_single_with_variant_is_priceable(self):
        self.assertTrue(
            is_priceable(ProductIdentity(kind=ItemKind.SINGLE, variant="512GB", confidence=0.8))
        )

    def test_multipack_is_priceable(self):
        # Multipacks are priceable now: rebased to per-unit against per-unit comparables.
        self.assertTrue(
            is_priceable(ProductIdentity(kind=ItemKind.MULTIPACK, quantity=4, confidence=0.85))
        )

    def test_bundle_abstains(self):
        self.assertFalse(is_priceable(ProductIdentity(kind=ItemKind.BUNDLE, confidence=0.6)))

    def test_subscription_abstains(self):
        self.assertFalse(
            is_priceable(ProductIdentity(kind=ItemKind.SUBSCRIPTION, confidence=0.9))
        )

    def test_low_confidence_falls_back_to_priceable(self):
        # A weak signal should not strip an otherwise ordinary listing.
        self.assertTrue(
            is_priceable(ProductIdentity(kind=ItemKind.BUNDLE, confidence=0.2))
        )


class PerUnitFieldsTests(unittest.TestCase):
    def test_single_item_none_identity_passes_through(self):
        self.assertEqual(per_unit_fields(100.0, 120.0, "desc", None), (100.0, 120.0, "desc", 1))

    def test_single_kind_passes_through(self):
        ident = ProductIdentity(kind=ItemKind.SINGLE, variant="512GB", confidence=0.8)
        self.assertEqual(per_unit_fields(50.0, None, "ssd", ident), (50.0, None, "ssd", 1))

    def test_multipack_divides_price_and_list_price(self):
        ident = ProductIdentity(kind=ItemKind.MULTIPACK, quantity=36, confidence=0.85)
        price, list_price, desc, qty = per_unit_fields(18.0, 36.0, "AAA 36-pack", ident)
        self.assertAlmostEqual(price, 0.5)
        self.assertAlmostEqual(list_price, 1.0)
        self.assertEqual(qty, 36)
        # The note must read as an instruction to value ONE unit (not a passive basis
        # remark) and must name the pack size; see per_unit_note.
        self.assertIn("ONE unit", desc)
        self.assertIn("pack of 36", desc)

    def test_display_description_appends_pack_suffix_when_missing(self):
        self.assertEqual(
            display_description("AAA Batteries" + per_unit_note(48), 48),
            "AAA Batteries (48-pack)",
        )
        # Legacy stored note wording is stripped too.
        self.assertEqual(
            display_description(
                "AAA Batteries (per-unit price; sold in packs of 48)", 48
            ),
            "AAA Batteries (48-pack)",
        )

    def test_display_description_skips_suffix_when_pack_already_mentioned(self):
        # The scanner's rephrasing often leads with the pack size; no duplicate suffix then.
        self.assertEqual(
            display_description("This 48-pack of Energizer MAX" + per_unit_note(48), 48),
            "This 48-pack of Energizer MAX",
        )
        self.assertEqual(
            display_description("Sold as a pack of 48" + per_unit_note(48), 48),
            "Sold as a pack of 48",
        )
        # Other quantities in the text (24 AA) must not suppress the 48-pack suffix.
        self.assertEqual(
            display_description("Includes 24 AA and 24 AAA" + per_unit_note(48), 48),
            "Includes 24 AA and 24 AAA (48-pack)",
        )

    def test_display_description_passes_single_items_through(self):
        self.assertEqual(display_description("A laptop", 1), "A laptop")

    def test_multipack_with_no_list_price(self):
        ident = ProductIdentity(kind=ItemKind.MULTIPACK, quantity=2, confidence=0.85)
        price, list_price, desc, qty = per_unit_fields(10.0, None, "x", ident)
        self.assertEqual(price, 5.0)
        self.assertIsNone(list_price)
        self.assertEqual(qty, 2)


if __name__ == "__main__":
    unittest.main()
