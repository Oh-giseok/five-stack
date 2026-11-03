"""Market semantics: no aggregate deletion, invented zeroes, or mixed reporters."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config
from oinq.preprocessing.comtrade import (ComtradeDataError, identify, make_panels, market_panel,
                                        typed, validate_panels, value_basis)


class ComtradePreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(ROOT)

    def scope(self, all_countries=False):
        cfg = deepcopy(self.config)
        if not all_countries:
            cfg["countries"]["countries"] = [r for r in cfg["countries"]["countries"] if r["iso3"] == "USA"]
            cfg["products"]["products"] = [r for r in cfg["products"]["products"] if r["hs_code"] == "330499"]
        cfg["splits"]["collection"].update(monthly_start="2025-01", monthly_end="2025-03")
        return cfg

    def panels(self, world=(100, 200, 300), korea=(10, 40, 90), *, all_countries=False):
        cfg = self.scope(all_countries)
        details, requests = [], []
        for country in cfg["countries"]["countries"]:
            for product in cfg["products"]["products"]:
                for role, values in (("world_import", world), ("korea_import", korea), ("export_comparison", (8, 30, 60))):
                    for month, value in enumerate(values, 1):
                        key = {"country_code": country["iso3"], "hs_code": product["hs_code"], "source_role": role,
                               "period_month": f"2025-{month:02d}"}
                        requests.append({**key, "source_file": "original.json", "collected_at": "2026-11-03T00:00:00+00:00",
                                         "request_success": True, "request_empty": False,
                                         "source_reporter_code": "410" if role == "export_comparison" else str(country["comtrade_reporter"]),
                                         "source_partner_code": str(country["comtrade_partner"]) if role == "export_comparison" else "0" if role == "world_import" else "410"})
                        if value is not None:
                            details.append({**key, "trade_value_usd": value, "net_weight_kg": value / 10,
                                            "valuation": "FOB" if role == "export_comparison" else "CIF", "hs_edition": "H6",
                                            "isOriginalClassification": True, "isReported": role == "export_comparison",
                                            "isAggregate": role != "export_comparison", "isNetWgtEstimated": role == "korea_import",
                                            "legacyEstimationFlag": 4 if role == "korea_import" else 0})
        details, requests = typed(pd.DataFrame(details)), typed(pd.DataFrame(requests))
        monthly, quarterly = make_panels(details, requests, cfg)
        return cfg, details, requests, monthly, quarterly

    def test_all_twenty_countries_eight_products_three_roles_preserved(self):
        cfg, details, _, monthly, quarterly = self.panels(all_countries=True)
        self.assertEqual(len(details), 1440)
        self.assertEqual(len(quarterly), 480)
        self.assertEqual(monthly.isAggregate.sum(), 960)
        mm, mq = market_panel(monthly, "period_month", cfg), market_panel(quarterly, "quarter", cfg)
        self.assertEqual(len(mq), 160)
        self.assertTrue(mq.korea_import_share.eq(140 / 600).all())
        validate_panels(details, monthly, quarterly, mm, mq, cfg)

    def test_quarter_share_is_ratio_of_sums_not_mean_of_monthly_shares(self):
        cfg, _, _, monthly, quarterly = self.panels()
        mm, mq = market_panel(monthly, "period_month", cfg), market_panel(quarterly, "quarter", cfg)
        self.assertAlmostEqual(mq.iloc[0].korea_import_share, 140 / 600)
        self.assertNotAlmostEqual(mq.iloc[0].korea_import_share, mm.korea_import_share.mean())

    def test_absent_korea_month_is_null_and_quarter_not_a_false_zero(self):
        cfg, _, _, monthly, quarterly = self.panels(korea=(10, None, 90))
        mm, mq = market_panel(monthly, "period_month", cfg), market_panel(quarterly, "quarter", cfg)
        self.assertTrue(pd.isna(mm.iloc[1].korea_import_value_usd))
        self.assertTrue(pd.isna(mm.iloc[1].korea_import_share))
        self.assertTrue(pd.isna(mq.iloc[0].korea_import_value_usd))
        self.assertTrue(pd.isna(mq.iloc[0].korea_import_share))
        q = quarterly.loc[quarterly.source_role.eq("korea_import")].iloc[0]
        self.assertEqual(q.observed_trade_value_usd, 100)
        self.assertEqual(q.valid_value_months, 2)

    def test_explicit_zero_import_is_preserved_and_zero_world_denominator_null(self):
        cfg, _, _, monthly, _ = self.panels(world=(0, 200, 300), korea=(0, 0, 30))
        result = market_panel(monthly, "period_month", cfg)
        self.assertEqual(result.iloc[0].market_import_value_usd, 0)
        self.assertTrue(pd.isna(result.iloc[0].korea_import_share))
        self.assertEqual(result.iloc[0].share_status, "zero_denominator")
        self.assertEqual(result.iloc[1].korea_import_share, 0)

    def test_missing_and_estimated_weight_never_discard_market_amount(self):
        cfg, details, requests, _, _ = self.panels()
        details.loc[details.source_role.eq("korea_import"), "net_weight_kg"] = pd.NA
        monthly, quarterly = make_panels(details, requests, cfg)
        result = market_panel(quarterly, "quarter", cfg)
        self.assertEqual(result.iloc[0].korea_import_value_usd, 140)
        self.assertTrue(pd.isna(result.iloc[0].korea_import_net_weight_kg))
        self.assertEqual(result.iloc[0].korea_import_estimated_weight_months, 3)
        self.assertAlmostEqual(result.iloc[0].korea_import_share, 140 / 600)

    def test_different_bases_or_editions_keep_amounts_and_flag_only_share(self):
        for column, value in (("valuation", "FOB"), ("hs_edition", "H5")):
            with self.subTest(column=column):
                cfg, _, _, monthly, _ = self.panels()
                monthly.loc[monthly.source_role.eq("korea_import"), column] = value
                result = market_panel(monthly, "period_month", cfg)
                self.assertTrue(result.korea_import_value_usd.notna().all())
                self.assertTrue(result.korea_import_share.isna().all())
                self.assertTrue(result.share_status.eq("basis_or_edition_mismatch").all())

    def test_korea_above_world_is_not_clipped_or_silently_discarded(self):
        cfg, _, _, monthly, _ = self.panels(korea=(101, 40, 90))
        result = market_panel(monthly, "period_month", cfg)
        self.assertEqual(result.iloc[0].korea_import_value_usd, 101)
        self.assertTrue(pd.isna(result.iloc[0].korea_import_share))
        self.assertEqual(result.iloc[0].share_status, "korea_exceeds_world")

    def test_duplicate_hs6_with_two_editions_requires_explicit_review(self):
        cfg, details, requests, _, _ = self.panels()
        duplicate = details.iloc[[0]].copy()
        duplicate["hs_edition"] = "H5"
        with self.assertRaises(ComtradeDataError):
            make_panels(pd.concat([details, duplicate], ignore_index=True), requests, cfg)

    def test_valuation_uses_values_not_country_assumptions(self):
        self.assertEqual(value_basis({"primaryValue": 100, "cifvalue": 100, "fobvalue": 90}, "world_import"), "CIF")
        self.assertEqual(value_basis({"primaryValue": 90, "cifvalue": None, "fobvalue": 90}, "world_import"), "FOB")
        self.assertEqual(value_basis({"primaryValue": 95, "cifvalue": 100, "fobvalue": 90}, "world_import"), "reported_unresolved")

    def test_numeric_country_code_and_iso_are_both_checked(self):
        countries = self.config["countries"]["countries"]
        india = next(r for r in countries if r["iso3"] == "IND")
        settings = self.config["sources"]["comtrade"]
        row = {"flowCode": "M", "reporterCode": 699, "reporterISO": "IND", "partnerCode": 410, "partnerISO": "KOR"}
        self.assertEqual(identify(row, countries, india, settings), ("IND", "korea_import"))
        row["reporterCode"] = 356
        with self.assertRaises(ComtradeDataError):
            identify(row, countries, india, settings)


if __name__ == "__main__":
    unittest.main()
