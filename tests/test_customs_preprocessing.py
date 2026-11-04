"""Guard against doubled totals, manufactured zeroes and partial targets."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config
from oinq.preprocessing.customs import CustomsDataError, make_panels, parse_response, save_table, validate_panels


class CustomsPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(ROOT)
        cls.fields = cls.config["sources"]["customs"]["response_fields"]

    def scope(self):
        config = deepcopy(self.config)
        config["countries"]["countries"] = [next(row for row in config["countries"]["countries"] if row["iso3"] == "USA")]
        config["products"]["products"] = [next(row for row in config["products"]["products"] if row["hs_code"] == "330499")]
        config["splits"]["collection"].update(monthly_start="2025-01", monthly_end="2025-03")
        return config

    def rows(self, values, *, extra=None):
        rows = []
        for month, amount in enumerate(values, start=1):
            if amount is None:
                continue
            rows.append({"country_code": "USA", "hs_code": "330499", "period_month": f"2025-{month:02d}",
                         "source_hs_code": "3304991000", "source_hs_digits": 10,
                         "export_value_usd": amount, "export_net_weight_kg": amount / 10})
        if extra:
            rows.extend(extra)
        requests = pd.DataFrame([{"country_code": "USA", "hs_code": "330499", "period_month": f"2025-{month:02d}",
                                  "source_file": "raw.xml", "collected_at": "2026-11-03T00:00:00+00:00",
                                  "request_success": True, "request_empty": False} for month in range(1, 4)])
        return pd.DataFrame(rows), requests

    def test_two_hs10_children_are_summed_once_and_quarter_is_complete(self):
        extra = [{"country_code": "USA", "hs_code": "330499", "period_month": "2025-01", "source_hs_code": "3304999000",
                  "source_hs_digits": 10, "export_value_usd": 5, "export_net_weight_kg": 1}]
        details, requests = self.rows([10, 20, 30], extra=extra)
        monthly, quarterly = make_panels(details, requests, self.scope())
        self.assertEqual(monthly.iloc[0]["export_value_usd"], 15)
        self.assertEqual(quarterly.iloc[0]["export_value_usd"], 65)
        self.assertTrue(quarterly.iloc[0]["target_value_complete"])
        validate_panels(details, monthly, quarterly, self.scope())

    def test_missing_month_stays_null_and_partial_sum_is_not_target(self):
        details, requests = self.rows([10, None, 30])
        monthly, quarterly = make_panels(details, requests, self.scope())
        self.assertTrue(pd.isna(monthly.iloc[1]["export_value_usd"]))
        self.assertEqual(quarterly.iloc[0]["observed_export_value_usd"], 40)
        self.assertEqual(quarterly.iloc[0]["observed_months"], 2)
        self.assertTrue(pd.isna(quarterly.iloc[0]["export_value_usd"]))
        self.assertEqual(quarterly.iloc[0]["data_status"], "partial")
        validate_panels(details, monthly, quarterly, self.scope())

    def test_explicit_zero_and_zero_weight_are_preserved(self):
        details, requests = self.rows([0, 10, 20])
        details.loc[1, "export_net_weight_kg"] = 0
        monthly, quarterly = make_panels(details, requests, self.scope())
        self.assertEqual(monthly.iloc[0]["export_value_usd"], 0)
        self.assertEqual(monthly.iloc[1]["export_net_weight_kg"], 0)
        self.assertEqual(quarterly.iloc[0]["export_value_usd"], 30)
        self.assertNotIn("export_unit_value_usd_per_kg", monthly.columns)

    def test_mixed_code_lengths_are_aggregated_in_the_same_hs6_product(self):
        extra = [{"country_code": "USA", "hs_code": "330499", "period_month": "2025-01", "source_hs_code": "330499",
                  "source_hs_digits": 6, "export_value_usd": 5, "export_net_weight_kg": 1}]
        details, requests = self.rows([10, 20, 30], extra=extra)
        monthly, quarterly = make_panels(details, requests, self.scope())
        self.assertEqual(len(details), 4)
        self.assertTrue(monthly.iloc[0]["hs_level_mixed"])
        self.assertFalse(monthly.iloc[0]["hs_level_review_required"])
        self.assertEqual(monthly.iloc[0]["export_value_usd"], 15)
        self.assertEqual(quarterly.iloc[0]["export_value_usd"], 65)
        self.assertEqual(quarterly.iloc[0]["observed_export_value_usd"], 65)
        validate_panels(details, monthly, quarterly, self.scope())

    def lip_scope(self, *, country="CHN", year="2016"):
        config = deepcopy(self.config)
        config["countries"]["countries"] = [next(row for row in config["countries"]["countries"] if row["iso3"] == country)]
        config["products"]["products"] = [next(row for row in config["products"]["products"] if row["hs_code"] == "330410")]
        config["splits"]["collection"].update(monthly_start=f"{year}-07", monthly_end=f"{year}-09")
        details, requests = self.rows([10, 20, 30])
        for frame in (details, requests):
            frame["country_code"] = country
            frame["hs_code"] = "330410"
            frame["period_month"] = [f"{year}-07", f"{year}-08", f"{year}-09"]
        details["source_hs_code"] = "3304101000"
        extra = details.iloc[[0]].copy()
        extra["source_hs_code"] = "330410"
        extra["source_hs_digits"] = 6
        extra["export_value_usd"] = 5
        extra["export_net_weight_kg"] = 1
        return config, pd.concat([details, extra], ignore_index=True), requests

    def test_user_directed_lip_sum_preserves_source_codes_and_flags_assumption(self):
        config, details, requests = self.lip_scope()
        monthly, quarterly = make_panels(details, requests, config, raw_snapshot="collection_20261103")
        self.assertEqual(monthly.iloc[0]["export_value_usd"], 15)
        self.assertEqual(quarterly.iloc[0]["export_value_usd"], 65)
        self.assertTrue(monthly.iloc[0]["hs_level_mixed"])
        self.assertTrue(monthly.iloc[0]["hs_level_user_approved"])
        self.assertFalse(monthly.iloc[0]["hs_level_review_required"])
        self.assertEqual(quarterly.iloc[0]["coverage_status"], "complete_with_configured_hs6_aggregation")
        self.assertEqual(quarterly.iloc[0]["aggregation_decision_basis"], "user_instruction")
        self.assertEqual(set(details["source_hs_code"]), {"330410", "3304101000"})
        validate_panels(details, monthly, quarterly, config)

    def test_context_policy_has_no_china_year_or_snapshot_exception(self):
        for country, year, snapshot in [("USA", "2016", "collection_20261103"),
                                         ("CHN", "2017", "collection_20261103"),
                                         ("CHN", "2016", "another_snapshot")]:
            with self.subTest(country=country, year=year, snapshot=snapshot):
                config, details, requests = self.lip_scope(country=country, year=year)
                monthly, quarterly = make_panels(details, requests, config, raw_snapshot=snapshot)
                self.assertFalse(monthly.iloc[0]["hs_level_review_required"])
                self.assertTrue(monthly.iloc[0]["hs_level_user_approved"])
                self.assertEqual(quarterly.iloc[0]["export_value_usd"], 65)

    def test_policy_covers_all_twenty_countries_and_eight_products(self):
        config = deepcopy(self.config)
        config["splits"]["collection"].update(monthly_start="2025-01", monthly_end="2025-03")
        details, requests = [], []
        for country in config["countries"]["countries"]:
            for product in config["products"]["products"]:
                for month, amount in enumerate([10, 20, 30], start=1):
                    base = {"country_code": country["iso3"], "hs_code": product["hs_code"],
                            "period_month": f"2025-{month:02d}"}
                    details.append({**base, "source_hs_code": product["hs_code"] + "0000", "source_hs_digits": 10,
                                    "export_value_usd": amount, "export_net_weight_kg": amount / 10})
                    requests.append({**base, "source_file": "raw.xml", "collected_at": "2026-11-03T00:00:00+00:00",
                                     "request_success": True, "request_empty": False})
                    if month == 1:
                        details.append({**base, "source_hs_code": product["hs_code"], "source_hs_digits": 6,
                                        "export_value_usd": 5, "export_net_weight_kg": 1})
        details = pd.DataFrame(details)
        monthly, quarterly = make_panels(details, pd.DataFrame(requests), config, raw_snapshot="any_snapshot")
        self.assertEqual(len(monthly), 480)
        self.assertEqual(len(quarterly), 160)
        self.assertTrue(quarterly["export_value_usd"].eq(65).all())
        self.assertFalse(monthly["hs_level_review_required"].any())
        validate_panels(details, monthly, quarterly, config)

    def test_missing_child_value_is_not_silently_ignored(self):
        extra = [{"country_code": "USA", "hs_code": "330499", "period_month": "2025-01", "source_hs_code": "3304999000",
                  "source_hs_digits": 10, "export_value_usd": None, "export_net_weight_kg": 1}]
        details, requests = self.rows([10, 20, 30], extra=extra)
        monthly, quarterly = make_panels(details, requests, self.scope())
        self.assertEqual(monthly.iloc[0]["observed_export_value_usd"], 10)
        self.assertTrue(pd.isna(monthly.iloc[0]["export_value_usd"]))
        self.assertTrue(pd.isna(quarterly.iloc[0]["export_value_usd"]))
        self.assertEqual(quarterly.iloc[0]["export_net_weight_kg"], 7)

    def test_absent_quarter_is_missing_without_zero_fill(self):
        details, requests = self.rows([10, 20, 30])
        next_quarter = requests.copy()
        next_quarter["period_month"] = ["2025-04", "2025-05", "2025-06"]
        requests = pd.concat([requests, next_quarter], ignore_index=True)
        config = self.scope()
        config["splits"]["collection"]["monthly_end"] = "2025-06"
        monthly, quarterly = make_panels(details, requests, config)
        self.assertTrue(monthly.iloc[3:]["export_value_usd"].isna().all())
        self.assertTrue(pd.isna(quarterly.iloc[1]["observed_export_value_usd"]))
        self.assertEqual(quarterly.iloc[1]["data_status"], "missing")
        validate_panels(details, monthly, quarterly, config)

    def test_detail_outside_request_scope_is_rejected(self):
        details, requests = self.rows([10, 20, 30])
        details.loc[0, "period_month"] = "2024-01"
        with self.assertRaises(CustomsDataError):
            make_panels(details, requests, self.scope())

    def test_source_detail_duplicate_is_rejected(self):
        details, requests = self.rows([10, 20, 30])
        details = pd.concat([details, details.iloc[[0]]], ignore_index=True)
        with self.assertRaises(CustomsDataError):
            make_panels(details, requests, self.scope())

    def xml(self, *, amount="10", period="2025.01", code="3304991000", country="US", total="10"):
        return (f'<response><header><resultCode>00</resultCode></header><body><items>'
                f'<item><year>총계</year><expDlr>{total}</expDlr><expWgt>1</expWgt></item>'
                f'<item><year>{period}</year><hsCd>{code}</hsCd><statCd>{country}</statCd>'
                f'<expDlr>{amount}</expDlr><expWgt>1</expWgt></item>'
                '</items></body></response>').encode("utf-8")

    def parse(self, body):
        return parse_response(body, {"collected_at": "2026-11-03T00:00:00+00:00"},
                              {"iso3": "USA", "customs": "US"}, {"hs_code": "330499"},
                              "2025-01", "2025-03", "raw.xml", self.fields)

    def test_total_row_is_removed_and_amount_reconciled(self):
        rows, reconciliation = self.parse(self.xml())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["export_value_usd"], 10)
        self.assertEqual(reconciliation["total_rows"], 1)
        self.assertEqual(reconciliation["export_value_difference_usd"], 0)

    def test_scope_mismatch_bad_amount_or_inconsistent_total_fails(self):
        for options in [{"period": "2025.04"}, {"code": "3305100000"}, {"country": "CN"},
                        {"amount": "-1"}, {"total": "999"}]:
            with self.subTest(options=options), self.assertRaises(CustomsDataError):
                self.parse(self.xml(**options))

    def test_parquet_roundtrip_keeps_null_and_hs_string(self):
        details, requests = self.rows([10, None, 30])
        for column in ["country_code", "hs_code", "period_month", "source_hs_code"]:
            details[column] = details[column].astype(pd.StringDtype(storage="python"))
        monthly, _ = make_panels(details, requests, self.scope())
        with tempfile.TemporaryDirectory() as folder:
            files = save_table(monthly, Path(folder), "monthly")
            self.assertEqual(len(files), 2)
            restored = pd.read_parquet(Path(folder) / "monthly.parquet")
            self.assertEqual(restored.iloc[0]["hs_code"], "330499")
            self.assertTrue(pd.isna(restored.iloc[1]["export_value_usd"]))
            csv = pd.read_csv(Path(folder) / "monthly.csv", encoding="utf-8-sig",
                              dtype={column: monthly[column].dtype for column in monthly},
                              keep_default_na=False, na_values=[""])
            csv = csv.astype({column: restored[column].dtype for column in restored
                              if isinstance(restored[column].dtype, pd.StringDtype)})
            pd.testing.assert_frame_equal(restored, csv)
            with self.assertRaises(CustomsDataError):
                save_table(monthly, Path(folder), "monthly")


if __name__ == "__main__":
    unittest.main()
