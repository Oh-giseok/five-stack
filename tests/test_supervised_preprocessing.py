"""Protect all calendars, source values, causal lags and frozen test origins."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config
from oinq.preprocessing.supervised import (SupervisedDataError, integrate_observations, make_origin_features,
                                           make_samples, safe_unit_value, typed, validate_outputs)


class SupervisedPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(ROOT)
        cls.config["splits"]["collection"].update(monthly_start="2013-01", monthly_end="2015-09")
        cls.config["splits"]["splits"] = {"train": {"start": "2013Q1", "end": "2013Q4"},
                                            "validation": {"start": "2014Q1", "end": "2014Q4"},
                                            "final_test": {"start": "2015Q1", "end": "2015Q3"}}
        cls.config["splits"]["final_test"]["origin_quarter"] = "2014Q4"
        months, quarters, markets_m, markets_q, annual = [], [], [], [], []
        for ci, country in enumerate(cls.config["countries"]["countries"]):
            for year in (2013, 2014, 2015):
                annual.append({"country_code": country["iso3"], "indicator_year": year,
                               "country_name": country["name_ko"], "region": country["region"],
                               **{name: float(year * 100 + ci) for name in cls.config["features"]["supervised"]["macro"]["columns"]}})
            for pi, product in enumerate(cls.config["products"]["products"]):
                common = {"country_code": country["iso3"], "country_name": country["name_ko"], "region": country["region"],
                          "hs_code": product["hs_code"], "product_name": product["name_ko"], "currency": "USD", "weight_unit": "kg",
                          "source_file": "raw.xml", "collected_at": "2026-11-03T00:00:00+00:00", "data_status": "actual"}
                for j, period in enumerate(pd.period_range("2013-01", "2015-09", freq="M")):
                    months.append({**common, "period_month": str(period), "quarter": str(period.asfreq("Q")),
                                   "export_value_usd": float(10000 * ci + 100 * pi + j), "export_net_weight_kg": float(j + 1)})
                    markets_m.append({**{k: v for k, v in common.items() if k != "source_file"}, "period_month": str(period),
                                      "market_import_value_usd": 1000., "korea_import_value_usd": 100., "korea_import_share": .1})
                for j, period in enumerate(pd.period_range("2013Q1", "2015Q3", freq="Q")):
                    quarters.append({**common, "quarter": str(period), "export_value_usd": float(10000 * ci + 100 * pi + j),
                                     "export_net_weight_kg": float(j + 1), "observed_months": 3})
                    markets_q.append({**{k: v for k, v in common.items() if k != "source_file"}, "quarter": str(period),
                                      "market_import_value_usd": 3000., "korea_import_value_usd": 300., "korea_import_share": .1})
        cls.months, cls.quarters, cls.market_m, cls.market_q, cls.annual = map(typed, map(pd.DataFrame, (months, quarters, markets_m, markets_q, annual)))
        cls.months, _ = integrate_observations(cls.months, cls.market_m, cls.annual, "period_month")
        cls.quarters, _ = integrate_observations(cls.quarters, cls.market_q, cls.annual, "quarter")

    def row(self, features, origin, country="USA", hs="330499"):
        return features.loc[features.country_code.eq(country) & features.hs_code.eq(hs) & features.origin_quarter.eq(origin)].iloc[0]

    def test_all_twenty_countries_eight_products_and_every_month_quarter_retained(self):
        self.assertEqual(len(self.months), 20 * 8 * 33)
        self.assertEqual(len(self.quarters), 20 * 8 * 11)
        self.assertEqual(self.months.month.nunique(), 12)
        self.assertTrue(self.quarters.period_months.str.count(";").eq(2).all())
        features, _ = make_origin_features(self.quarters, self.annual, self.config)
        samples = make_samples(self.quarters, features, self.config)
        validate_outputs(self.months, self.quarters, features, samples, self.config)

    def test_missing_amount_and_zero_weight_are_retained_without_fake_unit_values(self):
        base = self.quarters[[c for c in self.quarters if not c.startswith("comtrade_")]].copy()
        base.loc[0, "export_value_usd"] = pd.NA
        base.loc[1, "export_net_weight_kg"] = 0
        result = safe_unit_value(base)
        self.assertEqual(len(result), len(base))
        self.assertTrue(pd.isna(result.iloc[0].export_value_usd))
        self.assertTrue(pd.isna(result.iloc[0].export_unit_value_usd_per_kg))
        self.assertEqual(result.iloc[1].export_net_weight_kg, 0)
        self.assertTrue(pd.isna(result.iloc[1].export_unit_value_usd_per_kg))

    def test_lags_follow_origin_not_target_and_do_not_cross_country_or_product(self):
        features, _ = make_origin_features(self.quarters, self.annual, self.config)
        source = self.quarters.loc[self.quarters.country_code.eq("USA") & self.quarters.hs_code.eq("330499")].set_index("quarter")
        row = self.row(features, "2014Q1")
        self.assertEqual(row.export_value_usd_lag1, source.loc["2014Q1", "export_value_usd"])
        self.assertEqual(row.export_value_usd_lag2, source.loc["2013Q4", "export_value_usd"])
        self.assertEqual(row.export_value_usd_lag4, source.loc["2013Q2", "export_value_usd"])
        self.assertEqual(row.export_value_usd_rolling4_sum, source.loc[["2013Q2", "2013Q3", "2013Q4", "2014Q1"], "export_value_usd"].sum())

    def test_four_quarter_rolling_does_not_use_partial_sums_or_impute_gaps(self):
        q = self.quarters.copy()
        mask = q.country_code.eq("USA") & q.hs_code.eq("330499") & q.quarter.eq("2013Q4")
        q.loc[mask, "export_value_usd"] = pd.NA
        features, _ = make_origin_features(q, self.annual, self.config)
        row = self.row(features, "2014Q1")
        self.assertTrue(pd.isna(row.export_value_usd_lag2))
        self.assertTrue(pd.isna(row.export_value_usd_rolling4_sum))
        self.assertEqual(row.export_value_usd_rolling4_valid_quarters, 3)

    def test_macro_candidate_is_previous_calendar_year_not_same_year_observation(self):
        features, _ = make_origin_features(self.quarters, self.annual, self.config)
        row = self.row(features, "2014Q1")
        expected = self.annual.loc[self.annual.country_code.eq("USA") & self.annual.indicator_year.eq(2013)].iloc[0]
        self.assertEqual(row.gdp_usd, expected.gdp_usd)
        self.assertEqual(row.macro_candidate_year, 2013)
        self.assertFalse(row.training_approved)
        self.assertTrue(pd.isna(row.origin_cutoff_at))
        self.assertTrue(pd.isna(self.row(features, "2013Q1").gdp_usd))

    def test_future_values_do_not_change_earlier_features(self):
        earlier, numeric = make_origin_features(self.quarters, self.annual, self.config)
        altered = self.quarters.copy()
        altered.loc[altered.quarter.gt("2014Q1"), ["export_value_usd", "market_import_value_usd"]] = 999999999
        later, _ = make_origin_features(altered, self.annual, self.config)
        pd.testing.assert_series_equal(self.row(earlier, "2014Q1")[numeric], self.row(later, "2014Q1")[numeric])

    def test_missing_target_candidates_are_preserved_and_final_test_origin_frozen(self):
        q = self.quarters.copy()
        mask = q.country_code.eq("USA") & q.hs_code.eq("330499") & q.quarter.eq("2013Q1")
        q.loc[mask, "export_value_usd"] = pd.NA
        features, numeric = make_origin_features(q, self.annual, self.config)
        samples = make_samples(q, features, self.config)
        self.assertEqual(len(samples["train"]), 20 * 8 * 4 * 5)
        self.assertEqual((~samples["train"].target_available).sum(), 5)
        self.assertEqual(len(samples["final_test"]), 480)
        self.assertTrue(samples["final_test"].origin_quarter.eq("2014Q4").all())
        self.assertEqual(set(samples["final_test"].horizon_quarters), {1, 2, 3})
        self.assertNotIn("target_export_value_usd", numeric)
        self.assertNotIn("target_export_net_weight_kg", numeric)

    def test_missing_market_grid_or_duplicate_key_cannot_silently_drop_rows(self):
        with self.assertRaises(SupervisedDataError):
            integrate_observations(self.quarters, self.market_q.iloc[1:], self.annual, "quarter")
        with self.assertRaises(SupervisedDataError):
            integrate_observations(self.quarters, pd.concat([self.market_q, self.market_q.iloc[[0]]]), self.annual, "quarter")


if __name__ == "__main__":
    unittest.main()
