"""Protect temporal aggregation, missingness and strict adjacent comparisons."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config, validate_project_config, ConfigError
from oinq.preprocessing.unsupervised import make_macro_windows, aggregate_trade, add_changes, PROFILE


class UnsupervisedTests(unittest.TestCase):
    def setUp(self):
        self.config = load_project_config(ROOT)

    def test_cross_file_period_policy_rejects_bad_edits(self):
        for path, value in [(("features", "clustering", "row_key"), "country_code"),
                            (("features", "clustering", "shared_scaler_and_centroids_across_windows"), False),
                            (("models", "kmeans", "inputs", "fit_scope"), "per_window_refit")]:
            config = deepcopy(self.config)
            node = config
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            with self.assertRaises(ConfigError):
                validate_project_config(config)

    def test_macro_missing_year_is_not_partial_training_mean(self):
        cfg = deepcopy(self.config)
        cfg["countries"]["countries"] = [cfg["countries"]["countries"][0]]
        code = cfg["countries"]["countries"][0]["iso3"]
        rows = []
        for y in range(2013, 2027):
            row = {"country_code": code, "indicator_year": y}
            for name in PROFILE:
                row.update({name: float(y), name + "_source_file": "annual.json", name + "_collected_at": "2026-11-03"})
            rows.append(row)
        annual = pd.DataFrame(rows)
        annual.loc[annual.indicator_year.eq(2014), "internet_users_pct"] = float("nan")
        long, profile, features = make_macro_windows(annual, cfg, "2026-11-04")
        first = profile.iloc[0]
        self.assertTrue(pd.isna(first.internet_users_pct))
        self.assertEqual(first.internet_users_pct_observed_mean, 2014.0)
        self.assertFalse(first.eligible_for_fit)
        self.assertEqual(len(features), 5)
        self.assertTrue(features.iloc[-1].eligible_for_fit)
        self.assertEqual(profile.iloc[-1].population_observed_years, "[2025]")
        self.assertEqual(profile.iloc[-1].population, 2025.0)
        _, later_profile, _ = make_macro_windows(annual, cfg, "2028-01-01")
        self.assertEqual(later_profile.iloc[-1].population_observed_years, "[2025]")
        annual.loc[annual.indicator_year.eq(2025), "internet_users_pct"] = float("nan")
        _, _, missing = make_macro_windows(annual, cfg, "2026-11-04")
        self.assertFalse(missing.iloc[-1].eligible_for_fit)
        self.assertIn("missing", missing.iloc[-1].eligibility_reason)

    def test_trade_keeps_zero_and_missing_and_does_not_annualize(self):
        dates = pd.period_range("2025-01", "2026-09", freq="M").astype(str)
        monthly = pd.DataFrame({"country_code": "CHN", "hs_code": "330410", "country_name": "China",
            "region": "asia", "product_name": "lip", "period_month": dates, "export_value_usd": 0.0,
            "export_net_weight_kg": 0.0, "source_file": "source.xml", "collected_at": "2026-11-03",
            "missing_reason": pd.Series(pd.NA, index=range(len(dates)), dtype="string")})
        monthly.loc[0, "export_value_usd"] = float("nan")
        monthly.loc[0, "missing_reason"] = "not_reported"
        windows = [{"window_id": "2025_2026Q3", "start_year": 2025, "end_year": 2026,
                    "start_month": "2025-01", "end_month": "2026-09", "short_period": True}]
        row = aggregate_trade(monthly, windows, True, "window", "2026-11-04").iloc[0]
        self.assertTrue(pd.isna(row.export_value_usd))
        self.assertEqual(row.observed_export_value_usd, 0.0)
        self.assertEqual(row.missing_collected_value_cells, 1)
        self.assertEqual(row.expected_window_value_cells, 21)
        self.assertEqual(row.outside_collection_value_cells, 0)
        self.assertEqual(row.elapsed_outside_collection_value_cells, 0)
        self.assertEqual(row.current_month_outside_collection_value_cells, 0)
        self.assertEqual(row.future_value_cells, 0)
        self.assertEqual(row.disclosure_status, "final_test_restricted_local")
        monthly.loc[0, "export_value_usd"] = 0.0
        complete = aggregate_trade(monthly, windows, True, "window", "2026-11-04").iloc[0]
        self.assertEqual(complete.export_value_usd, 0.0)
        annual = aggregate_trade(monthly, windows, True, "annual", "2026-11-04")
        latest = annual.loc[annual.year.eq(2026)].iloc[0]
        self.assertEqual(latest.expected_window_value_cells, 9)
        self.assertEqual(latest.export_value_usd, 0.0)

    def test_changes_use_adjacent_complete_values_and_percent_points(self):
        rows = []
        for i, w in enumerate(self.config["features"]["clustering"]["context"]["trade"]["windows"]):
            rows.append({"country_code": "CHN", "window_id": w["window_id"],
                         **{name: 100.0 + i * 10 for name in PROFILE}, "export_value_usd": 0.0 if i == 0 else 100.0})
        frame = pd.DataFrame(rows)
        frame.loc[2, "gdp_usd"] = float("nan")
        result = add_changes(frame, self.config)
        self.assertEqual(result.iloc[1].internet_users_pct_change_pp, 10.0)
        self.assertAlmostEqual(result.iloc[1].gdp_per_capita_usd_change_pct, 10.0)
        self.assertTrue(pd.isna(result.iloc[1].export_value_usd_change_pct))
        self.assertTrue(pd.isna(result.iloc[3].gdp_usd_change_pct))
        self.assertTrue(pd.isna(result.iloc[-1].export_value_usd_change_pct))


if __name__ == "__main__":
    unittest.main()
