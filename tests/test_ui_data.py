"""Validation and aggregation checks for persisted Streamlit result data."""

import math
import unittest

import pandas as pd

from oinq.ui.data import annual_sum_from_quarters, audit_forecast_grid, filter_forecasts


class UIDataTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"country_code": country, "hs_code": hs, "target_quarter": quarter,
             "predicted_export_value_usd": 100.0}
            for country in ("CHN", "JPN")
            for hs in ("330410", "330420")
            for quarter in ("2027Q1", "2027Q2")
        ])

    def test_country_and_product_filters_apply_together(self):
        filtered = filter_forecasts(
            self.frame,
            country_codes=["CHN"],
            hs_codes=["330420"],
            target_quarters=["2027Q2"],
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["country_code"], "CHN")
        self.assertEqual(filtered.iloc[0]["hs_code"], "330420")
        self.assertEqual(filtered.iloc[0]["target_quarter"], "2027Q2")
        self.assertEqual(len(self.frame), 8, "filter must not mutate its input")

    def test_grid_audit_reports_missing_and_duplicate_keys(self):
        damaged = pd.concat([self.frame.iloc[1:], self.frame.iloc[[1]]], ignore_index=True)
        audit = audit_forecast_grid(
            damaged, ["CHN", "JPN"], ["330410", "330420"], ["2027Q1", "2027Q2"]
        )
        self.assertEqual(audit.expected_rows, 8)
        self.assertEqual(audit.observed_rows, 8)
        self.assertEqual(len(audit.missing_keys), 1)
        self.assertEqual(audit.missing_keys[0], ("CHN", "330410", "2027Q1"))
        self.assertEqual(audit.duplicate_keys, (("CHN", "330410", "2027Q2"),))
        self.assertEqual(audit.unexpected_keys, ())

    def test_incomplete_annual_sum_is_not_reported_as_complete(self):
        full = annual_sum_from_quarters([10.0, 20.0, 30.0, 40.0])
        incomplete = annual_sum_from_quarters([10.0, None, 30.0, 40.0])
        missing_quarter = annual_sum_from_quarters([10.0, 20.0, 30.0])
        self.assertEqual(full, 100.0)
        self.assertTrue(math.isnan(incomplete))
        self.assertTrue(math.isnan(missing_quarter))


if __name__ == "__main__":
    unittest.main()
