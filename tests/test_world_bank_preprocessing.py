"""Preserve annual values, missing distinctions and separate cluster candidates."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config
from oinq.preprocessing.world_bank import (WorldBankDataError, make_clustering_candidate,
                                          make_panels, normalize_row, save_table, validate_panels)


class WorldBankPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(ROOT)

    def scope(self):
        config = deepcopy(self.config)
        config["splits"]["collection"].update(monthly_start="2024-01", monthly_end="2026-09")
        return config

    def details(self, config):
        rows = []
        values = {"gdp_usd": 1000000000.123, "gdp_per_capita_usd": 1000.123456,
                  "population": 1000000, "inflation_pct": -0.2, "exchange_rate_lcu_per_usd": 1.4,
                  "internet_users_pct": 70.5}
        for country in config["countries"]["countries"]:
            for year in (2024, 2025):
                for name in config["sources"]["world_bank"]["indicators"]:
                    value = values[name]
                    if name == "internet_users_pct" and year == 2025 and country["iso3"] not in ("CHN", "IND"):
                        value = None
                    rows.append({"country_code": country["iso3"], "indicator_year": year, "indicator_name": name,
                                 "indicator_code": config["sources"]["world_bank"]["indicators"][name]["code"],
                                 "value": value, "source_file": "original.json", "source_row_number": len(rows),
                                 "collected_at": "2026-11-03T00:00:00+00:00", "source_lastupdated": "2026-10-08",
                                 "source_obs_status": ""})
        frame = pd.DataFrame(rows).convert_dtypes()
        frame["value"] = frame.value.astype("Float64")
        return frame

    def test_twenty_countries_six_indicators_retained_in_correct_grids(self):
        config = self.scope()
        details = self.details(config)
        long, annual = make_panels(details, config)
        candidate, _, year = make_clustering_candidate(annual, config)
        validate_panels(details, long, annual, candidate, config)
        self.assertEqual(len(long), 360)
        self.assertEqual(len(annual), 60)
        self.assertEqual(long.record_observed.sum(), 240)
        self.assertEqual(year, 2024)
        self.assertEqual(len(candidate), 20)

    def test_source_null_and_absent_year_are_separate_and_neither_is_zero(self):
        config = self.scope()
        long, annual = make_panels(self.details(config), config)
        null = long.loc[long.country_code.eq("USA") & long.indicator_name.eq("internet_users_pct") & long.indicator_year.eq(2025)].iloc[0]
        absent = long.loc[long.country_code.eq("USA") & long.indicator_name.eq("internet_users_pct") & long.indicator_year.eq(2026)].iloc[0]
        self.assertTrue(null.record_observed)
        self.assertFalse(absent.record_observed)
        self.assertEqual(null.missing_reason, "null_value_in_source")
        self.assertEqual(absent.missing_reason, "no_record_in_successful_response")
        self.assertTrue(pd.isna(null.value) and pd.isna(absent.value))
        self.assertTrue(annual.loc[annual.indicator_year.eq(2026), "internet_users_pct"].isna().all())

    def test_negative_inflation_zero_and_decimal_values_are_not_changed(self):
        config = self.scope()
        details = self.details(config)
        details.loc[details.indicator_name.eq("internet_users_pct") & details.indicator_year.eq(2024), "value"] = 0
        long, annual = make_panels(details, config)
        self.assertTrue(annual.loc[annual.indicator_year.eq(2024), "inflation_pct"].eq(-.2).all())
        self.assertTrue(annual.loc[annual.indicator_year.eq(2024), "internet_users_pct"].eq(0).all())
        self.assertEqual(annual.iloc[0].gdp_usd, 1000000000.123)
        self.assertEqual(long.quality_issue.notna().sum(), 0)

    def test_cluster_candidate_has_one_common_year_raw_values_no_training_approval(self):
        config = self.scope()
        _, annual = make_panels(self.details(config), config)
        candidate, coverage, year = make_clustering_candidate(annual, config)
        self.assertEqual(candidate.reference_year.unique().tolist(), [2024])
        self.assertEqual(candidate.iloc[0].internet_users_pct, 70.5)
        self.assertFalse(candidate.scaled.any())
        self.assertFalse(candidate.training_approved.any())
        self.assertFalse(candidate.historical_release_verified.any())
        self.assertEqual(coverage.loc[coverage.indicator_year.eq(2025), "countries_with_three_values"].iloc[0], 2)

    def test_no_common_year_does_not_mix_years_or_drop_countries_from_annual(self):
        config = self.scope()
        details = self.details(config)
        details.loc[details.country_code.eq("USA") & details.indicator_name.eq("internet_users_pct"), "value"] = pd.NA
        _, annual = make_panels(details, config)
        candidate, _, year = make_clustering_candidate(annual, config)
        self.assertIsNone(candidate)
        self.assertIsNone(year)
        self.assertEqual(annual.country_code.nunique(), 20)

    def test_out_of_range_value_is_flagged_and_original_value_retained(self):
        config = self.scope()
        details = self.details(config)
        mask = details.country_code.eq("USA") & details.indicator_year.eq(2024) & details.indicator_name.eq("internet_users_pct")
        details.loc[mask, "value"] = 101
        long, annual = make_panels(details, config)
        row = long.loc[long.country_code.eq("USA") & long.indicator_year.eq(2024) & long.indicator_name.eq("internet_users_pct")].iloc[0]
        self.assertEqual(row.value, 101)
        self.assertEqual(row.quality_issue, "internet_percent_out_of_range")
        self.assertEqual(annual.loc[annual.country_code.eq("USA") & annual.indicator_year.eq(2024), "internet_users_pct"].iloc[0], 101)

    def test_duplicate_country_indicator_year_is_never_summed(self):
        config = self.scope()
        details = self.details(config)
        with self.assertRaises(WorldBankDataError):
            make_panels(pd.concat([details, details.iloc[[0]]]), config)

    def test_api_lastupdated_decimal_and_blank_unit_do_not_change_value_or_release(self):
        config = self.scope()
        settings = config["sources"]["world_bank"]
        countries = {r["iso3"]: r for r in config["countries"]["countries"]}
        row = {"countryiso3code": "USA", "country": {"id": "US", "value": "United States"},
               "indicator": {"id": "NY.GDP.MKTP.CD", "value": "GDP (current US$)"},
               "date": "2024", "value": 1234567.89123, "unit": "", "obs_status": "", "decimal": 0}
        result = normalize_row(row, "gdp_usd", settings, countries, [2024, 2025, 2026], "raw.json", 0,
                               {"collected_at": "2026-11-03"}, {"lastupdated": "2026-10-08", "page": 1, "sourceid": "2"})
        self.assertEqual(result["value"], row["value"])
        self.assertEqual(result["source_unit"], "")
        self.assertEqual(result["unit"], "current_USD")
        self.assertNotIn("released_at", result)
        row["country"]["id"] = "CN"
        with self.assertRaises(WorldBankDataError):
            normalize_row(row, "gdp_usd", settings, countries, [2024], "raw.json", 0,
                          {"collected_at": "2026-11-03"}, {"page": 1, "sourceid": "2"})

    def test_csv_preserves_blank_source_unit_separately_from_missing_observation(self):
        frame = pd.DataFrame({"source_unit": pd.Series(["", pd.NA, "percent"], dtype="string[python]"),
                              "value": pd.Series([1.23456789012345, pd.NA, 0], dtype="Float64")})
        with tempfile.TemporaryDirectory(dir=ROOT / ".private") as folder:
            save_table(frame, Path(folder), "roundtrip")
            restored = pd.read_parquet(Path(folder) / "roundtrip.parquet").astype(frame.dtypes.to_dict())
            pd.testing.assert_frame_equal(frame, restored, check_exact=True)


if __name__ == "__main__":
    unittest.main()
