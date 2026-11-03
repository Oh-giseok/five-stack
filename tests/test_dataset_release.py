"""Release approval must preserve rows and keep final test labels private."""

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.preprocessing.release import RELEASE_METADATA, ReleaseError, release_columns, select_release


class DatasetReleaseTests(unittest.TestCase):
    def contract(self):
        return {"categorical": ["country_code", "hs_code", "region"], "known_future": ["target_year"],
                "numeric_candidates": ["export_value_usd_lag1"],
                "forbidden_feature_columns": ["target_export_value_usd", "target_available", "split"]}

    def frame(self):
        values = {c: ["metadata", "metadata"] for c in RELEASE_METADATA}
        values.update(country_code=["USA", "USA"], hs_code=["330499", "330499"], region=["americas", "americas"],
                      target_year=[2021, 2021], target_quarter=["2021Q1", "2021Q2"],
                      origin_quarter=["2020Q4", "2021Q1"], export_value_usd_lag1=[10., None],
                      target_export_value_usd=[20., None], target_available=[True, False], split=["train", "train"],
                      internal_source_file=["private.xml", "private.xml"])
        return pd.DataFrame(values).convert_dtypes()

    def test_release_keeps_missing_target_rows_and_selects_only_declared_columns(self):
        frame = self.frame()
        result = select_release(frame, self.contract(), {"start": "2013Q1", "end": "2021Q4"}, "train")
        self.assertEqual(len(result), 2)
        self.assertTrue(pd.isna(result.iloc[1].target_export_value_usd))
        self.assertNotIn("internal_source_file", result)
        self.assertIn("country_name", result)
        self.assertIn("target_period_months", result)
        pd.testing.assert_frame_equal(result, frame[release_columns(self.contract())])

    def test_final_test_target_cannot_be_released_even_with_wide_bounds(self):
        frame = self.frame()
        frame.loc[0, "target_quarter"] = "2026Q1"
        with self.assertRaises(ReleaseError):
            select_release(frame, self.contract(), {"start": "2013Q1", "end": "2026Q3"}, "train")

    def test_target_and_target_quality_cannot_be_declared_as_features(self):
        for forbidden in ("target_export_value_usd", "target_available", "split"):
            contract = self.contract()
            contract["numeric_candidates"].append(forbidden)
            with self.assertRaises(ReleaseError):
                release_columns(contract)

    def test_wrong_period_split_and_duplicate_sample_are_rejected(self):
        bounds = {"start": "2013Q1", "end": "2021Q4"}
        frame = self.frame()
        for data in (frame.assign(split="validation"), frame.assign(target_quarter="2022Q1"),
                     pd.concat([frame, frame.iloc[[0]]], ignore_index=True)):
            with self.assertRaises(ReleaseError):
                select_release(data, self.contract(), bounds, "train")


if __name__ == "__main__":
    unittest.main()
