"""Independently reconcile all period features and trade sums against source panels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.preprocessing.customs import sha256


def audit(version):
    folder = ROOT / "data/processed/unsupervised" / version
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    checked_files = 0
    for record in manifest["inputs"]:
        assert sha256(ROOT / record["path"]) == record["sha256"], "Input changed: " + record["path"]
    baseline_path = ROOT / ".private/unsupervised_v001_protected_inputs.json"
    protected = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    for name, digest in protected.items():
        assert sha256(ROOT / name) == digest, "Protected data changed: " + name
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert sha256(path) == record["sha256"]
        checked_files += 1
        if record["format"] != "parquet":
            continue
        f = pd.read_parquet(path)
        floats = {c: lambda value: pd.NA if value == "\\N" else float(value)
                  for c in f if pd.api.types.is_float_dtype(f[c].dtype)}
        csv = pd.read_csv(path.with_suffix(".csv"), encoding="utf-8-sig", keep_default_na=False, na_values=["\\N"],
                          dtype={c: t for c, t in f.dtypes.items() if c not in floats}, converters=floats,
                          low_memory=False).astype(f.dtypes.to_dict())
        pd.testing.assert_frame_equal(f, csv, check_exact=True)
    annual = pd.read_parquet(ROOT / "data/processed/world_bank/v001/macro_annual.parquet")
    pd.testing.assert_frame_equal(annual, pd.read_parquet(folder / "macro_annual.parquet"), check_exact=True)
    features = pd.read_parquet(folder / "country_window_features.parquet")
    profile = pd.read_parquet(folder / "country_window_profile.parquet")
    detail = pd.read_parquet(folder / "macro_window_long.parquet")
    assert len(features) == 100 and features.eligible_for_fit.sum() == manifest["fit_eligible_rows"]
    assert not features.end_year.gt(2026).any()
    latest = features.loc[features.window_id.eq("2025_2026Q3")]
    assert len(latest) == 20 and latest.period_month_count.eq(21).all()
    assert set(latest.loc[latest.eligible_for_fit, "country_code"]) == {"CHN", "IND"}
    assert not features.duplicated(["country_code", "window_id"]).any()
    assert features.groupby("window_id").country_code.nunique().eq(20).all()
    for r in detail.itertuples():
        vals = annual.loc[annual.country_code.eq(r.country_code) & annual.indicator_year.between(r.start_year, r.end_year), r.indicator_name].dropna()
        assert len(vals) == r.valid_year_count
        if len(vals):
            assert np.isclose(float(vals.mean()), r.observed_mean, rtol=1e-14)
        else:
            assert pd.isna(r.observed_mean)
        complete = len(vals) == r.expected_year_count and r.end_year < int(manifest["as_of_date"][:4])
        usable = len(vals) > 0 if r.period_status == "observed_to_cutoff" else complete
        assert r.complete == complete and (pd.notna(r.value) == usable)
        if usable:
            assert np.isclose(r.value, vals.mean(), rtol=1e-14)
    monthly = pd.read_parquet(ROOT / "data/processed/customs/v003/customs_monthly_hs6.parquet")
    aggregate_rows = 0
    for name, product in [("country_product_export_annual", True), ("country_export_annual", False),
                           ("country_product_export_window", True), ("country_export_window", False)]:
        table = pd.read_parquet(folder / (name + ".parquet"))
        for r in table.itertuples():
            data = monthly.loc[monthly.country_code.eq(r.country_code) & monthly.period_month.str[:4].astype(int).between(r.start_year, r.end_year)]
            if product:
                data = data.loc[data.hs_code.eq(r.hs_code)]
            n = data.export_value_usd.notna().sum()
            assert n == r.valid_value_cells and len(data) == r.expected_collected_value_cells
            for c in ["export_value_usd", "export_net_weight_kg"]:
                values = data[c].dropna()
                value, observed = getattr(r, c), getattr(r, "observed_" + c)
                assert pd.notna(value) == (len(values) == r.expected_window_value_cells)
                if len(values):
                    assert np.isclose(observed, values.sum(), rtol=1e-14)
                else:
                    assert pd.isna(observed)
                if pd.notna(value):
                    assert np.isclose(value, values.sum(), rtol=1e-14)
            assert r.missing_collected_value_cells == len(data) - n
            if r.start_year <= 2026 <= r.end_year:
                assert r.disclosure_status == "final_test_restricted_local"
            aggregate_rows += 1
    for name, n in [("country_window_context", 100), ("country_product_window_context", 800)]:
        frame = pd.read_parquet(folder / (name + ".parquet"))
        assert len(frame) == n
        key = ["country_code", "window_id"]
        check = frame.merge(profile[key + manifest["feature_columns"]], on=key, validate="many_to_one", suffixes=("", "_source"))
        for c in manifest["feature_columns"]:
            pd.testing.assert_series_equal(check[c], check[c + "_source"], check_names=False)
        keys = ["country_code"] + (["hs_code"] if "product" in name else []) + ["window_id"]
        index = frame.set_index(keys)
        for row in frame.itertuples():
            for c in ["gdp_usd", *manifest["feature_columns"], "export_value_usd"]:
                actual = getattr(row, c + ("_change_pp" if c == "internet_users_pct" else "_change_pct"))
                expected = np.nan
                if pd.notna(row.previous_window_id):
                    key = (row.country_code,) + ((row.hs_code,) if "product" in name else ()) + (row.previous_window_id,)
                    old, new = index.loc[key, c], getattr(row, c)
                    same_duration = c != "export_value_usd" or row.export_value_comparison_status == "same_duration"
                    if pd.notna(old) and pd.notna(new) and same_duration:
                        expected = new - old if c == "internet_users_pct" else (new - old) / old * 100 if old != 0 else np.nan
                assert pd.isna(actual) if pd.isna(expected) else np.isclose(actual, expected, rtol=1e-14, atol=1e-12)
        coverage = [c for c in [*keys, "country_name", "period_status", "eligible_for_fit", "eligibility_reason",
                    "valid_value_cells", "expected_collected_value_cells", "expected_window_value_cells",
                    "missing_collected_value_cells", "outside_collection_value_cells", "observed_period_start",
                    "observed_period_end", "coverage_status", "disclosure_status"] if c in frame]
        frame[coverage].to_csv(ROOT / "reports" / f"unsupervised_{version}_{name}_coverage.csv",
                               encoding="utf-8-sig", index=False, na_rep="\\N")
    result = {"status": "passed", "processing_version": version, "protected_existing_files": len(protected),
              "csv_parquet_files_verified": checked_files, "macro_window_cells_verified": len(detail),
              "trade_aggregate_rows_verified": aggregate_rows, "countries": 20, "products": 8,
              "feature_rows": 100, "fit_eligible_rows": int(features.eligible_for_fit.sum()),
              "new_model_fitted": False, "shared_release_created": False}
    (ROOT / "reports" / f"unsupervised_{version}_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.version.replace("_", "").replace("-", "").isalnum():
        parser.error("invalid version")
    audit(args.version)
