"""Independently audit source preservation, calendars, lags and sample labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.common.config import load_project_config
from oinq.collectors.raw import utc_now, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_number(actual, expected):
    if expected is None or pd.isna(expected):
        return bool(pd.isna(actual))
    return pd.notna(actual) and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9)


def assert_mapping(source, integrated, mapping, key, deduplicate=False):
    actual = integrated[[mapping[c] for c in source]].rename(columns={mapping[c]: c for c in source})
    if deduplicate:
        actual = actual.drop_duplicates(key)
    expected = source.copy()
    for name in key:
        dtype = "Int64" if name == "indicator_year" else "string[python]"
        expected[name] = expected[name].astype(dtype)
        actual[name] = actual[name].astype(dtype)
    expected = expected.set_index(key).sort_index()
    actual = actual.set_index(key).sort_index()
    pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=True)


def audit(version):
    config = load_project_config(ROOT)
    folder = ROOT / "data/processed/supervised" / version
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    # Evaluate a version against its own recorded periods after later revisions.
    config["splits"] = manifest.get("splits_config", config["splits"])
    protected = 0
    baseline = ROOT / f".private/supervised_{version}_protected_inputs.json"
    if version == "v001" and not baseline.exists():
        baseline = ROOT / ".private/supervised_protected_inputs.json"
    if baseline.exists():
        for name, sha in json.loads(baseline.read_text(encoding="utf-8")).items():
            assert digest(ROOT / name) == sha, f"Protected source/clustering/shared file changed: {name}"
            protected += 1
    inputs = {}
    for r in manifest["inputs"]:
        assert digest(ROOT / r["path"]) == r["sha256"]
        inputs[r["table"]] = pd.read_parquet(ROOT / r["path"])
    for r in manifest["input_manifests"]:
        assert digest(ROOT / r["path"]) == r["sha256"]
    for r in manifest["files"]:
        path = ROOT / r["path"]
        assert digest(path) == r["sha256"]
        if r["format"] != "parquet":
            continue
        frame = pd.read_parquet(path)
        token = manifest["csv_null_value"]
        converters = {c: lambda value: pd.NA if value == token else float(value)
                      for c in frame if pd.api.types.is_float_dtype(frame[c].dtype)}
        csv = pd.read_csv(path.with_suffix(".csv"), encoding="utf-8-sig", keep_default_na=False, na_values=[token],
                          dtype={c: t for c, t in frame.dtypes.items() if c not in converters},
                          converters=converters, low_memory=False).astype(frame.dtypes.to_dict())
        pd.testing.assert_frame_equal(frame.astype(csv.dtypes.to_dict()), csv, check_exact=True)
    monthly = pd.read_parquet(folder / "integrated_monthly.parquet")
    quarterly = pd.read_parquet(folder / "integrated_quarterly.parquet")
    annual = inputs["macro_annual"]
    source_columns_checked = 0
    for grain, period, integrated in (("monthly", "period_month", monthly), ("quarterly", "quarter", quarterly)):
        mapping = manifest["column_lineage"][grain]
        key = ["country_code", "hs_code", period]
        for source_name, input_name in (("customs", f"customs_{grain}_hs6"), ("comtrade", f"market_{grain}_hs6")):
            assert_mapping(inputs[input_name], integrated, mapping[source_name], key)
            source_columns_checked += len(inputs[input_name].columns)
        assert_mapping(annual, integrated, mapping["world_bank"], ["country_code", "indicator_year"], deduplicate=True)
        source_columns_checked += len(annual.columns)
        assert integrated[["country_code", "country_name", "region", "hs_code", "product_name", "year", "quarter_number"]].notna().all().all()
        assert not integrated.duplicated(key).any()
        expected_unit_value = (integrated.export_value_usd / integrated.export_net_weight_kg).where(integrated.export_net_weight_kg.gt(0), pd.NA)
        pd.testing.assert_series_equal(integrated.export_unit_value_usd_per_kg, expected_unit_value, check_names=False)
    months = pd.PeriodIndex(monthly.period_month, freq="M")
    assert (monthly.year.to_numpy() == months.year).all() and (monthly.month.to_numpy() == months.month).all()
    assert (monthly.quarter.to_numpy() == months.asfreq("Q").astype(str)).all()
    assert monthly.frequency.eq("monthly").all() and quarterly.frequency.eq("quarterly").all()
    for row in quarterly[["quarter", "year", "quarter_number", "period_months"]].drop_duplicates().itertuples(index=False):
        p = pd.Period(row.quarter, freq="Q")
        expected = ";".join(pd.period_range(p.asfreq("M", how="start"), p.asfreq("M", how="end"), freq="M").astype(str))
        assert row.period_months == expected and row.year == p.year and row.quarter_number == p.quarter
    # Build small dictionaries from immutable source observations, independent of
    # the production feature builder and its groupby/shift operations.
    q = inputs["customs_quarterly_hs6"]
    fields = ["export_value_usd", "export_net_weight_kg"]
    lookup = q.set_index(["country_code", "hs_code", "quarter"])[fields].to_dict("index")
    for values in lookup.values():
        amount, weight = values[fields[0]], values[fields[1]]
        values["export_unit_value_usd_per_kg"] = amount / weight if pd.notna(amount) and pd.notna(weight) and weight > 0 else None
    market_fields = config["features"]["supervised"]["market"]["columns"]
    markets = inputs["market_quarterly_hs6"].set_index(["country_code", "hs_code", "quarter"])[market_fields].to_dict("index")
    macro_fields = config["features"]["supervised"]["macro"]["columns"]
    macros = annual.set_index(["country_code", "indicator_year"])[macro_fields].to_dict("index")
    features = pd.read_parquet(folder / "origin_feature_candidates.parquet")
    lag_checks = 0
    for row in features.to_dict("records"):
        country, hs, origin = row["country_code"], row["hs_code"], pd.Period(row["origin_quarter"], freq="Q")
        for value in config["features"]["supervised"]["lag_columns"]:
            for lag in (1, 2, 4):
                expected = lookup.get((country, hs, str(origin - (lag - 1))), {}).get(value)
                assert same_number(row[f"{value}_lag{lag}"], expected), f"Wrong lag: {country}/{hs}/{origin}/{value}/{lag}"
                lag_checks += 1
            values = [lookup.get((country, hs, str(origin - i)), {}).get(value) for i in range(4)]
            present = [float(v) for v in values if v is not None and pd.notna(v)]
            assert row[f"{value}_rolling4_valid_quarters"] == len(present)
            expected_sum = math.fsum(present) if len(present) == 4 else None
            assert same_number(row[f"{value}_rolling4_sum"], expected_sum)
            assert same_number(row[f"{value}_rolling4_mean"], expected_sum / 4 if expected_sum is not None else None)
        current = lookup.get((country, hs, str(origin)), {}).get("export_value_usd")
        previous = lookup.get((country, hs, str(origin - 4)), {}).get("export_value_usd")
        expected_growth = current / previous - 1 if current is not None and pd.notna(current) and previous is not None and pd.notna(previous) and previous > 0 else None
        assert same_number(row["export_value_yoy_growth"], expected_growth)
        for name in market_fields:
            assert same_number(row[name], markets.get((country, hs, str(origin)), {}).get(name))
        assert row["macro_candidate_year"] == origin.year - 1
        for name in macro_fields:
            assert same_number(row[name], macros.get((country, origin.year - 1), {}).get(name))
        assert pd.isna(row["origin_cutoff_at"]) and not row["training_approved"]
    contract = manifest["feature_contract"]
    model_columns = contract["categorical"] + contract["known_future"] + contract["numeric_candidates"]
    assert len(model_columns) == len(set(model_columns))
    assert not set(model_columns).intersection(contract["forbidden_feature_columns"])
    split_counts = {}
    for split, name in (("train", "train_candidates"), ("validation", "validation_candidates"), ("final_test", "final_test_candidates")):
        record = next(r for r in manifest["files"] if Path(r["path"]).name == f"{name}.parquet")
        data = pd.read_parquet(ROOT / record["path"])
        assert set(model_columns).issubset(data.columns)
        assert data.origin_cutoff_at.isna().all() and not data.training_approved.any()
        assert not data.duplicated(["country_code", "hs_code", "origin_quarter", "target_quarter"]).any()
        assert data[["country_code", "country_name", "hs_code", "product_name", "target_year", "target_period_months", "origin_quarter", "target_quarter"]].notna().all().all()
        target = data.rename(columns={"target_quarter": "quarter"}).merge(
            q[["country_code", "hs_code", "quarter", *fields]], on=["country_code", "hs_code", "quarter"], how="left", validate="many_to_one")
        for original, target_column in zip(fields, ("target_export_value_usd", "target_export_net_weight_kg")):
            pd.testing.assert_series_equal(target[original], target[target_column], check_names=False)
        bounds = config["splits"]["splits"][split]
        assert data.target_quarter.between(bounds["start"], bounds["end"]).all()
        distances = pd.PeriodIndex(data.target_quarter, freq="Q").asi8 - pd.PeriodIndex(data.origin_quarter, freq="Q").asi8
        assert (data.horizon_quarters.to_numpy() == distances).all()
        assert (data.target_available == data.target_export_value_usd.notna()).all()
        if split == "final_test":
            assert data.origin_quarter.eq(config["splits"]["final_test"]["origin_quarter"]).all()
            assert Path(record["path"]).parts[0] == ".private"
        # Prove candidates retained every country/product/target/horizon,
        # including missing labels and pre-collection origins.
        unique_targets = q.loc[q.quarter.between(bounds["start"], bounds["end"])]
        factor = 1 if split == "final_test" else 5
        assert len(data) == len(unique_targets) * factor
        assert int(data.target_available.sum()) == int(unique_targets.export_value_usd.notna().sum()) * factor
        assert set(data.country_code) == set(q.country_code) and set(data.hs_code) == set(q.hs_code)
        expected_features = features.set_index(["country_code", "hs_code", "origin_quarter"])
        actual_features = data.set_index(["country_code", "hs_code", "origin_quarter"])
        for column in features.columns.difference(["country_code", "hs_code", "origin_quarter"]):
            expected = expected_features[column].reindex(actual_features.index).reset_index(drop=True)
            actual = actual_features[column].reset_index(drop=True)
            pd.testing.assert_series_equal(expected, actual, check_dtype=False, check_exact=True)
        split_counts[split] = {"rows": len(data), "target_available": int(data.target_available.sum()), "target_missing": int((~data.target_available).sum()),
                               "target_start": data.target_quarter.min(), "target_end": data.target_quarter.max(),
                               "target_quarters": int(data.target_quarter.nunique())}
    result = {"status": "passed", "version": version, "audited_at": utc_now(),
              "monthly_rows_verified": len(monthly), "quarterly_rows_verified": len(quarterly),
              "all_input_column_mappings_verified": source_columns_checked, "dropped_observation_cells": 0,
              "identifier_or_calendar_null_cells": 0, "source_amount_weight_and_macro_values_preserved": True,
              "origin_features_verified": len(features), "lag_values_verified": lag_checks,
              "model_feature_columns": len(model_columns), "target_or_target_quality_used_as_features": False,
              "candidate_splits": split_counts, "csv_parquet_exact_verified": True,
              "protected_source_shared_and_clustering_files_verified": protected,
              "clustering_data_modified": False, "shared_dataset_modified": False,
              "public_release_policy_approved": False, "training_approved": False}
    write_json(ROOT / "reports" / f"supervised_{version}_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v001")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.version):
        parser.error("version must be a simple directory name")
    audit(args.version)
