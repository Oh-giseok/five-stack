"""Independently compare every raw Comtrade row and output with source files."""

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


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def audit(snapshot, version):
    config = load_project_config(ROOT)
    raw = ROOT / "data/raw" / snapshot
    folder = ROOT / "data/processed/comtrade" / version
    interim = ROOT / "data/interim/comtrade" / version
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    check(digest(raw / "collection_manifest.json") == manifest["input_collection_manifest_sha256"], "Input manifest changed")
    cmanifest = ROOT / "data/processed/customs/v003/manifest.json"
    check(digest(cmanifest) == manifest["customs_manifest_sha256"], "Customs v003 manifest changed")
    checksums_verified = 0
    baseline = ROOT / ".private/comtrade_protected_inputs.json"
    if baseline.is_file():
        for name, sha in json.loads(baseline.read_text(encoding="utf-8")).items():
            check(digest(ROOT / name) == sha, f"Protected file changed: {name}")
            checksums_verified += 1
    for record in manifest["files"]:
        path = ROOT / record["path"]
        check(digest(path) == record["sha256"], f"Output checksum error: {path.name}")
        checksums_verified += 1
        if record["format"] != "parquet":
            continue
        parquet = pd.read_parquet(path)
        # Parse decimal text once, then restore nullable numeric dtypes.
        floats = {c: lambda value: pd.NA if value == "" else float(value)
                  for c in parquet if pd.api.types.is_float_dtype(parquet[c].dtype)}
        csv = pd.read_csv(path.with_suffix(".csv"), encoding="utf-8-sig",
                          dtype={c: t for c, t in parquet.dtypes.items() if c not in floats},
                          converters=floats, keep_default_na=False, na_values=[""], low_memory=False)
        csv = csv.astype(parquet.dtypes.to_dict())
        csv = csv.astype({c: parquet[c].dtype for c in parquet if isinstance(parquet[c].dtype, pd.StringDtype)})
        pd.testing.assert_frame_equal(parquet, csv, check_exact=True)
    details = pd.read_parquet(interim / "comtrade_details.parquet")
    lookup = details.set_index(["source_file", "source_row_number"])
    monthly = pd.read_parquet(folder / "comtrade_monthly_hs6.parquet")
    quarterly = pd.read_parquet(folder / "comtrade_quarterly_hs6.parquet")
    mkey = ["country_code", "hs_code", "source_role", "period_month"]
    qkey = ["country_code", "hs_code", "source_role", "quarter"]
    mlookup = monthly.set_index(mkey)
    qlookup = quarterly.set_index(qkey)
    by_reporter = {r["comtrade_reporter"]: r["iso3"] for r in config["countries"]["countries"]}
    by_partner = {r["comtrade_partner"]: r["iso3"] for r in config["countries"]["countries"]}
    # Use the original row values, not the production normalizer, for all sums.
    sums, present_keys, source_count, field_count, requests = {}, set(), 0, 0, 0
    for path in sorted((raw / "comtrade").glob("*.meta.json")):
        if path.name.startswith("probe_"):
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        body = path.parent / meta["body_file"]
        check(digest(body) == meta["sha256"], f"Raw checksum error: {body.name}")
        document = json.loads(body.read_bytes())
        source_file = body.relative_to(ROOT).as_posix()
        requests += 1
        for number, row in enumerate(document["data"]):
            normalized = lookup.loc[(source_file, number)]
            for field, value in row.items():
                actual = normalized[field]
                equal = pd.isna(actual) if value is None else bool(actual == value)
                check(equal, f"Original field changed: {source_file}/{number}/{field}")
                field_count += 1
            country = by_partner[row["partnerCode"]] if row["flowCode"] == "X" else by_reporter[row["reporterCode"]]
            role = "export_comparison" if row["flowCode"] == "X" else "world_import" if row["partnerCode"] == 0 else "korea_import"
            month = str(row["period"])
            mk = (country, row["cmdCode"], role, month[:4] + "-" + month[4:])
            check(mk not in present_keys, f"Duplicate source key: {mk}")
            present_keys.add(mk)
            check(normalized.country_code == country and normalized.source_role == role, f"Country/role mismatch: {mk}")
            m = mlookup.loc[mk]
            for raw_field, normalized_field in (("primaryValue", "trade_value_usd"), ("netWgt", "net_weight_kg")):
                value = row[raw_field]
                check(pd.isna(m[normalized_field]) if value is None else m[normalized_field] == value, f"Monthly value changed: {mk}")
            qk = (country, row["cmdCode"], role, f"{month[:4]}Q{(int(month[4:]) - 1) // 3 + 1}")
            bucket = sums.setdefault(qk, {"values": [], "weights": [], "months": 0})
            bucket["months"] += 1
            if row["primaryValue"] is not None:
                bucket["values"].append(row["primaryValue"])
            if row["netWgt"] is not None:
                bucket["weights"].append(row["netWgt"])
            source_count += 1
    check(source_count == len(details) == manifest["source_rows_retained"], "Source row count mismatch")
    check(int(monthly.record_observed.sum()) == source_count, "Source rows lost in monthly panel")
    for key, row in mlookup.iterrows():
        if key not in present_keys:
            check(pd.isna(row.trade_value_usd) and pd.isna(row.net_weight_kg) and not row.record_observed,
                  f"Invented absent monthly data: {key}")
    for key, row in qlookup.iterrows():
        bucket = sums.get(key, {"values": [], "weights": [], "months": 0})
        check(row.observed_months == bucket["months"], f"Quarter count error: {key}")
        for values, col in ((bucket["values"], "trade_value_usd"), (bucket["weights"], "net_weight_kg")):
            expected = math.fsum(values) if values else None
            observed = row[f"observed_{col}"]
            check(pd.isna(observed) if expected is None else math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-6),
                  f"Quarter observed sum error: {key}/{col}")
            check(pd.isna(row[col]) if len(values) != 3 else math.isclose(row[col], expected, rel_tol=1e-12, abs_tol=1e-6),
                  f"Partial quarter used as full value: {key}/{col}")
    shares_checked = {}
    for period, long, index in (("period_month", monthly, mlookup), ("quarter", quarterly, qlookup)):
        panel = pd.read_parquet(folder / f"market_{'monthly' if period == 'period_month' else 'quarterly'}_hs6.parquet")
        for _, row in panel.iterrows():
            country, hs, time = row.country_code, row.hs_code, row[period]
            world = index.loc[(country, hs, "world_import", time)]
            korea = index.loc[(country, hs, "korea_import", time)]
            for source, prefix in ((world, "market_import"), (korea, "korea_import")):
                for column, field in (("trade_value_usd", "value_usd"), ("net_weight_kg", "net_weight_kg")):
                    expected, actual = source[column], row[f"{prefix}_{field}"]
                    check(pd.isna(actual) if pd.isna(expected) else actual == expected, "Market component changed")
            available = (pd.notna(world.trade_value_usd) and pd.notna(korea.trade_value_usd)
                         and world.trade_value_usd > 0 and korea.trade_value_usd <= world.trade_value_usd
                         and world.valuation == korea.valuation and world.valuation in ("CIF", "FOB")
                         and world.hs_edition == korea.hs_edition)
            check(pd.notna(row.korea_import_share) == available, "Share completeness error")
            if available:
                check(math.isclose(row.korea_import_share, korea.trade_value_usd / world.trade_value_usd, rel_tol=1e-12), "Share ratio error")
        shares_checked[period] = int(panel.korea_import_share.notna().sum())
    export_comparison, difference_rows = {}, []
    for grain, period in (("monthly", "period_month"), ("quarterly", "quarter")):
        comparison = pd.read_parquet(interim / f"customs_v003_export_comparison_{grain}.parquet")
        original_customs = pd.read_parquet(ROOT / "data/processed/customs/v003" / f"customs_{grain}_hs6.parquet")
        long = monthly if grain == "monthly" else quarterly
        key = ["country_code", "hs_code", period]
        original_customs = original_customs.set_index(key).sort_index()
        compared = comparison.set_index(key).sort_index()
        comtrade_exports = long.loc[long.source_role.eq("export_comparison")].set_index(key).sort_index()
        for source, prefix, fields in ((original_customs, "customs", ("export_value_usd", "export_net_weight_kg")),
                                        (comtrade_exports, "comtrade", ("trade_value_usd", "net_weight_kg"))):
            for field, normalized in zip(fields, ("export_value_usd", "export_net_weight_kg")):
                pd.testing.assert_series_equal(source[field], compared[f"{prefix}_{normalized}"], check_names=False)
        pd.testing.assert_series_equal(compared.difference_usd,
                                       compared.comtrade_export_value_usd - compared.customs_export_value_usd,
                                       check_names=False)
        pair = comparison.loc[comparison.amounts_both_present]
        check(comparison.target_source.eq("customs_v003").all(), "Target source changed")
        changed = pair.loc[pair.difference_usd.ne(0)]
        export_comparison[grain] = {"paired_cells": len(pair), "equal_cells": int(pair.difference_usd.eq(0).sum()),
                                    "different_cells": len(changed),
                                    "over_10_percent_cells": int(pair.relative_difference.abs().gt(.1).sum())}
        for _, row in changed.iterrows():
            difference_rows.append({"grain": grain, "country_code": row.country_code, "hs_code": row.hs_code,
                                    "period": row[period], "amount_differs": True,
                                    "over_10_percent": bool(abs(row.relative_difference) > .1) if pd.notna(row.relative_difference) else False,
                                    "action": "retain_both_keep_customs_v003_target", "cause": "not_established"})
    pd.DataFrame(difference_rows).to_csv(ROOT / "reports" / f"comtrade_{version}_export_difference_review.csv",
                                       index=False, encoding="utf-8-sig")
    summary = {"status": "passed", "snapshot": snapshot, "version": version, "audited_at": utc_now(),
               "raw_requests_verified": requests, "source_rows_matched": source_count, "original_fields_matched": field_count,
               "normalized_rows": len(details), "dropped_source_rows": 0, "classification_held_rows": 0,
               "monthly_cells_verified": len(monthly), "quarterly_cells_verified": len(quarterly),
               "market_shares_checked": shares_checked, "protected_and_output_checksums_verified": checksums_verified,
               "csv_parquet_exact_values_and_nulls_verified": True, "customs_v003_unchanged": True,
               "export_comparison": export_comparison,
               "countries_reviewed": sorted(monthly.country_code.unique().tolist()),
               "products_reviewed": sorted(monthly.hs_code.unique().tolist())}
    write_json(ROOT / "reports" / f"comtrade_{version}_classification_audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="collection_20261103")
    parser.add_argument("--version", default="v001")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in (args.snapshot, args.version)):
        parser.error("snapshot/version must be simple directory names")
    audit(args.snapshot, args.version)
