"""Read-only review of the six requested products against original customs XML."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config
from oinq.collectors.raw import month_ranges, months_in_range


PRODUCTS = ["330420", "330499", "330590", "330491", "330300", "340130"]
SNAPSHOT = "collection_20261103"
VERSION = "v002"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = load_project_config(ROOT)
    raw = ROOT / "data/raw" / SNAPSHOT
    folder = ROOT / "data/processed/customs" / VERSION
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    before = {record["path"]: digest(ROOT / record["path"]) for record in manifest["files"]}
    assert all(before[record["path"]] == record["sha256"] for record in manifest["files"])
    m = pd.read_parquet(folder / "customs_monthly_hs6.parquet")
    q = pd.read_parquet(folder / "customs_quarterly_hs6.parquet")
    normalized = pd.read_parquet(ROOT / "data/interim/customs" / VERSION / "customs_details.parquet")
    raw_index = json.loads((raw / "collection_manifest.json").read_text(encoding="utf-8"))
    hashes = {record["path"].replace("\\", "/"): record["sha256"] for record in raw_index["files"]}
    raw_details, requests = [], {}
    raw_month_counts = Counter()
    request_count = 0
    for country in config["countries"]["countries"]:
        for product in PRODUCTS:
            for start, end in month_ranges(config):
                name = f"{country['iso3']}_{product}_{start.replace('-', '')}_{end.replace('-', '')}"
                relative = f"customs/{name}.xml"
                path = raw / relative
                assert digest(path) == hashes[relative]
                document = ET.fromstring(path.read_bytes())
                assert document.findtext(".//resultCode") == "00"
                rows = [{child.tag: child.text for child in item} for item in document.findall(".//item")]
                details = [row for row in rows if row.get("year") != "총계"]
                for month in months_in_range(start, end):
                    key = (country["iso3"], product, month[:4] + "-" + month[4:])
                    requests[key] = {"source_file": path.relative_to(ROOT).as_posix(),
                                     "request_empty": not details,
                                     "request_total_present": any(row.get("year") == "총계" for row in rows)}
                for row in details:
                    assert row["statCd"] == country["customs"]
                    assert row["hsCd"].startswith(product)
                    month = row["year"].replace(".", "-")
                    key = (country["iso3"], product, month)
                    assert key in requests
                    raw_month_counts[key] += 1
                    raw_details.append({"country_code": country["iso3"], "hs_code": product,
                                        "period_month": month, "source_hs_code": row["hsCd"],
                                        "source_product_name": row.get("statKor"),
                                        "export_value_usd": int(row["expDlr"]),
                                        "export_net_weight_kg": int(row["expWgt"]),
                                        "source_file": path.relative_to(ROOT).as_posix()})
                request_count += 1
    raw_frame = pd.DataFrame(raw_details)
    columns = list(raw_frame.columns)
    order = ["country_code", "source_hs_code", "period_month"]
    expected = raw_frame.sort_values(order).reset_index(drop=True)
    actual = normalized.loc[normalized["hs_code"].isin(PRODUCTS), columns].sort_values(order).reset_index(drop=True)
    expected = expected.astype(actual.dtypes.to_dict())
    pd.testing.assert_frame_equal(actual, expected)
    selected = m.loc[m["hs_code"].isin(PRODUCTS)]
    for row in selected.itertuples():
        key = (row.country_code, row.hs_code, row.period_month)
        assert bool(row.record_observed) == (raw_month_counts[key] > 0)
        assert int(row.source_detail_count) == raw_month_counts[key]
    held = selected.loc[selected["hs_level_review_required"]]
    held_details, held_records, products = [], [], []
    for product in PRODUCTS:
        a = held.loc[held["hs_code"].eq(product)]
        child_names = set()
        hs6_value, hs10_value = 0, 0
        for row in a.itertuples():
            details = raw_frame.loc[raw_frame["country_code"].eq(row.country_code)
                                    & raw_frame["hs_code"].eq(product)
                                    & raw_frame["period_month"].eq(row.period_month)]
            parent = details.loc[details["source_hs_code"].str.len().eq(6)]
            children = details.loc[details["source_hs_code"].str.len().eq(10)]
            assert len(parent) == 1 and len(children) > 0
            parent_value = int(parent["export_value_usd"].sum())
            child_value = int(children["export_value_usd"].sum())
            hs6_value += parent_value
            hs10_value += child_value
            child_names.update(children["source_hs_code"] + " " + children["source_product_name"])
            held_records.append({"country_code": row.country_code, "hs_code": product,
                                 "product_name": row.product_name, "period_month": row.period_month,
                                 "quarter": row.quarter, "hs6_source_name": parent.iloc[0]["source_product_name"],
                                 "hs6_reported_value_usd": parent_value, "hs10_reported_value_usd": child_value,
                                 "hs6_equals_hs10_sum": parent_value == child_value,
                                 "hs10_codes": ";".join(children["source_hs_code"]),
                                 "final_value_is_null": pd.isna(row.export_value_usd),
                                 "source_file": row.source_file})
            held_details.append(details)
        missing = selected.loc[selected["hs_code"].eq(product) & selected["data_status"].eq("missing")]
        q_held = q.loc[q["hs_code"].eq(product) & q["hs_level_review_months"].gt(0)]
        products.append({"hs_code": product,
                         "product_name": selected.loc[selected["hs_code"].eq(product), "product_name"].iloc[0],
                         "held_month_rows": len(a), "held_quarter_rows": len(q_held),
                         "held_months": sorted(a["period_month"].unique().tolist()),
                         "held_quarters": sorted(q_held["quarter"].unique().tolist()),
                         "hs10_source_codes_and_names": sorted(child_names),
                         "held_hs6_reported_usd": hs6_value, "held_hs10_reported_usd": hs10_value,
                         "missing_month_rows": len(missing),
                         "missing_country_counts": {key: int(value) for key, value in missing.groupby("country_name").size().items()},
                         "test_missing_months": {key: sorted(value["period_month"].tolist()) for key, value in
                                                 missing.loc[missing["quarter"].between("2026Q1", "2026Q3")].groupby("country_name")}})
    missing = selected.loc[selected["data_status"].eq("missing"),
                           ["country_code", "country_name", "hs_code", "product_name", "period_month", "quarter", "source_file"]].copy()
    missing["is_final_test_period"] = missing["quarter"].between("2026Q1", "2026Q3")
    missing["raw_request_empty"] = [requests[(r.country_code, r.hs_code, r.period_month)]["request_empty"] for r in missing.itertuples()]
    missing["raw_request_total_present"] = [requests[(r.country_code, r.hs_code, r.period_month)]["request_total_present"] for r in missing.itertuples()]
    missing["raw_month_detail_count"] = 0
    missing["year"] = missing["period_month"].str[:4]
    missing_years = missing.groupby(["country_code", "country_name", "hs_code", "product_name", "year"], observed=True).agg(
        missing_month_rows=("period_month", "size"), missing_months=("period_month", lambda values: ";".join(sorted(values)))).reset_index()
    reports = ROOT / "reports"
    prefix = f"customs_{VERSION}_six_product_review"
    held_table = pd.DataFrame(held_records)
    assert held_table["period_month"].str[:4].astype(int).lt(2026).all()
    held_table.to_csv(reports / f"{prefix}_held.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(reports / f"{prefix}_missing_months.csv", index=False, encoding="utf-8-sig")
    missing_years.to_csv(reports / f"{prefix}_missing_country_year.csv", index=False, encoding="utf-8-sig")
    summary = {"source_version": VERSION, "raw_snapshot": SNAPSHOT,
               "raw_requests_independently_checked": request_count,
               "raw_detail_rows_checked": len(expected), "normalization_dropped_rows": 0,
               "held_month_rows": len(held), "held_source_rows": sum(len(frame) for frame in held_details),
               "held_hs6_reported_usd": sum(row["held_hs6_reported_usd"] for row in products),
               "held_hs10_reported_usd": sum(row["held_hs10_reported_usd"] for row in products),
               "missing_month_rows": len(missing), "final_test_missing_month_rows": int(missing["is_final_test_period"].sum()),
               "products": products, "preprocessing_modified": False}
    assert all(digest(ROOT / name) == value for name, value in before.items())
    (reports / f"{prefix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
