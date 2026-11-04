"""Independently audit all configured customs country/product combinations."""

from collections import Counter, defaultdict
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = load_project_config(ROOT)
    snapshot, version = "collection_20261103", "v003"
    raw = ROOT / "data/raw" / snapshot
    processed = ROOT / "data/processed/customs" / version
    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    raw_manifest = json.loads((raw / "collection_manifest.json").read_text(encoding="utf-8"))
    index = {row["path"].replace("\\", "/"): row["sha256"] for row in raw_manifest["files"]}
    for record in manifest["files"]:
        assert digest(ROOT / record["path"]) == record["sha256"]
    assert digest(raw / "collection_manifest.json") == manifest["input_collection_manifest_sha256"]
    source_rows, request_counts, total_counts = [], Counter(), Counter()
    raw_counts, raw_amounts, raw_weights = Counter(), Counter(), Counter()
    for country in config["countries"]["countries"]:
        for product in config["products"]["products"]:
            hs = product["hs_code"]
            for start, end in month_ranges(config):
                name = f"{country['iso3']}_{hs}_{start.replace('-', '')}_{end.replace('-', '')}"
                relative = f"customs/{name}.xml"
                path = raw / relative
                assert digest(path) == index[relative]
                document = ET.fromstring(path.read_bytes())
                assert document.findtext(".//resultCode") == "00"
                rows = [{child.tag: child.text for child in item} for item in document.findall(".//item")]
                details = [row for row in rows if row.get("year") != "총계"]
                totals = [row for row in rows if row.get("year") == "총계"]
                assert len(totals) <= 1
                if details:
                    assert len(totals) == 1
                    assert sum(int(row["expDlr"]) for row in details) == int(totals[0]["expDlr"])
                total_counts[(country["iso3"], hs)] += len(totals)
                request_counts[(country["iso3"], hs)] += 1
                valid_months = set(months_in_range(start, end))
                for row in details:
                    assert row["statCd"] == country["customs"]
                    code = row["hsCd"]
                    assert len(code) in (6, 10) and code[:6] == hs
                    assert row["year"].replace(".", "") in valid_months
                    month = row["year"].replace(".", "-")
                    key = (country["iso3"], hs, month)
                    raw_counts[key] += 1
                    raw_amounts[key] += int(row["expDlr"])
                    raw_weights[key] += int(row["expWgt"])
                    source_rows.append({"country_code": country["iso3"], "hs_code": hs,
                                        "source_hs_code": code, "source_hs_digits": len(code),
                                        "source_product_name": row.get("statKor"), "period_month": month,
                                        "export_value_usd": int(row["expDlr"]),
                                        "export_net_weight_kg": int(row["expWgt"]),
                                        "source_file": path.relative_to(ROOT).as_posix()})
    source = pd.DataFrame(source_rows)
    actual = pd.read_parquet(ROOT / f"data/interim/customs/{version}/customs_details.parquet")
    keys = ["country_code", "source_hs_code", "period_month"]
    assert not source.duplicated(keys).any()
    cols = list(source.columns)
    expected = source.sort_values(keys).reset_index(drop=True).astype(actual[cols].dtypes.to_dict())
    pd.testing.assert_frame_equal(expected, actual[cols].sort_values(keys).reset_index(drop=True))
    monthly = pd.read_parquet(processed / "customs_monthly_hs6.parquet")
    quarterly = pd.read_parquet(processed / "customs_quarterly_hs6.parquet")
    for row in monthly.itertuples():
        key = (row.country_code, row.hs_code, row.period_month)
        assert row.source_detail_count == raw_counts[key]
        if raw_counts[key]:
            assert row.export_value_usd == raw_amounts[key]
            assert row.export_net_weight_kg == raw_weights[key]
        else:
            assert pd.isna(row.export_value_usd) and pd.isna(row.export_net_weight_kg)
        assert not row.hs_level_review_required
    for row in quarterly.itertuples():
        year, q = int(row.quarter[:4]), int(row.quarter[-1])
        months = [f"{year}-{month:02d}" for month in range(3 * q - 2, 3 * q + 1)]
        group = [(row.country_code, row.hs_code, month) for month in months]
        present = [key for key in group if raw_counts[key]]
        assert row.observed_months == len(present)
        assert row.hs_level_review_months == 0
        if len(present) == 3:
            assert row.export_value_usd == sum(raw_amounts[key] for key in group)
            assert row.export_net_weight_kg == sum(raw_weights[key] for key in group)
        else:
            assert pd.isna(row.export_value_usd) and pd.isna(row.export_net_weight_kg)
    names = {row["hs_code"]: row["name_ko"] for row in config["products"]["products"]}
    catalog = source.groupby(["hs_code", "source_hs_code", "source_product_name"], observed=True, dropna=False).agg(
        source_hs_digits=("source_hs_digits", "first"), detail_rows=("source_hs_digits", "size"),
        country_count=("country_code", "nunique"), first_month=("period_month", "min"),
        last_month=("period_month", "max")).reset_index()
    catalog["product_name"] = catalog["hs_code"].map(names)
    assert catalog["detail_rows"].sum() == len(source)
    assert catalog.groupby("source_hs_code")["hs_code"].nunique().eq(1).all()
    audit_rows = []
    for country in config["countries"]["countries"]:
        for product in config["products"]["products"]:
            c, h = country["iso3"], product["hs_code"]
            s = source.loc[source["country_code"].eq(c) & source["hs_code"].eq(h)]
            m = monthly.loc[monthly["country_code"].eq(c) & monthly["hs_code"].eq(h)]
            q = quarterly.loc[quarterly["country_code"].eq(c) & quarterly["hs_code"].eq(h)]
            audit_rows.append({"country_code": c, "country_name": country["name_ko"], "hs_code": h,
                               "product_name": product["name_ko"], "requests_checked": request_counts[(c, h)],
                               "total_rows_excluded": total_counts[(c, h)], "source_detail_rows": len(s),
                               "source_hs6_rows": int(s["source_hs_digits"].eq(6).sum()),
                               "source_hs10_rows": int(s["source_hs_digits"].eq(10).sum()),
                               "expected_month_rows": len(m), "queried_month_rows": int(m["record_observed"].sum()),
                               "missing_month_rows": int(m["data_status"].eq("missing").sum()),
                               "mixed_month_rows_integrated": int(m["hs_level_mixed"].sum()),
                               "held_month_rows": int(m["hs_level_review_required"].sum()),
                               "complete_quarter_rows": int(q["target_value_complete"].sum()),
                               "partial_quarter_rows": int(q["data_status"].eq("partial").sum()),
                               "missing_quarter_rows": int(q["data_status"].eq("missing").sum()),
                               "classification_and_amount_check": "passed"})
    country_product = pd.DataFrame(audit_rows)
    assert len(country_product) == 160 and country_product["requests_checked"].sum() == 2240
    assert country_product["source_detail_rows"].sum() == 46594
    changes = {}
    for name, period in [("customs_monthly_hs6", "period_month"), ("customs_quarterly_hs6", "quarter")]:
        old = pd.read_parquet(ROOT / f"data/processed/customs/v002/{name}.parquet").set_index(["country_code", "hs_code", period])
        new = pd.read_parquet(processed / f"{name}.parquet").set_index(["country_code", "hs_code", period])
        assert old.index.equals(new.index)
        for measure in ("export_value_usd", "export_net_weight_kg"):
            same = (old[measure].eq(new[measure]) | (old[measure].isna() & new[measure].isna())).fillna(False)
            allowed = old["hs_level_review_required"] if period == "period_month" else old["hs_level_review_months"].gt(0)
            assert same.eq(~allowed).all()
            assert new.loc[allowed, measure].notna().all()
            pd.testing.assert_series_equal(old[f"observed_{measure}"], new[f"observed_{measure}"])
            changes[f"{period}_{measure}_restored_rows"] = int((~same).sum())
    report = ROOT / "reports"
    catalog.to_csv(report / f"customs_{version}_source_code_map.csv", index=False, encoding="utf-8-sig")
    country_product.to_csv(report / f"customs_{version}_country_product_audit.csv", index=False, encoding="utf-8-sig")
    summary = {"version": version, "snapshot": snapshot, "country_count": 20, "product_count": 8,
               "country_product_combinations_checked": len(country_product),
               "requests_independently_checked": sum(request_counts.values()),
               "source_detail_rows_classified": len(source), "source_code_map_rows": len(catalog),
               "total_rows_excluded": sum(total_counts.values()), "source_detail_rows_dropped": 0,
               "classification_holds": int(monthly["hs_level_review_required"].sum()),
               "missing_month_rows": int(monthly["data_status"].eq("missing").sum()),
               "classification_and_amount_checks": "passed", "changes_vs_v002": changes}
    (report / f"customs_{version}_classification_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
