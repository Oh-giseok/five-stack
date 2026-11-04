"""Read-only source audit of annual macro data and the common-year candidate."""

import argparse
import hashlib
import json
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


def same(actual, expected):
    return bool(pd.isna(actual)) if expected is None else bool(actual == expected)


def audit(snapshot, version):
    config = load_project_config(ROOT)
    raw = ROOT / "data/raw" / snapshot
    folder = ROOT / "data/processed/world_bank" / version
    interim = ROOT / "data/interim/world_bank" / version
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert digest(raw / "collection_manifest.json") == manifest["input_collection_manifest_sha256"]
    protected = 0
    baseline = ROOT / ".private/world_bank_protected_inputs.json"
    if baseline.exists():
        for name, sha in json.loads(baseline.read_text(encoding="utf-8")).items():
            assert digest(ROOT / name) == sha, f"Existing input/output changed: {name}"
            protected += 1
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert digest(path) == record["sha256"], f"Output checksum mismatch: {path}"
        if record["format"] != "parquet":
            continue
        frame = pd.read_parquet(path)
        token = manifest["csv_null_value"]
        floats = {c: lambda value: pd.NA if value == token else float(value)
                  for c in frame if pd.api.types.is_float_dtype(frame[c].dtype)}
        csv = pd.read_csv(path.with_suffix(".csv"), encoding="utf-8-sig", keep_default_na=False, na_values=[token],
                          converters=floats, dtype={c: t for c, t in frame.dtypes.items() if c not in floats},
                          low_memory=False).astype(frame.dtypes.to_dict())
        pd.testing.assert_frame_equal(frame, csv, check_exact=True)
    details = pd.read_parquet(interim / "world_bank_details.parquet")
    long = pd.read_parquet(folder / "macro_annual_long.parquet")
    annual = pd.read_parquet(folder / "macro_annual.parquet")
    lookup = details.set_index(["source_file", "source_row_number"])
    llookup = long.set_index(["country_code", "indicator_year", "indicator_name"])
    alookup = annual.set_index(["country_code", "indicator_year"])
    indicators = config["sources"]["world_bank"]["indicators"]
    indicator_names = {r["code"]: name for name, r in indicators.items()}
    raw_keys, nrows, nfields, nrequests = set(), 0, 0, 0
    for meta_path in sorted((raw / "world_bank").glob("*.meta.json")):
        if meta_path.name.startswith("probe_"):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        path = meta_path.parent / meta["body_file"]
        assert digest(path) == meta["sha256"]
        header, rows = json.loads(path.read_bytes())
        source_file = path.relative_to(ROOT).as_posix()
        nrequests += 1
        for number, row in enumerate(rows):
            actual = lookup.loc[(source_file, number)]
            fields = {"indicator_code": row["indicator"]["id"], "indicator_description": row["indicator"]["value"],
                      "source_country_code": row["country"]["id"], "source_country_name": row["country"]["value"],
                      "source_country_iso3": row["countryiso3code"], "source_date": row["date"], "value": row["value"],
                      "source_unit": row["unit"], "source_obs_status": row["obs_status"], "source_decimal": row["decimal"]}
            for name, value in fields.items():
                assert same(actual[name], value), f"Original field changed: {source_file}/{number}/{name}"
                nfields += 1
            name = indicator_names[row["indicator"]["id"]]
            key = (row["countryiso3code"], int(row["date"]), name)
            assert key not in raw_keys
            raw_keys.add(key)
            normalized = llookup.loc[key]
            assert normalized.record_observed and same(normalized.value, row["value"])
            assert same(alookup.loc[key[:2], name], row["value"])
            assert normalized.source_lastupdated == header["lastupdated"]
            assert pd.isna(normalized.released_at) and pd.isna(normalized.release_evidence)
            assert normalized.unit == indicators[name]["unit"]
            if row["value"] is None:
                assert normalized.missing_reason == "null_value_in_source"
            nrows += 1
    assert nrows == len(details) == manifest["source_rows_retained"]
    assert int(long.record_observed.sum()) == nrows
    for key, row in llookup.iterrows():
        if key not in raw_keys:
            assert not row.record_observed and pd.isna(row.value)
            assert row.missing_reason == "no_record_in_successful_response"
        assert same(alookup.loc[key[:2], key[2]], None if pd.isna(row.value) else row.value)
    cluster_columns = config["features"]["clustering"]["columns"]
    common = []
    for year, group in annual.groupby("indicator_year"):
        if len(group) == len(config["countries"]["countries"]) and group[cluster_columns].notna().all().all():
            common.append(int(year))
    if common:
        candidate = pd.read_parquet(folder / "country_indicators_candidate.parquet")
        year = max(common)
        assert candidate.reference_year.eq(year).all() and candidate.country_code.nunique() == len(candidate) == 20
        for _, row in candidate.iterrows():
            for name in cluster_columns:
                assert row[name] == alookup.loc[(row.country_code, year), name]
        assert not candidate.training_approved.any() and not candidate.scaled.any() and not candidate.historical_release_verified.any()
    else:
        candidate, year = None, None
    result = {"status": "passed", "snapshot": snapshot, "version": version, "audited_at": utc_now(),
              "source_requests_verified": nrequests, "source_rows_matched": nrows, "original_fields_matched": nfields,
              "source_rows_dropped": 0, "annual_long_cells_verified": len(long), "annual_rows_verified": len(annual),
              "actual_values": int(long.value.notna().sum()), "source_null_cells": int((long.record_observed & long.value.isna()).sum()),
              "absent_record_cells": int((~long.record_observed).sum()), "csv_parquet_exact_verified": True,
              "blank_string_and_null_preserved": True, "release_dates_not_invented": True,
              "protected_existing_files_verified": protected, "new_output_checksums_verified": len(manifest["files"]),
              "countries_reviewed": sorted(long.country_code.unique().tolist()), "indicators_reviewed": sorted(indicators),
              "clustering_candidate_year": year, "clustering_candidate_countries": 0 if candidate is None else len(candidate),
              "training_dataset_generated": False}
    write_json(ROOT / "reports" / f"world_bank_{version}_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="collection_20261103")
    parser.add_argument("--version", default="v001")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (args.snapshot, args.version)):
        parser.error("snapshot/version must be simple directory names")
    audit(args.snapshot, args.version)
