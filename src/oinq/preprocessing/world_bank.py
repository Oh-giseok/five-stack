"""Preserve annual macro observations and prepare unscaled country candidates.

No quarterly replication, historical-release assumption, imputation, training,
or cross-source join is performed here.
"""

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oinq.collectors.raw import utc_now, write_json
from oinq.preprocessing.customs import sha256

STRING = pd.StringDtype(storage="python")
KEY = ["country_code", "indicator_year", "indicator_name"]
CSV_NULL = "\\N"


class WorldBankDataError(ValueError):
    """An annual response contradicts its configured scope or request."""


def require(condition, message):
    if not condition:
        raise WorldBankDataError(message)


def save_table(frame, folder, name):
    """Keep the API's empty string distinct from missing cells in both formats."""
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for column in frame.select_dtypes(include=["string", "object"]):
        require(not frame[column].eq(CSV_NULL).any(), "CSV null 표식과 실제 문자열 충돌")
    for extension in ("csv", "parquet"):
        path = folder / f"{name}.{extension}"
        require(not path.exists(), f"기존 파일을 덮어쓰지 않습니다: {path.name}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        if extension == "csv":
            frame.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep=CSV_NULL)
            floats = {c: lambda value: pd.NA if value == CSV_NULL else float(value)
                      for c in frame if pd.api.types.is_float_dtype(frame[c].dtype)}
            restored = pd.read_csv(temporary, encoding="utf-8-sig", keep_default_na=False, na_values=[CSV_NULL],
                                   dtype={c: t for c, t in frame.dtypes.items() if c not in floats},
                                   converters=floats, low_memory=False).astype(frame.dtypes.to_dict())
        else:
            frame.to_parquet(temporary, index=False, engine="pyarrow")
            restored = pd.read_parquet(temporary, engine="pyarrow").astype(frame.dtypes.to_dict())
        pd.testing.assert_frame_equal(frame, restored, check_exact=True)
        temporary.replace(path)
        records.append({"path": path, "rows": len(frame), "format": extension, "sha256": sha256(path),
                        "bytes": path.stat().st_size,
                        "schema": [{"name": c, "dtype": str(frame[c].dtype)} for c in frame]})
    return records


def years(config):
    period = config["splits"]["collection"]
    return list(range(int(period["monthly_start"][:4]), int(period["monthly_end"][:4]) + 1))


def normalize_row(row, indicator_name, settings, countries, year_scope, source_file, number, meta, header):
    country = countries.get(row["countryiso3code"])
    require(country is not None and row["country"]["id"] == country["iso2"], "World Bank 국가 매핑 불일치")
    indicator = settings["indicators"][indicator_name]
    require(row["indicator"]["id"] == indicator["code"] and row["date"].isdigit()
            and int(row["date"]) in year_scope, "지표·연도 범위 불일치")
    value = row["value"]
    require(value is None or (isinstance(value, (float, int)) and not isinstance(value, bool) and np.isfinite(value)),
            "거시지표 값이 유한 숫자 또는 null이 아닙니다.")
    return {"country_code": country["iso3"], "indicator_year": int(row["date"]), "indicator_name": indicator_name,
            "indicator_code": row["indicator"]["id"], "indicator_description": row["indicator"]["value"],
            "source_country_code": row["country"]["id"], "source_country_name": row["country"]["value"],
            "source_country_iso3": row["countryiso3code"], "source_date": row["date"], "value": value,
            "source_unit": row["unit"], "source_obs_status": row["obs_status"], "source_decimal": row["decimal"],
            "unit": indicator["unit"], "source_file": source_file, "source_row_number": number,
            "collected_at": meta["collected_at"], "source_lastupdated": header.get("lastupdated"),
            "source_page": int(header["page"]), "source_id": str(header["sourceid"])}


def dimensions(frame, config):
    frame = frame.copy()
    countries = config["countries"]["countries"]
    frame["country_name"] = frame.country_code.map({r["iso3"]: r["name_ko"] for r in countries})
    frame["region"] = frame.country_code.map({r["iso3"]: r["region"] for r in countries})
    frame["source"] = "world_bank"
    frame["frequency"] = "annual"
    frame["snapshot_kind"] = "current_revised_snapshot"
    frame["historical_release_status"] = "not_verified"
    for column in ("released_at", "release_evidence"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype=STRING)
    for column in frame:
        if frame[column].dtype == object or isinstance(frame[column].dtype, pd.StringDtype):
            frame[column] = frame[column].astype(STRING)
    return frame


def read_world_bank(root, config, snapshot):
    folder = root / config["sources"]["storage"]["raw_dir"] / snapshot
    manifest_path = folder / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["collection_complete"] and not manifest["integrity_errors"] and not manifest["api_errors"],
            "원본 수집 무결성 확인 필요")
    indexed = {r["path"].replace("\\", "/"): r for r in manifest["files"]}
    settings = config["sources"]["world_bank"]
    countries = {r["world_bank"]: r for r in config["countries"]["countries"]}
    year_scope = years(config)
    details, requests = [], []
    for name, indicator in settings["indicators"].items():
        prefix = f"{name}_{year_scope[0]}_{year_scope[-1]}_page"
        metas = sorted((folder / "world_bank").glob(f"{prefix}*.meta.json"))
        require(bool(metas), f"지표 요청 없음: {name}")
        seen_pages, source_rows, total_rows, expected_pages = set(), 0, None, None
        for meta_path in metas:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            params = meta["parameters"]
            endpoint = settings["base_url"] + settings["endpoint_template"].format(
                country=";".join(countries), indicator=indicator["code"])
            require(meta["source"] == "world_bank" and meta["endpoint"] == endpoint
                    and meta["transport_ok"] and meta["http_status"] == 200,
                    f"지표 API 요청 불일치: {name}")
            require(params["date"] == f"{year_scope[0]}:{year_scope[-1]}" and params["format"] == "json"
                    and int(params["source"]) == settings["source_id"]
                    and int(params["per_page"]) == settings["page_size"], f"지표 조회 범위 불일치: {name}")
            path = meta_path.parent / meta["body_file"]
            relative = path.relative_to(folder).as_posix()
            digest = sha256(path)
            require(digest == meta["sha256"] == indexed[relative]["sha256"], f"원본 체크섬 불일치: {path.name}")
            doc = json.loads(path.read_bytes())
            require(isinstance(doc, list) and len(doc) == 2 and isinstance(doc[0], dict)
                    and isinstance(doc[1], list), f"원본 응답 형식 오류: {path.name}")
            header, rows = doc
            page, pages, total = int(header["page"]), int(header["pages"]), int(header["total"])
            require(page == int(params["page"]) and page not in seen_pages and 1 <= page <= pages
                    and int(header["sourceid"]) == settings["source_id"], f"페이지 범위 오류: {name}")
            if total_rows is not None:
                require(total_rows == total and expected_pages == pages, f"페이지 총량 불일치: {name}")
            total_rows, expected_pages = total, pages
            seen_pages.add(page)
            source_rows += len(rows)
            source_file = path.relative_to(root).as_posix()
            for number, row in enumerate(rows):
                details.append(normalize_row(row, name, settings, countries, year_scope, source_file, number, meta, header))
            requests.append({"indicator_name": name, "source_file": source_file, "source_page": page,
                             "expected_pages": pages, "source_rows": len(rows), "total_source_rows": total,
                             "request_success": True, "source_lastupdated": header.get("lastupdated"),
                             "collected_at": meta["collected_at"], "source_sha256": digest})
        require(seen_pages == set(range(1, expected_pages + 1)) and source_rows == total_rows,
                f"미완료 페이지 또는 행 수 불일치: {name}")
    frame = pd.DataFrame(details)
    frame["value"] = frame["value"].astype("Float64")
    for column in ("indicator_year", "source_row_number", "source_page", "source_decimal"):
        frame[column] = frame[column].astype("Int64")
    require(not frame.duplicated(KEY).any(), "연간 국가·지표·연도 중복: 임의 합산하지 않습니다.")
    return dimensions(frame, config), pd.DataFrame(requests).convert_dtypes(), sha256(manifest_path)


def make_panels(details, config):
    require(not details.duplicated(KEY).any(), "국가·지표·연도 키 중복")
    indicators = config["sources"]["world_bank"]["indicators"]
    countries = [r["iso3"] for r in config["countries"]["countries"]]
    grid = pd.MultiIndex.from_product([countries, years(config), list(indicators)], names=KEY).to_frame(index=False)
    grid["indicator_year"] = grid.indicator_year.astype("Int64")
    require(pd.MultiIndex.from_frame(details[KEY]).isin(pd.MultiIndex.from_frame(grid[KEY])).all(), "격자 범위 밖 거시 관측")
    metadata = ["indicator_code", "value", "source_file", "source_row_number", "collected_at", "source_lastupdated", "source_obs_status"]
    long = grid.merge(details[KEY + metadata], on=KEY, how="left", validate="one_to_one", indicator=True)
    long["record_observed"] = long.pop("_merge").eq("both").astype("boolean")
    long["indicator_code"] = long.indicator_name.map({name: r["code"] for name, r in indicators.items()})
    long["unit"] = long.indicator_name.map({name: r["unit"] for name, r in indicators.items()})
    long["value"] = long.value.astype("Float64")
    long["value_missing"] = long.value.isna().astype("boolean")
    long["data_status"] = np.where(long.value.notna(), "actual", "missing")
    long["missing_reason"] = pd.Series(pd.NA, index=long.index, dtype=STRING)
    long.loc[~long.record_observed, "missing_reason"] = "no_record_in_successful_response"
    long.loc[long.record_observed & long.value.isna(), "missing_reason"] = "null_value_in_source"
    # Quality flags preserve the original value and never silently remove a country.
    long["quality_issue"] = pd.Series(pd.NA, index=long.index, dtype=STRING)
    nonnegative = long.indicator_name.isin(["gdp_usd", "gdp_per_capita_usd", "population"])
    long.loc[(nonnegative & long.value.lt(0)).fillna(False), "quality_issue"] = "negative_nonnegative_indicator"
    long.loc[(long.indicator_name.eq("exchange_rate_lcu_per_usd") & long.value.le(0)).fillna(False), "quality_issue"] = "nonpositive_exchange_rate"
    long.loc[(long.indicator_name.eq("internet_users_pct") & ~long.value.between(0, 100) & long.value.notna()).fillna(False), "quality_issue"] = "internet_percent_out_of_range"
    long = dimensions(long, config).sort_values(KEY).reset_index(drop=True)
    annual_key = ["country_code", "indicator_year"]
    annual = long[annual_key].drop_duplicates().reset_index(drop=True)
    fields = ["value", "value_missing", "record_observed", "missing_reason", "quality_issue", "unit", "source_file", "collected_at", "source_lastupdated", "source_obs_status"]
    for name in indicators:
        subset = long.loc[long.indicator_name.eq(name), annual_key + fields].copy()
        subset = subset.rename(columns={col: name if col == "value" else f"{name}_{col}" for col in fields})
        annual = annual.merge(subset, on=annual_key, how="left", validate="one_to_one")
    values = list(indicators)
    annual["available_indicator_count"] = annual[values].notna().sum(axis=1).astype("Int64")
    annual["all_indicators_present"] = annual[values].notna().all(axis=1).astype("boolean")
    cluster_columns = config["features"]["clustering"]["columns"]
    annual["clustering_values_complete"] = annual[cluster_columns].notna().all(axis=1).astype("boolean")
    annual = dimensions(annual, config).sort_values(annual_key).reset_index(drop=True)
    return long, annual


def make_clustering_candidate(annual, config):
    columns = config["features"]["clustering"]["columns"]
    year_quality = annual.groupby("indicator_year", sort=True).agg(
        countries=("country_code", "size"), countries_with_three_values=("clustering_values_complete", "sum"))
    expected = len(config["countries"]["countries"])
    common = year_quality.index[(year_quality.countries.eq(expected)) & year_quality.countries_with_three_values.eq(expected)]
    if not len(common):
        return None, year_quality.reset_index(), None
    reference_year = int(max(common))
    metadata = [f"{name}_{suffix}" for name in columns
                for suffix in ("value_missing", "unit", "source_file", "collected_at", "source_lastupdated", "quality_issue")]
    candidate = annual.loc[annual.indicator_year.eq(reference_year),
                           ["country_code", "country_name", "region", "indicator_year", *columns, *metadata]].copy()
    candidate = candidate.rename(columns={"indicator_year": "reference_year"})
    candidate["values_complete"] = candidate[columns].notna().all(axis=1).astype("boolean")
    candidate["selection_basis"] = "latest_common_observed_year"
    candidate["reference_year_status"] = "candidate_release_evidence_pending"
    candidate["availability_policy"] = config["features"]["clustering"]["reference_year"]
    candidate["historical_release_verified"] = pd.Series(False, index=candidate.index, dtype="boolean")
    candidate["training_approved"] = pd.Series(False, index=candidate.index, dtype="boolean")
    candidate["scaled"] = pd.Series(False, index=candidate.index, dtype="boolean")
    candidate["released_at"] = pd.Series(pd.NA, index=candidate.index, dtype=STRING)
    candidate["release_evidence"] = pd.Series(pd.NA, index=candidate.index, dtype=STRING)
    for name in candidate:
        if candidate[name].dtype == object:
            candidate[name] = candidate[name].astype(STRING)
    return candidate.sort_values("country_code").reset_index(drop=True), year_quality.reset_index().convert_dtypes(), reference_year


def validate_panels(details, long, annual, candidate, config):
    n = len(config["countries"]["countries"]) * len(years(config))
    values = list(config["sources"]["world_bank"]["indicators"])
    require(len(long) == n * len(values) and len(annual) == n, "국가·연도·지표 격자 수 불일치")
    require(not long.duplicated(KEY).any() and not annual.duplicated(["country_code", "indicator_year"]).any(), "중복 패널 키")
    require(int(long.record_observed.sum()) == len(details), "원본 행 누락·중복")
    original = details.astype({"country_code": STRING, "indicator_name": STRING, "indicator_year": "Int64"}).set_index(KEY).value.sort_index()
    observed = long.loc[long.record_observed].set_index(KEY).value.sort_index()
    pd.testing.assert_series_equal(original, observed)
    require(long.loc[~long.record_observed, "value"].isna().all(), "없는 관측을 채웠습니다.")
    for name in values:
        expected = long.loc[long.indicator_name.eq(name)].set_index(["country_code", "indicator_year"]).value.sort_index()
        actual = annual.set_index(["country_code", "indicator_year"])[name].sort_index()
        pd.testing.assert_series_equal(expected, actual, check_names=False)
    if candidate is not None:
        require(len(candidate) == len(config["countries"]["countries"]) and not candidate.country_code.duplicated().any()
                and candidate.reference_year.nunique() == 1, "군집 후보 국가당 한 행·동일 연도 위반")
        cluster_columns = config["features"]["clustering"]["columns"]
        require(candidate[cluster_columns].notna().all().all() and not candidate.training_approved.any()
                and not candidate.scaled.any(), "군집 후보의 결측·학습 승인·표준화 상태 불일치")


def coverage_tables(long, config):
    rows = []
    for (country, name), group in long.groupby(["country_code", "indicator_name"], sort=True):
        present = group.loc[group.value.notna()]
        rows.append({"country_code": country, "indicator_name": name, "indicator_code": group.indicator_code.iloc[0],
                     "expected_years": len(group), "source_rows": int(group.record_observed.sum()),
                     "available_years": len(present), "first_year": int(present.indicator_year.min()) if len(present) else pd.NA,
                     "last_year": int(present.indicator_year.max()) if len(present) else pd.NA,
                     "source_null_years": ";".join(map(str, group.loc[group.record_observed & group.value_missing, "indicator_year"])),
                     "no_record_years": ";".join(map(str, group.loc[~group.record_observed, "indicator_year"])),
                     "negative_values": int(group.value.lt(0).sum()), "zero_values": int(group.value.eq(0).sum()),
                     "quality_issues": int(group.quality_issue.notna().sum())})
    return pd.DataFrame(rows).convert_dtypes()


def prepare_world_bank(root, config, snapshot, version):
    root = Path(root).resolve()
    interim = root / config["sources"]["storage"]["interim_dir"] / "world_bank" / version
    processed = root / config["sources"]["storage"]["processed_dir"] / "world_bank" / version
    reports = root / config["sources"]["storage"]["reports_dir"]
    for folder in (interim, processed):
        require(folder.resolve().is_relative_to(root) and (not folder.exists() or not any(folder.iterdir())),
                f"기존 버전 또는 저장소 밖 경로: {folder}")
    require(not (reports / f"world_bank_{version}_preprocessing.md").exists(), "기존 보고서를 덮어쓰지 않습니다.")
    details, requests, input_hash = read_world_bank(root, config, snapshot)
    long, annual = make_panels(details, config)
    candidate, cluster_quality, reference_year = make_clustering_candidate(annual, config)
    validate_panels(details, long, annual, candidate, config)
    coverage = coverage_tables(long, config)
    files = []
    tables = [(details, interim, "world_bank_details"), (requests, interim, "request_reconciliation"),
              (long, processed, "macro_annual_long"), (annual, processed, "macro_annual")]
    if candidate is not None:
        tables.append((candidate, processed, "country_indicators_candidate"))
    for frame, folder, name in tables:
        files.extend(save_table(frame, folder, name))
    coverage.to_csv(reports / f"world_bank_{version}_country_indicator_audit.csv", index=False, encoding="utf-8-sig")
    long.loc[long.value_missing | long.quality_issue.notna(), KEY + ["record_observed", "missing_reason", "quality_issue", "source_file"]].to_csv(
        reports / f"world_bank_{version}_missing_review.csv", index=False, encoding="utf-8-sig")
    cluster_quality.to_csv(reports / f"world_bank_{version}_clustering_year_coverage.csv", index=False, encoding="utf-8-sig")
    summary = {"source": "world_bank", "processing_version": version, "raw_snapshot": snapshot,
               "processed_at": utc_now(), "status": "annual_prepared_with_gaps_and_release_evidence_pending",
               "requests_verified": len(requests), "source_rows": len(details), "source_rows_retained": len(details),
               "source_rows_dropped": 0, "source_null_values": int(details.value.isna().sum()),
               "countries_reviewed": len(config["countries"]["countries"]), "indicators_reviewed": len(config["sources"]["world_bank"]["indicators"]),
               "country_indicator_combinations_reviewed": len(coverage), "annual_long_rows": len(long), "annual_rows": len(annual),
               "available_values": int(long.value.notna().sum()), "missing_values": int(long.value.isna().sum()),
               "no_record_cells": int((~long.record_observed).sum()), "quality_issue_cells": int(long.quality_issue.notna().sum()),
               "negative_inflation_values_retained": int((long.indicator_name.eq("inflation_pct") & long.value.lt(0)).sum()),
               "clustering_candidate_reference_year": reference_year, "clustering_candidate_countries": 0 if candidate is None else len(candidate),
               "clustering_reference_year_approved": False, "public_release_dates_verified": False,
               "historical_vintage_verified": False, "imputation_applied": False, "scaling_applied": False,
               "quarterly_replication_applied": False, "cross_source_join_applied": False,
               "training_dataset_generated": False, "raw_modified": False, "previous_source_outputs_modified": False,
               "csv_parquet_null_semantics_verified": True, "csv_null_value": CSV_NULL,
               "csv_keep_default_na": False, "input_collection_manifest_sha256": input_hash,
               "indicators": {}, "config_sha256": {name: sha256(root / "configs" / f"{name}.yaml")
                                                     for name in ("countries", "sources", "features", "splits")},
               "environment": {name: importlib.metadata.version(name) for name in ("pandas", "pyarrow", "numpy")},
               "files": [{**record, "path": record["path"].relative_to(root).as_posix()} for record in files]}
    for name, setting in config["sources"]["world_bank"]["indicators"].items():
        sub = long.loc[long.indicator_name.eq(name)]
        summary["indicators"][name] = {"code": setting["code"], "unit": setting["unit"],
                                        "source_rows": int(sub.record_observed.sum()), "expected_cells": len(sub),
                                        "available_values": int(sub.value.notna().sum()), "missing_values": int(sub.value.isna().sum())}
    write_json(processed / "manifest.json", summary)
    write_report(reports / f"world_bank_{version}_preprocessing.md", summary, coverage, config)
    print(json.dumps({k: summary[k] for k in ("status", "source_rows", "source_rows_dropped", "annual_rows", "available_values", "missing_values", "clustering_candidate_reference_year", "indicators")}, ensure_ascii=False, indent=2))
    return summary


def write_report(path, summary, coverage, config):
    version = summary["processing_version"]
    lines = ["# World Bank 전처리·전체 국가 대조 결과", "", f"버전: `{version}` | 원본: `{summary['raw_snapshot']}`", "",
             "사양서 9·11절 및 features/sources/countries/splits YAML과 대조해 지도학습용 거시지표 6개와 K-means용 원래 지표 3개를 정리했다. 모든 20개국·6개 지표·수집 범위 연도를 검토했다.", "",
             "## 처리 결과", "",
             f"- 요청 {summary['requests_verified']}개, 실제 원본 {summary['source_rows']:,}행 전부 보존. 제외 {summary['source_rows_dropped']}행.",
             f"- 원본 행에는 null {summary['source_null_values']}개가 포함돼 있다. 원본 행이 없는 격자 {summary['no_record_cells']}개는 별도로 구분했다.",
             f"- 20개국×14년(2013~2026)×6개 지표의 long 격자 {summary['annual_long_rows']:,}행, 국가×연도 wide 표 {summary['annual_rows']}행을 생성했다.",
             f"- 실제 숫자 {summary['available_values']:,}/1,680개, 결측 {summary['missing_values']}개. 결측 대체·국가 제외·임의 반올림·분기 복제는 하지 않았다.", "",
             "| 지표 | 코드 | 단위 | 값 확보 / 280 |", "|---|---|---|---:|"]
    for name, r in summary["indicators"].items():
        lines.append(f"| {name} | {r['code']} | {r['unit']} | {r['available_values']} |")
    lines += ["", "## 모든 국가·지표 확보 범위", "", "| 국가 | GDP | 1인당 GDP | 인구 | 물가 | 환율 | 인터넷 |", "|---|---|---|---|---|---|---|"]
    order = list(config["sources"]["world_bank"]["indicators"])
    for country in config["countries"]["countries"]:
        parts = []
        for name in order:
            row = coverage.loc[coverage.country_code.eq(country["iso3"]) & coverage.indicator_name.eq(name)].iloc[0]
            parts.append(f"{row['first_year']}~{row['last_year']} ({row['available_years']}/14)")
        lines.append(f"| {country['name_ko']} ({country['iso3']}) | " + " | ".join(parts) + " |")
    lines += ["", "기간의 시작·끝은 숫자가 있는 첫해·마지막해다. 중간 결측은 아래 120개 국가·지표 대조표로 확인한다.", "",
              "## 결측·특이값 처리", "",
              "- 2026년은 20개국·6개 지표 모두 행이 없어 120개 셀을 null로 남겼다. 2026년 값을 2025년이나 2024년으로 덮어쓰지 않았다.",
              "- 2025년 미국 물가·베트남 환율·중국/인도 외 18개국 인터넷 이용률은 원본에 행이 있으나 값이 null이다. 총 20개로, 원본 미조회와 구분했다.",
              f"- 물가의 음수 {summary['negative_inflation_values_retained']}개는 그대로 보존했다. 음의 물가상승률을 잘못된 무역 금액처럼 삭제하지 않았다.",
              f"- 값 범위 확인 사항은 {summary['quality_issue_cells']}개다. 원천 unit이 빈 문자열이어도 공식 지표코드·설정 단위를 함께 남겼으며 source_decimal로 값 자체를 반올림하지 않았다.",
              "- 환율은 현지통화/USD의 연평균이고 1인당 GDP는 당해 명목 USD다. 환율을 뒤집거나 PPP 지표로 바꾸지 않았다. 인터넷 이용률·물가는 퍼센트 원래 값이며 이 단계에서 100으로 나누지 않았다.", "",
              "## 지도학습·군집 분리", "",
              "- macro_annual은 국가×연도 6개 지표와 지표별 결측·단위·원천·수집시각이다. 분기 교역자료와 결합하지 않은 원천 관측 표이며 예측 당시 알려진 값의 선택은 후속이다.",
              f"- K-means용 세 지표가 20개국 모두 있는 최신 연도는 {summary['clustering_candidate_reference_year']}년이다. country_indicators_candidate에 국가당 한 행으로 {summary['clustering_candidate_countries']}개국의 1인당 GDP·인구·인터넷 이용률을 저장했다.",
              "- 이 연도는 값 완전성에 따른 후보다. 설정의 latest_common_released_year에 필요한 과거 공개 근거는 미확보이므로 reference_year_status=candidate_release_evidence_pending·training_approved=false로 구분했다. 데이터를 보류·삭제한 것이 아니며 20개국 원래 값을 모두 제공한다.",
              "- 국가별로 서로 다른 연도를 섞지 않았고 StandardScaler·K-means 적합·군집 번호 생성은 하지 않았다. 현재 후보를 과거 모든 origin에 공통 사용하지 않는다.", "",
              "## 공개시점과 후속 확정", "",
              "- API lastupdated는 데이터셋 갱신일이다. 개별 연간 값의 최초 공개일·과거 빈티지로 쓰지 않으며 released_at·release_evidence는 null로 유지했다.",
              "- 기존 관세청 v003·Comtrade v001과 원본을 보존했다. 세 원천 통합·시차·분할·공통 학습 파일 생성은 이번 범위에서 실행하지 않았다.",
              "- [최종 자료 확정·결합·분리 참고 사양서](../docs/dataset_release_reference.md)의 공개 근거·사용 지연·결측·배포 기준을 확인한 뒤 통합한다.", "",
              "## 파일·재현", "",
              f"- `python scripts/prepare_data.py --source world_bank --snapshot {summary['raw_snapshot']} --version {version}`",
              f"- 정규화 상세·요청 검산 CSV/Parquet: `data/interim/world_bank/{version}/`",
              f"- 연간 long·wide·군집 후보 CSV/Parquet 및 manifest: `data/processed/world_bank/{version}/`",
              "- CSV null 표식은 `\\N`이며 빈 문자열과 구분한다. `keep_default_na=False, na_values=[r\"\\N\"]` 및 manifest 자료형을 적용하거나 Parquet를 읽는다.",
              f"- [120개 국가·지표 전체 대조](world_bank_{version}_country_indicator_audit.csv)",
              f"- [결측·범위 확인 목록](world_bank_{version}_missing_review.csv)",
              f"- [연도별 군집 세 지표 완전성](world_bank_{version}_clustering_year_coverage.csv)",
              f"- [독립 원본·파일 검산 결과](world_bank_{version}_audit.json): `python scripts/audit_world_bank_preprocessing.py --version {version}`", "",
              "## 공식 지표·API 기준", "",
              "- [World Bank API 호출·페이지 구조](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures)",
              "- [명목 USD 기준 1인당 GDP](https://data.worldbank.org/indicator/NY.GDP.PCAP.CD)",
              "- [현지통화/USD 연평균 환율](https://data.worldbank.org/indicator/PA.NUS.FCRF)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
