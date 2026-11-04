"""Normalize every configured Comtrade series without discarding UNSD aggregates.

Market features use importer-reported world/Korea imports. Exporter-reported
Comtrade exports remain a comparison series and never replace Customs targets.
"""

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oinq.collectors.raw import month_ranges, months_in_range, utc_now, write_json
from oinq.preprocessing.customs import save_table, sha256

STRING = pd.StringDtype(storage="python")
KEY = ["country_code", "hs_code"]
ROLES = ("world_import", "korea_import", "export_comparison")
ROLE_LABELS = {"world_import": "대상국 전체 수입", "korea_import": "대상국 한국산 수입",
               "export_comparison": "한국 신고 수출 비교"}
EDITIONS = {"H2": "2002", "H3": "2007", "H4": "2012", "H5": "2017", "H6": "2022"}
REFERENCE_DIR = "data/reference/comtrade/hs_20261103"


class ComtradeDataError(ValueError):
    """An input contradicts the declared collection scope or source semantics."""


def require(condition, message):
    if not condition:
        raise ComtradeDataError(message)


def typed(frame):
    """Use explicit nullable dtypes, including original flag and quantity fields."""
    frame = frame.copy()
    for name in frame:
        series = frame[name]
        if pd.api.types.is_bool_dtype(series.dtype):
            frame[name] = series.astype("boolean")
        elif pd.api.types.is_integer_dtype(series.dtype):
            frame[name] = series.astype("Int64")
        elif pd.api.types.is_float_dtype(series.dtype):
            frame[name] = series.astype("Float64")
        else:
            frame[name] = series.astype(STRING)
    return frame


def dimensions(frame, config):
    frame = frame.copy()
    countries = {r["iso3"]: r for r in config["countries"]["countries"]}
    products = {r["hs_code"]: r for r in config["products"]["products"]}
    frame["country_name"] = frame.country_code.map({k: v["name_ko"] for k, v in countries.items()})
    frame["region"] = frame.country_code.map({k: v["region"] for k, v in countries.items()})
    frame["product_name"] = frame.hs_code.map({k: v["name_ko"] for k, v in products.items()})
    for name, value in {"source": "comtrade", "currency": "USD", "weight_unit": "kg",
                        "snapshot_kind": "current_revised_snapshot", "finality_status": "not_provided"}.items():
        frame[name] = value
    for name in ("released_at", "release_evidence"):
        frame[name] = pd.Series(pd.NA, index=frame.index, dtype=STRING)
    return typed(frame)


def check_references(root, config):
    folder = root / REFERENCE_DIR
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed = {r["edition"]: r for r in manifest["files"]}
    review = []
    for edition, year in EDITIONS.items():
        path = folder / f"{edition}.json"
        require(sha256(path) == indexed[edition]["sha256"], f"공식 HS 사전 체크섬 오류: {edition}")
        reference = {str(r["id"]): r for r in json.loads(path.read_bytes())["results"]}
        for product in config["products"]["products"]:
            code = product["hs_code"]
            r = reference.get(code)
            require(r is not None and r["aggrlevel"] == 6 and r["parent"] == code[:4],
                    f"공식 HS 사전 코드·상위 분류 불일치: {edition}/{code}")
            review.append({"hs_edition": edition, "hs_edition_year": year, "hs_code": code,
                           "product_name": product["name_ko"], "parent_code": r["parent"],
                           "reference_description": r["text"], "reference_parent_description": reference[r["parent"]]["text"],
                           "reference_url": indexed[edition]["url"], "code_and_parent_verified": True})
    return typed(pd.DataFrame(review)), sha256(manifest_path)


def identify(row, countries, expected_country, settings):
    """Validate numeric statistical-area codes and their ISO labels together."""
    if expected_country is None:
        require(row["flowCode"] == settings["export_flow_code"]
                and row["reporterCode"] == settings["exporter_reporter_code"]
                and row["reporterISO"] == "KOR", "한국 신고 수출의 신고국·방향 불일치")
        mapping = {r["comtrade_partner"]: r for r in countries}
        country = mapping.get(row["partnerCode"])
        require(country is not None and row["partnerISO"] == country["iso3"], "수출 비교 상대국 매핑 불일치")
        return country["iso3"], "export_comparison"
    require(row["flowCode"] == settings["import_flow_code"]
            and row["reporterCode"] == expected_country["comtrade_reporter"]
            and row["reporterISO"] == expected_country["iso3"], "수입시장 신고국·방향 불일치")
    require(row["partnerCode"] in (settings["world_partner_code"], settings["korea_partner_code"]),
            "수입시장 상대국 범위 불일치")
    role = "world_import" if row["partnerCode"] == settings["world_partner_code"] else "korea_import"
    require(row["partnerISO"] == ("W00" if role == "world_import" else "KOR"), "수입시장 상대국 ISO 불일치")
    return expected_country["iso3"], role


def value_basis(row, role):
    """Use source values, not a country default, to identify CIF/FOB basis."""
    value = row.get("primaryValue")
    if value is None:
        return "not_provided"
    if role == "export_comparison" and value == row.get("fobvalue"):
        return "FOB"
    if value == row.get("cifvalue"):
        return "CIF"
    if value == row.get("fobvalue"):
        return "FOB"
    return "reported_unresolved"


def read_comtrade(root, config, snapshot):
    folder = root / config["sources"]["storage"]["raw_dir"] / snapshot
    manifest_path = folder / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["collection_complete"] and not manifest["integrity_errors"]
            and not manifest["api_errors"] and not manifest["duplicate_keys"], "원본 수집 무결성 확인 필요")
    indexed = {r["path"].replace("\\", "/"): r for r in manifest["files"]}
    countries = config["countries"]["countries"]
    products = [r["hs_code"] for r in config["products"]["products"]]
    settings = config["sources"]["comtrade"]
    details, requests, reconciliation = [], [], []
    for start, end in month_ranges(config):
        for country in [None, *countries]:
            code = "KOR" if country is None else country["iso3"]
            name = f"{code}_{'exports' if country is None else 'imports'}_{start.replace('-', '')}_{end.replace('-', '')}"
            meta = json.loads((folder / "comtrade" / f"{name}.meta.json").read_text(encoding="utf-8"))
            params = meta["parameters"]
            expected_partners = {r["comtrade_partner"] for r in countries} if country is None else {0, 410}
            require(meta["endpoint"] == settings["data_endpoint"] and meta["source"] == "comtrade"
                    and meta["http_status"] == 200 and meta["transport_ok"] and meta["api_ok"], f"요청 실패: {name}")
            require(int(params["reporterCode"]) == (410 if country is None else country["comtrade_reporter"])
                    and {int(x) for x in str(params["partnerCode"]).split(",")} == expected_partners
                    and params["flowCode"] == ("X" if country is None else "M")
                    and set(params["period"].split(",")) == set(months_in_range(start, end))
                    and set(params["cmdCode"].split(",")) == set(products)
                    and int(params["partner2Code"]) == 0 and params["customsCode"] == "C00"
                    and int(params["motCode"]) == 0, f"사양서·요청 범위 불일치: {name}")
            path = folder / "comtrade" / meta["body_file"]
            relative = path.relative_to(folder).as_posix()
            digest = sha256(path)
            require(digest == meta["sha256"] == indexed[relative]["sha256"], f"원본 체크섬 오류: {name}")
            document = json.loads(path.read_bytes())
            rows = document.get("data")
            require(isinstance(rows, list) and not document.get("error") and document.get("count") == len(rows)
                    and len(rows) < int(params["maxRecords"]), f"응답 오류·잘림: {name}")
            source_file = path.relative_to(root).as_posix()
            file_rows = []
            for row in rows:
                target_country, role = identify(row, countries, country, settings)
                hs = str(row["cmdCode"])
                month = str(row["period"])
                require(row["typeCode"] == "C" and row["freqCode"] == "M" and month in params["period"].split(",")
                        and str(row["refYear"]) == month[:4] and int(row["refMonth"]) == int(month[4:])
                        and hs in products and len(hs) == row["aggrLevel"] == 6
                        and row["partner2Code"] == 0 and row["customsCode"] == "C00" and row["motCode"] == 0
                        and str(row["mosCode"]).strip() == "0"
                        and row["classificationSearchCode"] == "HS" and row["classificationCode"] in EDITIONS,
                        f"원본 행의 기간·HS·통계 범위 불일치: {name}")
                for field in ("primaryValue", "netWgt"):
                    v = row.get(field)
                    require(v is None or (isinstance(v, (float, int)) and np.isfinite(v) and v >= 0),
                            f"금액·중량 오류: {name}/{hs}/{month}/{field}")
                normalized = {**row, "country_code": target_country, "hs_code": hs,
                              "period_month": month[:4] + "-" + month[4:], "source_role": role,
                              "trade_value_usd": row.get("primaryValue"), "net_weight_kg": row.get("netWgt"),
                              "valuation": value_basis(row, role), "hs_edition": row["classificationCode"],
                              "hs_edition_year": EDITIONS[row["classificationCode"]],
                              "source_reporter_code": str(row["reporterCode"]), "source_partner_code": str(row["partnerCode"]),
                              "source_file": source_file, "source_row_number": len(file_rows),
                              "collected_at": meta["collected_at"]}
                file_rows.append(normalized)
            details.extend(file_rows)
            reconciliation.append({"source_file": source_file, "request_country": code,
                                   "request_start": start, "request_end": end, "request_success": True,
                                   "request_empty": not rows, "source_rows": len(rows), "normalized_rows": len(file_rows),
                                   "source_sha256": digest, "collected_at": meta["collected_at"]})
            targets = countries if country is None else [country]
            roles = ["export_comparison"] if country is None else ["world_import", "korea_import"]
            for target in targets:
                for role in roles:
                    for hs in products:
                        for month in months_in_range(start, end):
                            requests.append({"country_code": target["iso3"], "hs_code": hs,
                                             "period_month": month[:4] + "-" + month[4:], "source_role": role,
                                             "source_file": source_file, "collected_at": meta["collected_at"],
                                             "request_success": True, "request_empty": not rows,
                                             "source_reporter_code": str(410 if role == "export_comparison" else target["comtrade_reporter"]),
                                             "source_partner_code": str(target["comtrade_partner"] if role == "export_comparison"
                                                                        else 0 if role == "world_import" else 410)})
    frame = pd.DataFrame(details)
    require(len(frame) == sum(manifest["counts"][f"comtrade_{role}_rows"] for role in ROLES), "원본·상세 수 불일치")
    for col in ("trade_value_usd", "net_weight_kg", "primaryValue", "netWgt", "qty", "altQty", "grossWgt", "cifvalue", "fobvalue"):
        frame[col] = frame[col].astype("Float64")
    return dimensions(frame, config), typed(pd.DataFrame(requests)), typed(pd.DataFrame(reconciliation)), sha256(manifest_path)


def make_panels(details, requests, config):
    key = [*KEY, "source_role", "period_month"]
    require(not details.duplicated(key).any(), "원본 상세 키 중복: HS 판본·집계행을 임의 합산하지 않습니다.")
    require(not requests.duplicated(key).any(), "요청 월 격자 중복")
    require(pd.MultiIndex.from_frame(details[key]).isin(pd.MultiIndex.from_frame(requests[key])).all(), "요청 격자 밖 상세")
    fields = ["trade_value_usd", "net_weight_kg", "valuation", "hs_edition", "isOriginalClassification",
              "isReported", "isAggregate", "isNetWgtEstimated", "legacyEstimationFlag"]
    monthly = requests.merge(details[key + fields], on=key, how="left", validate="one_to_one", indicator=True)
    monthly["record_observed"] = monthly.pop("_merge").eq("both")
    monthly["quarter"] = pd.PeriodIndex(monthly.period_month, freq="M").asfreq("Q").astype(str)
    for col in ("trade_value_usd", "net_weight_kg"):
        monthly[col] = monthly[col].astype("Float64")
    monthly["data_status"] = np.where(monthly.trade_value_usd.notna(), "actual", "missing")
    monthly["missing_reason"] = pd.Series(pd.NA, index=monthly.index, dtype=STRING)
    monthly.loc[~monthly.record_observed, "missing_reason"] = "no_record_in_successful_response"
    monthly.loc[monthly.record_observed & monthly.trade_value_usd.isna(), "missing_reason"] = "value_not_provided"
    monthly["weight_missing"] = monthly.net_weight_kg.isna()
    monthly = dimensions(monthly, config).sort_values(key).reset_index(drop=True)
    qkey = [*KEY, "source_role", "quarter"]
    grouped = monthly.groupby(qkey, sort=True, observed=True)
    quarterly = grouped.agg(observed_months=("record_observed", "sum"),
                            valid_value_months=("trade_value_usd", "count"),
                            valid_weight_months=("net_weight_kg", "count"),
                            expected_months=("period_month", "size"),
                            estimated_weight_months=("isNetWgtEstimated", "sum"),
                            source_reported_months=("isReported", "sum"),
                            unsd_aggregate_months=("isAggregate", "sum")).reset_index()
    for column in ("trade_value_usd", "net_weight_kg"):
        observed = grouped[column].sum(min_count=1).reset_index(name=f"observed_{column}")
        quarterly = quarterly.merge(observed, on=qkey, validate="one_to_one")
        valid = "valid_value_months" if column == "trade_value_usd" else "valid_weight_months"
        quarterly[column] = quarterly[f"observed_{column}"].where(quarterly[valid].eq(3), pd.NA).astype("Float64")
    def unique_values(values):
        return ";".join(sorted(set(str(v) for v in values.dropna()))) or pd.NA
    for column in ("valuation", "hs_edition", "source_file", "source_reporter_code", "source_partner_code"):
        quarterly = quarterly.merge(grouped[column].agg(unique_values).reset_index(), on=qkey, validate="one_to_one")
    quarterly = quarterly.merge(grouped.collected_at.max().reset_index(), on=qkey, validate="one_to_one")
    quarterly["value_complete"] = quarterly.trade_value_usd.notna()
    quarterly["weight_complete"] = quarterly.net_weight_kg.notna()
    quarterly["data_status"] = np.select([quarterly.value_complete, quarterly.observed_months.gt(0)],
                                          ["actual", "partial"], default="missing")
    quarterly["missing_reason"] = pd.Series(pd.NA, index=quarterly.index, dtype=STRING)
    quarterly.loc[~quarterly.value_complete & quarterly.observed_months.gt(0), "missing_reason"] = "incomplete_value_months"
    quarterly.loc[quarterly.observed_months.eq(0), "missing_reason"] = "no_record_in_successful_responses"
    quarterly = dimensions(quarterly, config)
    for col in ("observed_months", "valid_value_months", "valid_weight_months", "expected_months",
                "estimated_weight_months", "source_reported_months", "unsd_aggregate_months"):
        quarterly[col] = quarterly[col].astype("Int64")
    return monthly, quarterly


def market_panel(panel, period_column, config):
    key = [*KEY, period_column]
    columns = ["trade_value_usd", "net_weight_kg", "valuation", "hs_edition", "collected_at", "source_file", "data_status"]
    if period_column == "quarter":
        columns += ["valid_value_months", "valid_weight_months", "observed_months", "estimated_weight_months"]
    else:
        columns += ["record_observed", "isNetWgtEstimated"]
    result = None
    for role, prefix in (("world_import", "market_import"), ("korea_import", "korea_import")):
        subset = panel.loc[panel.source_role.eq(role), key + columns].copy()
        rename = {c: f"{prefix}_{c}" for c in columns}
        rename.update(trade_value_usd=f"{prefix}_value_usd", net_weight_kg=f"{prefix}_net_weight_kg")
        subset = subset.rename(columns=rename)
        result = subset if result is None else result.merge(subset, on=key, how="outer", validate="one_to_one")
    world, korea = result.market_import_value_usd, result.korea_import_value_usd
    comparable = (result.market_import_valuation.eq(result.korea_import_valuation)
                  & result.market_import_valuation.isin(["CIF", "FOB"])
                  & result.market_import_hs_edition.eq(result.korea_import_hs_edition)).fillna(False)
    valid = (world.gt(0) & korea.notna() & korea.ge(0) & korea.le(world) & comparable).fillna(False)
    result["korea_import_share"] = (korea / world).where(valid, pd.NA).astype("Float64")
    result["share_status"] = pd.Series("available", index=result.index, dtype=STRING)
    result.loc[world.isna() | korea.isna(), "share_status"] = "missing_component"
    both = world.notna() & korea.notna()
    result.loc[both & world.eq(0), "share_status"] = "zero_denominator"
    result.loc[both & world.gt(0) & ~comparable, "share_status"] = "basis_or_edition_mismatch"
    result.loc[both & korea.gt(world), "share_status"] = "korea_exceeds_world"
    result["market_values_complete"] = both
    result["data_status"] = np.where(both, "actual", np.where(world.notna() | korea.notna(), "partial", "missing"))
    # These are importer reports; never divide Customs FOB exports by this denominator.
    return dimensions(result, config).sort_values(key).reset_index(drop=True)


def compare_customs(root, panel, period_column):
    path = root / "data/processed/customs/v003" / f"customs_{'monthly' if period_column == 'period_month' else 'quarterly'}_hs6.parquet"
    customs = pd.read_parquet(path)
    key = [*KEY, period_column]
    comparison = panel.loc[panel.source_role.eq("export_comparison"), key + ["trade_value_usd", "net_weight_kg", "data_status"]].rename(
        columns={"trade_value_usd": "comtrade_export_value_usd", "net_weight_kg": "comtrade_export_net_weight_kg",
                 "data_status": "comtrade_data_status"})
    result = customs[key + ["export_value_usd", "export_net_weight_kg", "data_status"]].rename(
        columns={"export_value_usd": "customs_export_value_usd", "export_net_weight_kg": "customs_export_net_weight_kg",
                 "data_status": "customs_data_status"}).merge(comparison, on=key, how="outer", validate="one_to_one")
    result["amounts_both_present"] = result.customs_export_value_usd.notna() & result.comtrade_export_value_usd.notna()
    result["difference_usd"] = result.comtrade_export_value_usd - result.customs_export_value_usd
    result["relative_difference"] = (result.difference_usd / result.customs_export_value_usd).where(result.customs_export_value_usd.gt(0), pd.NA)
    result["target_source"] = "customs_v003"
    return typed(result)


def validate_panels(details, monthly, quarterly, market_monthly, market_quarterly, config):
    count = len(config["countries"]["countries"]) * len(config["products"]["products"])
    months = len(months_in_range(config["splits"]["collection"]["monthly_start"], config["splits"]["collection"]["monthly_end"]))
    require(len(monthly) == count * months * 3 and len(quarterly) == count * (months // 3) * 3,
            "20개국·8개 품목·세 용도의 격자 수 불일치")
    require(int(monthly.record_observed.sum()) == len(details), "원본 상세가 누락되거나 중복됐습니다.")
    for table, period in ((monthly, "period_month"), (quarterly, "quarter")):
        require(not table.duplicated([*KEY, "source_role", period]).any(), "패널 키 중복")
    key = [*KEY, "source_role", "period_month"]
    observed = monthly.loc[monthly.record_observed].set_index(key).sort_index()
    original = details.set_index(key).sort_index()
    for col in ("trade_value_usd", "net_weight_kg"):
        pd.testing.assert_series_equal(observed[col], original[col], check_dtype=False)
        qkey = [*KEY, "source_role", "quarter"]
        sums = monthly.groupby(qkey, observed=True)[col].sum(min_count=1).sort_index()
        pd.testing.assert_series_equal(sums, quarterly.set_index(qkey)[f"observed_{col}"].sort_index(), check_names=False)
        valid = "valid_value_months" if col == "trade_value_usd" else "valid_weight_months"
        require(quarterly[col].notna().eq(quarterly[valid].eq(3)).all(), "부분 분기를 완전한 값으로 사용했습니다.")
    for panel, period in ((market_monthly, "period_month"), (market_quarterly, "quarter")):
        require(not panel.duplicated([*KEY, period]).any(), "수입시장 키 중복")
        shares = panel.korea_import_share.dropna()
        require(shares.between(0, 1).all(), "점유율 범위 오류")
        valid = panel.share_status.eq("available")
        require(panel.korea_import_share.notna().eq(valid).all(), "점유율 상태 불일치")
        require(np.allclose(panel.loc[valid, "korea_import_share"].astype(float),
                            (panel.loc[valid, "korea_import_value_usd"] / panel.loc[valid, "market_import_value_usd"]).astype(float)),
                "점유율 계산 오류")


def build_coverage(monthly, quarterly):
    rows = []
    for key, group in monthly.groupby(["source_role", *KEY], sort=True, observed=True):
        role, country, hs = key
        q = quarterly.loc[quarterly.source_role.eq(role) & quarterly.country_code.eq(country) & quarterly.hs_code.eq(hs)]
        present = group.loc[group.record_observed]
        rows.append({"source_role": role, "country_code": country, "hs_code": hs,
                     "expected_months": len(group), "source_rows": len(present),
                     "valid_value_months": int(group.trade_value_usd.notna().sum()),
                     "missing_months": int(group.trade_value_usd.isna().sum()),
                     "missing_weight_rows": int(present.net_weight_kg.isna().sum()),
                     "zero_weight_rows": int(present.net_weight_kg.eq(0).sum()),
                     "estimated_weight_rows": int(present.isNetWgtEstimated.sum()),
                     "unsd_aggregate_rows": int(present.isAggregate.sum()),
                     "complete_value_quarters": int(q.value_complete.sum()),
                     "partial_quarters": int(q.data_status.eq("partial").sum()),
                     "missing_quarters": int(q.data_status.eq("missing").sum()),
                     "first_month": present.period_month.min() if len(present) else pd.NA,
                     "last_month": present.period_month.max() if len(present) else pd.NA,
                     "missing_month_list": ";".join(group.loc[group.trade_value_usd.isna(), "period_month"]),
                     "hs_editions": ";".join(sorted(set(present.hs_edition.dropna()))),
                     "valuation_bases": ";".join(sorted(set(present.valuation.dropna())))})
    return typed(pd.DataFrame(rows))


def prepare_comtrade(root, config, snapshot, version):
    root = Path(root).resolve()
    interim = root / config["sources"]["storage"]["interim_dir"] / "comtrade" / version
    processed = root / config["sources"]["storage"]["processed_dir"] / "comtrade" / version
    reports = root / config["sources"]["storage"]["reports_dir"]
    for folder in (interim, processed):
        require(folder.resolve().is_relative_to(root) and (not folder.exists() or not any(folder.iterdir())),
                f"기존 버전 또는 저장소 밖 경로: {folder}")
    require(not (reports / f"comtrade_{version}_preprocessing.md").exists(), "기존 보고서를 덮어쓰지 않습니다.")
    customs_manifest = root / "data/processed/customs/v003/manifest.json"
    customs_hash = sha256(customs_manifest)
    reference, reference_hash = check_references(root, config)
    details, requests, reconciliation, input_hash = read_comtrade(root, config, snapshot)
    monthly, quarterly = make_panels(details, requests, config)
    market_monthly = market_panel(monthly, "period_month", config)
    market_quarterly = market_panel(quarterly, "quarter", config)
    validate_panels(details, monthly, quarterly, market_monthly, market_quarterly, config)
    export_monthly = compare_customs(root, monthly, "period_month")
    export_quarterly = compare_customs(root, quarterly, "quarter")
    coverage = build_coverage(monthly, quarterly)
    files = []
    for frame, folder, name in ((details, interim, "comtrade_details"), (reconciliation, interim, "request_reconciliation"),
                                (monthly, processed, "comtrade_monthly_hs6"), (quarterly, processed, "comtrade_quarterly_hs6"),
                                (market_monthly, processed, "market_monthly_hs6"), (market_quarterly, processed, "market_quarterly_hs6"),
                                (export_monthly, interim, "customs_v003_export_comparison_monthly"),
                                (export_quarterly, interim, "customs_v003_export_comparison_quarterly")):
        files.extend(save_table(frame, folder, name))
    coverage.to_csv(reports / f"comtrade_{version}_country_product_audit.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(reports / f"comtrade_{version}_hs_edition_review.csv", index=False, encoding="utf-8-sig")
    review_columns = [*KEY, "source_role", "period_month", "record_observed", "missing_reason", "weight_missing", "valuation", "source_file"]
    monthly.loc[monthly.trade_value_usd.isna() | monthly.weight_missing, review_columns].to_csv(
        reports / f"comtrade_{version}_month_review.csv", index=False, encoding="utf-8-sig")
    quarterly[[*KEY, "source_role", "quarter", "observed_months", "valid_value_months", "valid_weight_months",
               "estimated_weight_months", "value_complete", "data_status", "missing_reason", "valuation", "hs_edition"]].to_csv(
        reports / f"comtrade_{version}_quarter_quality.csv", index=False, encoding="utf-8-sig")
    # No future/test monetary values in the public QA files.
    summary = {"source": "comtrade", "processing_version": version, "raw_snapshot": snapshot,
               "processed_at": utc_now(), "status": "comtrade_prepared_with_source_gaps",
               "requests_verified": len(reconciliation), "empty_success_requests": int(reconciliation.request_empty.sum()),
               "source_rows": len(details), "source_rows_retained": len(details), "source_rows_dropped": 0,
               "classification_held_rows": 0, "classification_completed": True,
               "countries_reviewed": len(config["countries"]["countries"]), "products_reviewed": len(config["products"]["products"]),
               "country_product_roles_reviewed": len(coverage), "monthly_rows": len(monthly), "quarterly_rows": len(quarterly),
               "market_monthly_rows": len(market_monthly), "market_quarterly_rows": len(market_quarterly),
               "input_collection_manifest_sha256": input_hash, "hs_reference_manifest_sha256": reference_hash,
               "customs_version": "v003", "customs_manifest_sha256": customs_hash,
               "customs_modified": False, "raw_modified": False, "world_bank_processed": False,
               "training_dataset_generated": False, "cross_source_target_fill_applied": False,
               "public_release_dates_verified": False, "historical_vintage_verified": False,
               "hs_codes_and_parents_verified_in_editions": list(EDITIONS),
               "csv_parquet_null_semantics_verified": True,
               "hs_editions": {str(k): int(v) for k, v in details.hs_edition.value_counts().items()},
               "unsd_aggregate_rows_retained": int(details.isAggregate.sum()),
               "estimated_weight_rows_retained": int(details.isNetWgtEstimated.sum()),
               "missing_weight_rows": int(details.net_weight_kg.isna().sum()),
               "zero_weight_rows": int(details.net_weight_kg.eq(0).sum()),
               "unresolved_valuation_rows": int(details.valuation.eq("reported_unresolved").sum()),
               "monthly_share_status": {str(k): int(v) for k, v in market_monthly.share_status.value_counts().items()},
               "quarterly_share_status": {str(k): int(v) for k, v in market_quarterly.share_status.value_counts().items()},
               "roles": {},
               "config_sha256": {name: sha256(root / "configs" / f"{name}.yaml") for name in ("countries", "products", "splits", "sources", "features")},
               "environment": {name: importlib.metadata.version(name) for name in ("pandas", "pyarrow", "numpy")},
               "files": [{**record, "path": record["path"].relative_to(root).as_posix()} for record in files]}
    for role in ROLES:
        m = monthly.loc[monthly.source_role.eq(role)]
        q = quarterly.loc[quarterly.source_role.eq(role)]
        summary["roles"][role] = {"source_rows": int(m.record_observed.sum()),
                                   "expected_months": len(m), "missing_months": int(m.trade_value_usd.isna().sum()),
                                   "complete_value_quarters": int(q.value_complete.sum()),
                                   "complete_weight_quarters": int(q.weight_complete.sum()),
                                   "quarterly_status": {str(k): int(v) for k, v in q.data_status.value_counts().items()}}
    require(sha256(customs_manifest) == customs_hash, "관세청 v003 명세 변경")
    write_report(reports / f"comtrade_{version}_preprocessing.md", summary, coverage, config)
    write_json(processed / "manifest.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "source_rows", "source_rows_dropped", "classification_held_rows", "roles", "monthly_share_status", "quarterly_share_status")}, ensure_ascii=False, indent=2))
    return summary


def write_report(path, summary, coverage, config):
    version = summary["processing_version"]
    lines = ["# UN Comtrade 전처리·전체 국가 대조 결과", "", f"버전: `{version}` | 원본: `{summary['raw_snapshot']}` | 관세청 확정본: `v003`", "",
             "20개국·8개 HS6·세 용도 전체를 사양서 2·3·6·9·16·18절 및 공통 YAML과 대조했다. 한국 수출 타깃은 관세청 v003으로 유지하고, Comtrade는 수입시장 독립변수와 수출 비교 자료로 정제했다.", "",
             "## 사양서와의 대응", "", "| 사양 | 처리 |", "|---|---|",
             "| 20개국·8개 HS6 | 모든 국가·제품·세 용도 480개 조합 전수 검토, 지정 국가 외 행·중복 키 없음 |",
             "| 2013-01~2026-09 월별 → 분기 | 165개월·55분기의 고정 격자, 3개월 금액·중량 완전성 각각 확인 |",
             "| 수입시장 독립변수 | market_import_value_usd·korea_import_value_usd·korea_import_share 생성 |",
             "| USD 및 kg | primaryValue·netWgt 사용, qty·altQty를 kg 또는 개수로 임의 치환하지 않음 |",
             "| 수출 타깃 | 관세청 v003 유지, Comtrade 수출은 비교용이며 합산·자동 결측 보충 안 함 |",
             "| 공개 시점 | collected_at과 released_at 분리, 과거 공개일·빈티지 근거는 미확보 |",
             "| 파일 저장 | CSV·Parquet 동일 값·null·자료형 재조회 확인, DB 및 학습 파일 미생성 |", "",
             "## 전체 처리량", "", "| 용도 | 원본 반영 행 | 미조회 월 / 26,400 | 금액 완전 분기 / 8,800 |", "|---|---:|---:|---:|"]
    for role in ROLES:
        r = summary["roles"][role]
        lines.append(f"| {ROLE_LABELS[role]} | {r['source_rows']:,} | {r['missing_months']:,} | {r['complete_value_quarters']:,} |")
    lines += ["", f"총 {summary['requests_verified']}개 요청, {summary['source_rows']:,}행 모두 보존·반영했다. 제외 {summary['source_rows_dropped']}행, 분류 보류 {summary['classification_held_rows']}행이다. 빈 성공 요청 {summary['empty_success_requests']}개는 0으로 대체하지 않았다.", "",
              f"점유율 확보: 월별 {summary['monthly_share_status']}, 분기별 {summary['quarterly_share_status']}. 분기 점유율은 양쪽 수입금액의 3개월 합계끼리 나눈 값이며 월별 점유율 평균이 아니다.", "",
              "## 모든 국가별 수입시장 확보 범위", "", "| 국가 | 전체 수입 월행 / 1,320 | 한국산 월행 / 1,320 | 전체 수입 완전 분기 / 440 | 한국산 완전 분기 / 440 | 전체 수입 최신 월 |", "|---|---:|---:|---:|---:|---|"]
    for country in config["countries"]["countries"]:
        sub = coverage.loc[coverage.country_code.eq(country["iso3"])]
        w, k = sub.loc[sub.source_role.eq("world_import")], sub.loc[sub.source_role.eq("korea_import")]
        lines.append(f"| {country['name_ko']} ({country['iso3']}) | {int(w.source_rows.sum()):,} | {int(k.source_rows.sum()):,} | {int(w.complete_value_quarters.sum()):,} | {int(k.complete_value_quarters.sum()):,} | {w.last_month.max()} |")
    lines += ["", "최신 월은 해당 국가의 8개 제품 중 가장 늦게 조회된 월이다. 모든 제품이 해당 월까지 완전하다는 의미는 아니다. 각 제품별 시작·끝·누락 월은 480행 대조 CSV를 따른다.", "",
              "## 제외하지 않고 처리한 항목", "",
              f"- UNSD 집계 표시 {summary['unsd_aggregate_rows_retained']:,}행을 모두 사용했다. HS6·세계 상대국 0의 유효한 집계값이며 관세청의 명시적 총계 행과 다르다. isReported=false만으로 삭제하지 않는다.",
              f"- 추정 순중량 {summary['estimated_weight_rows_retained']:,}행은 추정 표시와 함께 보존했다. 금액을 추정 중량 때문에 삭제하지 않았다.",
              f"- 원천 순중량 null {summary['missing_weight_rows']:,}행, 0 {summary['zero_weight_rows']:,}행을 그대로 보존했다. 금액은 유지하고 kg당 단가로 나누지 않았다.",
              "- HS H2(2002)·H3(2007)·H4(2012)·H5(2017)·H6(2022)를 보존했다. 공식 사전 5개에서 8개 코드와 상위 4자리 분류를 확인했다. 판본이 다르다는 이유로 제외하지 않고 같은 지정 HS6에 배치했다. 법률상 모든 주·예외 범위가 불변임을 입증한 것은 아니다.",
              "- 330499·330590의 기타 범주도 기존 지정 제품군에 포함했다. 표시 제품명만 보고 세부 원본을 삭제하지 않았다.",
              "- mosCode의 공백 차이는 총 통계영역 0으로 검증하되 원문도 보존했다. 10자리 하위코드 재집계나 세계 수입+한국산 수입의 중복 합산은 하지 않았다.", "",
              "## 실제로 남은 불확실성", "",
              "- 미조회 월은 미거래·미신고·미공개 중 원인을 단정할 수 없다. 원본에 행이 없는 셀만 null로 유지했다. 한국산 수입 행이 없다고 점유율 0으로 만들지 않았다.",
              "- 3개월 미확보 분기의 금액·중량은 측정값별로 null이며 관측 부분합은 observed_*에 별도 보관한다. 전체·한국산 수입 양쪽의 완전 금액·양수 분모·동일 평가기준·판본이 확인된 셀만 점유율을 계산한다.",
              f"- 평가기준 미확인 원본은 {summary['unresolved_valuation_rows']}행이다. 이번 원본은 캐나다·남아공 수입이 FOB, 나머지 대상국은 CIF로 식별됐다. CIF로 일괄 표기하거나 FOB 수출과 직접 더하지 않았다.",
              "- 과거 released_at·당시 수정 전 빈티지·잠정/확정 상태는 원본만으로 알 수 없다. 모든 금액을 삭제하는 대신 미확인 메타정보를 표시했다. 학습용 시점 결합은 공개 근거 또는 명시한 가정 검토 후 진행해야 한다.",
              "- 관세청과 Comtrade 수출의 차이는 통계영역·기록 시점·수정 등의 영향을 받을 수 있다. 차이를 계산했지만 원인을 임의 확정하거나 관세청 값을 수정하지 않았다. 전체 월별·분기 비교는 비공유 interim 파일에 보관했다.", "",
              "## 재현·검증 및 저장", "",
              f"- 실행: `python scripts/prepare_data.py --source comtrade --snapshot {summary['raw_snapshot']} --version {version}`. 기존 버전은 덮어쓰지 않는다.",
              f"- 정규화 64,649행·요청 검산·관세청 비교: `data/interim/comtrade/{version}/`.",
              f"- 세 용도 월별·분기 패널 및 수입시장 wide 패널: `data/processed/comtrade/{version}/`.",
              f"- [480개 국가·제품·용도 전수 대조](comtrade_{version}_country_product_audit.csv)",
              f"- [40개 HS·판본 공식 사전 대조](comtrade_{version}_hs_edition_review.csv)",
              f"- [미조회 월·중량 미제공 목록](comtrade_{version}_month_review.csv)",
              f"- [26,400개 분기 품질표](comtrade_{version}_quarter_quality.csv)", "",
              f"- 독립 검산: `python scripts/audit_comtrade_preprocessing.py --version {version}`. [전체 필드·패널·점유율·관세청 비교 감사](comtrade_{version}_classification_audit.json)와 [수출 원천 차이 목록](comtrade_{version}_export_difference_review.csv)을 생성한다. 차이 목록은 삭제 대상이 아니라 원인 미확정 비교 내역이다.", "",
              "원본 294개 체크섬·요청 범위·신고국/상대국 ISO·방향·HS6·판본·유일키·값 보존·분기 부분합·점유율 계산을 검증한다. 이 단계의 actual은 조회된 금액 관측값을 뜻하며 과거 공개 여부가 검증됐다는 뜻이 아니다. 시차·시장 as-of 결합·World Bank 전처리·최종 학습 데이터 배포는 후속 작업이다.", "",
              "## 공식 확인 근거", "",
              "- [UN Comtrade 수입 CIF·FOB 및 primaryValue 기준](https://uncomtrade.org/docs/trade-valuation/)",
              "- [UN Comtrade 집계·신고·중량 추정 표시 정의](https://comtradeplus-test.azurewebsites.net/en/reference-tables?table=dataitem)",
              "- [HS 판본별 제공 범위](https://uncomtrade.org/docs/why-are-some-converted-datasets-not-accessible-in-the-ui/)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
