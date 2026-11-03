"""Prepare customs-only monthly/quarterly panels with explicit coverage flags."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd

from oinq.collectors.raw import month_ranges, months_in_range, utc_now, write_json


class CustomsDataError(ValueError):
    """Raw data cannot safely be interpreted according to the shared scope."""


KEY = ["country_code", "hs_code"]
MEASURES = ["export_value_usd", "export_net_weight_kg"]
STRING = pd.StringDtype(storage="python")


def require(condition, message):
    if not condition:
        raise CustomsDataError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nullable_number(value):
    if value is None or value.strip() == "":
        return None
    require(re.fullmatch(r"\d+", value.strip()) is not None, "금액·중량에 음수 또는 정수 이외 값이 있습니다.")
    return int(value)


def parse_response(body, metadata, country, product, start, end, source_file, fields):
    """Remove only explicit total rows; preserve 6/10-digit source identities."""
    document = ET.fromstring(body)
    require(document.findtext(".//resultCode") == "00", f"성공 응답이 아닙니다: {source_file}")
    months = set(months_in_range(start, end))
    details, totals = [], []
    seen = set()
    for item in document.findall(".//item"):
        row = {child.tag: child.text for child in item}
        if row.get(fields["period"]) == "총계":
            totals.append(row)
            continue
        for field in fields.values():
            require(field in row, f"필수 응답 필드 누락: {field}, {source_file}")
        code = row[fields["hs_code"]] or ""
        period = row[fields["period"]] or ""
        require(re.fullmatch(r"\d{4}\.\d{2}", period) is not None, f"기간 형식 오류: {source_file}")
        month = period.replace(".", "")
        require(month in months and row[fields["country"]] == country["customs"],
                f"응답 국가·월이 요청과 다릅니다: {source_file}")
        require(len(code) in (6, 10) and code.isdigit() and code.startswith(product["hs_code"]),
                f"HS 범위·길이 오류: {source_file}")
        identity = (country["iso3"], code, month)
        require(identity not in seen, f"원본 상세 키 중복: {source_file}")
        seen.add(identity)
        details.append({
            "country_code": country["iso3"], "hs_code": product["hs_code"],
            "period_month": month[:4] + "-" + month[4:], "source_hs_code": code,
            "source_hs_digits": len(code), "source_country_code": row[fields["country"]],
            "source_product_name": row.get("statKor"),
            "export_value_usd": nullable_number(row[fields["export_value_usd"]]),
            "export_net_weight_kg": nullable_number(row[fields["export_net_weight_kg"]]),
            "source_file": source_file, "collected_at": metadata["collected_at"],
        })
    require(len(totals) <= 1, f"조회 총계 행이 여러 개입니다: {source_file}")
    reconciliation = {"source_file": source_file, "country_code": country["iso3"],
                      "hs_code": product["hs_code"], "start_month": start, "end_month": end,
                      "detail_rows": len(details), "total_rows": len(totals),
                      "response_empty": not details, "export_value_difference_usd": None,
                      "export_weight_difference_kg": None}
    if totals:
        for measure, output in [("export_value_usd", "export_value_difference_usd"),
                                ("export_net_weight_kg", "export_weight_difference_kg")]:
            total = nullable_number(totals[0].get(fields[measure]))
            values = [row[measure] for row in details]
            if total is not None and all(value is not None for value in values):
                difference = total - sum(values)
                reconciliation[output] = difference
                if measure == "export_value_usd":
                    require(difference == 0, f"수출금액 총계·상세 불일치: {source_file}")
    return details, reconciliation


def read_customs(root, config, snapshot):
    root = Path(root)
    folder = root / config["sources"]["storage"]["raw_dir"] / snapshot
    manifest_path = folder / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("collection_complete") is True, "완료 점검된 원본 수집 스냅샷이 필요합니다.")
    settings = config["sources"]["customs"]
    require(settings["currency"] == "USD" and settings["export_valuation"] == "FOB"
            and settings["net_weight_unit"] == "kg" and settings["amount_scale"] == 1,
            "USD·FOB·kg·금액배율 1 이외 설정은 별도 단위 검토가 필요합니다.")
    indexed = {str(row["path"]).replace("\\", "/"): row for row in manifest["files"]}
    details, reconciliations, requests = [], [], []
    fields = settings["request_fields"]
    for country in config["countries"]["countries"]:
        for product in config["products"]["products"]:
            for start, end in month_ranges(config):
                name = f"{country['iso3']}_{product['hs_code']}_{start.replace('-', '')}_{end.replace('-', '')}"
                relative = f"customs/{name}.xml"
                require(relative in indexed, f"수집 명세에 없는 요청입니다: {name}")
                body_path = folder / relative
                meta = json.loads(body_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
                expected = {fields["start"]: start.replace("-", ""), fields["end"]: end.replace("-", ""),
                            fields["country"]: country["customs"], fields["hs_code"]: product["hs_code"]}
                require(meta.get("parameters") == expected and meta.get("endpoint") == settings["endpoint"],
                        f"원본 요청과 현재 사양 불일치: {name}")
                require(meta.get("transport_ok") and meta.get("api_ok") and meta.get("http_status") == 200,
                        f"성공 확인되지 않은 요청: {name}")
                digest = sha256(body_path)
                require(digest == meta["sha256"] == indexed[relative]["sha256"], f"원본 체크섬 오류: {name}")
                source_file = body_path.relative_to(root).as_posix()
                rows, reconciliation = parse_response(body_path.read_bytes(), meta, country, product,
                                                       start, end, source_file, settings["response_fields"])
                policy = config["products"]["customs_aggregation"]
                if rows and policy["require_amount_reconciliation"]:
                    require(reconciliation["total_rows"] == 1
                            and reconciliation["export_value_difference_usd"] == 0,
                            f"수출금액 원천 총계 검산이 완료되지 않았습니다: {name}")
                details.extend(rows)
                reconciliations.append(reconciliation)
                for month in months_in_range(start, end):
                    requests.append({"country_code": country["iso3"], "hs_code": product["hs_code"],
                                     "period_month": month[:4] + "-" + month[4:], "source_file": source_file,
                                     "collected_at": meta["collected_at"], "request_success": True,
                                     "request_empty": reconciliation["response_empty"]})
    frame = pd.DataFrame(details)
    require(not frame.empty, "상세 데이터가 없습니다.")
    require(not frame.duplicated(["country_code", "source_hs_code", "period_month"]).any(), "상세 키 중복")
    require(len(frame) == manifest["counts"]["customs_detail_rows"], "수집 명세와 상세 행 수가 다릅니다.")
    for column in [*KEY, "period_month", "source_hs_code", "source_country_code", "source_product_name", "source_file", "collected_at"]:
        frame[column] = frame[column].astype(STRING)
    frame["source_hs_digits"] = frame["source_hs_digits"].astype("Int64")
    for column in MEASURES:
        frame[column] = frame[column].astype("Float64")
    return frame, pd.DataFrame(requests), pd.DataFrame(reconciliations), sha256(manifest_path)


def add_dimensions(frame, config):
    frame = frame.copy()
    countries = {row["iso3"]: row for row in config["countries"]["countries"]}
    products = {row["hs_code"]: row for row in config["products"]["products"]}
    frame["country_name"] = frame["country_code"].map({key: row["name_ko"] for key, row in countries.items()}).astype(STRING)
    frame["region"] = frame["country_code"].map({key: row["region"] for key, row in countries.items()}).astype(STRING)
    frame["product_name"] = frame["hs_code"].map({key: row["name_ko"] for key, row in products.items()}).astype(STRING)
    frame["source"] = pd.Series("customs", index=frame.index, dtype=STRING)
    frame["source_reporter_code"] = pd.Series("KR", index=frame.index, dtype=STRING)
    frame["source_partner_code"] = frame["country_code"].map({key: row["customs"] for key, row in countries.items()}).astype(STRING)
    for column, value in {"currency": "USD", "valuation": "FOB", "weight_unit": "kg",
                          "finality_status": "not_provided", "snapshot_kind": "current_revised_snapshot"}.items():
        frame[column] = pd.Series(value, index=frame.index, dtype=STRING)
    for column in ("released_at", "release_evidence", "hs_edition"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype=STRING)
    return frame


def apply_mixed_hs_rules(monthly, config, raw_snapshot):
    """Apply one declared HS6 context policy to every configured country/product."""
    policy = config["products"]["customs_aggregation"]
    require(policy["mode"] == "sum_reported_non_total_rows_within_hs6"
            and policy["decision_basis"] == "user_instruction"
            and policy["country_scope"] == "configured_countries"
            and policy["product_scope"] == "configured_hs6_products"
            and policy["period_scope"] == "configured_collection_period"
            and policy["require_amount_reconciliation"] is True
            and policy["exclude_explicit_total_rows"] is True
            and policy["retain_original_hs_codes"] is True
            and policy["hold_based_only_on_hs_length"] is False,
            "관세청 공통 HS6 집계 기준을 확인하세요.")
    countries = {row["iso3"] for row in config["countries"]["countries"]}
    products = {row["hs_code"] for row in config["products"]["products"]}
    period = config["splits"]["collection"]
    require(monthly["country_code"].isin(countries).all()
            and monthly["hs_code"].isin(products).all()
            and monthly["period_month"].between(period["monthly_start"], period["monthly_end"]).all(),
            "공통 집계 범위 밖의 국가·제품·월이 있습니다.")
    monthly["hs_level_user_approved"] = monthly["hs_level_mixed"].astype("boolean")
    monthly["aggregation_rule_id"] = pd.Series(pd.NA, index=monthly.index, dtype=STRING)
    monthly["aggregation_decision_basis"] = pd.Series(pd.NA, index=monthly.index, dtype=STRING)
    selected = monthly["hs_level_mixed"]
    monthly.loc[selected, "aggregation_rule_id"] = policy["rule_id"]
    monthly.loc[selected, "aggregation_decision_basis"] = policy["decision_basis"]
    monthly["hs_level_review_required"] = pd.Series(False, index=monthly.index, dtype="boolean")
    return monthly


def make_panels(details, requests, config, *, raw_snapshot=None):
    """Aggregate reported details by HS6; keep genuinely missing months null."""
    details = details.copy()
    for measure in MEASURES:
        details[measure] = details[measure].astype("Float64")
    group_key = KEY + ["period_month"]
    require(not requests.duplicated(group_key).any(), "월별 요청 키 중복")
    require(not details.duplicated(["country_code", "source_hs_code", "period_month"]).any(), "상세 키 중복")
    detail_keys = pd.MultiIndex.from_frame(details[group_key])
    request_keys = pd.MultiIndex.from_frame(requests[group_key])
    require(detail_keys.isin(request_keys).all(), "요청 범위 밖의 상세 행이 있습니다.")
    grouped = details.groupby(group_key, sort=True, observed=True)
    aggregated = grouped.size().rename("source_detail_count").to_frame()
    for measure in MEASURES:
        aggregated[f"observed_{measure}"] = grouped[measure].sum(min_count=1)
        aggregated[f"{measure}_detail_count"] = grouped[measure].count()
    digits = grouped["source_hs_digits"].agg(["min", "max"])
    aggregated["hs_level_mixed"] = digits["min"].ne(digits["max"])
    monthly = requests.merge(aggregated.reset_index(), on=group_key, how="left", validate="one_to_one")
    monthly["source_detail_count"] = monthly["source_detail_count"].fillna(0).astype("Int64")
    monthly["record_observed"] = monthly["source_detail_count"].gt(0)
    monthly["hs_level_mixed"] = monthly["hs_level_mixed"].fillna(False).astype("boolean")
    monthly = apply_mixed_hs_rules(monthly, config, raw_snapshot)
    for measure in MEASURES:
        count = f"{measure}_detail_count"
        monthly[count] = monthly[count].fillna(0).astype("Int64")
        valid = (monthly["record_observed"] & monthly[count].eq(monthly["source_detail_count"])
                 & ~monthly["hs_level_review_required"])
        monthly[f"observed_{measure}"] = monthly[f"observed_{measure}"].astype("Float64")
        monthly[measure] = monthly[f"observed_{measure}"].where(valid, pd.NA).astype("Float64")
    complete = monthly[MEASURES].notna().all(axis=1)
    monthly["data_status"] = pd.Series("actual", index=monthly.index, dtype=STRING)
    monthly.loc[~complete & monthly["record_observed"], "data_status"] = "partial"
    monthly.loc[~monthly["record_observed"], "data_status"] = "missing"
    monthly["coverage_status"] = monthly["data_status"].map({"actual": "complete", "partial": "review_required", "missing": "no_detail_returned"}).astype(STRING)
    monthly.loc[complete & monthly["hs_level_user_approved"], "coverage_status"] = "complete_with_configured_hs6_aggregation"
    monthly["missing_reason"] = pd.Series(pd.NA, index=monthly.index, dtype=STRING)
    monthly.loc[~monthly["record_observed"], "missing_reason"] = "no_detail_in_successful_response"
    monthly.loc[monthly["record_observed"] & ~complete, "missing_reason"] = "missing_component_or_hs_level_review"
    monthly.loc[monthly["hs_level_review_required"], "missing_reason"] = "mixed_hs6_hs10_requires_source_confirmation"
    month_number = monthly["period_month"].str[-2:].astype(int)
    monthly["quarter"] = (monthly["period_month"].str[:4] + "Q" + ((month_number - 1) // 3 + 1).astype(str)).astype(STRING)
    monthly["valid_value_month"] = monthly["export_value_usd"].notna()
    monthly["valid_weight_month"] = monthly["export_net_weight_kg"].notna()
    qgroup = monthly.groupby(KEY + ["quarter"], sort=True, observed=True)
    quarterly = qgroup.agg(observed_months=("record_observed", "sum"),
                           valid_value_months=("valid_value_month", "sum"),
                           valid_weight_months=("valid_weight_month", "sum"),
                           source_detail_count=("source_detail_count", "sum"),
                           hs_level_mixed_months=("hs_level_mixed", "sum"),
                           hs_level_user_approved_months=("hs_level_user_approved", "sum"),
                           hs_level_review_months=("hs_level_review_required", "sum"),
                           collected_at=("collected_at", "max"), source_file=("source_file", "first")).reset_index()
    for measure in MEASURES:
        observed = qgroup[f"observed_{measure}"].sum(min_count=1).reset_index(name=f"observed_{measure}")
        quarterly = quarterly.merge(observed, on=KEY + ["quarter"], validate="one_to_one")
        valid_column = "valid_value_months" if measure == "export_value_usd" else "valid_weight_months"
        quarterly[measure] = quarterly[f"observed_{measure}"].where(quarterly[valid_column].eq(3), pd.NA).astype("Float64")
    for column in ("observed_months", "valid_value_months", "valid_weight_months", "source_detail_count",
                   "hs_level_mixed_months", "hs_level_user_approved_months", "hs_level_review_months"):
        quarterly[column] = quarterly[column].astype("Int64")
    quarterly["expected_months"] = pd.Series(3, index=quarterly.index, dtype="Int64")
    quarterly["target_value_complete"] = quarterly["export_value_usd"].notna().astype("boolean")
    quarterly["data_status"] = pd.Series("actual", index=quarterly.index, dtype=STRING)
    full = quarterly[MEASURES].notna().all(axis=1)
    quarterly.loc[~full & quarterly["observed_months"].gt(0), "data_status"] = "partial"
    quarterly.loc[quarterly["observed_months"].eq(0), "data_status"] = "missing"
    quarterly["coverage_status"] = quarterly["data_status"].map({"actual": "complete", "partial": "incomplete_or_review", "missing": "no_detail_returned"}).astype(STRING)
    approved = monthly.loc[monthly["hs_level_user_approved"]]
    if len(approved):
        rules = approved.groupby(KEY + ["quarter"], observed=True)["aggregation_rule_id"].agg(
            lambda values: ";".join(sorted(values.dropna().unique()))).reset_index()
        quarterly = quarterly.merge(rules, on=KEY + ["quarter"], how="left", validate="one_to_one")
    else:
        quarterly["aggregation_rule_id"] = pd.Series(pd.NA, index=quarterly.index, dtype=STRING)
    quarterly["aggregation_decision_basis"] = pd.Series(pd.NA, index=quarterly.index, dtype=STRING)
    user_decision = quarterly["hs_level_user_approved_months"].gt(0)
    quarterly.loc[user_decision, "aggregation_decision_basis"] = "user_instruction"
    quarterly.loc[full & user_decision, "coverage_status"] = "complete_with_configured_hs6_aggregation"
    quarterly["missing_reason"] = pd.Series(pd.NA, index=quarterly.index, dtype=STRING)
    quarterly.loc[~full, "missing_reason"] = "missing_month_or_component"
    quarterly.loc[quarterly["hs_level_review_months"].gt(0), "missing_reason"] = "mixed_hs6_hs10_requires_source_confirmation"
    monthly = monthly.drop(columns=["valid_value_month", "valid_weight_month"])
    monthly = add_dimensions(monthly, config).sort_values(group_key).reset_index(drop=True)
    quarterly = add_dimensions(quarterly, config).sort_values(KEY + ["quarter"]).reset_index(drop=True)
    for frame in (monthly, quarterly):
        for column in [*KEY, "period_month", "quarter", "source_file", "collected_at", "aggregation_rule_id", "aggregation_decision_basis"]:
            if column in frame:
                frame[column] = frame[column].astype(STRING)
    return monthly, quarterly


def validate_panels(details, monthly, quarterly, config):
    months = months_in_range(config["splits"]["collection"]["monthly_start"], config["splits"]["collection"]["monthly_end"])
    combinations = len(config["countries"]["countries"]) * len(config["products"]["products"])
    require(len(monthly) == combinations * len(months), "월별 격자 행 수 불일치")
    require(not monthly.duplicated(KEY + ["period_month"]).any(), "월별 키 중복")
    require(not quarterly.duplicated(KEY + ["quarter"]).any(), "분기 키 중복")
    require(len(quarterly) == combinations * monthly["quarter"].nunique(), "분기 격자 행 수 불일치")
    for measure in MEASURES:
        original = details[measure].sum(min_count=1)
        require(original == monthly[f"observed_{measure}"].sum(min_count=1)
                == quarterly[f"observed_{measure}"].sum(min_count=1), f"합산 검산 실패: {measure}")
        require(monthly[measure].dropna().ge(0).all() and quarterly[measure].dropna().ge(0).all(), "음수 금액·중량")
        valid_column = "valid_value_months" if measure == "export_value_usd" else "valid_weight_months"
        require(quarterly[measure].notna().eq(quarterly[valid_column].eq(3)).all(), "미완전 분기 확정값 생성 오류")
    require(monthly.loc[~monthly["record_observed"], MEASURES].isna().all().all(), "미반환 월의 값 생성 오류")
    require(monthly.loc[monthly["hs_level_review_required"], MEASURES].isna().all().all(), "HS 혼재 월이 확정값에 포함됨")
    approved = monthly["hs_level_user_approved"]
    require(monthly.loc[approved, "hs_level_mixed"].all()
            and monthly.loc[approved, "aggregation_rule_id"].notna().all()
            and monthly.loc[approved, "aggregation_decision_basis"].eq("user_instruction").all(),
            "사용자 합산 행의 처리 근거가 없습니다.")


def save_table(frame, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for extension in ("csv", "parquet"):
        path = folder / f"{name}.{extension}"
        require(not path.exists(), f"기존 파일을 덮어쓰지 않습니다: {path.name}")
        temporary = path.with_suffix(path.suffix + ".tmp")
        if extension == "csv":
            frame.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep="")
            restored = pd.read_csv(temporary, encoding="utf-8-sig", dtype=frame.dtypes.to_dict(),
                                   keep_default_na=False, na_values=[""])
        else:
            frame.to_parquet(temporary, index=False, engine="pyarrow")
            restored = pd.read_parquet(temporary, engine="pyarrow")
        # Pandas may restore strings with a different memory backend.
        # Normalize storage only; still verify values, nulls and numeric dtypes.
        restored = restored.astype({column: frame[column].dtype for column in frame.columns
                                    if isinstance(frame[column].dtype, pd.StringDtype)})
        pd.testing.assert_frame_equal(frame, restored)
        temporary.replace(path)
        records.append({"path": path, "rows": len(frame), "format": extension,
                        "sha256": sha256(path), "bytes": path.stat().st_size,
                        "schema": [{"name": column, "dtype": str(frame[column].dtype)} for column in frame.columns]})
    return records


def prepare_customs(root, config, snapshot, version):
    root = Path(root).resolve()
    interim = root / config["sources"]["storage"]["interim_dir"] / "customs" / version
    processed = root / config["sources"]["storage"]["processed_dir"] / "customs" / version
    reports = root / config["sources"]["storage"]["reports_dir"]
    for folder in (interim, processed):
        require(folder.resolve().is_relative_to(root), "저장소 외부 출력 경로")
        require(not folder.exists() or not any(folder.iterdir()), f"기존 버전 경로가 있습니다: {folder.name}")
    require(not (reports / f"customs_{version}_preprocessing.md").exists(), "기존 보고서 버전이 있습니다.")
    details, requests, reconciliation, input_manifest_hash = read_customs(root, config, snapshot)
    monthly, quarterly = make_panels(details, requests, config, raw_snapshot=snapshot)
    validate_panels(details, monthly, quarterly, config)
    details = add_dimensions(details, config)
    mixed_keys = monthly.loc[monthly["hs_level_review_required"], KEY + ["period_month"]]
    held_details = details.merge(mixed_keys, on=KEY + ["period_month"], how="inner", validate="many_to_one")
    approved_keys = monthly.loc[monthly["hs_level_user_approved"], KEY + ["period_month", "aggregation_rule_id"]]
    approved_details = details.merge(approved_keys, on=KEY + ["period_month"], how="inner", validate="many_to_one")
    files = []
    tables = [(details, interim, "customs_details"),
                                (reconciliation, interim, "request_reconciliation"),
                                (monthly, processed, "customs_monthly_hs6"),
                                (quarterly, processed, "customs_quarterly_hs6")]
    if len(held_details):
        tables.append((held_details, interim, "hs_level_review_details"))
    if len(approved_details):
        tables.append((approved_details, interim, "hs_level_user_approved_details"))
    for frame, folder, name in tables:
        files.extend(save_table(frame, folder, name))
    quality_columns = [*KEY, "quarter", "observed_months", "valid_value_months", "valid_weight_months",
                       "hs_level_mixed_months", "hs_level_user_approved_months", "hs_level_review_months",
                       "aggregation_rule_id", "aggregation_decision_basis", "target_value_complete", "data_status", "missing_reason"]
    missing_columns = [*KEY, "period_month", "request_success", "request_empty", "record_observed", "missing_reason", "source_file"]
    quarterly[quality_columns].to_csv(reports / f"customs_{version}_quarter_quality.csv", index=False, encoding="utf-8-sig")
    monthly.loc[monthly["data_status"].ne("actual"), missing_columns].to_csv(
        reports / f"customs_{version}_month_review.csv", index=False, encoding="utf-8-sig")
    test_start = config["splits"]["splits"]["final_test"]["start"]
    test_end = config["splits"]["splits"]["final_test"]["end"]
    test = quarterly.loc[quarterly["quarter"].between(test_start, test_end)]
    summary = {"source": "customs", "processing_version": version, "raw_snapshot": snapshot,
               "processed_at": utc_now(), "status": "customs_prepared_with_source_gaps",
               "input_collection_manifest_sha256": input_manifest_hash,
               "requests_verified": len(reconciliation), "source_detail_rows": len(details),
               "source_hs_digits": {str(key): int(value) for key, value in details["source_hs_digits"].value_counts().items()},
               "total_rows_excluded": int(reconciliation["total_rows"].sum()),
               "empty_success_requests": int(reconciliation["response_empty"].sum()),
               "monthly_rows": len(monthly), "quarterly_rows": len(quarterly),
               "monthly_status": {str(key): int(value) for key, value in monthly["data_status"].value_counts().items()},
               "quarterly_status": {str(key): int(value) for key, value in quarterly["data_status"].value_counts().items()},
               "hs_mixed_months": int(monthly["hs_level_mixed"].sum()),
               "hs_mixed_quarters": int(quarterly["hs_level_mixed_months"].gt(0).sum()),
               "hs_held_months": int(monthly["hs_level_review_required"].sum()),
               "hs_held_quarters": int(quarterly["hs_level_review_months"].gt(0).sum()),
               "hs_user_approved_months": int(monthly["hs_level_user_approved"].sum()),
               "hs_user_approved_quarters": int(quarterly["hs_level_user_approved_months"].gt(0).sum()),
               "aggregation_policy": config["products"]["customs_aggregation"],
               "classification_scope_country_count": len(config["countries"]["countries"]),
               "classification_scope_product_count": len(config["products"]["products"]),
               "classification_completed": True,
               "source_detail_rows_retained": len(details), "source_detail_rows_dropped": 0,
               "target_complete_quarters": int(quarterly["target_value_complete"].sum()),
               "final_test_quarters": len(test), "final_test_complete_targets": int(test["target_value_complete"].sum()),
               "zero_weight_positive_value_months": int((monthly["export_net_weight_kg"].eq(0) & monthly["export_value_usd"].gt(0)).sum()),
               "amount_reconciliation_mismatches": int(reconciliation["export_value_difference_usd"].fillna(0).ne(0).sum()),
               "weight_reconciliation_mismatches": int(reconciliation["export_weight_difference_kg"].fillna(0).ne(0).sum()),
               "max_weight_difference_kg": float(reconciliation["export_weight_difference_kg"].abs().max()),
               "public_release_dates_verified": False, "hs_edition_verified": False,
               "no_record_filled_with_zero": False, "raw_modified": False,
               "other_sources_processed": False, "training_dataset_generated": False,
               "csv_parquet_null_semantics_verified": True,
               "config_sha256": {name: sha256(root / "configs" / f"{name}.yaml") for name in ("countries", "products", "splits", "sources")},
               "environment": {name: importlib.metadata.version(name) for name in ("pandas", "pyarrow", "numpy")},
               "files": [{**record, "path": record["path"].relative_to(root).as_posix()} for record in files]}
    write_json(processed / "manifest.json", summary)
    write_report(reports / f"customs_{version}_preprocessing.md", summary, config)
    print(json.dumps({key: summary[key] for key in ("status", "source_detail_rows", "total_rows_excluded", "monthly_rows", "monthly_status", "quarterly_rows", "quarterly_status", "hs_mixed_months", "hs_mixed_quarters", "final_test_complete_targets", "final_test_quarters")}, ensure_ascii=False, indent=2))
    return summary


def write_report(path, summary, config):
    lines = ["# 관세청 전처리 결과", "", f"버전: {summary['processing_version']} | 원본: {summary['raw_snapshot']}", "",
             "관세청 원본과 사양서의 국가·HS·기간·USD FOB·kg를 대조하고 월별 HS6와 달력 분기별 패널을 생성했다. Comtrade·World Bank 연결과 시차·단가·학습 파일 생성은 이번 범위에 포함하지 않았다.", "",
             "## 사양서 대조", "", "| 항목 | 결과 |", "|---|---|",
             f"| 국가 | YAML의 {len(config['countries']['countries'])}개 대상국, alpha-2→ISO3 매핑 확인 |",
             f"| 제품 | YAML의 {len(config['products']['products'])}개 HS6, 응답 코드의 접두·길이 확인 |",
             f"| 기간 | {config['splits']['collection']['monthly_start']}~{config['splits']['collection']['monthly_end']}, 요청 범위 밖 행 없음 |",
             "| 수출금액 | expDlr→export_value_usd, USD·FOB, 단위배율 1 |",
             "| 순중량 | expWgt→export_net_weight_kg, kg, 임의 반올림 보정 없음 |",
             "| 공개일·HS 판본 | 응답에 없어 null 유지, 과거 공개 근거·판본 확인은 별도 |", "",
             "## 처리 결과", "",
             f"- 검증한 요청 {summary['requests_verified']:,}개, 원본 상세 {summary['source_detail_rows']:,}행.",
             f"- 명시적인 총계 {summary['total_rows_excluded']:,}행을 집계에서 제외했다. 상세가 없는 성공 응답 {summary['empty_success_requests']}개는 월별 0으로 대체하지 않았다.",
             f"- 월별 HS6 패널 {summary['monthly_rows']:,}행: {summary['monthly_status']}.",
             f"- 분기별 HS6 패널 {summary['quarterly_rows']:,}행: {summary['quarterly_status']}.",
             f"- 2026Q1~Q3 분기 정답 금액 확보 {summary['final_test_complete_targets']}/{summary['final_test_quarters']}행. 실제 테스트 파일로 분리하거나 학습에 투입하지 않았다.", "",
             "## 미반환·부분 분기 처리", "",
             "- 원천에 숫자 0이 있으면 0을 보존한다. 성공 응답에 상세가 없으면 금액·중량은 null이며 미거래·미신고·미공개를 임의 판정하지 않는다.",
             "- 분기 내 3개월 모두 금액이 확인된 경우만 export_value_usd를 채운다. 중량도 같은 기준을 별도로 적용한다.",
             "- 부분 분기의 관측 합계는 observed_export_value_usd·observed_export_net_weight_kg에 보관한다. 이 열을 분기 정답으로 사용하지 않는다.",
             "- target_value_complete는 금액의 월별 완전성만 뜻한다. 예측 시점 공개 여부나 학습·검증 행 적합성까지 인증하는 플래그가 아니다.", "",
             "## HS6·HS10 혼재 처리", "",
             f"- 원본 코드 길이별 상세: {summary['source_hs_digits']}. 앞선 원본 소개의 46,594행 전체가 HS10이라는 설명을 바로잡는다.",
             f"- 혼재 월별 {summary['hs_mixed_months']}행·분기별 {summary['hs_mixed_quarters']}행 중 사용자 지시로 합산한 것은 월별 {summary['hs_user_approved_months']}행·분기별 {summary['hs_user_approved_quarters']}행이다.",
             f"- 분류로 인한 보류는 월별 {summary['hs_held_months']}행·분기별 {summary['hs_held_quarters']}행이다.",
             "- products.yaml의 공통 집계 기준을 모든 지정 국가·제품·수집 기간에 적용했다. 국가별 예외 없이 HS6 제품군의 비총계 조회 상세를 합산하고 수출금액 원천 총계와 검산했다.",
             "- 원본 코드·품명과 aggregation_rule_id·aggregation_decision_basis를 보존한다. 자리수 혼재만을 이유로 금액을 보류하지 않는다.",
             "- 자료가 조회되지 않은 월만 결측으로 유지한다. 일부 월만 조회된 분기의 observed_* 부분합은 완전한 분기 정답으로 대체하지 않는다.", "",
             "## 검증·남은 확인", "",
             "- 원본 체크섬·요청 조건·국가·HS·기간·유일키·금액·중량 비음수와 원본→월별 관측합→분기 관측합 보존을 검증했다.",
             f"- 조회 총계와 상세의 수출금액 차이 {summary['amount_reconciliation_mismatches']}건. 수출중량 차이는 {summary['weight_reconciliation_mismatches']}건, 최대 {summary['max_weight_difference_kg']:g}kg로 기록했으며 원본 값을 수정하지 않았다.",
             f"- 금액이 양수이고 원천 중량이 0인 월 {summary['zero_weight_positive_value_months']}개. 금액을 삭제하지 않았고 단가는 계산하지 않았다.",
             "- API 현재 수정치를 사용한 스냅샷이다. released_at·release_evidence·hs_edition은 null이며 조회시각을 공개일로 대입하지 않았다.",
             "- 원본과 최종 테스트 금액은 Git 공유 대상에 추가하지 않았다. 공개 보고서 CSV는 품질 플래그·누락 사유만 포함한다.", "",
             "## 파일", "",
             f"- 정규화 상세·조회 검산·혼재 집계 상세(CSV·Parquet): data/interim/customs/{summary['processing_version']}/",
             f"- 월별·분기별 패널(CSV·Parquet), manifest: data/processed/customs/{summary['processing_version']}/",
             f"- [월별 확인 목록](customs_{summary['processing_version']}_month_review.csv)",
             f"- [분기 품질표](customs_{summary['processing_version']}_quarter_quality.csv)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
