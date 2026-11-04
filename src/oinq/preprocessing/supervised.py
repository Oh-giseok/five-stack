"""Lossless source integration and explicitly unapproved supervised candidates.

Observation context and candidate model inputs are separate. Calendar alignment
does not prove historical publication availability. No model is fit here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oinq.collectors.raw import utc_now, write_json
from oinq.common.config import load_yaml
from oinq.preprocessing.customs import sha256
from oinq.preprocessing.world_bank import save_table

STRING = pd.StringDtype(storage="python")
PAIR = ["country_code", "hs_code"]
DIMENSIONS = ["country_code", "country_name", "region", "hs_code", "product_name"]


class SupervisedDataError(ValueError):
    """A source, join, calendar or feature would lose or misclassify data."""


def require(condition, message):
    if not condition:
        raise SupervisedDataError(message)


def typed(frame):
    result = frame.copy()
    for column in result:
        series = result[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            result[column] = series.astype("boolean")
        elif pd.api.types.is_integer_dtype(series.dtype):
            result[column] = series.astype("Int64")
        elif pd.api.types.is_float_dtype(series.dtype):
            result[column] = series.astype("Float64")
        else:
            result[column] = series.astype(STRING)
    return result


def validate_policy(policy):
    candidates = policy["candidate_features"]
    require(policy["observation_tables"]["trade_join"] == "exact_country_hs_period"
            and policy["observation_tables"]["macro_join"] == "same_calendar_year_observation_context_only",
            "관측 문맥 결합 기준 불일치")
    require(candidates["market_period"] == "exact_origin_quarter"
            and candidates["macro_year"] == "previous_calendar_year"
            and candidates["origin_cutoff_at"] is None
            and candidates["release_policy_status"] == "pending_user_review", "후보 기간 기준 또는 공개시점 상태 불일치")
    for name in ("carry_forward_missing_values", "historical_release_verified", "training_approved",
                 "shared_release_approved", "fill_feature_nulls", "fit_scaler_or_encoder", "generate_clustering_data"):
        require(candidates[name] is False, f"이번 단계에서 적용하지 않는 설정: {name}")
    require(candidates["retain_missing_targets_in_candidates"] is True
            and policy["output"]["preserve_all_country_product_period_cells"] is True, "전체 격자 보존 필요")


def read_inputs(root, config, policy):
    tables, records, manifests = {}, [], []
    table_names = {"customs": ["customs_monthly_hs6", "customs_quarterly_hs6"],
                   "comtrade": ["market_monthly_hs6", "market_quarterly_hs6"],
                   "world_bank": ["macro_annual"]}
    for source, names in table_names.items():
        version = policy["input_versions"][source]
        require(isinstance(version, str) and version.replace("_", "").isalnum(), "원천 버전 형식 오류")
        folder = root / config["sources"]["storage"]["processed_dir"] / source / version
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest["source"] == source and manifest["processing_version"] == version, "원천 명세 버전 불일치")
        indexed = {Path(record["path"]).name: record for record in manifest["files"]}
        manifests.append({"source": source, "version": version, "path": manifest_path.relative_to(root).as_posix(),
                          "sha256": sha256(manifest_path)})
        for name in names:
            path = folder / f"{name}.parquet"
            require(sha256(path) == indexed[path.name]["sha256"], f"원천 체크섬 불일치: {path.name}")
            frame = typed(pd.read_parquet(path))
            tables[name] = frame
            records.append({"table": name, "path": path.relative_to(root).as_posix(),
                            "sha256": sha256(path), "rows": len(frame), "columns": len(frame.columns)})
    return tables, records, manifests


def add_calendar(frame, period_column):
    frame = frame.copy()
    monthly = period_column == "period_month"
    periods = pd.PeriodIndex(frame[period_column], freq="M" if monthly else "Q")
    frame["year"] = pd.Series(periods.year, index=frame.index, dtype="Int64")
    frame["quarter_number"] = pd.Series(periods.quarter, index=frame.index, dtype="Int64")
    if monthly:
        frame["month"] = pd.Series(periods.month, index=frame.index, dtype="Int64")
        expected_quarter = pd.Series(periods.asfreq("Q").astype(str), index=frame.index, dtype=STRING)
        if "quarter" in frame:
            require(frame.quarter.eq(expected_quarter).all(), "월과 원천 분기 불일치")
        frame["quarter"] = expected_quarter
    else:
        frame["month_start"] = ((frame.quarter_number - 1) * 3 + 1).astype("Int64")
        frame["month_end"] = (frame.month_start + 2).astype("Int64")
        frame["period_months"] = pd.Series([";".join(str(p) for p in pd.period_range(q.asfreq("M", how="start"),
                                                            q.asfreq("M", how="end"), freq="M")) for q in periods],
                                             index=frame.index, dtype=STRING)
    return frame


def safe_unit_value(frame):
    frame = frame.copy()
    value, weight = frame.export_value_usd, frame.export_net_weight_kg
    frame["export_unit_value_usd_per_kg"] = (value / weight).where(value.notna() & weight.gt(0), pd.NA).astype("Float64")
    frame["export_unit_value_status"] = pd.Series("available", index=frame.index, dtype=STRING)
    frame.loc[value.isna(), "export_unit_value_status"] = "missing_export_value"
    frame.loc[value.notna() & weight.isna(), "export_unit_value_status"] = "missing_net_weight"
    frame.loc[value.notna() & weight.eq(0), "export_unit_value_status"] = "zero_net_weight"
    return frame


def integrate_observations(customs, market, macro, period_column):
    """Keep every source column, namespace collisions, prove each join's shape."""
    key = [*PAIR, period_column]
    for name, frame in (("customs", customs), ("market", market)):
        require(not frame.duplicated(key).any(), f"{name} 중복 키")
    ckeys, mkeys = pd.MultiIndex.from_frame(customs[key]), pd.MultiIndex.from_frame(market[key])
    require(ckeys.isin(mkeys).all() and mkeys.isin(ckeys).all(), "교역 원천의 전체 격자 불일치")
    # Dimensions are checked even though the original market columns are retained.
    for column in ("country_name", "product_name", "region", "currency", "weight_unit"):
        left = customs.set_index(key)[column].sort_index()
        right = market.set_index(key)[column].sort_index()
        require(left.eq(right).all(), f"국가·제품·단위 불일치: {column}")
    market_map = {column: column if column in key or column not in customs else f"comtrade_{column}"
                  for column in market}
    frame = customs.merge(market.rename(columns=market_map), on=key, how="left", validate="one_to_one")
    frame = add_calendar(frame, period_column)
    frame["frequency"] = "monthly" if period_column == "period_month" else "quarterly"
    macro_key = ["country_code", "indicator_year"]
    require(not macro.duplicated(macro_key).any(), "거시지표 국가·연도 중복")
    macro_map = {column: column if column == "country_code" or column not in frame else f"world_bank_{column}"
                 for column in macro}
    macro_map["indicator_year"] = "macro_observation_year"
    frame = frame.merge(macro.rename(columns=macro_map), left_on=["country_code", "year"],
                        right_on=["country_code", "macro_observation_year"], how="left", validate="many_to_one")
    require(len(frame) == len(customs), "결합 중 행이 누락되거나 증식했습니다.")
    require(frame.macro_observation_year.notna().all(), "원천 연간 격자 누락")
    frame["macro_join_semantics"] = "same_calendar_year_observation_context_only"
    frame["historical_release_verified"] = False
    frame["training_approved"] = False
    frame = typed(safe_unit_value(frame)).sort_values(key).reset_index(drop=True)
    return frame, {"customs": {col: col for col in customs}, "comtrade": market_map, "world_bank": macro_map}


def make_origin_features(quarterly, annual, config):
    """Use past observations; calendar candidate selection is not release proof."""
    cfg = config["features"]["supervised"]
    max_h = max(config["splits"]["validation"]["horizon_quarters"])
    observed_start = pd.Period(quarterly.quarter.min(), freq="Q")
    end = pd.Period(quarterly.quarter.max(), freq="Q")
    origins = pd.period_range(observed_start - max_h, end, freq="Q").astype(str)
    pairs = quarterly[PAIR].drop_duplicates().sort_values(PAIR)
    grid = pairs.merge(pd.DataFrame({"quarter": pd.Series(origins, dtype=STRING)}), how="cross")
    values = cfg["lag_columns"]
    actual = quarterly[PAIR + ["quarter", *values, "source_file", "collected_at", "data_status"]]
    frame = grid.merge(actual, on=PAIR + ["quarter"], how="left", validate="one_to_one").sort_values(PAIR + ["quarter"]).reset_index(drop=True)
    features = frame[PAIR + ["quarter"]].rename(columns={"quarter": "origin_quarter"}).copy()
    features["origin_in_collection_range"] = frame.quarter.ge(str(observed_start))
    features["origin_export_data_status"] = frame.data_status
    features["origin_customs_source_file"] = frame.source_file
    features["origin_customs_collected_at"] = frame.collected_at
    periods = pd.PeriodIndex(features.origin_quarter, freq="Q")
    features["origin_year"] = periods.year
    features["origin_quarter_number"] = periods.quarter
    features["origin_period_end_at"] = pd.Series(periods.end_time.strftime("%Y-%m-%dT23:59:59+00:00"), dtype=STRING)
    features["origin_cutoff_at"] = pd.Series(pd.NA, index=features.index, dtype=STRING)
    features["origin_cutoff_status"] = "not_set_publication_policy_pending"
    model_columns = []
    grouped = frame.groupby(PAIR, sort=False, observed=True)
    for value in values:
        for lag in cfg["lag_quarters"]:
            column = f"{value}_lag{lag}"
            features[column] = grouped[value].shift(lag - 1).astype("Float64")
            model_columns.append(column)
        window = cfg["rolling"]["window_quarters"]
        for stat in cfg["rolling"]["statistics"]:
            column = f"{value}_rolling{window}_{stat}"
            features[column] = grouped[value].transform(lambda s: getattr(s.rolling(window, min_periods=window), stat)()).astype("Float64")
            model_columns.append(column)
        features[f"{value}_rolling{window}_valid_quarters"] = grouped[value].transform(
            lambda s: s.rolling(window, min_periods=1).count()).astype("Int64")
    previous_year = grouped.export_value_usd.shift(cfg["growth"]["comparison_lag_quarters"])
    features["export_value_yoy_growth"] = ((frame.export_value_usd / previous_year) - 1).where(previous_year.gt(0), pd.NA).astype("Float64")
    model_columns.append("export_value_yoy_growth")
    market_columns = cfg["market"]["columns"]
    market_meta = [c for c in quarterly if c.startswith(("market_import_", "korea_import_")) and c not in market_columns]
    market = quarterly[PAIR + ["quarter", *market_columns, *market_meta]].rename(columns={"quarter": "origin_quarter"})
    features = features.merge(market, on=PAIR + ["origin_quarter"], how="left", validate="one_to_one")
    features["market_candidate_quarter"] = features.origin_quarter
    features["market_selection_status"] = "exact_origin_observation_release_unverified"
    model_columns.extend(market_columns)
    macro_columns = cfg["macro"]["columns"]
    features["macro_candidate_year"] = (features.origin_year - 1).astype("Int64")
    macro_meta = [c for c in annual if any(c.startswith(f"{name}_") for name in macro_columns)]
    macro = annual[["country_code", "indicator_year", *macro_columns, *macro_meta]].rename(columns={"indicator_year": "macro_candidate_year"})
    features = features.merge(macro, on=["country_code", "macro_candidate_year"], how="left", validate="many_to_one")
    features["macro_selection_status"] = "previous_calendar_year_candidate_release_unverified"
    features["macro_candidate_row_exists"] = features.macro_candidate_year.isin(annual.indicator_year.unique())
    model_columns.extend(macro_columns)
    features["historical_release_verified"] = False
    features["training_approved"] = False
    features["feature_selection_status"] = "calendar_candidates_pending_publication_policy"
    for column in model_columns:
        features[f"{column}_missing"] = features[column].isna()
    missing_columns = [f"{c}_missing" for c in model_columns]
    return typed(features), model_columns + missing_columns


def split_name(quarter, config):
    for name, bounds in config["splits"]["splits"].items():
        if bounds["start"] <= quarter <= bounds["end"]:
            return name
    raise SupervisedDataError(f"시간 분할 밖 대상 분기: {quarter}")


def make_samples(quarterly, features, config):
    target = quarterly[DIMENSIONS + ["quarter", "year", "quarter_number", "period_months", "month_start", "month_end",
                                     "export_value_usd", "export_net_weight_kg", "observed_months", "data_status"]].copy()
    target = target.rename(columns={"quarter": "target_quarter", "year": "target_year", "quarter_number": "target_quarter_number",
                                     "period_months": "target_period_months", "month_start": "target_month_start", "month_end": "target_month_end",
                                     "export_value_usd": "target_export_value_usd", "export_net_weight_kg": "target_export_net_weight_kg",
                                     "observed_months": "target_observed_months", "data_status": "target_data_status"})
    target["split"] = target.target_quarter.map(lambda q: split_name(q, config))
    samples = []
    for split in ("train", "validation", "final_test"):
        selected = target.loc[target.split.eq(split)].copy()
        if split == "final_test":
            selected["origin_quarter"] = config["splits"]["final_test"]["origin_quarter"]
            selected["horizon_quarters"] = [pd.Period(q, freq="Q").ordinal - pd.Period(config["splits"]["final_test"]["origin_quarter"], freq="Q").ordinal
                                             for q in selected.target_quarter]
        else:
            selected = selected.merge(pd.DataFrame({"horizon_quarters": config["splits"]["validation"]["horizon_quarters"]}), how="cross")
            selected["origin_quarter"] = [str(pd.Period(q, freq="Q") - int(h)) for q, h in zip(selected.target_quarter, selected.horizon_quarters)]
        before = len(selected)
        selected = selected.merge(features, on=PAIR + ["origin_quarter"], how="left", validate="many_to_one", indicator=True)
        require(len(selected) == before and selected.pop("_merge").eq("both").all(), "후보 표본 기준분기 연결 누락")
        selected["time_index"] = [pd.Period(q, freq="Q").ordinal - pd.Period(config["splits"]["collection"]["monthly_start"], freq="M").asfreq("Q").ordinal
                                  for q in selected.target_quarter]
        selected["target_available"] = selected.target_export_value_usd.notna()
        selected["label_status"] = np.where(selected.target_available, "observed_value_release_unverified", "source_target_missing")
        selected["candidate_only"] = True
        selected["currency"] = "USD"
        selected["valuation"] = "FOB"
        samples.append(typed(selected).sort_values(PAIR + ["origin_quarter", "target_quarter"]).reset_index(drop=True))
    return dict(zip(("train", "validation", "final_test"), samples))


def validate_outputs(monthly, quarterly, features, samples, config):
    expected_months = pd.period_range(config["splits"]["collection"]["monthly_start"], config["splits"]["collection"]["monthly_end"], freq="M")
    pairs = len(config["countries"]["countries"]) * len(config["products"]["products"])
    require(len(monthly) == pairs * len(expected_months) and len(quarterly) == pairs * len(expected_months.asfreq("Q").unique()), "전체 월·분기 격자 손실")
    for table, period in ((monthly, "period_month"), (quarterly, "quarter")):
        require(not table.duplicated(PAIR + [period]).any() and table[DIMENSIONS + ["year", "quarter_number"]].notna().all().all(), "국가·품목·달력 누락 또는 키 중복")
    require(monthly.month.notna().all() and quarterly.period_months.str.count(";").eq(2).all(), "월 정보 누락")
    require(not features.duplicated(PAIR + ["origin_quarter"]).any(), "기준분기 변수 중복")
    for split, table in samples.items():
        require(not table.duplicated(PAIR + ["origin_quarter", "target_quarter"]).any(), "표본 중복")
        require(table[DIMENSIONS + ["origin_quarter", "target_quarter", "target_year", "target_period_months", "horizon_quarters"]].notna().all().all(), "표본 식별·달력 누락")
        require(table.origin_quarter.lt(table.target_quarter).all() and not table.training_approved.any(), "예측 거리 또는 승인 상태 오류")
        require(table.macro_candidate_year.lt(table.origin_year).all(), "미래 연간 후보값 사용")
        if split == "final_test":
            require(table.origin_quarter.eq(config["splits"]["final_test"]["origin_quarter"]).all(), "테스트 origin 고정 위반")


def make_coverage(monthly, quarterly, config):
    rows = []
    for country in config["countries"]["countries"]:
        for product in config["products"]["products"]:
            m = monthly.loc[monthly.country_code.eq(country["iso3"]) & monthly.hs_code.eq(product["hs_code"])]
            q = quarterly.loc[quarterly.country_code.eq(country["iso3"]) & quarterly.hs_code.eq(product["hs_code"])]
            rows.append({"country_code": country["iso3"], "country_name": country["name_ko"], "hs_code": product["hs_code"],
                         "product_name": product["name_ko"], "monthly_rows": len(m), "quarterly_rows": len(q),
                         "monthly_export_values": int(m.export_value_usd.notna().sum()), "quarterly_export_values": int(q.export_value_usd.notna().sum()),
                         "monthly_shares": int(m.korea_import_share.notna().sum()), "quarterly_shares": int(q.korea_import_share.notna().sum()),
                         "first_month": m.period_month.min(), "last_month": m.period_month.max(),
                         "country_product_calendar_preserved": True})
    return typed(pd.DataFrame(rows))


def prepare_supervised(root, config, version):
    root = Path(root).resolve()
    policy = load_yaml(root / "configs/integration.yaml")
    validate_policy(policy)
    output = root / config["sources"]["storage"]["processed_dir"] / "supervised" / version
    private = root / config["sources"]["storage"]["final_test_dir"] / "supervised" / version
    reports = root / config["sources"]["storage"]["reports_dir"]
    for folder in (output, private):
        require(folder.resolve().is_relative_to(root) and (not folder.exists() or not any(folder.iterdir())), "기존 버전 또는 저장소 밖 경로")
    require(not (reports / f"supervised_{version}_integration.md").exists(), "기존 보고서 덮어쓰기 금지")
    tables, inputs, manifests = read_inputs(root, config, policy)
    annual = tables["macro_annual"]
    monthly, month_mapping = integrate_observations(tables["customs_monthly_hs6"], tables["market_monthly_hs6"], annual, "period_month")
    quarterly, quarter_mapping = integrate_observations(tables["customs_quarterly_hs6"], tables["market_quarterly_hs6"], annual, "quarter")
    features, numeric = make_origin_features(quarterly, annual, config)
    samples = make_samples(quarterly, features, config)
    validate_outputs(monthly, quarterly, features, samples, config)
    files = []
    for frame, folder, name in ((monthly, output, "integrated_monthly"), (quarterly, output, "integrated_quarterly"),
                                (features, output, "origin_feature_candidates"), (samples["train"], output, "train_candidates"),
                                (samples["validation"], output, "validation_candidates"),
                                (samples["final_test"], private, "final_test_candidates")):
        files.extend(save_table(frame, folder, name))
    coverage = make_coverage(monthly, quarterly, config)
    coverage.to_csv(reports / f"supervised_{version}_country_product_coverage.csv", index=False, encoding="utf-8-sig")
    month_quality = [*PAIR, "period_month", "year", "month", "quarter", "data_status", "missing_reason", "share_status"]
    monthly[month_quality].to_csv(reports / f"supervised_{version}_month_quality.csv", index=False, encoding="utf-8-sig")
    summary = {"source": "supervised", "processing_version": version, "processed_at": utc_now(),
               "status": "integrated_and_candidate_samples_created_release_policy_pending", "input_versions": policy["input_versions"],
               "splits_config": config["splits"],
               "train_period": config["splits"]["splits"]["train"],
               "validation_period": config["splits"]["splits"]["validation"],
               "final_test_period": config["splits"]["splits"]["final_test"],
               "split_by": config["splits"]["split_by"],
               "fit_protocol": {"validation": "expanding_origin", "final_test": "frozen_origin",
                                "fit_labels_must_be_available_by_origin_cutoff": True,
                                "candidate_split_does_not_approve_all_rows_for_each_historical_fit": True},
               "inputs": inputs, "input_manifests": manifests, "integration_policy": policy,
               "monthly_rows": len(monthly), "quarterly_rows": len(quarterly), "origin_feature_rows": len(features),
               "countries": len(config["countries"]["countries"]), "products": len(config["products"]["products"]),
               "country_product_combinations": len(coverage), "dropped_observation_cells": 0,
               "monthly_export_values": int(monthly.export_value_usd.notna().sum()), "quarterly_export_values": int(quarterly.export_value_usd.notna().sum()),
               "monthly_shares": int(monthly.korea_import_share.notna().sum()), "quarterly_shares": int(quarterly.korea_import_share.notna().sum()),
               "identifier_or_calendar_null_cells": 0, "column_lineage": {"monthly": month_mapping, "quarterly": quarter_mapping},
               "candidate_splits": {name: {"rows": len(frame), "target_available": int(frame.target_available.sum()),
                                             "target_missing": int((~frame.target_available).sum()),
                                             "target_start": frame.target_quarter.min(), "target_end": frame.target_quarter.max(),
                                             "target_quarters": int(frame.target_quarter.nunique()),
                                             "origin_before_collection": int((~frame.origin_in_collection_range).sum())}
                                    for name, frame in samples.items()},
               "feature_contract": {"categorical": config["features"]["supervised"]["categorical"],
                                      "known_future": config["features"]["supervised"]["known_future"], "numeric_candidates": numeric,
                                      "descriptive_only": config["features"]["supervised"]["descriptive_only"],
                                      "target": config["features"]["target"]["column"],
                                      "forbidden_feature_columns": ["target_export_value_usd", "target_export_net_weight_kg", "target_available", "target_data_status", "target_observed_months", "split"],
                                      "training_approved": False},
               "public_release_dates_verified": False, "origin_cutoff_at_set": False, "staleness_policy_approved": False,
               "imputation_applied": False, "scaler_or_encoder_fitted": False, "clustering_data_modified": False,
               "shared_dataset_modified": False, "raw_or_source_outputs_modified": False, "model_trained": False,
               "csv_null_value": "\\N", "csv_keep_default_na": False, "csv_parquet_exact_values_verified": True,
               "config_sha256": {name: sha256(root / "configs" / f"{name}.yaml") for name in ("integration", "features", "splits", "countries", "products")},
               "files": [{**record, "path": record["path"].relative_to(root).as_posix()} for record in files]}
    for record in inputs + manifests:
        require(sha256(root / record["path"]) == record["sha256"], "결합 중 원천 변경")
    write_json(output / "manifest.json", summary)
    write_report(reports / f"supervised_{version}_integration.md", summary)
    print(json.dumps({k: summary[k] for k in ("status", "monthly_rows", "quarterly_rows", "countries", "products", "candidate_splits", "dropped_observation_cells")}, ensure_ascii=False, indent=2))
    return summary


def write_report(path, summary):
    version = summary["processing_version"]
    lines = ["# 지도학습용 통합·전처리 결과", "", f"버전: `{version}` | 관세청 v003 · Comtrade v001 · World Bank v001", "",
             "사용자 지시에 따라 전체 통합자료와 지도학습 후보를 먼저 생성하고 공개시점·최종 공유 기준은 미확정으로 유지했다. 비지도학습 데이터는 변경하지 않았다.", "",
             "## 국가·품목·금액·연월분기 보존", "",
             f"- 20개국×8개 품목×165개월 월별 {summary['monthly_rows']:,}행, 55분기 분기별 {summary['quarterly_rows']:,}행을 보존했다. 국가·제품·기간 행 제외는 0개다.",
             "- 월별에는 국가 코드/이름·지역·HS6/제품명·year·month·quarter·quarter_number가 있다. 분기에는 year·quarter_number·month_start/end·period_months(구성 3개월)가 있다.",
             f"- 원천 수출금액은 월별 {summary['monthly_export_values']:,}개·분기별 {summary['quarterly_export_values']:,}개, 점유율은 월별 {summary['monthly_shares']:,}개·분기별 {summary['quarterly_shares']:,}개를 그대로 유지했다. 없는 금액을 0으로 채우지 않았다.",
             "- 금액은 한국 전체 수출금액 USD·FOB다. 기업 순이익이 아니며 Comtrade 수입액·비교용 수출액을 더하지 않는다. 중량·부분합·원천 상태·파일·수집시각도 보존했다.",
             "- 원천별 공통 열 이름은 접두어로 구분하고 manifest의 column_lineage에 모든 입력 열의 연결을 남겼다. 거시지표는 같은 연도 관측 문맥으로 붙인 통합표이며 해당 분기의 모델 입력으로 바로 쓰지 않는다.", "",
             "## 지도학습 후보", "", "| 구분 | 대상 기간 | 분기 수 | 후보 행 | 정답 금액 있음 | 정답 미확보 |", "|---|---|---:|---:|---:|---:|"]
    for name, r in summary["candidate_splits"].items():
        lines.append(f"| {name} | {r['target_start']}~{r['target_end']} | {r['target_quarters']} | {r['rows']:,} | {r['target_available']:,} | {r['target_missing']:,} |")
    lines += ["", "- 학습·검증은 각 대상 분기별 h=1~5 후보를 유지한다. 여러 행에 같은 대상 금액이 있는 것은 예측 기준분기·거리가 다른 표본이며 금액을 합산해 총매출로 해석하지 않는다.",
              "- 분할은 target_quarter 기준이다. 실제 이동 검증의 학습에는 해당 origin/cutoff까지 알려진 정답만 사용한다. 예컨대 2022Q1의 h=5 후보 origin은 2020Q4이므로 2021년 정답을 당시 학습에 넣지 않는다. 후보 파일의 전체 train 행을 모든 origin에 일괄 적합하지 않는다.",
              "- 최종 테스트는 2025Q4 origin으로 고정한 2026Q1~Q3·h=1~3 후보 480행이며 .private/evaluation/supervised 아래 별도 저장했다. 공개 품질표에는 테스트 금액을 넣지 않았다.",
              "- 원천 수집 시작 이전 origin 후보도 결측 상태로 보존하고 origin_in_collection_range=false로 표시했다. 타깃 미확보 후보도 보존하며 실제 학습·정답 평가는 target_available 및 후속 공개시점 기준을 적용한다.",
              "- 금액·중량·단가 lag1/2/4, 최근 4분기 sum/mean, 전년 대비 성장률, 변수별 결측 표시를 생성했다. 이동 통계는 4개 값이 모두 있을 때만 계산하며 부분합으로 대체하지 않았다.",
              "- 후보 시장 변수는 origin의 같은 분기 값, 거시는 origin 직전 달력연도 값이다. 이는 관측 기간을 맞춘 후보 규칙이며 공개 지연을 가정하거나 과거 사용 가능성을 입증한 것이 아니다. 자동 전방 채우기·다른 연도 fallback은 하지 않았다.",
              "- origin_period_end_at은 달력 분기 말이다. origin_cutoff_at은 null, historical_release_verified/training_approved는 false다. 변수·값을 삭제하지 않고 공개시점 확정 전 후보로 구분했다.",
              "- feature_contract의 명시 목록만 모델 입력 후보로 사용한다. 타깃·타깃 중량·타깃 완전성·분할·원본 경로·동일 연도 관측 문맥은 입력 목록에 포함하지 않는다.", "",
              "## 저장·검증", "",
              f"- 통합 월별/분기별·origin_feature_candidates·train_candidates·validation_candidates: `data/processed/supervised/{version}/`.",
              f"- 최종 테스트 후보: `.private/evaluation/supervised/{version}/`. shared 배포·Git 업로드·모델 학습은 실행하지 않았다.",
              "- 모든 표는 CSV·Parquet이며 CSV null 표식은 `\\N`이다. 빈 문자열과 null을 구분하고 manifest 스키마로 재조회한다.",
              f"- [160개 국가·제품 범위표](supervised_{version}_country_product_coverage.csv) · [26,400개 월별 품질표](supervised_{version}_month_quality.csv)",
              f"- 독립 검증: `python scripts/audit_supervised_preprocessing.py --version {version}`, [검산 결과](supervised_{version}_audit.json)", "",
              "## 후속 확정", "",
              "[결합·분리 참고 사양서](../docs/dataset_release_reference.md)에 따라 원천 공개시점·빈티지 가정, 오래된 값 허용 기간, 최종 학습 사용 행·변수와 공유 버전을 확정한다. 이번 후보를 공개시점 검증을 마친 배포본으로 해석하지 않는다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
