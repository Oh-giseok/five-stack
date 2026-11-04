"""Prepare period-wise country inputs and historical display context, without fitting.

The latest period ends at the actual collection cutoff, never a future year.
Its macro observations retain their actual annual coverage; trade completeness
is measured over the selected months rather than a hypothetical full window.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oinq.collectors.raw import utc_now, write_json
from oinq.preprocessing.customs import sha256
from oinq.preprocessing.world_bank import save_table

FEATURES = ["gdp_per_capita_usd", "population", "internet_users_pct"]
PROFILE = ["gdp_usd", *FEATURES]
KEY = ["country_code", "window_id"]


class UnsupervisedDataError(ValueError):
    """Source scope, aggregation or preservation contracts were violated."""


def require(condition, message):
    if not condition:
        raise UnsupervisedDataError(message)


def typed(frame):
    frame = frame.reset_index(drop=True).copy()
    for c in frame:
        dtype = frame[c].dtype
        if pd.api.types.is_bool_dtype(dtype):
            frame[c] = frame[c].astype("boolean")
        elif pd.api.types.is_integer_dtype(dtype):
            frame[c] = frame[c].astype("Int64")
        elif pd.api.types.is_float_dtype(dtype):
            frame[c] = frame[c].astype("Float64")
        elif dtype == object or isinstance(dtype, pd.StringDtype):
            frame[c] = frame[c].astype(pd.StringDtype(storage="python"))
    return frame


def serialized(values):
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def make_macro_windows(annual, config, as_of_date):
    """Keep full-period means and usable latest-period observed means distinct."""
    current_year = pd.Timestamp(as_of_date).year
    rows, profiles = [], []
    windows = config["features"]["clustering"]["context"]["trade"]["windows"]
    for country in config["countries"]["countries"]:
        source = annual.loc[annual.country_code.eq(country["iso3"])].set_index("indicator_year")
        for w in windows:
            years = list(range(w["start_year"], w["end_year"] + 1))
            short = w.get("short_period", False)
            start_month = w.get("start_month", f"{w['start_year']}-01")
            end_month = w.get("end_month", f"{w['end_year']}-12")
            profile = {"country_code": country["iso3"], "country_name": country["name_ko"],
                       "region": country["region"], **w,
                       "start_month": start_month, "end_month": end_month,
                       "period_month_count": len(pd.period_range(start_month, end_month, freq="M")),
                       "short_period": short,
                       "is_short_period": short,
                       "period_status": "observed_to_cutoff" if short else "ended",
                       "aggregation": "available_annual_mean" if short else "complete_three_year_mean",
                       "historical_release_verified": False, "snapshot_kind": "current_revised_snapshot"}
            for name in PROFILE:
                series = source[name].reindex(years)
                observed = [int(y) for y in series.index[series.notna()]]
                complete = len(observed) == len(years) and w["end_year"] < current_year
                usable = bool(observed) if short else complete
                if short:
                    # A partial 2026 annual observation cannot stand in for Q1-Q3.
                    series = series.where([f"{int(y)}-12" <= end_month for y in series.index])
                    observed = [int(y) for y in series.index[series.notna()]]
                    complete = False
                    usable = bool(observed)
                row = {**{k: profile[k] for k in KEY + ["start_year", "end_year", "period_status"]},
                       "indicator_name": name, "unit": config["sources"]["world_bank"]["indicators"][name]["unit"],
                       "value": float(series.mean()) if usable else np.nan,
                       "value_basis": "available_annual_mean" if short else "complete_three_year_mean",
                       "observed_mean": float(series.mean()) if observed else np.nan,
                       "valid_year_count": len(observed), "expected_year_count": len(years),
                       "observed_years": serialized(observed),
                       "missing_source_years": serialized(y for y in years if y in source.index and pd.isna(series.loc[y])),
                       "outside_source_scope_years": serialized(y for y in years if y not in source.index),
                       "future_years": serialized(y for y in years if y > current_year),
                       "in_progress_years": serialized(y for y in years if y == current_year),
                       "complete": complete,
                       "source_files": serialized(sorted(set(source.loc[source.index.isin(observed), name + "_source_file"].dropna()))),
                       "collected_at": source.loc[source.index.isin(observed), name + "_collected_at"].dropna().max()
                       if observed else pd.NA,
                       "historical_release_verified": False}
                rows.append(row)
                profile[name] = row["value"]
                profile[name + "_complete"] = complete
                for suffix, value in [("observed_mean", row["observed_mean"]),
                                      ("valid_year_count", len(observed)), ("observed_years", row["observed_years"]),
                                      ("source_files", row["source_files"]), ("collected_at", row["collected_at"])]:
                    profile[name + "_" + suffix] = value
            profile["eligible_for_fit"] = all(pd.notna(profile[name]) for name in FEATURES)
            observed_sets = {profile[name + "_observed_years"] for name in FEATURES}
            profile["eligible_for_fit"] = profile["eligible_for_fit"] and len(observed_sets) == 1
            profile["eligibility_reason"] = ("available_short_period_features" if short else "complete_ended_window") if profile["eligible_for_fit"] else "missing_or_mismatched_annual_feature_values"
            profiles.append(profile)
    profile = typed(pd.DataFrame(profiles).sort_values(KEY))
    features = profile[KEY + ["country_name", "region", "start_year", "end_year", "start_month", "end_month",
                              "period_month_count", "is_short_period", "aggregation", *FEATURES,
                              *[name + "_observed_years" for name in FEATURES],
                              "period_status", "eligible_for_fit", "eligibility_reason", "historical_release_verified"]].copy()
    return typed(pd.DataFrame(rows).sort_values(KEY + ["indicator_name"])), profile, features


def aggregate_trade(monthly, windows, by_product, frequency, as_of_date):
    """Keep complete sums, observed partial sums and missing cells distinct."""
    rows = []
    group = ["country_code"] + (["hs_code"] if by_product else [])
    countries = monthly.groupby(group, sort=True, observed=True)
    collection = set(monthly.period_month.unique())
    now_month = pd.Period(as_of_date, freq="M")
    periods = windows if frequency == "window" else [
        {"start_year": y, "end_year": y, "start_month": f"{y}-01",
         "end_month": min(f"{y}-12", monthly.period_month.max())}
        for y in sorted(monthly.period_month.str[:4].astype(int).unique())]
    for key, data in countries:
        key = key if isinstance(key, tuple) else (key,)
        dims = dict(zip(group, key))
        product_count = 1 if by_product else data.hs_code.nunique()
        for period in periods:
            start, end = int(period["start_year"]), int(period["end_year"])
            start_month = period.get("start_month", f"{start}-01")
            end_month = period.get("end_month", f"{end}-12")
            months = pd.period_range(start_month, end_month, freq="M")
            scope = data.loc[data.period_month.between(start_month, end_month)]
            expected = len(months) * product_count
            collected = sum(str(m) in collection for m in months) * product_count
            require(len(scope) == collected, "무역 월별 격자 손실 또는 중복")
            row = {**dims, "country_name": str(data.country_name.iloc[0]), "region": str(data.region.iloc[0]),
                   **({"product_name": str(data.product_name.iloc[0])} if by_product else {}),
                   **({"window_id": period["window_id"]} if frequency == "window" else {"year": start}),
                   "start_year": start, "end_year": end,
                   "start_month": start_month, "end_month": end_month,
                   "period_month_count": len(months),
                   "is_short_period": len(months) < (end-start+1)*12,
                   "period_status": "observed_to_cutoff" if end_month[-2:] != "12" else "ended",
                   "expected_window_value_cells": expected, "expected_collected_value_cells": collected,
                   "outside_collection_value_cells": expected - collected,
                   "future_value_cells": sum(m > now_month for m in months) * product_count,
                   "elapsed_outside_collection_value_cells": sum(m < now_month and str(m) not in collection for m in months) * product_count,
                   "current_month_outside_collection_value_cells": sum(m == now_month and str(m) not in collection for m in months) * product_count,
                   "currency": "USD", "valuation": "FOB", "weight_unit": "kg",
                   "source_version": "customs_v003", "historical_release_verified": False,
                   "disclosure_status": "final_test_restricted_local" if start <= 2026 <= end else "candidate_not_shared"}
            for name, stem in [("export_value_usd", "value"), ("export_net_weight_kg", "weight")]:
                valid = scope[name].notna()
                n = int(valid.sum())
                row[name] = float(scope[name].sum(min_count=1)) if n == expected else np.nan
                row["observed_" + name] = float(scope[name].sum(min_count=1)) if n else np.nan
                row["valid_" + stem + "_cells"] = n
                row["missing_collected_" + stem + "_cells"] = collected - n
            row["coverage_status"] = ("complete_to_collection_cutoff" if row["missing_collected_value_cells"] == 0
                                       else "partial" if row["valid_value_cells"] else "no_observations")
            observed = scope.loc[scope.export_value_usd.notna()]
            row["observed_period_start"] = observed.period_month.min() if len(observed) else pd.NA
            row["observed_period_end"] = observed.period_month.max() if len(observed) else pd.NA
            row["latest_observed_export_month"] = row["observed_period_end"]
            row["latest_month_observed_product_count"] = int(observed.loc[
                observed.period_month.eq(row["observed_period_end"]), "hs_code"].nunique()) if len(observed) else 0
            row["source_missing_reasons"] = serialized(sorted(scope.loc[scope.export_value_usd.isna(), "missing_reason"].dropna().unique()))
            row["source_files"] = serialized(sorted(scope.source_file.dropna().unique()))
            row["collected_at"] = scope.collected_at.dropna().max() if scope.collected_at.notna().any() else pd.NA
            amount, weight = row["export_value_usd"], row["export_net_weight_kg"]
            row["export_unit_value_usd_per_kg"] = amount / weight if pd.notna(amount) and pd.notna(weight) and weight > 0 else np.nan
            rows.append(row)
    return typed(pd.DataFrame(rows).sort_values(group + (["window_id"] if frequency == "window" else ["year"])))


def add_changes(frame, config, product=False):
    """Changes compare adjacent complete windows only; internet differences are pp."""
    frame = frame.copy()
    windows = config["features"]["clustering"]["context"]["trade"]["windows"]
    previous = {w["window_id"]: (windows[i-1]["window_id"] if i else None) for i, w in enumerate(windows)}
    durations = {w["window_id"]: len(pd.period_range(w.get("start_month", f"{w['start_year']}-01"),
                  w.get("end_month", f"{w['end_year']}-12"), freq="M")) for w in windows}
    frame["previous_window_id"] = frame.window_id.map(previous).astype("string")
    metrics = [*PROFILE, "export_value_usd"]
    keys = ["country_code"] + (["hs_code"] if product else [])
    lookup = frame[keys + ["window_id", *metrics]].rename(columns={
        "window_id": "previous_window_id", **{name: "previous_" + name for name in metrics}})
    frame = frame.merge(lookup, on=keys + ["previous_window_id"], how="left", validate="many_to_one")
    for name in metrics:
        old = frame["previous_" + name]
        if name == "internet_users_pct":
            frame[name + "_change_pp"] = frame[name] - old
        else:
            frame[name + "_change_pct"] = ((frame[name] - old) / old.where(old.ne(0))) * 100
        if name == "export_value_usd":
            same_duration = frame.window_id.map(durations).eq(frame.previous_window_id.map(durations))
            frame[name + "_change_pct"] = frame[name + "_change_pct"].where(same_duration)
            frame["export_value_comparison_status"] = np.where(same_duration, "same_duration", "different_duration_or_no_previous")
        frame = frame.drop(columns="previous_" + name)
    return typed(frame)


def load_sources(root, config):
    inputs = []
    for folder in ["data/processed/world_bank/v001", "data/processed/customs/v003"]:
        manifest_path = root / folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs.append({"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256(manifest_path)})
        for record in manifest["files"]:
            path = root / record["path"]
            require(path.exists() and sha256(path) == record["sha256"], f"원천 체크섬 불일치: {path.name}")
            inputs.append({"path": path.relative_to(root).as_posix(), "sha256": record["sha256"]})
    annual = pd.read_parquet(root / "data/processed/world_bank/v001/macro_annual.parquet")
    monthly = pd.read_parquet(root / "data/processed/customs/v003/customs_monthly_hs6.parquet")
    country_codes = {r["iso3"] for r in config["countries"]["countries"]}
    hs_codes = {r["hs_code"] for r in config["products"]["products"]}
    require(not annual.duplicated(["country_code", "indicator_year"]).any(), "거시 연도 중복")
    require(set(annual.country_code) == country_codes and len(annual) == 280, "거시 국가/연도 범위 불일치")
    annual_keys = pd.MultiIndex.from_frame(annual[["country_code", "indicator_year"]])
    require(set(annual_keys) == set(pd.MultiIndex.from_product([sorted(country_codes), range(2013, 2027)])),
            "거시 국가별 연도 격자 불일치")
    for name in PROFILE:
        vals = annual[name].dropna().to_numpy(dtype=float)
        require(np.isfinite(vals).all() and (vals >= 0).all(), "경제지표 유한 수치/범위 오류")
    require(not annual.internet_users_pct.gt(100).any(), "인터넷 이용률 범위 오류")
    expected = pd.MultiIndex.from_product([sorted(country_codes), sorted(hs_codes),
               pd.period_range(config["splits"]["collection"]["monthly_start"],
                               config["splits"]["collection"]["monthly_end"], freq="M").astype(str)],
               names=["country_code", "hs_code", "period_month"])
    actual = pd.MultiIndex.from_frame(monthly[list(expected.names)])
    require(actual.is_unique and set(actual) == set(expected), "관세청 전체 국가·제품·월 격자 불일치")
    require(set(monthly.currency) == {"USD"} and set(monthly.valuation) == {"FOB"}, "수출 통화·평가 기준 불일치")
    require(not monthly.period_month.gt("2026-09").any(), "설정 외 수출 관측 기간")
    for name in ["export_value_usd", "export_net_weight_kg"]:
        require(np.isfinite(monthly[name].dropna().to_numpy(dtype=float)).all() and not monthly[name].lt(0).any(), "무역 수치 범위 오류")
    return annual, monthly, inputs


def prepare_unsupervised(root, config, version, as_of_date):
    root = Path(root)
    folder = root / "data/processed/unsupervised" / version
    require(not folder.exists(), f"기존 처리 버전을 덮어쓰지 않습니다: {version}")
    annual, monthly, inputs = load_sources(root, config)
    windows = config["features"]["clustering"]["context"]["trade"]["windows"]
    macro_long, profile, features = make_macro_windows(annual, config, as_of_date)
    annual_product = aggregate_trade(monthly, windows, True, "annual", as_of_date)
    annual_country = aggregate_trade(monthly, windows, False, "annual", as_of_date)
    window_product = aggregate_trade(monthly, windows, True, "window", as_of_date)
    window_country = aggregate_trade(monthly, windows, False, "window", as_of_date)
    duplicate_metadata = ["country_name", "region", "start_year", "end_year", "start_month", "end_month",
                          "period_month_count", "is_short_period", "period_status", "historical_release_verified"]
    trade_context = window_country.drop(columns=duplicate_metadata)
    context = add_changes(profile.merge(trade_context, on=KEY, validate="one_to_one"), config)
    product_context = add_changes(profile.merge(window_product.drop(columns=duplicate_metadata),
                                               on=KEY, validate="one_to_many"), config, product=True)
    tables = {"macro_annual": annual, "macro_window_long": macro_long, "country_window_profile": profile,
              "country_window_features": features, "country_product_export_annual": annual_product,
              "country_export_annual": annual_country, "country_product_export_window": window_product,
              "country_export_window": window_country, "country_window_context": context,
              "country_product_window_context": product_context}
    expected_rows = [280, 400, 100, 100, 2240, 280, 800, 100, 100, 800]
    keys = [["country_code", "indicator_year"], KEY + ["indicator_name"], KEY, KEY,
            ["country_code", "hs_code", "year"], ["country_code", "year"],
            ["country_code", "hs_code", "window_id"], KEY, KEY, ["country_code", "hs_code", "window_id"]]
    files = []
    for (name, frame), n, key in zip(tables.items(), expected_rows, keys):
        require(len(frame) == n and not frame.duplicated(key).any(), f"출력 격자 또는 중복 오류: {name}")
        files.extend(save_table(frame, folder, name))
    by_window = []
    for w in windows:
        f = features.loc[features.window_id.eq(w["window_id"])]
        t = window_product.loc[window_product.window_id.eq(w["window_id"])]
        by_window.append({**w, "countries": len(f), "fit_eligible": int(f.eligible_for_fit.sum()),
                          "trade_country_product_rows": len(t), "complete_trade_rows": int(t.export_value_usd.notna().sum()),
                          "valid_monthly_value_cells": int(t.valid_value_cells.sum()),
                          "missing_collected_value_cells": int(t.missing_collected_value_cells.sum())})
    summary = {"processing_version": version, "spec_version": "v1.3", "processed_at": utc_now(),
               "as_of_date": as_of_date, "status": "preprocessed_local_pending_release_review",
               "countries": 20, "products": 8, "windows": by_window,
               "feature_columns": FEATURES, "feature_key": KEY,
               "feature_rows": len(features), "fit_eligible_rows": int(features.eligible_for_fit.sum()),
               "annual_macro_rows_preserved": len(annual), "source_monthly_trade_rows_reviewed": len(monthly),
               "scaling_applied": False, "model_fitted": False, "cluster_assignments_generated": False,
               "shared_scaler_and_centroids_required": True, "fit_scope": "eligible_observed_periods_retrospective_only",
               "historical_backtest_claim": False, "historical_release_verified": False,
               "imputation_applied": False, "source_modified": False, "supervised_release_modified": False,
               "approved_for_shared_release": False, "contains_private_final_test_amounts": True,
               "release_policy": "local_only; exclude_or_redact_2026_test_amounts_before_common_release",
               "csv_null_value": "\\N", "csv_keep_default_na": False, "csv_parquet_roundtrip_exact": True,
               "inputs": inputs, "config_sha256": {p: sha256(root / p) for p in [
                   "configs/features.yaml", "configs/models/kmeans.yaml", "configs/countries.yaml", "configs/products.yaml"]},
               "files": [{**r, "path": r["path"].relative_to(root).as_posix(),
                          "local_only": True} for r in files]}
    write_json(folder / "manifest.json", summary)
    report = root / "reports" / f"unsupervised_{version}_preprocessing.md"
    lines = ["# 비지도학습 데이터셋 전처리 결과", "", f"버전: {version} / 사양 v1.3 / 기준일: {as_of_date}", "",
             "20개국·8개 제품·2013년 이후 전체 이력을 검토했다. 기존 원천과 지도학습 공유본은 수정하지 않았다.", "",
             "| 구간 | 보존 국가 | 학습 가능 국가×구간 | 수출 국가×제품 | 완전 수출 합계 | 수집 범위 내 월 금액 결측 |",
             "|---|---:|---:|---:|---:|---:|"]
    for w in by_window:
        label = "2025~2026Q3" if w.get("short_period") else f"{w['start_year']}~{w['end_year']}"
        lines.append(f"| {label} | {w['countries']} | {w['fit_eligible']} | {w['trade_country_product_rows']} | {w['complete_trade_rows']} | {w['missing_collected_value_cells']} |")
    lines += ["", "- 10개 논리 표를 CSV·Parquet 각 1개로 저장하고 값·null·자료형을 재조회 대조했다.",
              f"- 전체 100행 중 {summary['fit_eligible_rows']}행은 세 입력이 실제 확보돼 적합 가능하다. 마지막 구간은 3년 미완결을 이유로 제외하지 않는다.",
              "- 마지막 구간은 2025-01~2026-09의 21개월이다. 수출 완전성은 품목당 21개·국가 전체 168개 월별 값으로 판단한다. 2026년 연간 표도 1~9월 실적으로 명시한다.",
              "- 과거 네 구간 경제지표는 완전 3개년 평균, 마지막은 실제 확보 연도의 평균과 실제 연도를 저장한다. 현재 GDP·인구는 2025년 20개국, 인터넷은 중국·인도 2개국만 있고 18개국은 실제 원천 결측이다.",
              "- 수출은 완전 합계와 관측 부분합을 구분한다. 관측된 실제 0은 유지하고 결측은 0으로 채우지 않았다.",
              "- 인터넷 변화는 %p, 경제지표 변화는 %이며 관측 연도 수를 함께 읽는다. 수출 증감률은 같은 길이의 구간만 계산한다. 마지막 21개월을 이전 36개월 누적과 직접 나누지 않는다.",
              "- 공통 스케일러·군집 중심은 후속 학습에서 적합한다. 군집 번호·모델·화면 운영 결과는 아직 없다.",
              "- 이 자료는 현재 수정치 기반 과거 변화 탐색용이다. 과거 공개시점이나 누수 없는 역사 예측을 입증하지 않는다.",
              "- 전체 파일은 data/processed/unsupervised/" + version + "/의 로컬 처리본이다. 2026년 테스트 금액을 포함하므로 자동 공용 배포하지 않는다.",
              "- 표준화 전 원래 단위 보존: GDP·1인당 GDP는 명목 USD, 인구는 명, 인터넷은 %. 개인 소득·실질 구매력으로 해석하지 않는다."]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["status", "feature_rows", "fit_eligible_rows", "windows"]}, ensure_ascii=False, indent=2))
    return summary
