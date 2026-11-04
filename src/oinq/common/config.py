"""Load shared YAML and check cross-file project constraints without API access."""

from collections import Counter
from pathlib import Path, PureWindowsPath
import re

import yaml


class ConfigError(ValueError):
    """A shared configuration is invalid or inconsistent."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in result:
                raise ConfigError(f"중복 YAML 키: {key} (줄 {key_node.start_mark.line + 1})")
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise ConfigError("YAML 키는 문자열 등 단일 값이어야 합니다.") from exc
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_yaml(path):
    """Read one versioned mapping; duplicate keys must never silently override."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = yaml.load(handle, Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"YAML 읽기 실패: {path.name} ({type(exc).__name__})") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ConfigError(f"{path.name}: schema_version=1인 YAML mapping이 필요합니다.")
    return data


def load_project_config(root):
    """Return plain dictionaries keyed by config filename and model ID."""
    folder = Path(root) / "configs"
    data = {name: load_yaml(folder / f"{name}.yaml")
            for name in ("countries", "products", "splits", "features", "sources")}
    data["models"] = {name: load_yaml(folder / "models" / f"{name}.yaml")
                      for name in ("xgboost", "lightgbm", "catboost", "kmeans")}
    return data


def quarter_number(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}Q[1-4]", value):
        raise ConfigError(f"분기 형식은 YYYYQ1~Q4여야 합니다: {value!r}")
    return int(value[:4]) * 4 + int(value[-1]) - 1


def _require(condition, message):
    if not condition:
        raise ConfigError(message)


def _relative_path(value):
    if not isinstance(value, str) or not value:
        return False
    native, windows = Path(value), PureWindowsPath(value)
    return not (native.is_absolute() or windows.anchor
                or ".." in native.parts or ".." in windows.parts)


def validate_project_config(data):
    """Check shared scope, chronology, feature mappings and output isolation.

    This checks configuration only, not credentials, API coverage, or data leakage
    inside a dataset. Runtime processing must enforce those separate checks.
    """
    try:
        return _validate(data)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ConfigError(f"설정 필수 필드 누락 또는 자료형 오류: {exc}") from exc


def _validate(data):
    country_cfg, product_cfg = data["countries"], data["products"]
    splits, features, sources = data["splits"], data["features"], data["sources"]
    countries, products = country_cfg["countries"], product_cfg["products"]
    _require(len(countries) == country_cfg["expected_count"] == 20, "대상국은 20개여야 합니다.")
    for field in ("iso3", "iso2", "customs", "comtrade_reporter", "comtrade_partner", "world_bank"):
        _require(len({row[field] for row in countries}) == 20, f"국가 매핑 중복: {field}")
    _require(Counter(row["region"] for row in countries) == country_cfg["expected_regions"],
             "국가 지역별 수와 expected_regions가 다릅니다.")
    for row in countries:
        _require(re.fullmatch(r"[A-Z]{3}", row["iso3"]) is not None, "ISO3 형식 오류")
        _require(re.fullmatch(r"[A-Z]{2}", row["iso2"]) is not None, "ISO2 형식 오류")
        _require(row["customs"] == row["iso2"] and row["world_bank"] == row["iso3"],
                 "원천 국가코드와 내부 코드 매핑을 확인하세요.")
        _require(all(type(row[field]) is int and row[field] > 0
                     for field in ("comtrade_reporter", "comtrade_partner")), "Comtrade 코드 형식 오류")
    by_country = {row["iso3"]: row for row in countries}
    for iso3, code in {"IND": 699, "FRA": 251, "USA": 842}.items():
        _require(all(by_country[iso3][field] == code
                     for field in ("comtrade_reporter", "comtrade_partner")),
                 f"{iso3}: 확인한 Comtrade 매핑은 {code}입니다.")
    _require(country_cfg["exporter"]["iso3"] == "KOR"
             and country_cfg["exporter"]["comtrade_reporter"] == 410,
             "수출 신고국은 한국(410)입니다.")
    codes = [row["hs_code"] for row in products]
    _require(len(codes) == product_cfg["expected_count"] == 8, "제품은 8개여야 합니다.")
    _require(all(isinstance(code, str) and re.fullmatch(r"\d{6}", code) for code in codes),
             "HS 코드는 따옴표로 감싼 6자리 문자열이어야 합니다.")
    _require(len(set(codes)) == 8, "HS 제품 중복")

    periods = splits["splits"]
    bounds = {key: (quarter_number(row["start"]), quarter_number(row["end"]))
              for key, row in periods.items()}
    _require(all(start <= end for start, end in bounds.values()), "기간 시작이 종료보다 늦습니다.")
    _require(bounds["train"][1] < bounds["validation"][0]
             and bounds["validation"][1] < bounds["final_test"][0], "학습·검증·테스트 기간이 겹칩니다.")
    _require(bounds["train"][1] + 1 == bounds["validation"][0]
             and bounds["validation"][1] + 1 == bounds["final_test"][0], "학습·검증·테스트 사이에 누락 분기가 있습니다.")
    _require(splits["split_by"] == "target_quarter" and splits["shuffle"] is False,
             "대상 분기 기준 시계열 분할이어야 합니다.")
    validation, test, forecast = splits["validation"], splits["final_test"], splits["forecast"]
    _require(validation["final_test_for_tuning"] is False
             and validation["refit_preprocessing_per_origin"] is True
             and validation["require_label_available_before_fit"] is True,
             "테스트 튜닝 금지·기준시점별 변환 적합·공개 정답 제한이 필요합니다.")
    _require(validation["horizon_quarters"] == [1, 2, 3, 4, 5]
             and validation["primary_horizons"] == [2, 3, 4, 5], "검증 예측 거리 설정 오류")
    test_quarters = [quarter_number(value) for value in test["target_quarters"]]
    _require(test_quarters == list(range(bounds["final_test"][0], bounds["final_test"][1] + 1)),
             "테스트 대상 목록과 분할 기간이 다릅니다.")
    test_origin = quarter_number(test["origin_quarter"])
    _require(test["horizon_quarters"] == [value - test_origin for value in test_quarters]
             and bounds["validation"][1] <= test_origin < test_quarters[0], "테스트 origin·예측 거리 오류")
    _require(test["strategy"] == "frozen_origin"
             and test["selection_data_must_be_available_by_origin"] is True,
             "테스트 전에 선택 모델과 정보 마감을 고정해야 합니다.")
    start, end = quarter_number(forecast["start"]), quarter_number(forecast["end"])
    _require(forecast["start"] == "2027Q1" and forecast["end"] == "2027Q4", "예측 기간은 2027년 4분기입니다.")
    for origin_key, horizon_key in (("preferred_origin_quarter", "horizon_quarters"),
                                    ("updated_origin_quarter", "updated_horizon_quarters")):
        origin = quarter_number(forecast[origin_key])
        _require(forecast[horizon_key] == [value - origin for value in range(start, end + 1)],
                 f"2027년 예측 거리와 {origin_key}가 다릅니다.")
        _require(set(forecast[horizon_key]).issubset(validation["horizon_quarters"]),
                 "운영 예측 거리는 검증 거리 안에 있어야 합니다.")
    _require(forecast["partial_quarter_as_actual"] is False
             and forecast["require_actual_coverage_before_origin_use"] is True,
             "일부 월 실적을 완전한 분기로 사용할 수 없습니다.")

    supervised, clustering = features["supervised"], features["clustering"]
    _require(features["target"]["column"] == "target_export_value_usd"
             and features["target"]["currency"] == "USD" and features["target"]["valuation"] == "FOB",
             "타깃은 한국 수출금액 USD·FOB입니다.")
    macro = supervised["macro"]
    _require(macro["enabled"] is True and macro["future_actuals_allowed"] is False
             and macro["join"] == "latest_released_at_origin", "거시지표 공개시점 제한이 필요합니다.")
    _require(set(macro["columns"]) == set(sources["world_bank"]["indicators"]), "거시 컬럼과 원천 지표가 다릅니다.")
    _require(all(isinstance(row, dict) and isinstance(row.get("code"), str) and row["code"]
                 and isinstance(row.get("unit"), str)
                 for row in sources["world_bank"]["indicators"].values()), "거시 지표 코드·단위 누락")
    _require(set(clustering["columns"]) == {"gdp_per_capita_usd", "population", "internet_users_pct"}
             and set(clustering["columns"]).issubset(macro["columns"]), "K-means 입력 3개를 확인하세요.")
    _require(clustering["allow_mixed_years"] is False
             and clustering["cluster_label_as_supervised_feature"] is False
             and clustering["scaler"] == "StandardScaler", "군집 기준연도·표준화·참고용 정책 오류")
    _require(clustering["row_key"] == ["country_code", "window_id"]
             and clustering["reference_year"] == "three_year_window"
             and clustering["aggregation"] == "complete_period_mean_with_observed_mean_for_latest_short_period"
             and clustering["shared_scaler_and_centroids_across_windows"] is True
             and clustering["historical_backtest_claim"] is False,
             "군집은 같은 3년 구간의 세 지표 평균과 공통 비교 기준을 사용합니다.")
    trade = clustering["context"]["trade"]
    windows = trade["windows"]
    _require(trade["window_years"] == 3 and trade["window_anchor_year"] == 2013
             and trade["overlapping_windows"] is False and len(windows) == 5
             and [(w["start_year"], w["end_year"]) for w in windows]
             == [(2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024), (2025, 2026)]
             and all(w["window_id"] == f"{w['start_year']}_{w['end_year']}" for w in windows[:-1])
             and windows[-1]["window_id"] == "2025_2026Q3"
             and windows[-1]["start_month"] == "2025-01"
             and windows[-1]["end_month"] == trade["collection_end_month"] == "2026-09"
             and windows[-1]["short_period"] is True,
             "마지막 군집 분석기간은 2025-01~2026-09의 실제 확보 구간입니다.")
    _require(features["quality"]["allow_future_observed_features"] is False
             and features["quality"]["fill_unreported_trade_with_zero"] is False
             and features["quality"]["interpolation_across_splits"] is False,
             "미래값·미신고 0 대체·분할 간 보간은 허용하지 않습니다.")
    _require(supervised["rolling"]["end_at_origin"] is True, "이동 통계는 origin에서 끝나야 합니다.")
    _require(supervised["ablation"]["compare_same_rows_and_origins"] is True
             and supervised["ablation"]["selection_split"] == "validation",
             "거시지표 비교는 같은 검증 행·기준시점으로 수행해야 합니다.")

    storage = sources["storage"]
    _require(storage["mode"] == "files", "저장 방식은 DB 없이 파일입니다.")
    path_keys = (key for key in storage if key.endswith("_dir") or key.endswith("_output")
                 or key == "production_manifest")
    _require(all(_relative_path(storage[key]) for key in path_keys), "저장 경로는 저장소 내부 상대 경로여야 합니다.")
    _require(PureWindowsPath(storage["final_test_dir"]).parts[0] == ".private", "최종 테스트는 개인 경로에 둡니다.")
    _require(sources["request_defaults"]["log_secret_values"] is False, "인증키를 로그에 기록할 수 없습니다.")
    _require(sources["comtrade"]["preview_for_production_dataset"] is False
             and sources["comtrade"]["require_no_truncation"] is True, "Comtrade 잘림 없는 정식 응답을 사용하세요.")
    _require(sources["comtrade"]["exporter_reporter_code"] == country_cfg["exporter"]["comtrade_reporter"],
             "한국 신고국 코드가 원천 설정과 다릅니다.")
    _require(sources["world_bank"]["future_year_actuals_allowed"] is False, "미래 연도 거시 실측값 사용 금지")

    for name, model in data["models"].items():
        _require(model["model_id"] == name, f"모델 ID와 파일명 불일치: {name}")
        if name == "kmeans":
            _require(model["inputs"]["columns"] == clustering["columns"]
                     and model["inputs"]["row_key"] == clustering["row_key"]
                     and model["inputs"]["reference_year"] == clustering["reference_year"]
                     and all(model["inputs"][key] == clustering[key] for key in (
                         "aggregation", "fit_scope", "shared_scaler_and_centroids_across_windows", "historical_backtest_claim"))
                     and model["inputs"]["scaler"] == clustering["scaler"], "K-means 입력 설정 불일치")
            _require(model["output"]["supervised_feature"] is False
                     and model["output"]["cluster_id_is_rank"] is False, "군집 번호는 순위·지도학습 변수가 아닙니다.")
        else:
            training = model["training"]
            _require(training["selection_split"] == "validation"
                     and training["selection_metric"] == supervised["ablation"]["primary_metric"]
                     and training["comparison_variants"] == supervised["ablation"]["variants"],
                     f"{name}: 공통 검증·거시 비교 조건 불일치")
    return {
        "countries": len(countries), "products": len(products),
        "quarters": {key: end - start + 1 for key, (start, end) in bounds.items()},
        "forecast_quarters": end - start + 1,
        "models": list(data["models"]),
    }
