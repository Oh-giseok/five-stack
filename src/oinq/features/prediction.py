"""Build future calendar rows from existing origin features without fitting."""
import numpy as np
import pandas as pd

from oinq.common.config import quarter_number


PAIR = ["country_code", "hs_code"]


class PredictionInputError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise PredictionInputError(message)


def build_prediction_inputs(origin_features, config, manifest, origin_quarter):
    """Preserve the full country/product grid and extend it to forecast quarters.

    Lags/rolling values already computed by preprocessing are reused exactly.
    Source nulls are retained; missing flags are checked rather than imputed.
    """
    quarter_number(origin_quarter)
    contract = manifest["feature_contract"]
    numeric = contract["numeric_candidates"]
    required = PAIR + ["origin_quarter"] + numeric
    missing_columns = sorted(set(required) - set(origin_features.columns))
    _require(not missing_columns, f"예측 입력 원천에 필요한 열이 없습니다: {missing_columns}")
    source = origin_features.copy()
    for column in PAIR + ["origin_quarter"]:
        source[column] = source[column].astype("string")
    source = source.loc[source.origin_quarter.eq(origin_quarter)].copy()
    _require(not source.empty, f"{origin_quarter} 기준 변수 후보가 없습니다. 최신 전처리 파일을 지정하세요.")
    _require(not source[PAIR].isna().any().any(), "국가·제품 키에 결측이 있습니다.")
    _require(not source.duplicated(PAIR).any(), "같은 국가·제품·기준분기의 입력이 중복됐습니다.")

    countries = config["countries"]["countries"]
    products = config["products"]["products"]
    country_codes = [c["iso3"] for c in countries]
    hs_codes = [p["hs_code"] for p in products]
    _require(len(set(country_codes)) == len(country_codes) and len(set(hs_codes)) == len(hs_codes), "설정의 국가·제품 키가 중복됐습니다.")
    _require(source.country_code.isin(country_codes).all() and source.hs_code.isin(hs_codes).all(), "설정 밖 국가·제품이 입력에 있습니다.")
    schemas = {c["name"]: c["dtype"] for c in manifest["schemas"]["train"]}
    for column in numeric:
        try:
            source[column] = source[column].astype(schemas[column])
        except (ValueError, TypeError) as error:
            raise PredictionInputError(f"{column}의 자료형이 명세와 다릅니다.") from error
    values = [c for c in numeric if not c.endswith("_missing")]
    for column in values:
        _require(not np.isinf(source[column].to_numpy(dtype=float, na_value=np.nan)).any(), f"무한대 입력: {column}")
        flag = f"{column}_missing"
        _require(flag in source and source[flag].notna().all() and source[flag].eq(source[column].isna()).all(), f"{column}의 결측 표시가 실제 값과 다릅니다.")

    macro_year = int(origin_quarter[:4]) - 1
    if "macro_candidate_year" in source:
        _require(pd.to_numeric(source.macro_candidate_year, errors="coerce").eq(macro_year).all(), "거시지표 후보 연도가 학습 입력 정책과 다릅니다.")
    if "market_candidate_quarter" in source:
        _require(source.market_candidate_quarter.eq(origin_quarter).all(), "수입시장 기준분기가 학습 입력 정책과 다릅니다.")

    dimensions = pd.DataFrame([{"country_code": c["iso3"], "country_name": c["name_ko"], "region": c["region"],
                                "hs_code": p["hs_code"], "product_name": p["name_ko"]}
                               for c in countries for p in products])
    for column in PAIR + ["country_name", "region", "product_name"]:
        dimensions[column] = dimensions[column].astype("string")
    source = source.drop(columns=[c for c in ["country_name", "product_name", "region"] if c in source])
    source["source_row_available"] = True
    base = dimensions.merge(source, on=PAIR, how="left", validate="one_to_one")
    base["source_row_available"] = base.source_row_available.fillna(False).astype(bool)
    base["origin_quarter"] = origin_quarter
    if "origin_in_collection_range" in base:
        base["source_row_available"] &= base.origin_in_collection_range.fillna(False).astype(bool)
    _require(base.source_row_available.any(), "기준분기에 사용 가능한 원천 행이 없습니다.")
    unavailable = ~base.source_row_available
    for column in values:
        base.loc[unavailable, column] = pd.NA
        base.loc[unavailable, f"{column}_missing"] = True
    base["macro_candidate_year"] = macro_year
    base["market_candidate_quarter"] = origin_quarter
    if "origin_cutoff_at" not in base:
        base["origin_cutoff_at"] = pd.NA
    if "historical_release_verified" not in base:
        base["historical_release_verified"] = False
    base["historical_release_verified"] = base.historical_release_verified.fillna(False).astype(bool)

    forecast = config["splits"]["forecast"]
    periods = pd.period_range(forecast["start"], forecast["end"], freq="Q")
    _require(len(periods) == 4 and len(set(periods.year)) == 1, "예측 대상은 한 해의 4개 분기여야 합니다.")
    horizons = [quarter_number(str(p)) - quarter_number(origin_quarter) for p in periods]
    allowed = config["splits"]["validation"]["horizon_quarters"]
    _require(all(h in allowed for h in horizons), f"학습한 예측 거리 {allowed} 밖의 예측입니다: {horizons}")
    if origin_quarter == forecast["preferred_origin_quarter"]:
        _require(horizons == forecast["horizon_quarters"], "예측 거리 설정이 맞지 않습니다.")
    elif origin_quarter == forecast.get("updated_origin_quarter"):
        _require(horizons == forecast["updated_horizon_quarters"], "갱신 예측 거리 설정이 맞지 않습니다.")
    first = pd.Period(config["splits"]["collection"]["monthly_start"], freq="M").asfreq("Q")
    calendar = pd.DataFrame({"target_quarter": periods.astype(str), "target_year": periods.year,
                             "target_quarter_number": periods.quarter, "horizon_quarters": horizons,
                             "time_index": [p.ordinal-first.ordinal for p in periods]})
    # Ignore any caller-supplied future calendar/label fields; build only known dates.
    base = base.drop(columns=[c for c in base if c in calendar.columns or c.startswith("target_export_")])
    result = base.merge(calendar, how="cross")
    for column in manifest["feature_columns"]:
        result[column] = result[column].astype(schemas[column])
    for column in ["origin_quarter", "target_quarter", "origin_cutoff_at"]:
        result[column] = result[column].astype("string")
    _require(len(result) == len(countries)*len(products)*4, "예측 입력 격자 행 수가 맞지 않습니다.")
    _require(not result.duplicated(PAIR+["origin_quarter", "target_quarter"]).any(), "예측 입력 키가 중복됐습니다.")
    return result.sort_values(PAIR+["target_quarter"]).reset_index(drop=True)


def prepare_model_inputs(frame, loaded):
    """Match saved feature order and dtypes; do not fit or impute here."""
    columns = loaded["feature_columns"]
    missing = sorted(set(columns)-set(frame.columns))
    _require(not missing, f"모델 입력 변수가 없습니다: {missing}")
    categorical = set(loaded["categorical_columns"])
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in categorical:
            result[column] = frame[column].astype("string").fillna("__MISSING__").astype(object)
        else:
            result[column] = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            _require(not np.isinf(result[column]).any(), f"무한대 모델 입력: {column}")
    return result


def reusable_origin_features(inputs, manifest):
    """Return a compact 160-row origin table reusable via --features on another PC."""
    columns = PAIR + ["origin_quarter"] + manifest["feature_contract"]["numeric_candidates"]
    columns += ["macro_candidate_year", "market_candidate_quarter", "origin_cutoff_at", "historical_release_verified"]
    result = inputs[columns + ["source_row_available"]].drop_duplicates(PAIR).copy()
    result["origin_in_collection_range"] = result.pop("source_row_available")
    return result.reset_index(drop=True)
