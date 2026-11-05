"""Read and validate persisted GEM UI results without running model code."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


QUARTER_KEYS = ("country_code", "hs_code", "origin_quarter", "target_quarter")
ANNUAL_KEYS = ("country_code", "hs_code", "origin_quarter", "target_year")
QUARTER_REQUIRED = {
    *QUARTER_KEYS,
    "country_name",
    "product_name",
    "target_year",
    "predicted_export_value_usd",
    "prediction_status",
    "input_missing_feature_count",
    "model_id",
    "model_variant",
    "run_id",
    "dataset_version",
    "currency",
    "valuation",
}
ANNUAL_REQUIRED = {
    *ANNUAL_KEYS,
    "country_name",
    "product_name",
    "available_quarters",
    "predicted_partial_sum_usd",
    "predicted_annual_export_value_usd",
    "annual_status",
    "model_id",
    "model_variant",
    "run_id",
    "dataset_version",
    "currency",
    "valuation",
}


class ResultDataError(ValueError):
    """A persisted result is malformed or combines different forecast runs."""


@dataclass(frozen=True)
class GridAudit:
    expected_rows: int
    observed_rows: int
    missing_keys: tuple[tuple[str, ...], ...]
    duplicate_keys: tuple[tuple[str, ...], ...]
    unexpected_keys: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ForecastRun:
    run_dir: Path
    metadata: dict
    quarterly: pd.DataFrame
    annual: pd.DataFrame
    grid_audit: GridAudit


def discover_forecast_runs(project_root: str | Path) -> list[str]:
    """Return complete run IDs found under the private forecast output folder."""
    folder = Path(project_root) / ".private" / "forecasts"
    if not folder.is_dir():
        return []
    required = ("run.json", "quarterly_forecast.parquet", "annual_forecast.parquet")
    return sorted(
        (child.name for child in folder.iterdir()
         if child.is_dir() and all((child / name).is_file() for name in required)),
        reverse=True,
    )


def audit_forecast_grid(
    frame: pd.DataFrame,
    country_codes: Iterable[str],
    hs_codes: Iterable[str],
    target_quarters: Iterable[str],
) -> GridAudit:
    """Find missing, duplicate, and out-of-scope country/product/quarter keys."""
    keys = ("country_code", "hs_code", "target_quarter")
    absent = set(keys) - set(frame.columns)
    if absent:
        raise ResultDataError(f"분기 예측 키 열이 없습니다: {sorted(absent)}")
    expected = set(product(
        map(str, country_codes), map(str, hs_codes), map(str, target_quarters)
    ))
    observed_rows = [tuple(str(row[key]) for key in keys) for _, row in frame.iterrows()]
    counts = pd.Series(observed_rows, dtype="object").value_counts() if observed_rows else pd.Series(dtype="int64")
    observed = set(observed_rows)
    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    return GridAudit(
        expected_rows=len(expected),
        observed_rows=len(frame),
        missing_keys=tuple(sorted(expected - observed)),
        duplicate_keys=duplicates,
        unexpected_keys=tuple(sorted(observed - expected)),
    )


def filter_forecasts(
    frame: pd.DataFrame,
    *,
    country_codes: Iterable[str] | None = None,
    hs_codes: Iterable[str] | None = None,
    target_quarters: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter forecast rows on selected dimensions, preserving null amounts."""
    result = frame.copy()
    for column, values in (
        ("country_code", country_codes),
        ("hs_code", hs_codes),
        ("target_quarter", target_quarters),
    ):
        if values is not None:
            selected = list(map(str, values))
            result = result.loc[result[column].astype("string").isin(selected)]
    return result.copy()


def sum_if_complete(values: Iterable[object], expected_count: int) -> float:
    """Sum a group only when every expected value is present."""
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    if expected_count <= 0 or len(series) != expected_count or series.isna().any():
        return float("nan")
    return float(series.sum())


def annual_sum_from_quarters(values: Iterable[object]) -> float:
    """Return a yearly amount only when all four quarterly forecasts exist."""
    return sum_if_complete(values, expected_count=4)


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ResultDataError(f"{label} 결과 형식이 맞지 않습니다. 누락 열: {missing}")


def _require_unique(frame: pd.DataFrame, keys: tuple[str, ...], label: str) -> None:
    duplicate_mask = frame.duplicated(list(keys), keep=False)
    if duplicate_mask.any():
        examples = frame.loc[duplicate_mask, list(keys)].head(5).to_dict("records")
        raise ResultDataError(f"{label} 결과 키가 중복됐습니다: {examples}")


def validate_forecast_results(
    quarterly: pd.DataFrame,
    annual: pd.DataFrame,
    metadata: dict,
    *,
    country_codes: Iterable[str],
    hs_codes: Iterable[str],
) -> GridAudit:
    """Check schemas and ensure the quarterly and annual files share one run."""
    _require_columns(quarterly, QUARTER_REQUIRED, "분기 예측")
    _require_columns(annual, ANNUAL_REQUIRED, "연간 예측")
    _require_unique(quarterly, QUARTER_KEYS, "분기 예측")
    _require_unique(annual, ANNUAL_KEYS, "연간 예측")
    if metadata.get("status") != "completed":
        raise ResultDataError("완료 상태의 예측 실행만 조회할 수 있습니다.")
    shared = ("run_id", "model_id", "model_variant", "dataset_version", "origin_quarter")
    for column in shared:
        q_values = set(quarterly[column].dropna().astype(str).unique())
        a_values = set(annual[column].dropna().astype(str).unique())
        expected = metadata.get(column)
        if len(q_values) != 1 or q_values != a_values or (expected is not None and q_values != {str(expected)}):
            raise ResultDataError(f"분기·연간 결과가 같은 실행의 {column} 값으로 묶이지 않았습니다.")
    for label, frame in (("분기", quarterly), ("연간", annual)):
        if set(frame["currency"].dropna().astype(str)) != {"USD"}:
            raise ResultDataError(f"{label} 예측 금액은 USD여야 합니다.")
        if set(frame["valuation"].dropna().astype(str)) != {"FOB"}:
            raise ResultDataError(f"{label} 예측 금액은 FOB여야 합니다.")
    quarters = metadata.get("target_quarters") or sorted(quarterly["target_quarter"].astype(str).unique())
    quarters = list(map(str, quarters))
    if len(quarters) != 4 or len(set(quarters)) != 4:
        raise ResultDataError("예측 실행에는 중복 없는 4개 대상 분기가 필요합니다.")
    audit = audit_forecast_grid(quarterly, country_codes, hs_codes, quarters)
    if audit.duplicate_keys or audit.unexpected_keys:
        raise ResultDataError("분기 결과가 국가·품목·분기 전체 조합 형식과 다릅니다.")
    years = set(pd.to_numeric(annual["target_year"], errors="coerce").dropna().astype(int))
    if years != {int(str(quarters[0])[:4])}:
        raise ResultDataError("연간 결과의 대상 연도가 분기 예측과 다릅니다.")
    complete = annual["annual_status"].astype("string").eq("complete")
    incomplete = annual["annual_status"].astype("string").eq("incomplete")
    if (annual.loc[complete, "available_quarters"].ne(4).any()
            or annual.loc[complete, "predicted_annual_export_value_usd"].isna().any()):
        raise ResultDataError("완전 연간 합계에는 네 분기 예측이 모두 있어야 합니다.")
    if annual.loc[incomplete, "predicted_annual_export_value_usd"].notna().any():
        raise ResultDataError("불완전 연간 예측값이 완전한 합계로 표시됐습니다.")
    if (~(complete | incomplete)).any():
        raise ResultDataError("연간 예측 상태는 complete 또는 incomplete여야 합니다.")
    quarter_groups = {}
    group_keys = ("country_code", "hs_code", "origin_quarter", "target_year")
    for key, group in quarterly.groupby(list(group_keys), dropna=False, sort=False):
        quarter_values = dict(zip(group["target_quarter"].astype(str), group["predicted_export_value_usd"]))
        quarter_groups[tuple(str(value) for value in key)] = annual_sum_from_quarters(
            quarter_values.get(quarter, float("nan")) for quarter in quarters
        )
    for _, row in annual.loc[complete].iterrows():
        key = tuple(str(row[column]) for column in group_keys)
        expected_sum = quarter_groups.get(key, float("nan"))
        actual_sum = float(row["predicted_annual_export_value_usd"])
        if not pd.notna(expected_sum) or not abs(actual_sum - expected_sum) <= max(1e-7, abs(expected_sum) * 1e-12):
            raise ResultDataError("완전 연간 예측 합계가 같은 실행의 네 분기 예측과 다릅니다.")
    return audit


def load_forecast_run(
    run_dir: str | Path,
    *,
    country_codes: Iterable[str],
    hs_codes: Iterable[str],
) -> ForecastRun:
    """Load one completed run from disk; this function never invokes a model."""
    folder = Path(run_dir)
    required = ("run.json", "quarterly_forecast.parquet", "annual_forecast.parquet")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"예측 실행 파일이 없습니다: {', '.join(missing)}")
    try:
        metadata = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        quarterly = pd.read_parquet(folder / "quarterly_forecast.parquet")
        annual = pd.read_parquet(folder / "annual_forecast.parquet")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ResultDataError(f"예측 결과 파일을 읽지 못했습니다: {error}") from error
    audit = validate_forecast_results(
        quarterly, annual, metadata,
        country_codes=country_codes, hs_codes=hs_codes,
    )
    return ForecastRun(folder, metadata, quarterly, annual, audit)
