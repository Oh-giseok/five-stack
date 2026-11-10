"""Read evaluated models and matching saved forecasts for the comparison UI.

No widgets, fitting, inference, model loading, or writes occur in this module.
The page can select one metric for every model, rather than selecting a model.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from oinq.evaluation.comparison import performance_table, sha256
from oinq.ui.data import ForecastRun, ResultDataError, load_forecast_run

METRICS = {
    "예상 수출액": ("forecast_2027_usd", "백만 USD", False, 1e6),
    "검증 WAPE": ("validation_wape", "%", True, 1.0),
    "테스트 WAPE": ("test_wape", "%", True, 1.0),
    "R²": ("test_r2", "R²", False, 1.0),
    "MAE": ("test_mae_usd", "백만 USD", True, 1e6),
    "RMSE": ("test_rmse_usd", "백만 USD", True, 1e6),
}
IDENTITY = ["model_id", "experiment", "variant"]
SHARED_SUMMARY = "data/shared/model_comparison/summary.json"


@dataclass(frozen=True)
class ModelComparisonData:
    records: list[dict]
    models: pd.DataFrame
    forecasts: dict[str, ForecastRun]
    metadata: dict

    def select_metric(self, metric: str, *, ascending: bool | None = None) -> pd.DataFrame:
        """Return all configurations, missing values last in either sort direction."""
        if metric not in METRICS:
            raise ValueError(f"지원하지 않는 세부지표입니다: {metric}")
        column, unit, default_ascending, divisor = METRICS[metric]
        if ascending is None:
            ascending = default_ascending
        result = self.models.copy()
        result["comparison_value"] = result[column] / divisor
        result["comparison_unit"] = unit
        result["comparison_metric"] = metric
        result = result.sort_values(["comparison_value", *IDENTITY],
                                    ascending=[ascending, True, True, True], na_position="last", kind="stable")
        return result.reset_index(drop=True)

    def performance_table(self, metric: str = "검증 WAPE", *, ascending: bool | None = None) -> pd.DataFrame:
        """Keep the established ten columns, using the graph's exact row order."""
        ordered = self.select_metric(metric, ascending=ascending)
        lookup = {tuple(record[key] for key in IDENTITY): record for record in self.records}
        return performance_table([lookup[tuple(row[key] for key in IDENTITY)] for _, row in ordered.iterrows()])


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def comparison_source_signature(root: str | Path, connection_path: str = SHARED_SUMMARY) -> tuple:
    """Invalidate UI caching when connected inputs, models or results change."""
    root = Path(root).resolve()
    if str(connection_path).replace("\\", "/") == SHARED_SUMMARY:
        summary = _json(root / connection_path)
        paths = {connection_path, *summary["shared_inputs_sha256"]}
        paths.update(record["model_path"] for record in summary["models"] if record.get("model_path"))
        return _path_signature(root, paths)
    paths = {connection_path, "data/shared/v002/manifest.json",
             "data/shared/v002/forecast/origin_features_2026Q3.parquet"}
    connection = _json(root / connection_path)
    paths.add(connection["comparison_report"])
    report = _json(root / connection["comparison_report"])
    paths.update(record["model_path"] for record in report["models"] if record.get("model_path"))
    for link in connection["forecast_runs"]:
        if link.get("run_dir") is None:
            continue
        folder = Path(link["run_dir"])
        for name in ["run.json", "quarterly_forecast.parquet", "annual_forecast.parquet"]:
            paths.add((folder / name).as_posix())
        info = _json(root / folder / "run.json")
        if "origin_features.parquet" in info.get("outputs", {}):
            paths.add((folder / "origin_features.parquet").as_posix())
    return _path_signature(root, paths)


def _path_signature(root: Path, paths) -> tuple:
    signature = []
    for relative in sorted(paths):
        file = (root / relative).resolve()
        if not file.is_relative_to(root):
            raise ResultDataError("비교 자료 경로가 프로젝트 밖입니다.")
        stat = file.stat()
        signature.append((relative, stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def _checked_path(root: Path, path: str, digest: str) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ResultDataError("비교 자료 경로가 프로젝트 밖입니다.")
    if not resolved.is_file() or not digest or sha256(resolved) != digest:
        raise ResultDataError(f"비교 자료가 없거나 체크섬이 다릅니다: {path}")
    return resolved


def _verify_forecast_inputs(root: Path, folder: Path, metadata: dict) -> str:
    """Accept legacy full-history inputs only with an identical saved origin table."""
    sources = metadata.get("source_paths_sha256", {})

    def recorded(relative):
        return [digest for path, digest in sources.items()
                if path.replace("\\", "/").endswith("/" + relative) or path.replace("\\", "/") == relative]

    manifest_name = "data/shared/v002/manifest.json"
    if recorded(manifest_name) != [sha256(root / manifest_name)]:
        raise ResultDataError("예측에 사용한 공통 데이터 명세가 현재 자료와 다릅니다.")
    features_name = "data/shared/v002/forecast/origin_features_2026Q3.parquet"
    declared = recorded(features_name)
    if declared:
        if declared != [sha256(root / features_name)]:
            raise ResultDataError("예측에 사용한 기준분기 입력 체크섬이 현재 자료와 다릅니다.")
        return "shared_input_checksum"

    # Earlier runs used origin_feature_candidates.parquet and selected 2026Q3.
    # Verify the exact input saved by that run instead of trusting its old path.
    digest = metadata.get("outputs", {}).get("origin_features.parquet")
    snapshot = _checked_path(root, (folder / "origin_features.parquet").relative_to(root).as_posix(), digest)
    previous, current = pd.read_parquet(snapshot), pd.read_parquet(root / features_name)
    keys = ["country_code", "hs_code", "origin_quarter"]
    for frame in [previous, current]:
        if (not set(keys).issubset(frame.columns) or len(frame) != 160 or frame[keys].isna().any().any()
                or frame.duplicated(keys).any() or not frame.origin_quarter.eq("2026Q3").all()):
            raise ResultDataError("저장 입력의 국가·품목·기준분기 160행을 확인할 수 없습니다.")
    try:
        pd.testing.assert_frame_equal(
            previous.sort_values(keys).reset_index(drop=True).sort_index(axis=1),
            current.sort_values(keys).reset_index(drop=True).sort_index(axis=1),
            check_dtype=False, check_exact=True,
        )
    except AssertionError as error:
        raise ResultDataError("예전 실행의 저장 입력과 현재 기준분기 입력 값이 다릅니다.") from error
    return "saved_origin_snapshot"


def load_model_comparison(root: str | Path, connection_path: str | Path = SHARED_SUMMARY) -> ModelComparisonData:
    """Read one explicit result connection; never discover and sum repeated runs."""
    root = Path(root).resolve()
    if str(connection_path).replace("\\", "/") == SHARED_SUMMARY:
        return _load_shared_summary(root)
    connection = _json(root / connection_path)
    if connection.get("schema_version") != 1:
        raise ResultDataError("지원하지 않는 비교 연결 형식입니다.")
    source = _checked_path(root, connection["comparison_report"], connection["comparison_report_sha256"])
    report = _json(source)
    meta = report["metadata"]
    if (meta.get("dataset_version") != "v002" or meta.get("validation_period") != ["2022Q1", "2025Q4"]
            or meta.get("test_origin") != "2025Q4" or meta.get("test_period") != ["2026Q1", "2026Q3"]
            or meta.get("primary_horizons") != [2, 3, 4, 5] or meta.get("test_horizons") != [1, 2, 3]
            or meta.get("internal_currency") != "USD" or meta.get("valuation") != "FOB"):
        raise ResultDataError("확정된 공통 평가조건과 비교 결과가 다릅니다.")
    records = report["models"]
    identities = [tuple(record[key] for key in IDENTITY) for record in records]
    if len(set(identities)) != len(identities):
        raise ResultDataError("비교 모델 구성이 중복됐습니다.")
    links = connection["forecast_runs"]
    link_keys = [tuple(link[key] for key in IDENTITY) for link in links]
    if len(set(link_keys)) != len(link_keys) or set(link_keys) != set(identities):
        raise ResultDataError("평가·예측 연결의 모델 구성이 중복되거나 누락됐습니다.")
    links = dict(zip(link_keys, links))
    countries, products = tuple(connection["country_codes"]), tuple(connection["hs_codes"])
    if len(countries) != 20 or len(products) != 8 or len(set(countries)) != 20 or len(set(products)) != 8:
        raise ResultDataError("비교 범위는 국가 20개·품목 8개여야 합니다.")
    manifest_name = "data/shared/v002/manifest.json"
    manifest_path = _checked_path(root, manifest_name, meta["source_checksums"][manifest_name])
    manifest = _json(manifest_path)
    if set(countries) != set(manifest["country_codes"]) or set(products) != set(manifest["hs_codes"]):
        raise ResultDataError("비교 국가·품목이 공통 데이터 명세와 다릅니다.")
    models, forecasts = [], {}
    for identity, record in zip(identities, records):
        validation, test = record["validation"], record["test"]
        if record["status"] == "평가 완료":
            if not validation or not test or validation["n"] != 9164 or test["n"] != 447:
                raise ResultDataError("비교 모델의 평가 건수가 다릅니다.")
            if record.get("trained_through_target_quarter") != "2025Q4":
                raise ResultDataError("테스트용 모델의 학습 종료 분기가 다릅니다.")
        elif validation is not None or test is not None:
            raise ResultDataError("비교 제외 모델에 평가 점수를 붙일 수 없습니다.")
        model_path = _checked_path(root, record["model_path"], record["model_sha256"]) if record.get("model_path") else None
        row = {key: record[key] for key in [*IDENTITY, "model_name", "status", "detail"]}
        row.update(validation_wape=validation["wape"] if validation else np.nan,
                   test_wape=test["wape"] if test else np.nan, test_r2=test["r2"] if test else np.nan,
                   test_mae_usd=test["mae"] if test else np.nan, test_rmse_usd=test["rmse"] if test else np.nan,
                   forecast_2027_usd=np.nan, forecast_status="예측 자료 없음", forecast_run_id=None,
                   forecast_input_verification=None,
                   model_sha256=record.get("model_sha256"), model_path=record.get("model_path"),
                   trained_through_target_quarter=record.get("trained_through_target_quarter"))
        link = links[identity]
        if link.get("run_dir") is not None:
            if model_path is None:
                raise ResultDataError("저장 모델이 없는 구성에 예측을 연결할 수 없습니다.")
            folder = (root / link["run_dir"]).resolve()
            if not folder.is_relative_to(root):
                raise ResultDataError("예측 경로가 프로젝트 밖입니다.")
            _checked_path(root, (folder / "run.json").relative_to(root).as_posix(), link["run_metadata_sha256"])
            forecast = load_forecast_run(folder, country_codes=countries, hs_codes=products)
            info = forecast.metadata
            if (info.get("model_sha256") != record["model_sha256"] or info.get("model_variant") != record["variant"]
                    or info.get("model_id") != model_path.parent.parent.name or info.get("dataset_version") != "v002"
                    or info.get("origin_quarter") != "2026Q3" or info.get("target_quarters") != [f"2027Q{i}" for i in range(1, 5)]
                    or info.get("horizons") != [2, 3, 4, 5] or info.get("refit_performed") is not False):
                raise ResultDataError("평가한 모델·기간과 연결된 예측 모델·기간이 다릅니다.")
            row["forecast_input_verification"] = _verify_forecast_inputs(root, folder, info)
            for name in ["quarterly_forecast.parquet", "annual_forecast.parquet"]:
                _checked_path(root, (folder / name).relative_to(root).as_posix(), info["outputs"][name])
            if forecast.grid_audit.missing_keys or len(forecast.quarterly) != 640 or len(forecast.annual) != 160:
                raise ResultDataError("예측 조합 640행 또는 연간 160행이 부족합니다.")
            annual_keys = ["country_code", "hs_code", "origin_quarter", "target_year"]
            if forecast.annual.duplicated(annual_keys).any() or not forecast.annual.annual_status.eq("complete").all():
                raise ResultDataError("연간 예측 조합이 중복되거나 불완전합니다.")
            values = forecast.quarterly.predicted_export_value_usd.to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ResultDataError("예상 수출액에 결측·무한대·음수가 있습니다.")
            total = float(values.sum())
            if not np.isclose(total, forecast.annual.predicted_annual_export_value_usd.sum(), rtol=1e-12, atol=1e-6):
                raise ResultDataError("분기와 연간 예측 합계가 다릅니다.")
            row.update(forecast_2027_usd=total, forecast_status="예측 완료", forecast_run_id=info["run_id"])
            forecasts[info["run_id"]] = forecast
        models.append(row)
    model_frame = pd.DataFrame(models)
    for column in [value[0] for value in METRICS.values()]:
        values = pd.to_numeric(model_frame[column], errors="raise")
        if np.isinf(values).any():
            raise ResultDataError("비교 지표에 무한대가 있습니다.")
        model_frame[column] = values
    return ModelComparisonData(records, model_frame, forecasts,
                               {**meta, "forecast_scope": "20 countries x 8 products x 4 quarters",
                                "forecast_origin": "2026Q3", "forecast_period": ["2027Q1", "2027Q4"],
                                "comparison_report": connection["comparison_report"], "connection_path": str(connection_path)})


def _load_shared_summary(root: Path) -> ModelComparisonData:
    """Load the one shared runtime summary; no private report or run is needed."""
    summary = _json(root / SHARED_SUMMARY)
    meta = summary["metadata"]
    if (summary.get("schema_version") != 1 or meta.get("dataset_version") != "v002"
            or meta.get("validation_period") != ["2022Q1", "2025Q4"] or meta.get("primary_horizons") != [2, 3, 4, 5]
            or meta.get("test_origin") != "2025Q4" or meta.get("test_period") != ["2026Q1", "2026Q3"]
            or meta.get("test_horizons") != [1, 2, 3] or meta.get("internal_currency") != "USD" or meta.get("valuation") != "FOB"
            or meta.get("forecast_origin") != "2026Q3" or meta.get("forecast_period") != ["2027Q1", "2027Q4"]
            or meta.get("forecast_horizons") != [2, 3, 4, 5] or meta.get("forecast_rows_per_configuration") != 640):
        raise ResultDataError("공유 비교 자료의 평가·예측 조건이 확정 사양과 다릅니다.")
    required_inputs = {"data/shared/v002/manifest.json", "data/shared/v002/forecast/origin_features_2026Q3.parquet"}
    if set(summary["shared_inputs_sha256"]) != required_inputs:
        raise ResultDataError("공유 비교 자료의 입력 명세가 다릅니다.")
    for path, digest in summary["shared_inputs_sha256"].items():
        _checked_path(root, path, digest)
    manifest = _json(root / "data/shared/v002/manifest.json")
    if (len(summary["country_codes"]) != 20 or len(summary["hs_codes"]) != 8
            or set(summary["country_codes"]) != set(manifest["country_codes"])
            or set(summary["hs_codes"]) != set(manifest["hs_codes"])):
        raise ResultDataError("공유 비교 자료의 국가·품목 범위가 다릅니다.")
    records, rows, seen = summary["models"], [], set()
    for record in records:
        identity = tuple(record[key] for key in IDENTITY)
        if identity in seen or record["variant"] not in {"with_macro", "without_macro"}:
            raise ResultDataError("공유 비교 모델 구성이 중복되거나 잘못됐습니다.")
        seen.add(identity)
        _checked_path(root, record["model_path"], record["model_sha256"])
        validation, test = record["validation"], record["test"]
        if record["status"] == "평가 완료":
            if (not validation or not test or validation["n"] != 9164 or test["n"] != 447
                    or record["trained_through_target_quarter"] != "2025Q4"):
                raise ResultDataError("공유 비교 모델의 평가 건수·학습 종료 분기가 다릅니다.")
        elif record["status"] == "운영용·비교 제외":
            if validation is not None or test is not None:
                raise ResultDataError("운영용 모델에 테스트 점수를 붙일 수 없습니다.")
        else:
            raise ResultDataError("지원하지 않는 공유 비교 평가 상태입니다.")
        for score in [validation, test]:
            if score is not None:
                for name in ["wape", "mae", "rmse", "r2"]:
                    value = score.get(name)
                    if value is not None and (not np.isfinite(value) or (name != "r2" and value < 0)):
                        raise ResultDataError("공유 비교 지표가 비정상 값입니다.")
        amount = record["forecast_2027_usd"]
        if amount is None or not np.isfinite(amount) or amount < 0:
            raise ResultDataError("공유 예상 수출액이 결측·비정상 값입니다.")
        row = {key: record[key] for key in [*IDENTITY, "model_name", "status", "detail", "model_path", "model_sha256", "trained_through_target_quarter"]}
        row.update(validation_wape=validation["wape"] if validation else np.nan,
                   test_wape=test["wape"] if test else np.nan, test_r2=test["r2"] if test else np.nan,
                   test_mae_usd=test["mae"] if test else np.nan, test_rmse_usd=test["rmse"] if test else np.nan,
                   forecast_2027_usd=float(amount), forecast_status="예측 완료", forecast_run_id=None,
                   forecast_input_verification="shared_summary")
        rows.append(row)
    return ModelComparisonData(records, pd.DataFrame(rows), {},
                               {**meta, "source_kind": "shared_summary", "forecast_detail_available": False,
                                "summary_path": SHARED_SUMMARY, "summary_sha256": sha256(root / SHARED_SUMMARY)})
