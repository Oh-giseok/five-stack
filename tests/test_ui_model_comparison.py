"""Data-only UI comparison: preserve missing scores and pair exact model versions."""
import json

import numpy as np
import pandas as pd
import pytest

from oinq.evaluation.comparison import COLUMNS, sha256
from oinq.ui.data import ResultDataError
from oinq.ui.model_comparison import METRICS, load_model_comparison, comparison_source_signature
from oinq.ui.model_comparison import SHARED_SUMMARY


@pytest.fixture
def connection(tmp_path):
    countries = [f"C{i:02d}" for i in range(20)]
    products = [f"3304{i:02d}" for i in range(8)]
    manifest = tmp_path / "data/shared/v002/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"country_codes": countries, "hs_codes": products}))
    records = []
    for name, r2, wape in [("a", -.2, 120.), ("b", .9, 20.), ("c", None, None)]:
        path = tmp_path / f"artifacts/shared_models/{name}/with_macro/pipeline.joblib"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not deserialised by UI result reader")
        records.append({"model_id": name, "model_name": name, "experiment": "basic", "variant": "with_macro",
                        "status": "평가 완료" if r2 is not None else "운영용·비교 제외", "detail": "",
                        "model_path": path.relative_to(tmp_path).as_posix(), "model_sha256": sha256(path),
                        "trained_through_target_quarter": "2025Q4" if r2 is not None else "2026Q3",
                        "validation": {"n": 9164, "wape": wape} if wape is not None else None,
                        "test": {"n": 447, "wape": wape + 1, "r2": r2, "mae": 3e6, "rmse": 15e6} if r2 is not None else None})
    report = {"metadata": {"dataset_version": "v002", "validation_period": ["2022Q1", "2025Q4"], "primary_horizons": [2, 3, 4, 5],
                           "test_origin": "2025Q4", "test_period": ["2026Q1", "2026Q3"], "test_horizons": [1, 2, 3],
                           "internal_currency": "USD", "valuation": "FOB", "source_checksums": {manifest.relative_to(tmp_path).as_posix(): sha256(manifest)}},
              "models": records}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    config = {"schema_version": 1, "comparison_report": "report.json", "comparison_report_sha256": sha256(report_path),
              "country_codes": countries, "hs_codes": products,
              "forecast_runs": [{**{key: record[key] for key in ["model_id", "experiment", "variant"]}, "run_dir": None} for record in records]}
    config_path = tmp_path / "connection.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return tmp_path, config_path, report_path, config


@pytest.mark.parametrize("metric", list(METRICS))
@pytest.mark.parametrize("ascending", [True, False])
def test_six_metrics_sort_both_directions_and_keep_all_models(connection, metric, ascending):
    root, config_path, *_ = connection
    data = load_model_comparison(root, config_path)
    selected = data.select_metric(metric, ascending=ascending)
    assert len(selected) == 3
    actual = selected.comparison_value.dropna().tolist()
    assert actual == sorted(actual, reverse=not ascending)
    assert selected.comparison_value.isna().tolist() == sorted(selected.comparison_value.isna().tolist())
    table = data.performance_table(metric, ascending=ascending)
    assert table.columns.tolist() == COLUMNS
    assert table["모델명"].tolist() == selected.model_name.tolist()


def test_no_forecasts_are_not_zero_and_default_directions_are_preserved(connection):
    root, path, *_ = connection
    data = load_model_comparison(root, path)
    assert data.models.forecast_2027_usd.isna().all()
    assert data.select_metric("검증 WAPE").model_id.tolist() == ["b", "a", "c"]
    assert data.select_metric("R²").model_id.tolist() == ["b", "a", "c"]
    assert data.select_metric("R²").comparison_value.iloc[1] == -.2
    assert data.select_metric("검증 WAPE").comparison_value.iloc[1] == 120
    assert data.select_metric("MAE").comparison_value.iloc[0] == 3
    assert data.select_metric("RMSE").comparison_value.iloc[0] == 15
    assert data.models.test_mae_usd.iloc[0] == 3e6


def test_changed_report_is_rejected(connection):
    root, path, report, _ = connection
    report.write_text("{}")
    with pytest.raises(ResultDataError, match="체크섬"):
        load_model_comparison(root, path)


def test_duplicate_or_missing_forecast_model_links_rejected(connection):
    root, path, _, config = connection
    config["forecast_runs"].append(config["forecast_runs"][0])
    path.write_text(json.dumps(config))
    with pytest.raises(ResultDataError, match="중복되거나 누락"):
        load_model_comparison(root, path)


def test_operational_model_cannot_borrow_test_score(connection):
    root, path, report_path, config = connection
    report = json.loads(report_path.read_text("utf-8"))
    report["models"][2]["test"] = report["models"][0]["test"]
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    config["comparison_report_sha256"] = sha256(report_path)
    path.write_text(json.dumps(config))
    with pytest.raises(ResultDataError, match="비교 제외"):
        load_model_comparison(root, path)


@pytest.fixture
def linked_forecast(connection):
    root, config_path, report_path, config = connection
    report = json.loads(report_path.read_text("utf-8"))
    record = report["models"][0]
    folder = root / ".private/forecasts/fixture"
    folder.mkdir(parents=True)
    shared = {"origin_quarter": "2026Q3", "target_year": 2027, "model_id": "a", "model_variant": "with_macro",
              "run_id": "fixture", "dataset_version": "v002", "currency": "USD", "valuation": "FOB"}
    quarterly, annual = [], []
    for country in config["country_codes"]:
        for hs in config["hs_codes"]:
            common = {**shared, "country_code": country, "country_name": country, "hs_code": hs, "product_name": hs}
            for q in range(1, 5):
                quarterly.append({**common, "target_quarter": f"2027Q{q}", "horizon_quarters": q + 1,
                                  "predicted_export_value_usd": 10., "prediction_status": "predicted", "input_missing_feature_count": 0})
            annual.append({**common, "available_quarters": 4, "predicted_partial_sum_usd": 40.,
                           "predicted_annual_export_value_usd": 40., "annual_status": "complete"})
    pd.DataFrame(quarterly).to_parquet(folder / "quarterly_forecast.parquet", index=False)
    pd.DataFrame(annual).to_parquet(folder / "annual_forecast.parquet", index=False)
    features = root / "data/shared/v002/forecast/origin_features_2026Q3.parquet"
    features.parent.mkdir(parents=True)
    features.write_bytes(b"input snapshot fingerprint only; never used for prediction")
    manifest = root / "data/shared/v002/manifest.json"
    meta = {**shared, "status": "completed", "model_sha256": record["model_sha256"], "horizons": [2, 3, 4, 5],
            "target_quarters": [f"2027Q{q}" for q in range(1, 5)], "refit_performed": False,
            "source_paths_sha256": {manifest.relative_to(root).as_posix(): sha256(manifest), features.relative_to(root).as_posix(): sha256(features)},
            "outputs": {name: sha256(folder / name) for name in ["quarterly_forecast.parquet", "annual_forecast.parquet"]}}
    metadata_path = folder / "run.json"
    metadata_path.write_text(json.dumps(meta), encoding="utf-8")
    config["forecast_runs"][0].update(run_dir=folder.relative_to(root).as_posix(), run_metadata_sha256=sha256(metadata_path))
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return root, config_path, folder, config, meta


def test_one_matching_run_is_summed_once_without_deserializing_models(linked_forecast, monkeypatch):
    root, path, *_ = linked_forecast
    def prohibited(*args, **kwargs):
        raise AssertionError("The UI data reader must not load or fit models")
    monkeypatch.setattr("joblib.load", prohibited)
    data = load_model_comparison(root, path)
    assert len(data.forecasts) == 1
    assert data.models.loc[data.models.model_id.eq("a"), "forecast_2027_usd"].iloc[0] == 6400.
    assert len(data.forecasts["fixture"].quarterly) == 640
    assert len(data.forecasts["fixture"].annual) == 160
    assert data.select_metric("예상 수출액").comparison_value.iloc[0] == .0064


def test_model_replacement_changes_ui_cache_signature(linked_forecast):
    root, _, _, _, _ = linked_forecast
    before = comparison_source_signature(root, "connection.json")
    model = root / "artifacts/shared_models/a/with_macro/pipeline.joblib"
    model.write_bytes(model.read_bytes() + b"new revision")
    assert comparison_source_signature(root, "connection.json") != before


@pytest.mark.parametrize("field,value", [
    ("model_sha256", "another-checkpoint"), ("origin_quarter", "2026Q4"),
    ("model_variant", "without_macro"), ("dataset_version", "v003"),
    ("refit_performed", True), ("horizons", [1, 2, 3, 4]),
])
def test_wrong_model_or_forecast_conditions_are_rejected(linked_forecast, field, value):
    root, path, folder, config, meta = linked_forecast
    meta[field] = value
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    config["forecast_runs"][0]["run_metadata_sha256"] = sha256(folder / "run.json")
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ResultDataError):
        load_model_comparison(root, path)


def test_modified_forecast_file_is_rejected(linked_forecast):
    root, path, folder, *_ = linked_forecast
    frame = pd.read_parquet(folder / "quarterly_forecast.parquet")
    frame["country_name"] = "modified"
    frame.to_parquet(folder / "quarterly_forecast.parquet", index=False)
    with pytest.raises(ResultDataError, match="체크섬"):
        load_model_comparison(root, path)


def test_missing_quarter_is_not_treated_as_complete_year(linked_forecast):
    root, path, folder, config, meta = linked_forecast
    file = folder / "quarterly_forecast.parquet"
    pd.read_parquet(file).iloc[1:].to_parquet(file, index=False)
    meta["outputs"][file.name] = sha256(file)
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    config["forecast_runs"][0]["run_metadata_sha256"] = sha256(folder / "run.json")
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ResultDataError):
        load_model_comparison(root, path)


@pytest.fixture
def legacy_forecast(linked_forecast):
    root, path, folder, config, meta = linked_forecast
    relative = "data/shared/v002/forecast/origin_features_2026Q3.parquet"
    features = pd.DataFrame([{"country_code": c, "hs_code": h, "origin_quarter": "2026Q3", "export_value_usd_lag1": 12.}
                             for c in config["country_codes"] for h in config["hs_codes"]])
    features.to_parquet(root / relative, index=False)
    # Order and dtype metadata may differ; the actual feature values must not.
    features.iloc[::-1].to_parquet(folder / "origin_features.parquet", index=False)
    meta["source_paths_sha256"].pop(relative)
    meta["source_paths_sha256"]["data/processed/supervised/v002/origin_feature_candidates.parquet"] = "legacy-full-history-fingerprint"
    meta["outputs"]["origin_features.parquet"] = sha256(folder / "origin_features.parquet")
    return root, path, folder, config, meta


def _save_metadata(path, folder, config, meta):
    (folder / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    config["forecast_runs"][0]["run_metadata_sha256"] = sha256(folder / "run.json")
    path.write_text(json.dumps(config), encoding="utf-8")


def test_legacy_source_accepted_only_when_saved_input_values_match(legacy_forecast):
    root, path, folder, config, meta = legacy_forecast
    _save_metadata(path, folder, config, meta)
    data = load_model_comparison(root, path)
    model = data.models.loc[data.models.model_id.eq("a")].iloc[0]
    assert model.forecast_input_verification == "saved_origin_snapshot"
    assert model.forecast_2027_usd == 6400.


@pytest.mark.parametrize("change", ["value", "missing_row", "duplicate", "missing_column", "checksum"])
def test_legacy_source_rejects_changed_or_incomplete_saved_inputs(legacy_forecast, change):
    root, path, folder, config, meta = legacy_forecast
    snapshot = folder / "origin_features.parquet"
    frame = pd.read_parquet(snapshot)
    if change == "value":
        frame.loc[0, "export_value_usd_lag1"] = 13.
    elif change == "missing_row":
        frame = frame.iloc[:-1]
    elif change == "duplicate":
        frame.iloc[0] = frame.iloc[1]
    elif change == "missing_column":
        frame = frame.drop(columns="export_value_usd_lag1")
    else:
        frame.loc[0, "export_value_usd_lag1"] = 13.
    frame.to_parquet(snapshot, index=False)
    if change != "checksum":
        meta["outputs"][snapshot.name] = sha256(snapshot)
    _save_metadata(path, folder, config, meta)
    with pytest.raises(ResultDataError):
        load_model_comparison(root, path)


@pytest.fixture
def shared_summary(connection):
    root, _, report_path, _ = connection
    summary = json.loads(report_path.read_text("utf-8"))
    summary["schema_version"] = 1
    summary["metadata"].update(forecast_origin="2026Q3", forecast_period=["2027Q1", "2027Q4"],
                               forecast_horizons=[2, 3, 4, 5], forecast_rows_per_configuration=640)
    manifest = json.loads((root / "data/shared/v002/manifest.json").read_text())
    summary.update(country_codes=manifest["country_codes"], hs_codes=manifest["hs_codes"])
    inputs = root / "data/shared/v002/forecast/origin_features_2026Q3.parquet"
    inputs.parent.mkdir(parents=True)
    inputs.write_bytes(b"released prediction inputs")
    summary["shared_inputs_sha256"] = {name: sha256(root / name) for name in ["data/shared/v002/manifest.json", inputs.relative_to(root).as_posix()]}
    for index, record in enumerate(summary["models"]):
        record["forecast_2027_usd"] = float((index + 1) * 1e6)
    path = root / SHARED_SUMMARY
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    return root, path, summary


def test_shared_runtime_works_without_private_reports_or_model_loading(shared_summary, monkeypatch):
    root, _, _ = shared_summary
    assert not (root / ".private").exists()
    def prohibited(*args, **kwargs):
        raise AssertionError("Shared UI runtime must not load models")
    monkeypatch.setattr("joblib.load", prohibited)
    data = load_model_comparison(root)
    assert len(data.models) == 3 and data.metadata["source_kind"] == "shared_summary"
    assert data.select_metric("예상 수출액").comparison_value.tolist() == [3., 2., 1.]
    assert data.models.test_wape.notna().sum() == 2
    assert comparison_source_signature(root)


def test_shared_runtime_does_not_depend_on_local_private_connection(shared_summary):
    root, _, _ = shared_summary
    private = root / ".private/model_comparison/connection.json"
    private.parent.mkdir(parents=True)
    private.write_text("not a valid private configuration")
    assert len(load_model_comparison(root).models) == 3


@pytest.mark.parametrize("change", ["model", "inputs", "duplicate", "amount", "evaluation", "operational_score"])
def test_shared_runtime_rejects_inconsistent_data(shared_summary, change):
    root, path, summary = shared_summary
    if change == "model":
        (root / summary["models"][0]["model_path"]).write_bytes(b"different model")
    elif change == "inputs":
        (root / "data/shared/v002/forecast/origin_features_2026Q3.parquet").write_bytes(b"different inputs")
    elif change == "duplicate":
        summary["models"].append(summary["models"][0])
    elif change == "amount":
        summary["models"][0]["forecast_2027_usd"] = -1.
    elif change == "evaluation":
        summary["models"][0]["validation"]["n"] = 9000
    elif change == "operational_score":
        summary["models"][2]["test"] = summary["models"][0]["test"]
    path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ResultDataError):
        load_model_comparison(root)
