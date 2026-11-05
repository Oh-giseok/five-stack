"""Only offer persisted results for the exact saved regressor and variant."""

import hashlib
import json
from pathlib import Path

import pytest

from oinq.ui.forecasting import saved_forecast_models, matching_forecast_runs, create_forecast_run


def model_fixture(root):
    config = root / "configs/engine.yaml"
    config.parent.mkdir()
    config.write_text("schema_version: 1\npaths:\n  models_dir: artifacts/shared_models\n", encoding="utf-8")
    for name in ["random_forest/with_macro/pipeline.joblib", "random_forest/without_macro/pipeline.joblib",
                 "elastic_net/with_macro/pipeline.joblib", "catboost/with_macro/model.cbm", "kmeans/pipeline.joblib"]:
        path = root / "artifacts/shared_models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"saved-model-fixture")


def write_run(root, run_id, **overrides):
    folder = root / ".private/forecasts" / run_id
    folder.mkdir(parents=True)
    metadata = {"status": "completed", "model_id": "random_forest", "model_variant": "with_macro",
                "model_sha256": hashlib.sha256(b"saved-model-fixture").hexdigest(), "dataset_version": "v002",
                "origin_quarter": "2026Q3", "target_quarters": [f"2027Q{i}" for i in range(1, 5)],
                "generated_at_utc": "2026-11-05T00:00:00+00:00"}
    metadata.update(overrides)
    (folder / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in ["quarterly_forecast.parquet", "annual_forecast.parquet"]:
        (folder / name).touch()


def test_discovery_includes_new_regressors_and_cbm_but_excludes_flat_kmeans(tmp_path):
    model_fixture(tmp_path)
    models = saved_forecast_models(tmp_path)
    assert set(models) == {"random_forest/with_macro", "random_forest/without_macro",
                           "elastic_net/with_macro", "catboost/with_macro"}
    assert not (tmp_path / ".private").exists()
    with pytest.raises(ValueError, match="선택한 저장 모델"):
        create_forecast_run(tmp_path, "kmeans/with_macro")
    assert not (tmp_path / ".private").exists()


def test_run_filter_rejects_wrong_variant_model_revision_period_and_incomplete_runs(tmp_path):
    model_fixture(tmp_path)
    write_run(tmp_path, "valid")
    write_run(tmp_path, "valid_new", generated_at_utc="2026-11-05T01:00:00+00:00")
    for name, changes in {
        "wrong_model": {"model_id": "elastic_net"}, "wrong_variant": {"model_variant": "without_macro"},
        "old_model": {"model_sha256": "outdated"}, "old_origin": {"origin_quarter": "2025Q4"},
        "incomplete": {"status": "failed"}, "old_dataset": {"dataset_version": "v001"},
        "wrong_year": {"target_quarters": [f"2026Q{i}" for i in range(1, 5)]},
    }.items():
        write_run(tmp_path, name, **changes)
    model = saved_forecast_models(tmp_path)["random_forest/with_macro"]
    assert matching_forecast_runs(tmp_path, "random_forest/with_macro", model) == ["valid_new", "valid"]
    model.write_bytes(b"newly-saved-model")
    assert matching_forecast_runs(tmp_path, "random_forest/with_macro", model) == []
