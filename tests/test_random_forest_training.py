from pathlib import Path

import numpy as np
import pandas as pd

from oinq.common.config import load_yaml, quarter_number
from oinq.modeling.loader import load_model
from oinq.modeling.supervised.lightgbm import TARGET, load_inputs, prepare_x, training_rows
from oinq.modeling.supervised.random_forest import feature_sets, fit_pipeline, save_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_shared_time_and_feature_contract():
    train, validation, manifest, config, _, _, _, _ = load_inputs(ROOT)
    pool = pd.concat([train, validation], ignore_index=True)
    early = training_rows(pool, "2021Q1")
    later = training_rows(pool, "2023Q1")
    assert early["_target_index"].max() <= quarter_number("2021Q1")
    assert later["_target_index"].max() <= quarter_number("2023Q1")
    assert not early["split"].eq("validation").any()
    assert later["split"].eq("validation").any()
    assert later[TARGET].notna().all()
    features = feature_sets(manifest, config)
    macro = config["supervised"]["macro"]["columns"]
    excluded = set(macro) | {f"{name}_missing" for name in macro}
    assert features["with_macro"] == manifest["feature_columns"]
    assert features["without_macro"] == [
        name for name in features["with_macro"] if name not in excluded
    ]
    assert not set(features["with_macro"]).intersection(
        manifest["feature_contract"]["forbidden_feature_columns"]
    )


def test_pipeline_keeps_codes_and_reloads_without_fit(tmp_path, monkeypatch):
    rows = pd.DataFrame({
        "country_code": ["004", "004", "008"],
        "hs_code": ["001234", "001234", "001234"],
        "lag": [1.0, np.nan, 3.0],
        "all_missing": [np.nan, np.nan, np.nan],
        TARGET: [5.0, 7.0, 9.0],
    })
    categorical = ["country_code", "hs_code"]
    columns = categorical + ["lag", "all_missing"]
    params = load_yaml(ROOT / "configs/models/random_forest.yaml")["params"]
    pipeline = fit_pipeline(rows, columns, categorical, params | {"n_estimators": 3, "n_jobs": 1})
    preprocessor = pipeline.named_steps["preprocessor"]
    assert preprocessor.named_transformers_["cat"].categories_[1].tolist() == ["001234"]
    assert preprocessor.named_transformers_["num"].statistics_[0] == 2
    sample = prepare_x(rows.assign(country_code="999"), columns, categorical)
    path = tmp_path / "pipeline.joblib"

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Prediction must not fit the pipeline")

    monkeypatch.setattr(type(pipeline), "fit", forbidden_fit)
    save_pipeline(path, pipeline, columns, categorical, "v002", "2025Q4", sample)
    loaded = load_model(path)
    assert loaded["feature_columns"] == columns
    assert loaded["categorical_columns"] == categorical
    np.testing.assert_array_equal(loaded["model"].predict(sample), pipeline.predict(sample))
