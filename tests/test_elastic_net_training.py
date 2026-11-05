"""Fast contract checks; no training against private test data is required."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from oinq.modeling.supervised.elastic_net import (
    MODEL_ID, TARGET, make_pipeline, metric_values, prepare_x, training_rows,
    _save_bundle,
)


def _toy():
    return pd.DataFrame({
        "country_code": ["KOR", "USA", "KOR", "USA"],
        "hs_code": ["330499"] * 4,
        "lag": [np.nan, np.nan, 10.0, 20.0],
        "gdp": [1.0, 2.0, 3.0, 4.0],
        TARGET: [5., 7., 8., np.nan],
        "target_available": [True, True, True, False],
        "origin_quarter": ["2020Q3", "2020Q4", "2021Q1", "2021Q2"],
        "target_quarter": ["2020Q4", "2021Q1", "2021Q2", "2021Q3"],
        "_origin_index": [2020 * 4 + 2, 2020 * 4 + 3, 2021 * 4, 2021 * 4 + 1],
        "_target_index": [2020 * 4 + 3, 2021 * 4, 2021 * 4 + 1, 2021 * 4 + 2],
    })


def test_expanding_origin_only_available_past_labels():
    rows = training_rows(_toy(), "2021Q1")
    assert len(rows) == 2
    assert set(rows.target_quarter) == {"2020Q4", "2021Q1"}


def test_manifest_only_features_and_numeric_conversion():
    df = _toy()
    features = ["country_code", "hs_code", "lag", "gdp"]
    x = prepare_x(df, features, ["country_code", "hs_code"])
    assert x.columns.tolist() == features
    assert TARGET not in x.columns
    assert isinstance(x.country_code.iloc[0], str)
    assert x.lag.isna().sum() == 2


def test_fold_only_fit_keeps_all_missing_feature_and_reload(tmp_path):
    df = _toy().iloc[:3]
    features = ["country_code", "hs_code", "lag", "gdp"]
    cat = ["country_code", "hs_code"]
    x_train = prepare_x(df.iloc[:2], features, cat)
    x_eval = prepare_x(df.iloc[2:], features, cat)
    params = {"alpha": 1., "l1_ratio": 0.5, "max_iter": 1000, "tol": 1e-4}
    model = make_pipeline(features, cat, params)
    model.fit(x_train, df.iloc[:2][TARGET])
    assert list(model.named_steps) == ["preprocessor", "model"]
    assert "cat" in model.named_steps["preprocessor"].named_transformers_
    imputer = model.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 0.0  # lag entirely missing in the TRAIN fold
    assert model.named_steps["preprocessor"].transform(x_eval).shape[1] == model.named_steps["preprocessor"].transform(x_train).shape[1]
    path = tmp_path / "with_macro" / "pipeline.joblib"
    assert _save_bundle(path, model, features, cat, "2025Q4", x_eval)
    saved = joblib.load(path)
    assert saved["dataset_version"] == "v002"
    assert saved["feature_columns"] == features
    np.testing.assert_array_equal(model.predict(x_eval), saved["pipeline"].predict(x_eval))


def test_metric_wape_and_nonnegative_scoring_policy():
    raw = np.array([-3., 8.])
    scored = np.maximum(raw, 0.)
    values = metric_values([2., 10.], scored)
    assert values["n"] == 2
    assert values["wape"] == pytest.approx(100 * (2 + 2) / 12)
    assert int((raw < 0).sum()) == 1


def test_team_manifest_macro_ablation_contract():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/shared/v002/manifest.json"
    if not path.exists():
        pytest.skip("Shared v002 data not installed")
    m = json.loads(path.read_text(encoding="utf-8"))
    config = root / "configs/features.yaml"
    if not config.exists():
        pytest.skip("Shared feature config not installed")
    from oinq.common.config import load_yaml
    macros = load_yaml(config)["supervised"]["macro"]["columns"]
    removal = [x for x in m["feature_columns"] if x in macros or x in {f"{c}_missing" for c in macros}]
    assert len(removal) == 12
    assert MODEL_ID == "elastic_net"
