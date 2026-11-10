from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oinq.common.config import load_yaml
from oinq.modeling.loader import load_model
from oinq.modeling.supervised.lightgbm import TARGET, prepare_x, training_rows
from oinq.modeling.supervised.linear_regression import fit_pipeline, select_ridge_alpha
from oinq.modeling.supervised.random_forest import save_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_ridge_selection_uses_h2_5_instead_of_better_h1_5_score():
    records = []
    for variant in ["with_macro", "without_macro"]:
        for alpha, full_wape, primary_wape in [(1., 10., 40.), (10., 20., 30.)]:
            for scope, wape in [("h1_5", full_wape), ("h2_5", primary_wape)]:
                records.append(dict(model_id="ridge", variant=variant, alpha=alpha,
                                    group_by="overall", scope=scope, wape=wape))
    assert select_ridge_alpha(pd.DataFrame(records)) == {"with_macro": 10., "without_macro": 10.}


@pytest.mark.parametrize("model,alpha", [("linear_regression", None), ("ridge", 1)])
def test_time_cutoff_and_reload(model, alpha, tmp_path, monkeypatch):
    data = pd.DataFrame({
        "country_code": ["001", "001", "002", "003", "003"],
        "hs_code": ["000001"] * 5,
        "origin_quarter": ["2024Q4"] * 5,
        "target_quarter": ["2025Q1", "2025Q1", "2025Q1", "2025Q2", "2025Q1"],
        "_origin_index": [2024 * 4 + 4] * 5,
        "_target_index": [2025 * 4 + 1] * 3 + [2025 * 4 + 2, 2025 * 4 + 1],
        "target_available": [True, True, True, True, False],
        TARGET: [10, 30, 50, 100, np.nan],
        "lag": [1, 3, np.nan, 100, 500],
    })
    from oinq.common.config import quarter_number
    data["_origin_index"] = data.origin_quarter.map(quarter_number)
    data["_target_index"] = data.target_quarter.map(quarter_number)
    fit = training_rows(data, "2025Q1")
    cats = ["country_code", "hs_code"]
    columns = cats + ["lag"]
    config = load_yaml(ROOT / "configs/models/linear_regression.yaml")
    pipe = fit_pipeline(fit, columns, cats, model, config, alpha)
    assert len(fit) == 3
    processor = pipe.named_steps["preprocessor"]
    assert processor.named_transformers_["cat"].categories_[0].tolist() == ["001", "002"]
    assert processor.named_transformers_["num"].named_steps["imputer"].statistics_[0] == 2
    assert processor.named_transformers_["num"].named_steps["scaler"].mean_[0] == 2
    sample = prepare_x(data.iloc[[3]], columns, cats)
    expected = pipe.predict(sample)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reload must not fit preprocessing")

    monkeypatch.setattr(type(processor), "fit", forbidden)
    monkeypatch.setattr(type(processor), "fit_transform", forbidden)
    path = tmp_path / "pipeline.joblib"
    save_pipeline(path, pipe, columns, cats, "v002", "2025Q1", sample)
    np.testing.assert_array_equal(load_model(path)["model"].predict(sample), expected)
