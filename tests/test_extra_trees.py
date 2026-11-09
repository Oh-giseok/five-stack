from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from oinq.common.config import load_yaml, quarter_number
from oinq.modeling.loader import load_model
from oinq.modeling.supervised.extra_trees import (
    TARGET, PRED, KEY, load_inputs, training_rows, feature_sets, prepare_x,
    fit_pipeline, predict_rows, save_pipeline, metrics, pruning_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_actual_contract_and_origin_label_cutoff():
    train, valid, manifest, config, _, _, _, _ = load_inputs(ROOT)
    features = feature_sets(manifest, config)
    assert len(features['with_macro']) == 57
    assert len(features['without_macro']) == 45
    pool = pd.concat([train, valid], ignore_index=True)
    for origin in ('2020Q4', '2021Q4', '2023Q1', '2025Q4'):
        rows = training_rows(pool, origin)
        assert rows[TARGET].notna().all()
        assert rows._target_index.max() <= quarter_number(origin)
    test = pd.read_parquet(ROOT/'data/shared/v002/final_test.parquet')
    assert (len(test), test[TARGET].notna().sum(), test[TARGET].isna().sum()) == (480,447,33)
    assert not test.duplicated(KEY).any()


def test_fit_local_preprocessing_unknown_categories_and_reload(tmp_path, monkeypatch):
    rows = pd.DataFrame({'country_code':['004','004','008'], 'lag':[1.,np.nan,3.],
                         'empty':[np.nan]*3, TARGET:[5.,7.,9.]})
    columns = ['country_code','lag','empty']
    cats = ['country_code']
    params = dict(n_estimators=4, random_state=42, n_jobs=1)
    pipe = fit_pipeline(rows, columns, cats, params)
    assert pipe.named_steps['preprocessor'].named_transformers_['num'].statistics_.tolist() == [2.,0.]
    heldout = rows.assign(country_code='999', lag=99999.)
    before = predict_rows(pipe, heldout, columns, cats)
    assert pipe.named_steps['preprocessor'].named_transformers_['num'].statistics_[0] == 2.
    def forbidden(*args, **kwargs):
        raise AssertionError('Reload/predict must not fit')
    monkeypatch.setattr(type(pipe), 'fit', forbidden)
    path = tmp_path/'pipeline.joblib'
    sample = prepare_x(heldout, columns, cats)
    save_pipeline(path, pipe, columns, cats, sample)
    loaded = load_model(path)
    np.testing.assert_array_equal(before, loaded['model'].predict(sample))
    assert loaded['feature_columns'] == columns
    payload = joblib.load(path)
    assert payload['trained_through_target_quarter'] == '2025Q4'
    with pytest.raises(FileExistsError):
        save_pipeline(path, pipe, columns, cats, sample)


def test_metric_preserves_missing_target_rows():
    rows = pd.DataFrame({TARGET:[100.,np.nan,200.], PRED:[90.,777.,180.],
                         'target_available':[True,False,True]})
    score = metrics(rows)
    assert score['n_predictions'] == 3 and score['n_evaluated'] == 2
    assert score['WAPE'] == 10.
    assert score['MAE'] == 15.
    assert rows[TARGET].isna().sum() == 1


def test_data_scaled_pruning_changes_tree_complexity():
    rng = np.random.default_rng(42)
    rows = pd.DataFrame({'x':rng.normal(size=120), TARGET:rng.uniform(0,1e7,120)})
    params = dict(n_estimators=3, random_state=42, n_jobs=1, max_features=1.0)
    values = pruning_candidates(rows, ['x'], [], params, [.8,.95])
    assert values[0] == 0 and values[-1] > 1
    raw = fit_pipeline(rows, ['x'], [], params)
    pruned = fit_pipeline(rows, ['x'], [], params | {'ccp_alpha':values[-1]})
    assert sum(t.tree_.node_count for t in pruned['model'].estimators_) < sum(t.tree_.node_count for t in raw['model'].estimators_)


def test_no_publish_or_overwrite_when_notebooks_exist(tmp_path):
    from oinq.modeling.supervised.extra_trees import run_training
    path = tmp_path/'notebooks/extra_trees_depth.ipynb'
    path.parent.mkdir()
    path.write_text('{}')
    with pytest.raises(FileExistsError):
        run_training(tmp_path, 'guard', ('depth',), publish=False)
    assert path.read_text() == '{}'
