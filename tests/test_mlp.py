from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from oinq.modeling.loader import load_model
from oinq.features.prediction import prepare_model_inputs
from oinq.modeling.supervised.mlp import (
    TARGET, load_config, smoke_check, fit_pipeline, prepare_x, save_pipeline,
    shared_model_path, select_validation, load_inputs, training_rows, feature_sets,
)

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_usd_and_both_modes():
    report = smoke_check(ROOT)
    assert report['status']=='passed', report
    assert [x['feature_count'] for x in report['results']]==[57,45]
    for x in report['results']:
        assert x['converged'] and x['usd_inverse_exact'] and x['reload_exact']
        assert len(x['loss_curve'])==x['n_iter']<300
    assert not report['shared_models_written'] and not report['final_test_run']


def test_nonconvergence_is_logged_and_cannot_be_saved(tmp_path):
    _, params = load_config(ROOT)
    rows = pd.DataFrame({'x':np.linspace(0,1,32), TARGET:np.linspace(1e6,2e6,32)})
    pipe, audit = fit_pipeline(rows, ['x'], [], params | {'max_iter':1,'batch_size':16})
    assert not audit['converged'] and audit['n_iter']==1 and audit['convergence_warnings']
    assert isinstance(pipe,Pipeline) and isinstance(pipe['model'],TransformedTargetRegressor)
    with pytest.raises(ValueError, match='Non-converged'):
        save_pipeline(tmp_path/'pipeline.joblib',pipe,['x'],[],dataset_version='test',
                      train_end='synthetic',sample=prepare_x(rows,['x'],[]))
    assert not (tmp_path/'pipeline.joblib').exists()


def test_train_local_scalers_and_common_loader(tmp_path, monkeypatch):
    _, params=load_config(ROOT)
    rows=pd.DataFrame({'code':['001234']*288,'x':np.linspace(-1,1,288),
                       'empty':[np.nan]*288,TARGET:1e6+2e5*np.linspace(-1,1,288)})
    pipe,audit=fit_pipeline(rows,['code','x','empty'],['code'],params)
    assert audit['converged']
    pre=pipe['preprocessor'];num=pre.named_transformers_['num']
    assert num['imputer'].statistics_[1]==0
    np.testing.assert_allclose(num['scaler'].mean_[0],0,atol=1e-12)
    target_mean=pipe['model'].transformer_.mean_.copy()
    sample=prepare_x(rows.iloc[:4].assign(code='UNSEEN',x=99999),['code','x','empty'],['code'])
    def forbidden(*args,**kwargs):raise AssertionError('Inference must not fit')
    monkeypatch.setattr(Pipeline,'fit',forbidden)
    path=tmp_path/'pipeline.joblib'
    save_pipeline(path,pipe,['code','x','empty'],['code'],dataset_version='synthetic',train_end='synthetic',sample=sample)
    loaded=load_model(path)
    engine_sample=prepare_model_inputs(rows.iloc[:4].assign(code='UNSEEN',x=99999),loaded)
    pd.testing.assert_frame_equal(sample,engine_sample)
    np.testing.assert_array_equal(pipe.predict(sample),loaded['model'].predict(sample))
    np.testing.assert_array_equal(pipe['model'].transformer_.mean_,target_mean)
    assert loaded['feature_columns']==['code','x','empty'] and loaded['categorical_columns']==['code']
    with pytest.raises(FileExistsError):
        save_pipeline(path,pipe,['code','x','empty'],['code'],dataset_version='synthetic',train_end='synthetic',sample=sample)


def test_validation_rejects_nonconverged_candidates():
    records=pd.DataFrame([dict(macro_mode=m,candidate='bad',eligible=False,WAPE=1.)
                          for m in ['with_macro','without_macro']])
    with pytest.raises(ValueError,match='No fully converged'):
        select_validation(records)


def test_frozen_and_operational_model_paths_differ():
    a=shared_model_path(ROOT,'with_macro',train_end='2025Q4')
    b=shared_model_path(ROOT,'with_macro',train_end='2026Q3')
    assert a!=b and 'mlp_64_32' in a.parts and 'mlp_64_32_operational_2026Q3' in b.parts


def test_actual_contract_without_fitting():
    train,valid,manifest,config,_,_,_,_=load_inputs(ROOT)
    sets=feature_sets(manifest,config)
    assert len(sets['with_macro'])==57 and len(sets['without_macro'])==45
    fit=training_rows(pd.concat([train,valid]),'2021Q4')
    assert fit.target_quarter.max()<='2021Q4' and fit[TARGET].notna().all()
    assert not fit['split'].eq('validation').any()
