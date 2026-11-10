"""MLP (64, 32): USD target scaling, origin-local preprocessing, no auto-publish.

Only the synthetic smoke entry point is enabled by the current CLI. Validation
helpers do not open final_test or run the common forecast engine.
"""
from pathlib import Path
import json
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oinq.common.config import load_yaml
from oinq.modeling.loader import load_model
from oinq.modeling.supervised.lightgbm import (
    KEY, TARGET, PRED, load_inputs, training_rows, prepare_x, metric_values, frame_hash,
)
from oinq.modeling.supervised.lightgbm_final_test import select_scored_rows

MODEL_ID = 'mlp_64_32'
MODES = ('with_macro', 'without_macro')
LIMITATION = ('Current revised snapshots; historical publication dates/vintages '
              'are unverified. No historically reproducible as-of claim.')


def load_config(root):
    config = load_yaml(Path(root)/'configs/models/mlp.yaml')
    if config['model_id'] != MODEL_ID:
        raise ValueError('Unexpected MLP model ID')
    params = dict(config['params'])
    params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
    if params['hidden_layer_sizes'] != (64, 32):
        raise ValueError('The authorized architecture is (64, 32)')
    if params['early_stopping'] or params['shuffle']:
        raise ValueError('Random validation/batch shuffling is disabled')
    return config, params


def feature_sets(manifest, feature_config):
    columns = list(manifest['feature_columns'])
    macro = feature_config['supervised']['macro']['columns']
    excluded = set(macro) | {f'{x}_missing' for x in macro}
    if len(macro) != 6 or len(excluded) != 12 or not excluded.issubset(columns):
        raise ValueError('Expected six macro values and six missing flags')
    return {'with_macro': columns, 'without_macro': [c for c in columns if c not in excluded]}


def make_pipeline(columns, categorical, params):
    categorical = list(categorical)
    if not set(categorical).issubset(columns) or len(set(columns)) != len(columns):
        raise ValueError('Invalid feature contract')
    numeric = [c for c in columns if c not in categorical]
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scaler', StandardScaler()),
        ]), numeric),
    ], sparse_threshold=0.0)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor=MLPRegressor(**params), transformer=StandardScaler(), check_inverse=True)),
    ])


def fit_pipeline(rows, columns, categorical, params):
    y = rows[TARGET].to_numpy(dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Fit requires finite observed nonnegative USD targets')
    pipe = make_pipeline(columns, categorical, params)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pipe.fit(prepare_x(rows, columns, categorical), y)
    model = pipe['model'].regressor_
    curve = [float(x) for x in model.loss_curve_]
    convergence_warnings = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    finite = bool(np.isfinite(curve).all() and np.isfinite(model.loss_))
    converged = bool(finite and not convergence_warnings and model.n_iter_ < model.max_iter)
    audit = dict(status='converged' if converged else 'not_converged', converged=converged,
                 n_iter=int(model.n_iter_), max_iter=int(model.max_iter), loss=float(model.loss_),
                 loss_curve=curve, loss_units='standardized_target_squared_error_plus_regularization',
                 convergence_warnings=convergence_warnings,
                 warnings=[{'category': w.category.__name__, 'message': str(w.message)} for w in caught])
    # Custom metadata lives on the standard Pipeline, no custom estimator class.
    pipe.mlp_fit_audit_ = audit
    return pipe, audit


def predict_rows(pipe, rows, columns, categorical):
    raw = np.asarray(pipe.predict(prepare_x(rows, columns, categorical)), dtype=float)
    if raw.ndim != 1 or len(raw) != len(rows) or not np.isfinite(raw).all():
        raise ValueError('MLP must return one finite USD amount per row')
    return raw, np.maximum(raw, 0.0)


def evaluate(rows, values):
    result = rows.copy()
    result[PRED] = values
    scored, _ = select_scored_rows(result)
    if not np.isfinite(result[PRED].to_numpy(float)).all():
        raise ValueError('Nonfinite predictions may not silently leave evaluation')
    score = metric_values(scored[TARGET], scored[PRED])
    return dict(WAPE=score['wape'], MAE=score['mae'], RMSE=score['rmse'],
                R2=float(r2_score(scored[TARGET], scored[PRED])) if len(scored)>1 else None,
                n_predictions=len(result), n_evaluated=len(scored))


def save_pipeline(path, pipe, columns, categorical, *, dataset_version, train_end, sample):
    """Never publish a non-converged or uncertified fit; never overwrite."""
    if not getattr(pipe, 'mlp_fit_audit_', {}).get('converged', False):
        raise ValueError('Non-converged fit cannot be saved as a completed model')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    before = pipe.predict(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        joblib.dump(dict(pipeline=pipe, feature_columns=list(columns),
            categorical_columns=list(categorical), dataset_version=dataset_version,
            trained_through_target_quarter=train_end), f, compress=3)
    loaded = load_model(path)
    np.testing.assert_array_equal(before, loaded['model'].predict(sample))
    return dict(reload_exact=True, rows=len(sample), max_absolute_difference=0.0)


def shared_model_path(root, mode, *, train_end):
    if mode not in MODES:
        raise ValueError('Unknown macro mode')
    if train_end not in ('2025Q4', '2026Q3'):
        raise ValueError('Explicit test/operational cutoff is required')
    # Team-designated name for the frozen test model; future operational version
    # gets a separate experiment ID. The engine accepts either via --model.
    model_id = MODEL_ID if train_end == '2025Q4' else MODEL_ID+'_operational_2026Q3'
    return Path(root)/'artifacts/shared_models'/model_id/mode/'pipeline.joblib'


def validate_candidates(root, candidates):
    """Explicit future full-validation API. Not called by smoke CLI/notebook.

    Every fit warning/loss/iteration count is returned. A candidate is eligible
    only if every origin converged. No final_test/forecast access or saving.
    """
    train, valid, manifest, features, split, _, _, _ = load_inputs(Path(root))
    _, initial = load_config(root)
    columns_by_mode = feature_sets(manifest, features)
    categorical = manifest['feature_contract']['categorical']
    pool = pd.concat([train, valid], ignore_index=True)
    records, audits = [], []
    for mode, columns in columns_by_mode.items():
        for candidate, overrides in candidates.items():
            params = initial | overrides
            if (tuple(params['hidden_layer_sizes']) != (64,32) or params['early_stopping'] or params['shuffle']):
                raise ValueError('Candidate violates fixed architecture/time protocol')
            blocks, eligible = [], True
            for origin, scoring in valid.sort_values(KEY).groupby('origin_quarter', sort=True):
                fit = training_rows(pool, origin)
                pipe, audit = fit_pipeline(fit, columns, categorical, params)
                eligible &= audit['converged']
                audits.append(dict(macro_mode=mode, candidate=candidate, origin=origin,
                    fit_rows=len(fit), last_target=fit.target_quarter.max(),
                    fit_key_hash=frame_hash(fit, KEY), **audit))
                _, values = predict_rows(pipe, scoring, columns, categorical)
                blocks.append(scoring.assign(**{PRED: values}))
            block = pd.concat(blocks, ignore_index=True)
            primary = block[block.horizon_quarters.isin(split['validation']['primary_horizons'])]
            records.append(dict(macro_mode=mode, candidate=candidate, eligible=bool(eligible),
                                params=params, **evaluate(primary, primary[PRED].to_numpy(float))))
    return pd.DataFrame(records), audits


def select_validation(records):
    eligible = records.loc[records.eligible & records.WAPE.notna()].copy()
    choices = {}
    for mode in MODES:
        rank = eligible[eligible.macro_mode.eq(mode)].sort_values(['WAPE','candidate'])
        if rank.empty:
            raise ValueError(f'No fully converged validation candidate: {mode}')
        choices[mode] = rank.iloc[0].to_dict()
    return choices


def smoke_check(root):
    """Synthetic 288-row fit per macro mode; no shared training/test reads.

    Temporary joblibs are labelled synthetic, outside shared_models, removed
    on exit. This is a wiring check, never a validation/performance claim.
    """
    root = Path(root)
    config, params = load_config(root)
    manifest = json.loads((root/'data/shared/v002/manifest.json').read_text(encoding='utf-8'))
    feature = load_yaml(root/'configs/features.yaml')
    columns_by_mode = feature_sets(manifest, feature)
    cats = manifest['feature_contract']['categorical']
    rows = pd.DataFrame({c:np.zeros(288) for c in manifest['feature_columns']})
    for c in cats:
        rows[c] = '001234' if c=='hs_code' else ('AAA' if c=='country_code' else 'test')
    x = np.linspace(-1, 1, len(rows))
    rows['export_value_usd_lag1'] = x
    rows['gdp_usd'] = 1e9
    rows[TARGET] = 1e6 + 2e5*x
    rows.loc[0,'export_value_usd_lag2'] = np.nan
    rows['export_net_weight_kg_lag4'] = np.nan
    sample = rows.iloc[:8].copy()
    sample['country_code'] = 'UNSEEN'
    sample['gdp_usd'] = 9e12
    reports = []
    for mode, columns in columns_by_mode.items():
        pipe, audit = fit_pipeline(rows, columns, cats, params)
        raw, clipped = predict_rows(pipe, sample, columns, cats)
        ttr = pipe['model']
        prepared = pipe['preprocessor'].transform(prepare_x(sample, columns, cats))
        standardized = ttr.regressor_.predict(prepared)
        manual = ttr.transformer_.inverse_transform(standardized.reshape(-1,1)).ravel()
        np.testing.assert_array_equal(raw, manual)
        np.testing.assert_allclose(ttr.transformer_.mean_[0], rows[TARGET].mean())
        reload = dict(reload_exact=False, skipped_reason='not_converged')
        if audit['converged']:
            with tempfile.TemporaryDirectory(prefix='mlp_smoke_') as temp:
                reload = save_pipeline(Path(temp)/'pipeline.joblib', pipe, columns, cats,
                    dataset_version='synthetic_smoke', train_end='synthetic_only',
                    sample=prepare_x(sample, columns, cats))
        reports.append(dict(macro_mode=mode, training_rows=len(rows), feature_count=len(columns),
            **audit, usd_inverse_exact=True, finite_usd_predictions=bool(np.isfinite(raw).all()),
            prediction_min_usd=float(raw.min()), prediction_max_usd=float(raw.max()), **reload))
    passed = all(x['converged'] and x['reload_exact'] for x in reports)
    return dict(status='passed' if passed else 'not_converged', synthetic_only=True,
                real_validation_run=False, final_test_run=False, forecast_run=False,
                shared_models_written=False, params=params, results=reports, limitation=LIMITATION)
