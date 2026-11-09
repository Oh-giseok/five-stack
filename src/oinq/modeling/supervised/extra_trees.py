"""Extra Trees experiments; immutable shared contracts and no operational refit.

WAPE is expressed as a percentage, matching the common evaluator.
Current revised input vintages are not historically verified as-of snapshots.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import re
import traceback
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import ExtraTreeRegressor

from oinq.common.config import load_yaml, quarter_number
from oinq.modeling.loader import load_model
from oinq.modeling.supervised.lightgbm import (
    KEY, TARGET, PRED, load_inputs, prepare_x, training_rows, frame_hash,
    sha256, write_json, metric_values,
)
from oinq.modeling.supervised.lightgbm_final_test import check_test_inputs, select_scored_rows

EXPERIMENTS = ('depth', 'leaf', 'features', 'pruned')
MODES = ('with_macro', 'without_macro')
LIMITATION = ('현재 수정치·달력 정렬 자료이며 과거 공개일 및 빈티지는 미검증입니다. '
              'origin 이후 정답은 학습에서 제외하지만 역사적 as-of 재현 백테스트를 보장하지 않습니다.')


def feature_sets(manifest, config):
    columns = manifest['feature_columns']
    macro = config['supervised']['macro']['columns']
    excluded = set(macro) | {f'{x}_missing' for x in macro}
    if len(excluded) != 12 or not excluded.issubset(columns):
        raise ValueError('Macro feature contract mismatch')
    return dict(with_macro=list(columns), without_macro=[x for x in columns if x not in excluded])


def make_pipeline(features, categorical, params):
    return Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
            ('num', SimpleImputer(strategy='median', keep_empty_features=True),
             [x for x in features if x not in categorical]),
        ])),
        ('model', ExtraTreesRegressor(**params)),
    ])


def fit_pipeline(rows, features, categorical, params):
    if rows.empty or rows[TARGET].isna().any():
        raise ValueError('Fit requires nonempty observed targets')
    pipe = make_pipeline(features, categorical, params)
    pipe.fit(prepare_x(rows, features, categorical), rows[TARGET].to_numpy(dtype=float))
    pipe.set_params(model__n_jobs=1)  # deterministic summation and exact reload
    return pipe


def predict_rows(pipe, rows, features, categorical):
    values = pipe.predict(prepare_x(rows, features, categorical))
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Nonfinite or negative Extra Trees prediction')
    return values


def metrics(rows):
    scored, _ = select_scored_rows(rows)
    if not np.isfinite(scored[PRED].to_numpy(float)).all():
        raise ValueError('Invalid prediction must not silently leave evaluation')
    values = metric_values(scored[TARGET], scored[PRED])
    return {'WAPE': values['wape'], 'MAE': values['mae'], 'RMSE': values['rmse'],
            'R2': float(r2_score(scored[TARGET], scored[PRED])) if len(scored) > 1 else None,
            'n_predictions': len(rows), 'n_evaluated': len(scored)}


def save_pipeline(path, pipe, columns, categorical, sample):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(pipeline=pipe, feature_columns=columns, categorical_columns=categorical,
                   dataset_version='v002', trained_through_target_quarter='2025Q4')
    with path.open('xb') as handle:
        joblib.dump(payload, handle, compress=3)
    loaded = load_model(path)
    np.testing.assert_array_equal(pipe.predict(sample), loaded['model'].predict(sample))


def pruning_candidates(fit, columns, categorical, common, quantiles):
    """Use only earliest-origin training labels; freeze absolute candidates."""
    pilot = fit_pipeline(fit, columns, categorical, common | {'n_estimators': 1})
    x = pilot.named_steps['preprocessor'].transform(prepare_x(fit, columns, categorical))
    tree = ExtraTreeRegressor(random_state=common['random_state'], max_features=common['max_features'])
    path = tree.cost_complexity_pruning_path(x, fit[TARGET].to_numpy(float))
    positive = path.ccp_alphas[np.isfinite(path.ccp_alphas) & (path.ccp_alphas > 0)]
    if not len(positive):
        raise ValueError('No positive data-derived pruning alpha')
    alphas = sorted(set(float(np.quantile(positive, q)) for q in quantiles))
    return [0.0] + alphas


def _protected_snapshot(root):
    paths = [root/'app.py', root/'pyproject.toml']
    for folder in ('data', 'configs', 'src', 'scripts', 'tests', 'notebooks', 'artifacts/shared_models'):
        paths += [p for p in (root/folder).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts]
    return {str(p.relative_to(root)): sha256(p) for p in paths if p.is_file()}


def audit_data(root, output):
    train, valid, manifest, feature, split, _, _, checksums = load_inputs(root)
    report = {'limitation': LIMITATION, 'partitions': {}, 'manifest_limitation': manifest['availability_limitation']}
    for item in manifest['files']:
        if sha256(root/item['path']) != item['sha256']:
            raise ValueError(f"Shared release checksum mismatch: {item['path']}")
    frames = {}
    for name in ('train', 'validation', 'final_test'):
        df = pd.read_parquet(root/f'data/shared/v002/{name}.parquet')
        frames[name] = df
        if len(df) != manifest['row_counts'][name]['rows'] or df.duplicated(KEY).any():
            raise ValueError(f'Row/key mismatch: {name}')
        if df[KEY].isna().any().any():
            raise ValueError('Null sample key')
        if not df.target_available.eq(df[TARGET].notna()).all():
            raise ValueError('Label availability mismatch')
        for col in manifest['feature_contract']['categorical']:
            if not all(isinstance(v, str) for v in df[col].dropna()):
                raise ValueError(f'Categorical dtype mismatch: {col}')
        prepare_x(df, manifest['feature_columns'], manifest['feature_contract']['categorical'])
        observed = df[TARGET].dropna().to_numpy(float)
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError('Invalid observed target')
        bounds = split['splits'][name]
        if not df.target_quarter.map(quarter_number).between(quarter_number(bounds['start']), quarter_number(bounds['end'])).all():
            raise ValueError('Target period mismatch')
        report['partitions'][name] = dict(rows=len(df), columns=len(df.columns),
            observed_targets=int(df[TARGET].notna().sum()), missing_targets=int(df[TARGET].isna().sum()),
            target_start=df.target_quarter.min(), target_end=df.target_quarter.max(),
            origin_start=df.origin_quarter.min(), origin_end=df.origin_quarter.max(),
            dtypes=df.dtypes.astype(str).to_dict(), null_counts=df.isna().sum().astype(int).to_dict(),
            duplicate_keys=int(df.duplicated(KEY).sum()))
    test = frames['final_test']
    check_test_inputs(test, split['final_test'], manifest['country_codes'], manifest['hs_codes'])
    if (len(test), int(test[TARGET].notna().sum()), int(test[TARGET].isna().sum())) != (480, 447, 33):
        raise ValueError('Expected 480 predictions / 447 labels / 33 missing')
    report['feature_columns'] = feature_sets(manifest, feature)
    write_json(output/'data_audit.json', report)
    return train, valid, manifest, feature, split


def _frame(rows, values):
    result = rows[KEY+['horizon_quarters', TARGET, 'target_available']].copy()
    result[PRED] = values
    return result


def _complexity(pipe):
    trees = pipe.named_steps['model'].estimators_
    return dict(mean_nodes=float(np.mean([t.tree_.node_count for t in trees])),
                mean_depth=float(np.mean([t.tree_.max_depth for t in trees])),
                mean_leaves=float(np.mean([t.tree_.n_leaves for t in trees])))


def run_training(root, run_id=None, experiments=EXPERIMENTS, *, publish=True, forecast=True, notebooks=True):
    root = Path(root).resolve()
    run_id = run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', run_id):
        raise ValueError('Invalid run ID')
    experiments = tuple(experiments)
    if not experiments or len(set(experiments)) != len(experiments) or not set(experiments).issubset(EXPERIMENTS):
        raise ValueError('Unknown/duplicate experiment')
    targets = [root/f'artifacts/shared_models/extra_trees_{e}/{mode}/pipeline.joblib'
               for e in experiments for mode in MODES]
    notebook_paths = [root/f'notebooks/extra_trees_{e}.ipynb' for e in (*experiments, 'experiment')]
    for path in ((targets if publish else []) + (notebook_paths if notebooks else [])):
        if path.exists():
            raise FileExistsError(f'Existing destination will not be overwritten: {path}')
    protected = _protected_snapshot(root)
    out = root/'.private/extra_trees'/run_id
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protected_before.json', protected)
    write_json(out/'status.json', {'status': 'running', 'experiments': experiments})
    try:
        result = _run(root, out, experiments, publish, forecast)
        if notebooks:
            create_notebooks(root, out, experiments)
        changed = [p for p, digest in protected.items() if not (root/p).is_file() or sha256(root/p) != digest]
        write_json(out/'protected_verification.json', {'checked_files': len(protected), 'changed': changed})
        if changed:
            raise ValueError(f'Preexisting files changed: {changed}')
        write_json(out/'status.json', {'status': 'completed', 'experiments': experiments,
                                     'shared_models': len(targets) if publish else 0})
        return result
    except Exception:
        (out/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(out/'status.json', {'status': 'failed', 'error': traceback.format_exc()})
        raise


def _run(root, out, experiments, publish, forecast):
    train, valid, manifest, feature, split = audit_data(root, out)
    config = load_yaml(root/'configs/models/extra_trees.yaml')
    write_json(out/'config_snapshot.json', config)
    columns_by_mode = feature_sets(manifest, feature)
    categorical = manifest['feature_contract']['categorical']
    pool = pd.concat([train, valid], ignore_index=True)
    common = config['params']
    origins = sorted(valid.origin_quarter.unique())
    earliest = training_rows(pool, origins[0])
    candidates = {}
    for mode, columns in columns_by_mode.items():
        candidates[mode] = {'baseline': common.copy()}
        for exp in experiments:
            spec = config['experiments'][exp]
            values = (pruning_candidates(earliest, columns, categorical, common, spec['path_quantiles'])
                      if exp == 'pruned' else spec['values'])
            for i, value in enumerate(values):
                candidates[mode][f'{exp}_{i}'] = common | {spec['parameter']: value}
    write_json(out/'candidate_definitions.json', {'earliest_origin': origins[0],
        'alpha_fit_last_target': earliest.target_quarter.max(), 'candidates': candidates})
    validation_frames, audits, complexity = [], [], []
    for origin, scoring in valid.sort_values(KEY).groupby('origin_quarter', sort=True):
        fit = training_rows(pool, origin)
        if fit.target_quarter.max() > origin:
            raise ValueError('Future label leakage')
        audits.append(dict(origin=origin, fit_rows=len(fit), last_target=fit.target_quarter.max(),
                           fit_key_hash=frame_hash(fit, KEY)))
        print(f'Validation {origin}: fit={len(fit)}, predict={len(scoring)}', flush=True)
        for mode, columns in columns_by_mode.items():
            cache = {}
            for candidate, params in candidates[mode].items():
                signature = json.dumps(params, sort_keys=True)
                if signature not in cache:
                    started = perf_counter()
                    pipe = fit_pipeline(fit, columns, categorical, params)
                    cache[signature] = (predict_rows(pipe, scoring, columns, categorical), _complexity(pipe))
                    print(f'  {mode}/{candidate}: {perf_counter()-started:.1f}s', flush=True)
                values, stats = cache[signature]
                validation_frames.append(_frame(scoring, values).assign(macro_mode=mode, candidate=candidate))
                complexity.append(dict(origin=origin, macro_mode=mode, candidate=candidate, **stats))
        pd.DataFrame(audits).to_csv(out/'validation_fit_audit.csv', index=False)
        pd.DataFrame(complexity).to_csv(out/'validation_complexity.csv', index=False)
    predictions = pd.concat(validation_frames, ignore_index=True)
    predictions.to_csv(out/'validation_predictions.csv', index=False)
    records = []
    for (mode, candidate), block in predictions.groupby(['macro_mode', 'candidate'], sort=True):
        for scope, horizons in [('h1_5', split['validation']['horizon_quarters']),
                                ('h2_5', split['validation']['primary_horizons'])]:
            records.append(dict(macro_mode=mode, candidate=candidate, scope=scope,
                                **metrics(block[block.horizon_quarters.isin(horizons)])))
    validation_metrics = pd.DataFrame(records)
    validation_metrics.to_csv(out/'validation_candidate_metrics.csv', index=False)
    selection = []
    for exp in experiments:
        for mode in MODES:
            rank = validation_metrics.loc[validation_metrics.macro_mode.eq(mode)
                & validation_metrics.scope.eq('h2_5') & validation_metrics.candidate.str.startswith(exp+'_')]
            if rank.WAPE.isna().any():
                raise ValueError('Undefined selection WAPE')
            choice = rank.sort_values(['WAPE', 'candidate']).iloc[0]
            selection.append(dict(model_id='extra_trees_'+exp, macro_mode=mode,
                                  candidate=choice.candidate, params=candidates[mode][choice.candidate]))
    write_json(out/'selection_lock.json', {'selection': selection, 'criterion': 'minimum validation h2_5 WAPE; tie=candidate ID',
        'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'post_test_changes': False,
        'historical_asof_verified': False, 'limitation': LIMITATION})
    lock_hash = sha256(out/'selection_lock.json')
    fit = training_rows(pool, '2025Q4')
    if fit.target_quarter.max() != '2025Q4':
        raise ValueError('Final fit cutoff mismatch')
    test = pd.read_parquet(root/'data/shared/v002/final_test.parquet').sort_values(KEY)
    scored, missing = select_scored_rows(test)
    test_key_hash = frame_hash(scored, KEY)
    summaries, baselines, reloads, final_complexity, forecast_checks = [], [], [], [], []
    for mode, columns in columns_by_mode.items():
        jobs = [('extra_trees_baseline', 'baseline', common)] + [
            (s['model_id'], s['candidate'], s['params']) for s in selection if s['macro_mode'] == mode]
        for model_id, candidate, params in jobs:
            print(f'Final {model_id}/{mode}: {params}', flush=True)
            pipe = fit_pipeline(fit, columns, categorical, params)
            result = _frame(test, predict_rows(pipe, test, columns, categorical))
            result = result.assign(model_id=model_id, macro_mode=mode)
            result.to_csv(out/f'{model_id}_{mode}_test_predictions.csv', index=False)
            val = validation_metrics.loc[validation_metrics.macro_mode.eq(mode)
                & validation_metrics.candidate.eq(candidate) & validation_metrics.scope.eq('h2_5')].iloc[0]
            score = metrics(result)
            row = dict(model_id=model_id, macro_mode=mode, hyperparameters=json.dumps(params, sort_keys=True),
                dataset_version='v002', train_end='2025Q4', candidate=candidate,
                validation_n_predictions=int(val.n_predictions), validation_n_evaluated=int(val.n_evaluated),
                **{f'validation_{k}': val[k] for k in ('WAPE', 'MAE', 'RMSE', 'R2')},
                **{f'test_{k}': score[k] for k in ('WAPE', 'MAE', 'RMSE', 'R2')},
                n_predictions=score['n_predictions'], n_evaluated=score['n_evaluated'],
                test_evaluation_key_hash=test_key_hash, historical_asof_verified=False, limitation=LIMITATION)
            final_complexity.append(dict(model_id=model_id, macro_mode=mode, **_complexity(pipe)))
            if model_id == 'extra_trees_baseline':
                baselines.append(row)
                continue
            destination = (root/f'artifacts/shared_models/{model_id}/{mode}/pipeline.joblib' if publish
                           else out/f'models/{model_id}/{mode}/pipeline.joblib')
            sample = prepare_x(test, columns, categorical)
            save_pipeline(destination, pipe, columns, categorical, sample)
            row['model_path'] = str(destination.relative_to(root))
            row['reload_exact'] = True
            summaries.append(row)
            reloads.append(dict(model_id=model_id, macro_mode=mode, reload_exact=True, rows=len(test),
                                max_absolute_difference=0.0, model_path=row['model_path']))
            if forecast and publish:
                forecast_checks.append(check_forecast(root, out, model_id, mode))
    summary = pd.DataFrame(summaries)
    summary['validation_WAPE_rank'] = summary.validation_WAPE.rank(method='min').astype(int)
    summary['test_WAPE_rank'] = summary.test_WAPE.rank(method='min').astype(int)
    summary.to_csv(out/'model_comparison.csv', index=False)
    pd.DataFrame(baselines).to_csv(out/'baseline_comparison.csv', index=False)
    pd.DataFrame(reloads).to_csv(out/'reload_verification.csv', index=False)
    comp = pd.DataFrame(final_complexity)
    comp.to_csv(out/'final_complexity.csv', index=False)
    if 'pruned' in experiments:
        history = pd.DataFrame(complexity)
        for mode in MODES:
            h = history[history.macro_mode.eq(mode)]
            base = h[h.candidate.eq('baseline')].set_index('origin').mean_nodes
            positive = [k for k,v in candidates[mode].items() if k.startswith('pruned_') and v['ccp_alpha'] > 0]
            for candidate in positive:
                delta = base - h[h.candidate.eq(candidate)].set_index('origin').mean_nodes
                if not (delta > 0).any():
                    raise ValueError(f'Pruning did not reduce nodes: {mode}/{candidate}')
    write_json(out/'forecast_verification.json', forecast_checks)
    if sha256(out/'selection_lock.json') != lock_hash:
        raise ValueError('Selection lock changed after testing')
    write_json(out/'run_metadata.json', dict(dataset_version='v002', train_end='2025Q4', final_fit_rows=len(fit),
        models=len(summary), test_predictions=480, test_evaluated=len(scored), test_missing=len(missing),
        test_evaluation_key_hash=test_key_hash, selection_lock_sha256=lock_hash, post_test_changes=False,
        limitation=LIMITATION, python=platform.python_version(),
        libraries={x:version(x) for x in ('numpy','pandas','scikit-learn','pyarrow','joblib','PyYAML','lightgbm')},
        baseline_shared=False, forecast_is_connection_check_only=True, operational_refit=False))
    return out


def check_forecast(root, out, model_id, mode):
    """Exercise the unchanged common engine on the frozen 2025Q4 model."""
    from oinq.forecasting.engine import run_forecast
    identity = dict(model_id=model_id, macro_mode=mode, operational_refit=False,
                    model_train_end='2025Q4', prediction_origin='2026Q3')
    try:
        run = run_forecast(root=root, model_id=model_id, variant=mode, origin_quarter='2026Q3',
            features_path='data/shared/v002/forecast/origin_features_2026Q3.parquet',
            output_dir=str(out.relative_to(root)/'forecasts'), run_id=f'{model_id}_{mode}')
        folder = Path(run['output_dir'])
        path = folder/'quarterly_forecast.csv'
        if not path.exists():
            path = folder/'quarterly_forecast.parquet'
        df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
        values = df[PRED].to_numpy(float)
        checks = dict(rows=len(df), duplicate_keys=int(df.duplicated(['country_code','hs_code','target_quarter']).sum()),
            missing_predictions=int(pd.isna(values).sum()), nonfinite=int((~np.isfinite(values)).sum()),
            negative=int((values < 0).sum()), target_quarters=sorted(df.target_quarter.unique().tolist()),
            output_dir=str(folder))
        passed = (checks['rows']==640 and checks['duplicate_keys']==0 and checks['nonfinite']==0
                  and checks['negative']==0 and checks['target_quarters']==['2027Q1','2027Q2','2027Q3','2027Q4'])
        return identity | checks | {'status': 'passed' if passed else 'failed'}
    except Exception:
        return identity | {'status': 'failed', 'error': traceback.format_exc()}


def create_notebooks(root, out, experiments=EXPERIMENTS):
    """Execute result-reader cells in order, capture real stdout, emit nbformat 4.

    Standard-library serialization avoids adding notebook packages. The cells
    remain independently rerunnable in Jupyter and call this common module.
    Training is explicit opt-in to avoid overwriting completed shared models.
    """
    purposes = dict(depth='max_depth로 트리 깊이를 제한합니다.', leaf='min_samples_leaf로 리프의 최소 표본 수를 조절합니다.',
        features='max_features로 분할 시 고려하는 변수 비율을 조절합니다.',
        pruned='ccp_alpha로 비용 복잡도 가지치기를 수행합니다. 학습 데이터의 ExtraTree pruning path에서 양의 alpha 분위수를 추출합니다.',
        experiment='네 가지 실험 × 거시지표 포함·제외 8개 모델을 공통 기준 모델 2개와 비교합니다.')
    for exp in (*experiments, 'experiment'):
        path = root/f'notebooks/extra_trees_{exp}.ipynb'
        cells = []
        namespace = {}
        def markdown(text):
            cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True)))
        def code(source):
            buffer = StringIO()
            with redirect_stdout(buffer):
                exec(compile(source, str(path), 'exec'), namespace)
            output = buffer.getvalue()
            cells.append(dict(cell_type='code', metadata={}, source=source.splitlines(keepends=True),
                execution_count=1+sum(c['cell_type']=='code' for c in cells),
                outputs=[dict(output_type='stream', name='stdout', text=output.splitlines(keepends=True))] if output else []))
        markdown(f'# Extra Trees: {exp}\n\n{purposes[exp]}\n\n{LIMITATION}\n\n'
            'WAPE 단위는 %입니다. 검증 h=2~5의 전체 절대오차 합 / 실제금액 합으로 설정을 선정했습니다. '
            '최종 테스트는 2025Q4 고정 모델의 h=1~3입니다. 결측 정답은 예측에 보존하고 평가에서만 제외합니다. '
            '2027년 예측은 2025Q4 저장 모델의 연결 점검이며 운영용 재학습 결과가 아닙니다.')
        code(f"from pathlib import Path\nimport sys, json\nimport pandas as pd\nROOT = Path.cwd().resolve()\nif not (ROOT / 'configs').exists():\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\nfrom oinq.modeling.supervised.extra_trees import run_training, LIMITATION\nOUT = ROOT / {str(out.relative_to(root)).replace(chr(92), '/')!r}\nEXPERIMENT = {exp!r}\nprint('실행 결과:', OUT)\nprint(LIMITATION)\n")
        markdown('## 공통 데이터 및 기간\n\nmanifest, features.yaml, splits.yaml의 입력 순서와 분할을 유지합니다. 각 origin마다 전처리를 새로 fit합니다. '
                 '최초 검증 origin=2020Q4이며 target 기준 분할이므로 2022년 이전 origin이 포함됩니다. 원본 결측·표본은 변경하지 않습니다.')
        code("audit = json.loads((OUT / 'data_audit.json').read_text(encoding='utf-8'))\nprint(pd.DataFrame({k:{x:v[x] for x in ['rows','columns','observed_targets','missing_targets','target_start','target_end','origin_start','origin_end','duplicate_keys']} for k,v in audit['partitions'].items()}).to_string())\nprint('입력 개수:', {k:len(v) for k,v in audit['feature_columns'].items()})\nprint('자료형 및 결측 수:')\nfor name, partition in audit['partitions'].items():\n    print(name)\n    print(pd.DataFrame({'dtype':partition['dtypes'], 'nulls':partition['null_counts']}).to_string())\n")
        markdown('## 후보값 및 실제 학습 호출\n\n나머지 설정은 공통값으로 유지합니다. 가지치기 후보값은 최초 검증 origin의 학습 데이터에서만 생성하여 고정했습니다. '
                 '아래 RUN_TRAINING을 켜면 공통 모듈을 호출합니다. 기존 저장 경로가 있으면 덮어쓰기를 거부합니다. 재실행은 publish=False, notebooks=False로 별도 결과 폴더를 만듭니다.')
        code("definitions = json.loads((OUT / 'candidate_definitions.json').read_text(encoding='utf-8'))\nprint('alpha 생성 학습 종료:', definitions['alpha_fit_last_target'])\nfor mode, candidates in definitions['candidates'].items():\n    print(mode)\n    print(json.dumps({k:v for k,v in candidates.items() if EXPERIMENT=='experiment' or k=='baseline' or k.startswith(EXPERIMENT+'_')}, indent=2, ensure_ascii=False))\nRUN_TRAINING = False\nif RUN_TRAINING:\n    selected_experiments = ('depth','leaf','features','pruned') if EXPERIMENT=='experiment' else (EXPERIMENT,)\n    rerun = run_training(ROOT, experiments=selected_experiments, publish=False, forecast=False, notebooks=False)\n")
        markdown('## 이동 검증 및 선정 근거\n\n거시지표 조건별 검증 WAPE가 가장 작은 후보를 선정했으며, 동률은 후보 ID 순입니다. 테스트 결과에 따른 설정 변경은 없습니다.')
        code("candidate_metrics = pd.read_csv(OUT / 'validation_candidate_metrics.csv')\nmask = candidate_metrics.candidate.eq('baseline') | (candidate_metrics.candidate.str.startswith(EXPERIMENT+'_') if EXPERIMENT!='experiment' else True)\nprint(candidate_metrics.loc[mask].to_string(index=False))\nlock = json.loads((OUT / 'selection_lock.json').read_text(encoding='utf-8'))\nprint(json.dumps([s for s in lock['selection'] if EXPERIMENT=='experiment' or s['model_id']=='extra_trees_'+EXPERIMENT], indent=2, ensure_ascii=False))\n")
        markdown('## 최종 8개 모델 비교·평가 건수·WAPE 순위\n\n공통 기준 Extra Trees는 별도 표에 표시하고 공유 8개 모델에서 제외합니다. 검증은 h=2~5, 테스트는 h=1~3이므로 두 지표의 예측 범위가 다릅니다.')
        code("summary = pd.read_csv(OUT / 'model_comparison.csv')\nselected = summary if EXPERIMENT=='experiment' else summary[summary.model_id.eq('extra_trees_'+EXPERIMENT)]\nprint(selected.sort_values('validation_WAPE').to_string(index=False))\nprint('공통 기준 모델:')\nprint(pd.read_csv(OUT / 'baseline_comparison.csv').to_string(index=False))\n")
        markdown('## 실제 가지치기 및 모델 복잡도\n\n양의 alpha 후보와 가지치기 전 모델의 노드 수·깊이·리프 수를 비교합니다. 최적 alpha가 0인 경우도 성능 선택 결과 그대로 보존합니다.')
        code("complexity = pd.read_csv(OUT / 'validation_complexity.csv')\nmask = complexity.candidate.eq('baseline') | (complexity.candidate.str.startswith(EXPERIMENT+'_') if EXPERIMENT!='experiment' else True)\nprint(complexity.loc[mask].groupby(['macro_mode','candidate'])[['mean_nodes','mean_depth','mean_leaves']].mean().to_string())\nprint(pd.read_csv(OUT / 'final_complexity.csv').to_string(index=False))\n")
        markdown('## 모델 저장·재로딩·2027년 연결 검증\n\n공통 로더로 다시 읽어 전체 480행 예측의 완전 일치를 검증했습니다. 2027년은 기존 공통 예측 엔진으로 연결만 점검하며 2026Q3 운영용 재학습을 하지 않았습니다.')
        code("print(pd.read_csv(OUT / 'reload_verification.csv').to_string(index=False))\nprint(json.dumps(json.loads((OUT / 'forecast_verification.json').read_text(encoding='utf-8')), indent=2, ensure_ascii=False))\nprint((OUT / 'run_metadata.json').read_text(encoding='utf-8'))\n")
        notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
            language_info=dict(name='python', version=platform.python_version())), cells=cells)
        for i, cell in enumerate(cells):
            cell['id'] = f'extra-trees-{i:03d}'
        with path.open('x', encoding='utf-8') as handle:
            json.dump(notebook, handle, ensure_ascii=False, indent=1)
