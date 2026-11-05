"""LightGBM initial validation against the immutable shared-data contract.

Local implementation of documented evaluation rules, not a replacement for the
team's common evaluator. No private test data or operational forecasts are read.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from oinq.common.config import load_yaml, quarter_number

KEY = ["country_code", "hs_code", "origin_quarter", "target_quarter"]
TARGET = "target_export_value_usd"
PRED = "predicted_export_value_usd"
PROTOCOL = "lightgbm-initial-expanding-current-revised-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False, default=str), encoding="utf-8")


def frame_hash(frame, columns):
    data = frame[columns].sort_values(columns).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_inputs(root):
    root = Path(root)
    folder = root / "data/shared/v002"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    feature = load_yaml(root / "configs/features.yaml")
    split = load_yaml(root / "configs/splits.yaml")
    model = load_yaml(root / "configs/models/lightgbm.yaml")
    if manifest["dataset_version"] != "v002" or not manifest["dataset_release_approved"]:
        raise ValueError("Expected approved shared dataset v002")
    if split["validation"]["method"] != "expanding_origin":
        raise ValueError("Unsupported validation protocol")
    features = manifest["feature_columns"]
    if TARGET != manifest["feature_contract"]["target"]:
        raise ValueError("Target contract mismatch")
    forbidden = set(manifest["feature_contract"]["forbidden_feature_columns"])
    if len(set(features)) != len(features) or forbidden.intersection(features):
        raise ValueError("Duplicated or forbidden input features")
    if features != (manifest["feature_contract"]["categorical"] +
                    manifest["feature_contract"]["known_future"] +
                    manifest["feature_contract"]["numeric_candidates"]):
        raise ValueError("Input order does not match manifest contract")
    macro = feature["supervised"]["macro"]["columns"]
    removed = [x for x in features if x in macro or x in [f"{v}_missing" for v in macro]]
    if len(removed) != 2 * len(macro):
        raise ValueError("Macro values and their missing indicators must be paired")
    protected = {}
    for item in manifest["files"]:
        # The release also contains final_test and unsupervised files. This
        # validation run only opens the train/validation partitions.
        if Path(item["path"]).name not in {"train.csv", "train.parquet", "validation.csv", "validation.parquet"}:
            continue
        path = root / item["path"]
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {item['path']}")
        protected[item["path"]] = actual
    for path in [folder / "manifest.json", root / "configs/features.yaml",
                 root / "configs/splits.yaml", root / "configs/models/lightgbm.yaml"]:
        protected[str(path.relative_to(root)).replace("\\", "/")] = sha256(path)
    frames = []
    for name in ("train", "validation"):
        df = pd.read_parquet(folder / f"{name}.parquet")
        if len(df) != manifest["row_counts"][name]["rows"]:
            raise ValueError(f"Row count mismatch: {name}")
        if df[KEY].isna().any().any() or df.duplicated(KEY).any():
            raise ValueError(f"Missing or duplicate keys: {name}")
        bounds = split["splits"][name]
        q = df["target_quarter"].map(quarter_number)
        if not q.between(quarter_number(bounds["start"]), quarter_number(bounds["end"])).all():
            raise ValueError(f"Target partition mismatch: {name}")
        if not df["split"].eq(name).all():
            raise ValueError(f"Split labels mismatch: {name}")
        if not set(df["country_code"]).issubset(manifest["country_codes"]):
            raise ValueError("Country scope mismatch")
        if not set(df["hs_code"]).issubset(manifest["hs_codes"]):
            raise ValueError("HS scope mismatch")
        for col in ("country_code", "hs_code"):
            if not all(isinstance(x, str) for x in df[col]):
                raise ValueError(f"Identifiers must be strings: {col}")
        oq = df["origin_quarter"].map(quarter_number)
        if not (q - oq).eq(df["horizon_quarters"]).all():
            raise ValueError("Origin/target/horizon mismatch")
        if not df["horizon_quarters"].isin(split["validation"]["horizon_quarters"]).all():
            raise ValueError("Unexpected horizon")
        if not df["currency"].eq("USD").all() or not df["valuation"].eq("FOB").all():
            raise ValueError("Target unit mismatch")
        if not df["target_available"].eq(df[TARGET].notna()).all():
            raise ValueError("Target availability flag mismatch")
        if (df[TARGET].dropna() < 0).any() or not np.isfinite(df[TARGET].dropna().to_numpy(dtype=float)).all():
            raise ValueError("Non-finite/negative actual target")
        df = df.copy()
        df["_target_index"], df["_origin_index"] = q, oq
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(KEY).any():
        raise ValueError("Overlapping split keys")
    labels = combined[combined[TARGET].notna()]
    if labels.groupby(["country_code", "hs_code", "target_quarter"])[TARGET].nunique().gt(1).any():
        raise ValueError("Conflicting actuals across horizons")
    return frames[0], frames[1], manifest, feature, split, model, removed, protected


def training_rows(pool, origin):
    """Calendar availability only; no verified historical publication cutoff."""
    mask = (pool[TARGET].notna() & pool["target_available"].fillna(False)
            & pool["_target_index"].le(quarter_number(origin)))
    result = pool.loc[mask].sort_values(KEY).copy()
    if result.empty or not result["_origin_index"].lt(result["_target_index"]).all():
        raise ValueError(f"Invalid historical training pool at {origin}")
    return result


def prepare_x(df, features, categorical):
    result = pd.DataFrame(index=df.index)
    for col in features:
        if col in categorical:
            result[col] = df[col].astype("string").fillna("__MISSING__").astype(object)
        else:
            result[col] = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(result[col]).any():
                raise ValueError(f"Infinite feature values: {col}")
    return result


def make_pipeline(fit, features, categorical, params):
    # Keep explicit manifest missing flags; do not create duplicate flags using
    # a fit-dependent SimpleImputer.add_indicator feature universe.
    dropped = [x for x in features if x not in categorical and fit[x].isna().all()]
    used = [x for x in features if x not in dropped]
    cats = [x for x in used if x in categorical]
    numeric = [x for x in used if x not in categorical]
    transformer = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cats),
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ], sparse_threshold=1.0, verbose_feature_names_out=True)
    return Pipeline([("transformer", transformer), ("model", lgb.LGBMRegressor(**params))]), used, dropped


def metric_values(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if not len(a):
        return dict(n=0, excluded_nonfinite=int((~valid).sum()), wape=None, mae=None, rmse=None,
                    smape=None, actual_sum=0.0, absolute_error_sum=0.0, zero_wape_denominator=True)
    error = np.abs(a - p)
    denominator = np.abs(a).sum()
    smape = np.divide(200 * error, np.abs(a) + np.abs(p), out=np.zeros(len(a)),
                      where=(np.abs(a) + np.abs(p)) != 0)
    return dict(n=len(a), excluded_nonfinite=int((~valid).sum()),
                wape=float(100 * error.sum() / denominator) if denominator else None,
                mae=float(error.mean()), rmse=float(np.sqrt(np.mean((a-p)**2))),
                smape=float(smape.mean()), actual_sum=float(a.sum()),
                absolute_error_sum=float(error.sum()), zero_wape_denominator=bool(denominator == 0))


def grouped_metrics(predictions, primary):
    rows = []
    for (model, variant), group in predictions.groupby(["model_id", "variant"], sort=True):
        for scope, subset in [("h1_5", group), ("h2_5", group[group.horizon_quarters.isin(primary)])]:
            row = dict(model_id=model, variant=variant, scope=scope, group_by="overall", group_value="all")
            rows.append(row | metric_values(subset[TARGET], subset[PRED]))
            for column in ["country_code", "hs_code", "horizon_quarters"]:
                for value, part in subset.groupby(column, sort=True):
                    row = dict(model_id=model, variant=variant, scope=scope,
                               group_by=column, group_value=str(value))
                    rows.append(row | metric_values(part[TARGET], part[PRED]))
    return pd.DataFrame(rows)


def baseline_predictions(evaluation, pool):
    history = pool.drop_duplicates(["country_code", "hs_code", "target_quarter"])
    lookup = {(r.country_code, r.hs_code, quarter_number(r.target_quarter)): r.target_export_value_usd
              for r in history.itertuples()}
    out = evaluation[KEY + ["horizon_quarters", TARGET]].copy()
    predictions, source_quarters = [], []
    for row in out.itertuples():
        target, origin = quarter_number(row.target_quarter), quarter_number(row.origin_quarter)
        # Most recent same-season observation, including h=5 => target-8.
        candidate = target - 4
        while candidate > origin:
            candidate -= 4
        minimum = pool["_target_index"].min()
        value = np.nan
        while candidate >= minimum:
            value = lookup.get((row.country_code, row.hs_code, candidate), np.nan)
            if pd.notna(value):
                break
            candidate -= 4
        predictions.append(float(value) if pd.notna(value) else np.nan)
        source_quarters.append(f"{candidate // 4}Q{candidate % 4 + 1}" if pd.notna(value) else None)
    out[PRED] = predictions
    out["baseline_source_quarter"] = source_quarters
    out["prediction_status"] = np.where(out[PRED].notna(), "predicted", "no_historical_same_season_label")
    out["model_id"], out["variant"] = "seasonal_naive", "baseline"
    return out


def annual_results(predictions):
    """Two explicit consecutive-four-quarter windows, without mixing origins."""
    cols = ["model_id", "variant", "country_code", "hs_code", "origin_quarter"]
    # Aggregate the same completeness rules in groups once, rather than
    # constructing tens of thousands of small frames for all experiments.
    universe = predictions[cols].drop_duplicates()
    complete, skipped = [], []
    for window, horizons in [("h1_4", {1,2,3,4}), ("h2_5", {2,3,4,5})]:
        block = predictions.loc[predictions.horizon_quarters.isin(horizons)]
        stats = block.groupby(cols, sort=True).agg(
            row_count=(TARGET, "size"), horizon_count=("horizon_quarters", "nunique"),
            actual_count=(TARGET, "count"), prediction_count=(PRED, "count"),
            actual_sum_usd=(TARGET, "sum"), predicted_sum_usd=(PRED, "sum"),
            target_start=("target_quarter", "min"), target_end=("target_quarter", "max"),
        ).reset_index()
        stats = universe.merge(stats, on=cols, how="left", validate="one_to_one")
        valid = stats[["row_count", "horizon_count", "actual_count", "prediction_count"]].eq(4).all(axis=1)
        selected = stats.loc[valid, cols + ["actual_sum_usd", "predicted_sum_usd", "target_start", "target_end"]].copy()
        selected["window"] = window
        complete.append(selected)
        missing = stats.loc[~valid, cols].copy()
        missing["window"], missing["reason"] = window, "incomplete_four_quarters"
        skipped.append(missing)
    sums = pd.concat(complete, ignore_index=True)
    metrics = []
    if not sums.empty:
        for key, block in sums.groupby(["model_id", "variant", "window"]):
            metrics.append(dict(zip(["model_id", "variant", "window"], key)) |
                           metric_values(block.actual_sum_usd, block.predicted_sum_usd))
    return sums, pd.DataFrame(metrics), pd.concat(skipped, ignore_index=True)


def git_state(root):
    def call(*args):
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8")
        return proc.stdout.strip() if proc.returncode == 0 else "unavailable"
    return {"commit": call("rev-parse", "HEAD"), "branch": call("branch", "--show-current"),
            "status": call("status", "--short")}


def resolve_model_params(config, reg_alpha=None, reg_lambda=None):
    """Run-local L1/L2 overrides; never mutate shared model configuration."""
    params = config["params"].copy()
    for name, value in [("reg_alpha", reg_alpha), ("reg_lambda", reg_lambda)]:
        if value is not None:
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            params[name] = float(value)
    return params


def run_validation(root, run_id=None, max_origins=None, reg_alpha=None, reg_lambda=None):
    root = Path(root).resolve()
    train, validation, manifest, feature, split, model, removed, protected = load_inputs(root)
    run_id = run_id or ("lightgbm_v002_initial_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or run_id == "production":
        raise ValueError("Invalid experimental run_id")
    params = resolve_model_params(model, reg_alpha, reg_lambda)
    output = root / "artifacts" / run_id
    output.mkdir(exist_ok=False)
    pool = pd.concat([train, validation], ignore_index=True)
    eligible = validation[TARGET].notna() & validation["target_available"].fillna(False)
    evaluation = validation.loc[eligible].sort_values(KEY).copy()
    origins = sorted(evaluation.origin_quarter.unique())
    if max_origins is not None:
        if max_origins < 1:
            raise ValueError("max_origins must be positive")
        origins = origins[:max_origins]
        evaluation = evaluation[evaluation.origin_quarter.isin(origins)].copy()
    categorical = manifest["feature_contract"]["categorical"]
    variants = model["training"]["comparison_variants"]
    if variants != ["with_macro", "without_macro"]:
        raise ValueError("Expected paired macro experiments")
    metadata = {
        "run_id": run_id, "protocol_id": PROTOCOL, "model_id": "lightgbm",
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_only": max_origins is not None, "origins": origins,
        "dataset_version": manifest["dataset_version"], "protected_checksums": protected,
        "manifest_sha256": sha256(root / "data/shared/v002/manifest.json"),
        "model_params": params, "training_config": model["training"],
        "parameter_overrides": {k: float(v) for k,v in [("reg_alpha",reg_alpha),("reg_lambda",reg_lambda)] if v is not None},
        "split_config": split, "macro_removed_columns": removed,
        "fit_pool_policy": "expanding: train plus earlier validation targets <= current origin; missing targets excluded",
        "historical_asof_verified": False, "origin_cutoff_at": None,
        "availability_limitation": manifest["availability_limitation"],
        "ac_alignment_status": "common expanding pool; cross-model row/fit hashes checked by scripts/evaluate.py; no tuning",
        "git": git_state(root), "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name) for name in
                      ["lightgbm", "scikit-learn", "pandas", "numpy", "scipy", "pyarrow", "joblib", "PyYAML"]},
        "negative_prediction_policy": "clip to zero before scoring",
        "missing_indicators": "use manifest flags; no duplicate auto-generated flags",
        "all_missing_numeric_policy": "drop per fit; save dropped list; retain manifest missing flags",
        "evaluation_rows_sha256": frame_hash(evaluation, KEY),
        "validation_input_rows": len(validation), "validation_target_missing": int((~eligible).sum()),
        "selected_evaluation_rows": len(evaluation), "final_test_used": False,
        "annual_windows": {"h1_4": [1,2,3,4], "h2_5": [2,3,4,5]},
        "annual_interpretation": "consecutive four-quarter sums, not necessarily calendar year",
    }
    write_json(output / "run_metadata.json", metadata)
    installed = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                        if d.metadata['Name'] and d.metadata['Name'].lower() not in
                        {"oinq-export-forecast", "pip", "setuptools"}})
    (output / "requirements-lock.txt").write_text("\n".join(installed) + "\n", encoding="utf-8")
    evaluation[KEY + ["horizon_quarters"]].to_parquet(output / "evaluation_rows.parquet", index=False)
    exclusions = validation.loc[~eligible, KEY + ["horizon_quarters", "target_available"]].copy()
    exclusions["reason"] = "missing_actual_target"
    exclusions.to_parquet(output / "evaluation_exclusions.parquet", index=False)
    code = output / "code_snapshot"
    code.mkdir()
    for path in [Path(__file__), root / "scripts/train_lightgbm.py", root / "tests/test_lightgbm_training.py"]:
        if path.exists():
            (code / path.name).write_bytes(path.read_bytes())
    write_json(code / "checksums.json", {p.name: sha256(p) for p in code.iterdir() if p.is_file()})
    # Copies are run-local evidence; original configuration stays untouched.
    for name, cfg in [("model_config", model), ("features_config", feature), ("manifest", manifest)]:
        write_json(output / f"{name}.json", cfg)
    fits, predictions = [], []
    for number, origin in enumerate(origins, 1):
        fit = training_rows(pool, origin)
        scoring = evaluation[evaluation.origin_quarter.eq(origin)].copy()
        fit_dir = output / "fit_rows"
        fit_dir.mkdir(exist_ok=True)
        fit[KEY + ["horizon_quarters", "split"]].to_parquet(fit_dir / f"{origin}.parquet", index=False)
        counts = {"origin_quarter": origin, "fit_rows": len(fit),
                  "past_train_rows": int(fit.split.eq("train").sum()),
                  "past_validation_rows": int(fit.split.eq("validation").sum()),
                  "min_fit_target": fit.target_quarter.min(), "max_fit_target": fit.target_quarter.max(),
                  "future_label_rows_excluded": int((pool._target_index > quarter_number(origin)).sum()),
                  "missing_label_rows_excluded": int((pool[TARGET].isna() &
                      pool._target_index.le(quarter_number(origin))).sum()),
                  "fit_rows_sha256": frame_hash(fit, KEY), "score_rows": len(scoring),
                  "origin_cutoff_at": None, "historical_release_verified": False}
        for variant in variants:
            features = [x for x in manifest["feature_columns"] if variant == "with_macro" or x not in removed]
            pipe, used, dropped = make_pipeline(fit, features, categorical, params)
            x_fit, x_score = prepare_x(fit, used, categorical), prepare_x(scoring, used, categorical)
            pipe.fit(x_fit, fit[TARGET].to_numpy(dtype=float))
            raw = pipe.predict(x_score)
            if not np.isfinite(raw).all():
                raise ValueError("Non-finite model predictions")
            pred = np.maximum(raw, feature["target"]["minimum_prediction"])
            path = output / variant / origin
            path.mkdir(parents=True)
            joblib.dump(pipe, path / "pipeline.joblib", compress=3)
            # Native Windows file APIs can reject the Korean workspace path.
            # Serialize in memory and let Python write the exact model text.
            (path / "model.txt").write_text(pipe.named_steps["model"].booster_.model_to_string(), encoding="utf-8")
            transformed = pipe.named_steps["transformer"].get_feature_names_out().tolist()
            write_json(path / "input_columns.json", {"manifest_selected": features, "raw_input_order": used,
                        "dropped_all_missing": dropped, "transformed_order": transformed,
                        "categorical": [x for x in categorical if x in used]})
            restored = joblib.load(path / "pipeline.joblib")
            replay = np.maximum(restored.predict(x_score), 0)
            if not np.array_equal(pred, replay):
                raise AssertionError("Saved pipeline replay differs")
            write_json(path / "fit_metadata.json", counts | {"variant": variant, "selected_features": len(features),
                        "raw_features": len(used), "transformed_features": len(transformed),
                        "reload_max_abs_diff": float(np.max(np.abs(pred-replay))),
                        "negative_raw_predictions": int((raw < 0).sum())})
            part = scoring[KEY + ["horizon_quarters", TARGET, "target_available"]].copy()
            part[PRED], part["raw_predicted_usd"] = pred, raw
            part["model_id"], part["variant"], part["prediction_status"] = "lightgbm", variant, "predicted"
            part["run_id"], part["dataset_version"] = run_id, manifest["dataset_version"]
            part["origin_cutoff_at"] = None
            predictions.append(part)
            fits.append(counts | {"variant": variant})
        print(f"[{number}/{len(origins)}] {origin}: fit={len(fit)} score={len(scoring)} both variants replay OK", flush=True)
    learned = pd.concat(predictions, ignore_index=True)
    for variant in variants:
        block = learned[learned.variant.eq(variant)]
        if frame_hash(block, KEY) != metadata["evaluation_rows_sha256"]:
            raise AssertionError("Paired evaluation keys differ")
    baseline = baseline_predictions(evaluation, pool)
    baseline["run_id"], baseline["dataset_version"] = run_id, manifest["dataset_version"]
    all_predictions = pd.concat([learned, baseline], ignore_index=True)
    all_predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    all_predictions.to_csv(output / "validation_predictions.csv", index=False, encoding="utf-8-sig", na_rep="\\N")
    metrics = grouped_metrics(all_predictions, split["validation"]["primary_horizons"])
    metrics.to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    # Baseline intersections reported separately; missing baseline predictions
    # never silently remove valid rows from the paired LightGBM comparison.
    baseline_keys = baseline[baseline[PRED].notna()][KEY]
    common = all_predictions.merge(baseline_keys, on=KEY, validate="many_to_one")
    grouped_metrics(common, split["validation"]["primary_horizons"]).to_csv(
        output / "metrics_baseline_common_rows.csv", index=False, encoding="utf-8-sig")
    sums, annual, excluded = annual_results(all_predictions)
    sums.to_parquet(output / "four_quarter_sums.parquet", index=False)
    annual.to_csv(output / "four_quarter_metrics.csv", index=False, encoding="utf-8-sig")
    excluded.to_parquet(output / "four_quarter_exclusions.parquet", index=False)
    pd.DataFrame(fits).to_csv(output / "fit_audit.csv", index=False, encoding="utf-8-sig")
    for relative, expected in protected.items():
        if sha256(root / relative) != expected:
            raise AssertionError(f"Protected input changed: {relative}")
    summary = metrics[(metrics.group_by == "overall") & (metrics.scope == "h2_5")]
    metadata.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    fitted_pipelines=len(fits), paired_rows_verified=True, reload_verified=True,
                    protected_inputs_unchanged=True, baseline_missing_predictions=int(baseline[PRED].isna().sum()))
    write_json(output / "run_metadata.json", metadata)
    (output / "README.md").write_text(run_readme(root, run_id, summary, params), encoding="utf-8")
    checksums = {str(p.relative_to(output)).replace("\\", "/"): sha256(p)
                 for p in output.rglob("*") if p.is_file() and p.name != "output_checksums.json"}
    write_json(output / "output_checksums.json", checksums)
    print(summary[["model_id", "variant", "n", "wape", "mae", "rmse", "smape"]].to_string(index=False), flush=True)
    print(f"Saved: {output}", flush=True)
    return output


def run_readme(root, run_id, summary, params):
    return f'''# LightGBM first validation: {run_id}

This is a current-revised-snapshot experiment, NOT a historically reproduced publication-as-of backtest.
No 2026 final-test data, clustering data or operational predictions were read/generated.
Cross-model row/fit agreement is checked separately by scripts/evaluate.py; no tuning has been performed.

## Re-run from repository root (PowerShell)
```
.\\.venv\\Scripts\\python.exe scripts/train_lightgbm.py --run-id NEW_UNIQUE_RUN_ID --reg-alpha {params.get('reg_alpha', 0.0)} --reg-lambda {params.get('reg_lambda', 0.0)}
```
Smoke check: append `--max-origins 2`. A smoke run is not full validation.
Use Python 3.12 and versions recorded in run_metadata.json. Run ID directories must not already exist.
Install the recorded environment with `python -m pip install -r artifacts/{run_id}/requirements-lock.txt`.
Source snapshots and configuration snapshots are included. Model code lives in src/oinq/modeling/supervised/lightgbm.py.
Effective model parameters (shared YAML is unchanged): {params}

## Protocol
Target partitions stay 2013Q1-2021Q4 train / 2022Q1-2025Q4 validation.
At each validation origin, fit on train plus earlier validation rows whose target <= origin and target is available.
Preprocessing is refitted at that origin. This calendar constraint does not establish historical publication availability.
All horizons are pooled with horizon/target calendar features. Preserve duplicate target values across distinct supervised keys;
do not sum supervised rows as unique trade. With/without macro share the exact scoring keys; 6 macro values and 6 flags are removed.
Explicit manifest missing flags are retained, all-null numeric values dropped per fit, unknown categories ignored at prediction.
Negative predictions are clipped to zero for scoring. Validation early stopping is disabled.

## Files
- validation_predictions: per-row actuals/predictions, country, HS, origin, target, horizon, variant; baseline included.
- evaluation_rows / evaluation_exclusions: fixed eligible keys / target-missing rows and reasons.
- fit_rows and fit_audit.csv: per-origin training keys, original partition, cutoff and excluded counts.
- metrics.csv: all eligible rows; h1_5 and h2_5, overall / country / HS / horizon.
- metrics_baseline_common_rows.csv: strict common rows with available seasonal baseline, for fair baseline comparison.
- four_quarter_*: complete consecutive h1-4 / h2-5 windows per country/HS/origin, not necessarily calendar-year totals.
- with_macro, without_macro: per-origin pipeline.joblib, native model, raw/transformed input orders, fit/reload metadata.
- run_metadata.json / output_checksums.json: parameters, seed, versions, input hashes, protocol and output integrity.

## Load without refitting
```python
import json, joblib, pandas as pd
from oinq.modeling.supervised.lightgbm import prepare_x
from pathlib import Path
p = Path("artifacts/{run_id}/with_macro/2020Q4")
cols = json.loads((p / "input_columns.json").read_text(encoding="utf-8"))
model = joblib.load(p / "pipeline.joblib")
rows = pd.read_parquet("data/shared/v002/validation.parquet")
rows = rows[rows.origin_quarter.eq("2020Q4")]
predicted_usd = model.predict(prepare_x(rows, cols["raw_input_order"], cols["categorical"])).clip(min=0)
```
Run from an environment where `src` is importable (the entry script inserts it); for interactive use set PYTHONPATH=src.
Each saved pipeline was reloaded and checked against all of its scoring rows; exact prediction replay was required.

## Primary h2-5 summary
```
{summary[["model_id", "variant", "n", "wape", "mae", "rmse", "smape"]].to_string(index=False)}
```
WAPE/sMAPE: percent; MAE/RMSE: USD. Undefined zero-denominator WAPE is null, both-zero sMAPE is zero.
Overall WAPE is computed from pooled errors and actuals, not the mean of country WAPEs.
Reports remain here until approved for reports/. No git push/PR/merge was performed.
'''
