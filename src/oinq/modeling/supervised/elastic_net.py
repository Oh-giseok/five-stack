"""ElasticNet expanding-origin selection, frozen test and deployable pipelines.

Run from the repository root using scripts/train_elastic_net.py.
Only model files are written under artifacts/shared_models; all diagnostics are private.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import re
import warnings
from sklearn.exceptions import ConvergenceWarning
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from oinq.common.config import load_yaml, quarter_number

MODEL_ID = "elastic_net"
TARGET = "target_export_value_usd"
PRED = "predicted_export_value_usd"
KEY = ["country_code", "hs_code", "origin_quarter", "target_quarter"]
VARIANTS = ("with_macro", "without_macro")


def _json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str,
                                     allow_nan=False), encoding="utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_columns(df, columns, name):
    missing = set(columns).difference(df.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def load_inputs(root):
    """Read, but never modify, the released data and team configuration."""
    root = Path(root)
    shared = root / "data/shared/v002"
    manifest = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    features_cfg = load_yaml(root / "configs/features.yaml")
    split_cfg = load_yaml(root / "configs/splits.yaml")
    config = load_yaml(root / "configs/models/elastic_net.yaml")
    if manifest.get("dataset_version") != "v002" or not manifest.get("dataset_release_approved"):
        raise ValueError("Expected released and approved dataset v002")
    if split_cfg["validation"]["method"] != "expanding_origin" or split_cfg.get("shuffle"):
        raise ValueError("Only non-shuffled expanding-origin evaluation is supported")
    if manifest["feature_contract"]["target"] != TARGET:
        raise ValueError("Unexpected target")
    features = manifest["feature_columns"]
    fc = manifest["feature_contract"]
    if features != fc["categorical"] + fc["known_future"] + fc["numeric_candidates"]:
        raise ValueError("Feature order does not match manifest")
    if len(set(features)) != len(features) or set(features) & set(fc["forbidden_feature_columns"]):
        raise ValueError("Forbidden or duplicate feature")
    macros = features_cfg["supervised"]["macro"]["columns"]
    removed = [c for c in features if c in macros or c in {f"{m}_missing" for m in macros}]
    if len(macros) != 6 or len(removed) != 12:
        raise ValueError("Expected six macro variables and six missing flags")
    if config.get("model_id") != MODEL_ID or config["training"]["variants"] != list(VARIANTS):
        raise ValueError("ElasticNet YAML model/variant mismatch")
    tables = {}
    manifest_files = {x["path"].replace("\\", "/"): x for x in manifest["files"]}
    for name in ("train", "validation", "final_test"):
        relative = f"data/shared/v002/{name}.parquet"
        path = root / relative
        record = manifest_files.get(relative)
        if record is None or _sha256(path) != record["sha256"]:
            raise ValueError(f"Released checksum mismatch: {relative}")
        df = pd.read_parquet(path)
        _require_columns(df, features + KEY + [TARGET, "target_available", "split", "horizon_quarters"], name)
        if len(df) != manifest["row_counts"][name]["rows"]:
            raise ValueError(f"Unexpected {name} row count")
        if df[KEY].isna().any().any() or df.duplicated(KEY).any():
            raise ValueError(f"Missing or repeated {name} keys")
        if not df["split"].eq(name).all():
            raise ValueError(f"{name} partition labels differ")
        bound = split_cfg["splits"][name]
        q = df.target_quarter.map(quarter_number)
        o = df.origin_quarter.map(quarter_number)
        if not q.between(quarter_number(bound["start"]), quarter_number(bound["end"])).all():
            raise ValueError(f"{name} target quarters outside split")
        if not (q - o).eq(df.horizon_quarters).all():
            raise ValueError(f"{name}: origin/target/horizon inconsistency")
        if not df.horizon_quarters.isin(split_cfg["validation"]["horizon_quarters"]).all():
            raise ValueError(f"{name}: unexpected horizon")
        if not df.target_available.eq(df[TARGET].notna()).all():
            raise ValueError(f"{name}: availability does not match target")
        if (df[TARGET].dropna() < 0).any() or not np.isfinite(df[TARGET].dropna().to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: invalid target")
        for c in ("country_code", "hs_code"):
            if not all(isinstance(x, str) for x in df[c]):
                raise ValueError(f"{name}: {c} must be string")
        df = df.copy()
        df["_target_index"], df["_origin_index"] = q, o
        tables[name] = df
    return tables, manifest, features_cfg, split_cfg, config, removed


def training_rows(pool, origin):
    """Require actual label at or before origin; calendar availability only."""
    selected = pool.loc[pool[TARGET].notna() & pool.target_available.fillna(False)
                        & pool._target_index.le(quarter_number(origin))].copy()
    selected = selected.sort_values(KEY)
    if selected.empty or not selected._origin_index.lt(selected._target_index).all():
        raise ValueError(f"Invalid training pool at {origin}")
    return selected


def prepare_x(df, feature_columns, categorical_columns):
    """Preserve manifest order and safe numeric NaNs; do not fit preprocessing."""
    _require_columns(df, feature_columns, "input")
    x = pd.DataFrame(index=df.index)
    for col in feature_columns:
        if col in categorical_columns:
            x[col] = df[col].astype("string").fillna("__MISSING__").astype(object)
        else:
            x[col] = pd.to_numeric(df[col], errors="raise").to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(x[col].to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite feature: {col}")
    return x


def make_pipeline(feature_columns, categorical_columns, params):
    """Constant imputation also keeps all-missing numeric columns fit-safe."""
    cat = [c for c in feature_columns if c in categorical_columns]
    num = [c for c in feature_columns if c not in categorical_columns]
    if not cat or not num:
        raise ValueError("Categorical and numeric features are both required")
    preprocessor = ColumnTransformer([
        ("cat", Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0,
                                                    keep_empty_features=True)),
                          ("scaler", StandardScaler())]), num),
    ], remainder="drop", sparse_threshold=0.0)
    return Pipeline([("preprocessor", preprocessor),
                     ("model", ElasticNet(**params))])


def metric_values(actual, predicted):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    a, p = a[valid], p[valid]
    if not len(a):
        return {"n": 0, "excluded_nonfinite": int((~valid).sum()),
                "wape": None, "mae": None, "rmse": None, "smape": None}
    err = np.abs(a - p)
    denom = np.abs(a).sum()
    smape = np.divide(200 * err, np.abs(a) + np.abs(p), out=np.zeros(len(a)),
                      where=(np.abs(a) + np.abs(p)) != 0)
    return {"n": int(len(a)), "excluded_nonfinite": int((~valid).sum()),
            "wape": float(100 * err.sum() / denom) if denom else None,
            "mae": float(err.mean()), "rmse": float(np.sqrt(np.mean((a-p)**2))),
            "smape": float(smape.mean())}


def grouped_metrics(frame, horizons):
    rows = []
    for scope, block in (("h1_5", frame), ("h2_5", frame[frame.horizon_quarters.isin(horizons)])):
        for col, groups in (("overall", [("all", block)]),
                            *[(c, list(block.groupby(c, sort=True))) for c in
                              ("country_code", "hs_code", "horizon_quarters")]):
            for value, section in groups:
                rows.append({"model_id": MODEL_ID, "variant": str(frame.variant.iloc[0]),
                             "scope": scope, "group_by": col, "group_value": str(value),
                             **metric_values(section[TARGET], section[PRED])})
    return pd.DataFrame(rows)


def _predict(pipe, scoring, features, categorical, variant):
    x = prepare_x(scoring, features, categorical)
    raw = np.asarray(pipe.predict(x), dtype=float)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite prediction")
    result = scoring[KEY + ["horizon_quarters", TARGET]].copy()
    result["raw_predicted_usd"] = raw
    result[PRED] = np.maximum(raw, 0)
    result["negative_raw_prediction"] = raw < 0
    result["variant"] = variant
    result["model_id"] = MODEL_ID
    return result


def _fit_and_predict(fit, scoring, features, cats, params, variant):
    model = make_pipeline(features, cats, params)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(prepare_x(fit, features, cats), fit[TARGET].to_numpy(dtype=float))
    model._convergence_warnings = sum(issubclass(w.category, ConvergenceWarning) for w in recorded)
    return model, _predict(model, scoring, features, cats, variant)


def _params(config):
    base = {k: config["params"][k] for k in ("max_iter", "tol", "selection", "random_state")
            if k in config["params"]}
    candidates = [{**base, "alpha": float(a), "l1_ratio": float(r)}
                  for a, r in itertools.product(config["search"]["alpha"], config["search"]["l1_ratio"])]
    if not candidates or any(p["alpha"] <= 0 or not 0 <= p["l1_ratio"] <= 1 for p in candidates):
        raise ValueError("Invalid ElasticNet parameter grid")
    return candidates


def _make_run_dir(root, run_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Unsafe run ID")
    path = root / ".private/elastic_net" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_bundle(path, pipe, features, cats, trained_quarter, x_check):
    if path.exists():
        raise FileExistsError(f"Shared model already exists: {path}")
    bundle = {"pipeline": pipe, "feature_columns": list(features),
              "categorical_columns": list(cats), "dataset_version": "v002",
              "trained_through_target_quarter": trained_quarter}
    before = np.asarray(pipe.predict(x_check), dtype=float)
    if not np.isfinite(before).all():
        raise ValueError("Nonfinite saved-model check predictions")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(bundle, path, compress=3)
        restored = joblib.load(path)
        if restored["feature_columns"] != features or restored["categorical_columns"] != cats:
            raise AssertionError("Reloaded feature metadata differs")
        after = np.asarray(restored["pipeline"].predict(x_check), dtype=float)
        np.testing.assert_array_equal(before, after)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return True


def run_training(root, run_id=None, max_origins=None, publish=True):
    root = Path(root).resolve()
    tables, manifest, feature_cfg, splits, config, removed = load_inputs(root)
    run_id = run_id or "elastic_net_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    private = _make_run_dir(root, run_id)
    metadata = {"run_id": run_id, "model_id": MODEL_ID, "dataset_version": "v002",
                "historical_asof_verified": False,
                "note": "Calendar-only label availability; publication-at-origin not independently verified.",
                "selection_split": "validation", "final_test_for_tuning": False,
                "deployment_uses_latest_actuals_after_frozen_test": True,
                "published": False}
    _json(private / "run_metadata.json", metadata)
    train, validation, test = (tables[k] for k in ("train", "validation", "final_test"))
    pool = pd.concat([train, validation], ignore_index=True)
    categorical = manifest["feature_contract"]["categorical"]
    primary = splits["validation"]["primary_horizons"]
    eligible_val = validation[TARGET].notna() & validation.target_available.fillna(False)
    val = validation.loc[eligible_val].copy()
    origins = sorted(val.origin_quarter.unique(), key=quarter_number)
    if max_origins is not None:
        if max_origins < 1:
            raise ValueError("max_origins must be positive")
        origins = origins[:max_origins]
        val = val[val.origin_quarter.isin(origins)].copy()
        publish = False  # smoke test cannot create a production model
    # Select on the same eligible rows/origins for all candidate parameters.
    if val.empty:
        raise ValueError("No validation rows")
    candidates = _params(config)
    search_rows, winning = [], {}
    for variant in VARIANTS:
        features = [c for c in manifest["feature_columns"] if variant == "with_macro" or c not in removed]
        trial_results = []
        for i, params in enumerate(candidates):
            portions = []
            convergence_warnings = 0
            for origin in origins:
                fit = training_rows(pool, origin)
                score = val[val.origin_quarter.eq(origin)]
                model, predicted = _fit_and_predict(fit, score, features, categorical, params, variant)
                portions.append(predicted)
                convergence_warnings += model._convergence_warnings
                print(f"[{variant}] candidate {i+1}/{len(candidates)}, origin {origin}: "
                      f"fit={len(fit)}, score={len(score)}", flush=True)
            predictions = pd.concat(portions, ignore_index=True)
            primary_rows = predictions[predictions.horizon_quarters.isin(primary)]
            score = metric_values(primary_rows[TARGET], primary_rows[PRED])
            if score["n"] != len(primary_rows) or score["wape"] is None:
                raise ValueError("Undefined or incomplete primary validation metric")
            record = {"variant": variant, "candidate": i, **params, **score,
                      "convergence_warnings": convergence_warnings,
                      "negative_raw_predictions": int(predictions.negative_raw_prediction.sum())}
            search_rows.append(record)
            trial_results.append((convergence_warnings, score["wape"], i, params, predictions))
        converged = [r for r in trial_results if r[0] == 0]
        if not converged:
            raise ValueError(f"All {variant} parameter candidates had convergence warnings; "
                             "increase max_iter/tol or revise search grid in elastic_net.yaml")
        _, _, _, best_params, best_predictions = min(converged, key=lambda v: (v[1], v[2]))
        winning[variant] = (features, best_params, best_predictions)
    pd.DataFrame(search_rows).to_csv(private / "parameter_search.csv", index=False)
    all_val = []
    for variant, (features, params, predicted) in winning.items():
        all_val.append(predicted)
        grouped_metrics(predicted, primary).to_csv(private / f"validation_metrics_{variant}.csv",
                                                   index=False, encoding="utf-8-sig")
    pd.concat(all_val, ignore_index=True).to_parquet(private / "validation_predictions.parquet", index=False)
    # The smoke path stops before final test or publishing.
    if not publish:
        metadata.update(status="validation_only", selected_params={v: p for v, (_, p, _) in winning.items()})
        _json(private / "run_metadata.json", metadata)
        return private
    test_origin = splits["final_test"]["origin_quarter"]
    if test_origin != "2025Q4":
        raise ValueError("Unexpected frozen test origin")
    eligible_test = (test.origin_quarter.eq(test_origin) & test[TARGET].notna()
                     & test.target_available.fillna(False)
                     & test.target_quarter.isin(splits["final_test"]["target_quarters"]))
    test_rows = test.loc[eligible_test].copy()
    if test_rows.empty:
        raise ValueError("No frozen test labels")
    fit_test = training_rows(pool, test_origin)
    all_test, test_summaries = [], {}
    for variant, (features, params, _) in winning.items():
        pipe, predicted = _fit_and_predict(fit_test, test_rows, features, categorical, params, variant)
        if pipe._convergence_warnings:
            raise ValueError(f"Frozen test fit did not converge: {variant}")
        all_test.append(predicted)
        grouped_metrics(predicted, primary).to_csv(private / f"final_test_metrics_{variant}.csv",
                                                   index=False, encoding="utf-8-sig")
        test_summaries[variant] = metric_values(predicted[TARGET], predicted[PRED]) | {
            "negative_raw_predictions": int(predicted.negative_raw_prediction.sum())}
    pd.concat(all_test, ignore_index=True).to_parquet(private / "final_test_predictions.parquet", index=False)
    # Only AFTER recording the frozen final-test results, fit through latest actuals.
    latest_pool = pd.concat([pool, test], ignore_index=True)
    fit_deploy = training_rows(latest_pool, "2026Q3")
    trained_through = max(fit_deploy.target_quarter.unique(), key=quarter_number)
    if trained_through != "2026Q3":
        raise ValueError("Deployment labels do not reach 2026Q3")
    files = {}
    # Avoid half-publishing when the other shared model path already exists.
    for variant in VARIANTS:
        p = root / "artifacts/shared_models" / MODEL_ID / variant / "pipeline.joblib"
        if p.exists():
            raise FileExistsError(f"Existing shared model is protected: {p}")
        files[variant] = p
    created = []
    try:
        for variant, (features, params, _) in winning.items():
            pipe = make_pipeline(features, categorical, params)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always", ConvergenceWarning)
                pipe.fit(prepare_x(fit_deploy, features, categorical),
                         fit_deploy[TARGET].to_numpy(dtype=float))
            if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
                raise ValueError(f"Deployment fit did not converge: {variant}")
            x_check = prepare_x(test_rows.head(12), features, categorical)
            _save_bundle(files[variant], pipe, features, categorical, trained_through, x_check)
            created.append(files[variant])
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    metadata.update(status="completed", published=True, saved_paths={v: str(p) for v, p in files.items()},
                    selected_params={v: p for v, (_, p, _) in winning.items()},
                    validation_primary={v: metric_values(pred[pred.horizon_quarters.isin(primary)][TARGET],
                                                         pred[pred.horizon_quarters.isin(primary)][PRED]) | {
                        "negative_raw_predictions": int(pred.negative_raw_prediction.sum())}
                                        for v, (_, _, pred) in winning.items()},
                    frozen_test=test_summaries, trained_through_target_quarter=trained_through,
                    pipeline_reload_exact=True)
    _json(private / "run_metadata.json", metadata)
    print(json.dumps({"validation_primary": metadata["validation_primary"],
                      "frozen_test": test_summaries, "saved_paths": metadata["saved_paths"]},
                     ensure_ascii=False, indent=2), flush=True)
    return private
