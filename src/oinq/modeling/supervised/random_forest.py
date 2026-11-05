"""Random Forest validation and frozen-origin evaluation on shared GEM data."""
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import platform
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from oinq.common.config import load_yaml
from oinq.modeling.loader import load_model
from oinq.modeling.supervised.lightgbm import (
    KEY, PRED, TARGET, frame_hash, grouped_metrics, load_inputs, prepare_x,
    sha256, training_rows, write_json,
)
from oinq.modeling.supervised.lightgbm_final_test import (
    check_test_inputs, select_scored_rows, test_metrics,
)

MODEL_ID = "random_forest"
VARIANTS = ("with_macro", "without_macro")


def feature_sets(manifest, features_config):
    columns = manifest["feature_columns"]
    macro = features_config["supervised"]["macro"]["columns"]
    excluded = set(macro) | {f"{name}_missing" for name in macro}
    if len(macro) != 6 or len(excluded) != 12 or not excluded.issubset(columns):
        raise ValueError("Expected six macro values and their six missing indicators")
    return {
        "with_macro": list(columns),
        "without_macro": [column for column in columns if column not in excluded],
    }


def make_pipeline(features, categorical, params):
    numeric = [column for column in features if column not in categorical]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(**params)),
    ])


def fit_pipeline(rows, features, categorical, params):
    pipeline = make_pipeline(features, categorical, params)
    pipeline.fit(prepare_x(rows, features, categorical), rows[TARGET].to_numpy(dtype=float))
    # A fixed prediction order makes reload verification bit-for-bit reproducible.
    pipeline.set_params(model__n_jobs=1)
    return pipeline


def predict_rows(pipeline, rows, features, categorical, minimum):
    raw = pipeline.predict(prepare_x(rows, features, categorical))
    if not np.isfinite(raw).all():
        raise ValueError("Random Forest returned non-finite predictions")
    return np.maximum(raw, minimum)


def prediction_frame(rows, values, variant):
    result = rows[KEY + ["horizon_quarters", TARGET]].copy()
    result[PRED] = values
    return result.assign(model_id=MODEL_ID, variant=variant)


def add_r2(metrics, predictions, horizons_by_scope):
    metrics = metrics.copy()
    metrics["r2"] = np.nan
    for index, row in metrics.iterrows():
        subset = predictions.loc[
            predictions["variant"].eq(row["variant"])
            & predictions["horizon_quarters"].isin(horizons_by_scope[row["scope"]])
        ]
        if row["group_by"] != "overall":
            subset = subset.loc[subset[row["group_by"]].astype(str).eq(row["group_value"])]
        if len(subset) >= 2 and subset[TARGET].nunique() > 1:
            metrics.at[index, "r2"] = r2_score(subset[TARGET], subset[PRED])
    return metrics


def save_pipeline(path, pipeline, features, categorical, dataset_version, last_quarter, sample):
    before = pipeline.predict(sample)
    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": features,
        "categorical_columns": categorical,
        "dataset_version": dataset_version,
        "trained_through_target_quarter": last_quarter,
    }, path, compress=3)
    loaded = load_model(path)
    np.testing.assert_array_equal(before, loaded["model"].predict(sample))


def run_training(root, run_id=None, *, publish=True):
    root = Path(root).resolve()
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run ID")
    shared = root / "artifacts/shared_models" / MODEL_ID
    destinations = {variant: shared / variant / "pipeline.joblib" for variant in VARIANTS}
    if publish and any(path.exists() for path in destinations.values()):
        raise FileExistsError("Shared models already exist. Use --no-publish for a private rerun.")
    output = root / ".private" / MODEL_ID / run_id
    output.mkdir(parents=True, exist_ok=False)

    train, validation, manifest, features_config, split, _, _, protected = load_inputs(root)
    config_path = root / "configs/models/random_forest.yaml"
    config = load_yaml(config_path)
    if tuple(config["training"]["comparison_variants"]) != VARIANTS:
        raise ValueError("Both macro variants are required")
    candidates = {
        name: config["params"] | overrides
        for name, overrides in config["training"]["candidates"].items()
    }
    features = feature_sets(manifest, features_config)
    categorical = manifest["feature_contract"]["categorical"]
    primary = split["validation"]["primary_horizons"]
    minimum = features_config["target"]["minimum_prediction"]
    pool = pd.concat([train, validation], ignore_index=True)
    evaluation, validation_missing = select_scored_rows(validation.sort_values(KEY))
    validation_predictions = []
    fit_audit = []
    for origin, scoring in evaluation.groupby("origin_quarter", sort=True):
        fit = training_rows(pool, origin)
        fit_audit.append({
            "origin_quarter": origin, "fit_rows": len(fit),
            "last_target": fit.target_quarter.max(), "fit_rows_sha256": frame_hash(fit, KEY),
        })
        print(f"Validation origin {origin} ({len(fit)} training rows)", flush=True)
        for variant, columns in features.items():
            for name, params in candidates.items():
                pipeline = fit_pipeline(fit, columns, categorical, params)
                values = predict_rows(pipeline, scoring, columns, categorical, minimum)
                validation_predictions.append(
                    prediction_frame(scoring, values, variant).assign(candidate=name)
                )

    predictions = pd.concat(validation_predictions, ignore_index=True)
    metrics = pd.concat([
        add_r2(grouped_metrics(part, primary), part, {
            "h1_5": split["validation"]["horizon_quarters"], "h2_5": primary,
        }).assign(candidate=name)
        for name, part in predictions.groupby("candidate", sort=True)
    ], ignore_index=True)
    ranking = metrics.loc[
        metrics.scope.eq("h2_5") & metrics.group_by.eq("overall")
    ].sort_values(["wape", "candidate"])
    if ranking.wape.isna().any():
        raise ValueError("Validation WAPE is undefined")
    selected = ranking.drop_duplicates("variant").set_index("variant")["candidate"].to_dict()
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    write_json(output / "selection_lock.json", {
        "criterion": "Lowest validation h=2–5 WAPE per variant; ties by candidate name",
        "selected": {variant: {"candidate": name, "params": candidates[name]}
                     for variant, name in selected.items()},
        "preferred_variant": ranking.iloc[0]["variant"],
    })

    spec = split["final_test"]
    fit = training_rows(pool, spec["origin_quarter"])
    fitted = {
        variant: fit_pipeline(fit, columns, categorical, candidates[selected[variant]])
        for variant, columns in features.items()
    }
    test_item = next(item for item in manifest["files"] if item["file"] == "final_test.parquet")
    test_path = root / test_item["path"]
    protected[test_item["path"]] = test_item["sha256"]
    if sha256(test_path) != test_item["sha256"]:
        raise ValueError("Final-test checksum mismatch")
    test = pd.read_parquet(test_path)
    check_test_inputs(test, spec, manifest["country_codes"], manifest["hs_codes"])
    if (len(test) != manifest["row_counts"]["final_test"]["rows"]
            or not test["split"].eq("final_test").all()
            or not test["currency"].eq("USD").all() or not test["valuation"].eq("FOB").all()):
        raise ValueError("Final-test partition contract mismatch")
    scoring, test_missing = select_scored_rows(test.sort_values(KEY))
    final_predictions, final_metrics, saved_paths = [], [], {}
    for variant, pipeline in fitted.items():
        columns = features[variant]
        values = predict_rows(pipeline, scoring, columns, categorical, minimum)
        result = prediction_frame(scoring, values, variant)
        final_predictions.append(result)
        final_metrics.append(add_r2(test_metrics(result, MODEL_ID, primary, variant=variant), result, {
            "h1_3": spec["horizon_quarters"], "h2_3": primary,
        }))
        path = output / f"{variant}.joblib"
        save_pipeline(path, pipeline, columns, categorical, manifest["dataset_version"],
                      fit.target_quarter.max(), prepare_x(scoring, columns, categorical))
        saved_paths[variant] = path

    pd.concat(final_predictions).to_csv(output / "final_test_predictions.csv", index=False)
    pd.concat(final_metrics).to_csv(output / "final_test_metrics.csv", index=False)
    pd.DataFrame(fit_audit).to_csv(output / "validation_fit_audit.csv", index=False)
    for name, digest in protected.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Protected input changed during training: {name}")
    write_json(output / "run_metadata.json", {
        "dataset_version": manifest["dataset_version"], "protected_checksums": protected,
        "availability_limitation": manifest["availability_limitation"],
        "split_config": split, "candidates": candidates, "prediction_n_jobs": 1,
        "evaluation_rows_sha256": frame_hash(evaluation, KEY),
        "feature_columns": features, "categorical_columns": categorical,
        "validation_excluded_missing": len(validation_missing),
        "final_test_excluded_missing": len(test_missing), "final_fit_rows": len(fit),
        "trained_through_target_quarter": fit.target_quarter.max(), "reload_exact": True,
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in (
            "scikit-learn", "numpy", "pandas", "pyarrow", "scipy", "joblib", "PyYAML"
        )},
    })
    if publish:
        published = []
        try:
            for variant, path in destinations.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(saved_paths[variant], path)
                published.append(path)
        except OSError:
            for path in published:
                path.unlink()
            raise
    return output
