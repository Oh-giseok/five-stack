"""Expanding-origin linear baselines with a frozen test and separate deployment fits."""
from importlib.metadata import version
import os
from pathlib import Path
import platform
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from oinq.common.config import load_yaml, quarter_number
from oinq.modeling.supervised.lightgbm import (
    KEY, PRED, TARGET, frame_hash, grouped_metrics, load_inputs, prepare_x,
    sha256, training_rows, write_json,
)
from oinq.modeling.supervised.lightgbm_final_test import (
    check_test_inputs, select_scored_rows, test_metrics,
)
from oinq.modeling.supervised.random_forest import add_r2, feature_sets, save_pipeline

MODELS = ("linear_regression", "ridge")


def make_pipeline(columns, cats, model_id, config, alpha=None):
    numeric = [column for column in columns if column not in cats]
    processor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric),
    ])
    if model_id == "linear_regression":
        model = LinearRegression(**config["linear_regression"])
    else:
        model = Ridge(alpha=alpha, solver=config["ridge"]["solver"])
    return Pipeline([("preprocessor", processor), ("model", model)])


def fit_pipeline(rows, columns, cats, model_id, config, alpha=None):
    return make_pipeline(columns, cats, model_id, config, alpha).fit(
        prepare_x(rows, columns, cats), rows[TARGET].to_numpy(dtype=float)
    )


def score(pipe, rows, columns, cats, model_id, variant, minimum):
    raw = pipe.predict(prepare_x(rows, columns, cats))
    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite predictions: {model_id}/{variant}")
    return rows[KEY + ["horizon_quarters", TARGET]].assign(
        model_id=model_id, variant=variant, raw_prediction=raw,
        **{PRED: np.maximum(raw, minimum)}
    )


def fit_record(rows, origin):
    return {"origin_quarter": origin, "fit_rows": len(rows),
            "last_target": rows.target_quarter.max(), "fit_rows_sha256": frame_hash(rows, KEY)}


def run_training(root, run_id, *, publish=True):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("Invalid run ID")
    with threadpool_limits(limits=1):
        return _run_training(Path(root).resolve(), run_id, publish)


def _run_training(root, run_id, publish):
    shared = root / "artifacts/shared_models"
    train, validation, manifest, features_cfg, split, _, _, protected = load_inputs(root)
    features = feature_sets(manifest, features_cfg)
    cats = manifest["feature_contract"]["categorical"]
    destinations = {(model, variant): shared / model / variant / "pipeline.joblib"
                    for model in MODELS for variant in features}
    if publish and any(path.exists() for path in destinations.values()):
        raise FileExistsError("A shared model already exists; use --no-publish for a private run.")
    output = root / ".private/linear_regression" / run_id
    output.mkdir(parents=True, exist_ok=False)
    config_path = root / "configs/models/linear_regression.yaml"
    config = load_yaml(config_path)
    protected[config_path.relative_to(root).as_posix()] = sha256(config_path)
    alphas = config["ridge"]["alphas"]
    if not alphas or not all(np.isfinite(alpha) and alpha > 0 for alpha in alphas):
        raise ValueError("Ridge alpha values must be finite and positive")
    candidates = [("linear_regression", None), *(("ridge", alpha) for alpha in alphas)]
    horizons = split["validation"]["horizon_quarters"]
    primary = split["validation"]["primary_horizons"]
    minimum = features_cfg["target"]["minimum_prediction"]
    pool = pd.concat([train, validation], ignore_index=True)
    evaluation, val_missing = select_scored_rows(validation.sort_values(KEY))
    predictions, audit = [], []
    for origin, rows in evaluation.groupby("origin_quarter", sort=True):
        fit = training_rows(pool, origin)
        audit.append(fit_record(fit, origin))
        for variant, columns in features.items():
            for model, alpha in candidates:
                pipe = fit_pipeline(fit, columns, cats, model, config, alpha)
                predictions.append(score(pipe, rows, columns, cats, model, variant, minimum)
                                   .assign(alpha=np.nan if alpha is None else alpha))
        print(f"Validated {origin}", flush=True)

    predictions = pd.concat(predictions, ignore_index=True)
    metrics = pd.concat([
        add_r2(grouped_metrics(part, primary), part, {"h1_5": horizons, "h2_5": primary})
        .assign(alpha=np.nan if alpha is None else alpha)
        for (_, alpha), part in predictions.groupby(["model_id", "alpha"], dropna=False)
    ], ignore_index=True)
    ranking = metrics.loc[metrics.model_id.eq("ridge") & metrics.scope.eq("h1_5")
                          & metrics.group_by.eq("overall")].sort_values(["wape", "alpha"])
    if ranking.wape.isna().any():
        raise ValueError("Validation WAPE is undefined")
    selected = ranking.drop_duplicates("variant").set_index("variant").alpha.to_dict()
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "validation_fit_audit.csv", index=False)
    write_json(output / "selection_lock.json", {
        "criterion": "Lowest pooled validation h=1–5 WAPE; ties by smaller alpha",
        "selected_alpha": selected, "grid_edge": {
            variant: alpha in (min(alphas), max(alphas)) for variant, alpha in selected.items()
        },
    })

    spec = split["final_test"]
    frozen = training_rows(pool, spec["origin_quarter"])
    test_item = next(item for item in manifest["files"] if item["file"] == "final_test.parquet")
    test_path = root / test_item["path"]
    protected[test_item["path"]] = test_item["sha256"]
    if sha256(test_path) != test_item["sha256"]:
        raise ValueError("Final test checksum mismatch")
    test = pd.read_parquet(test_path)
    check_test_inputs(test, spec, manifest["country_codes"], manifest["hs_codes"])
    if not (test.split.eq("final_test") & test.currency.eq("USD") & test.valuation.eq("FOB")).all():
        raise ValueError("Final test partition or unit mismatch")
    scoring, test_missing = select_scored_rows(test.sort_values(KEY))
    results, reports = [], []
    for model, variant in destinations:
        columns = features[variant]
        alpha = selected[variant] if model == "ridge" else None
        pipe = fit_pipeline(frozen, columns, cats, model, config, alpha)
        result = score(pipe, scoring, columns, cats, model, variant, minimum)
        results.append(result.assign(alpha=np.nan if alpha is None else alpha))
        reports.append(add_r2(test_metrics(result, model, primary, variant), result, {
            "h1_3": spec["horizon_quarters"], "h2_3": primary,
        }).assign(alpha=np.nan if alpha is None else alpha))
    pd.concat(results).to_csv(output / "final_test_predictions.csv", index=False)
    pd.concat(reports).to_csv(output / "final_test_metrics.csv", index=False)

    deployment_origin = split["forecast"]["preferred_origin_quarter"]
    test = test.assign(_target_index=test.target_quarter.map(quarter_number),
                       _origin_index=test.origin_quarter.map(quarter_number))
    deployment = training_rows(pd.concat([pool, test], ignore_index=True), deployment_origin)
    if deployment.target_quarter.max() != deployment_origin:
        raise ValueError("Deployment labels do not reach the configured origin")
    staged = {}
    for model, variant in destinations:
        columns = features[variant]
        alpha = selected[variant] if model == "ridge" else None
        pipe = fit_pipeline(deployment, columns, cats, model, config, alpha)
        path = output / f"{model}_{variant}.joblib"
        sample = prepare_x(scoring.head(32), columns, cats)
        save_pipeline(path, pipe, columns, cats, manifest["dataset_version"],
                      deployment.target_quarter.max(), sample)
        sample.to_parquet(path.with_suffix(".parquet"), index=False)
        np.save(path.with_suffix(".npy"), pipe.predict(sample))
        staged[model, variant] = path

    if any(sha256(root / name) != digest for name, digest in protected.items()):
        raise ValueError("Protected input changed")
    write_json(output / "run_metadata.json", {
        "dataset_version": manifest["dataset_version"], "protected_checksums": protected,
        "availability_limitation": manifest["availability_limitation"], "config": config,
        "evaluation_rows_sha256": frame_hash(evaluation, KEY),
        "final_test_rows_sha256": frame_hash(scoring, KEY),
        "final_test_fit": fit_record(frozen, spec["origin_quarter"]),
        "deployment_fit": fit_record(deployment, deployment_origin),
        "excluded_missing": {"validation": len(val_missing), "final_test": len(test_missing)},
        "feature_columns": features, "categorical_columns": cats, "python": platform.python_version(),
        "libraries": {name: version(name) for name in (
            "scikit-learn", "pandas", "numpy", "scipy", "pyarrow", "joblib", "PyYAML", "threadpoolctl"
        )},
    })
    if publish:
        published = []
        try:
            for key, path in destinations.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                os.link(staged[key], path)
                published.append(path)
        except OSError:
            for path in published:
                path.unlink()
            raise
    return output
