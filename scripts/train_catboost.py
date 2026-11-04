"""Train pooled CatBoost regressors and run expanding-origin validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
NULL_CATEGORY = "__MISSING__"
TARGET = "target_export_value_usd"
VALIDATION_COLUMNS = [
    "country_code", "hs_code", "origin_quarter", "target_quarter",
    "horizon_quarters", TARGET,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame(folder: Path, split: str, schema: dict) -> tuple[pd.DataFrame, str]:
    """Read Parquet when possible, otherwise use the manifest-typed CSV copy."""
    parquet = folder / f"{split}.parquet"
    csv = folder / f"{split}.csv"
    if parquet.exists():
        try:
            return pd.read_parquet(parquet), str(parquet)
        except ImportError:
            pass
    if not csv.exists():
        raise FileNotFoundError(f"Parquet 읽기 엔진이 없고 CSV도 없습니다: {parquet}, {csv}")
    dtype_map = {column["name"]: column["dtype"] for column in schema}
    frame = pd.read_csv(csv, encoding="utf-8-sig", na_values=[r"\N"], dtype=dtype_map)
    return frame, str(csv)


def quarter_key(series: pd.Series) -> pd.Series:
    """Convert YYYYQn labels into sortable integer quarter indexes."""
    parts = series.astype("string").str.extract(r"^(\d{4})Q([1-4])$")
    if parts.isna().any(axis=None):
        bad = series[parts.isna().any(axis=1)].head(3).tolist()
        raise ValueError(f"분기 라벨 형식이 예상과 다릅니다: {bad}")
    return parts[0].astype(int) * 4 + parts[1].astype(int)


def build_features(frame: pd.DataFrame, columns: list[str], categorical: list[str]) -> pd.DataFrame:
    x = frame.loc[:, columns].copy()
    for name in categorical:
        x[name] = x[name].astype("string").fillna(NULL_CATEGORY).astype(str)
    for name in columns:
        if name not in categorical:
            x[name] = pd.to_numeric(x[name], errors="coerce").astype("float64")
    return x


def metric_values(actual: pd.Series, predicted: pd.Series) -> dict:
    y = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)
    error = p - y
    denominator = np.abs(y).sum()
    smape_denominator = np.abs(y) + np.abs(p)
    smape = np.divide(200 * np.abs(error), smape_denominator,
                      out=np.zeros_like(error), where=smape_denominator != 0)
    return {
        "n_rows": int(len(y)),
        "actual_sum_usd": float(y.sum()),
        "wape_pct": float(100 * np.abs(error).sum() / denominator) if denominator else None,
        "mae_usd": float(np.abs(error).mean()),
        "rmse_usd": float(np.sqrt(np.mean(error**2))),
        "smape_pct": float(smape.mean()),
    }


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for variant, subset in predictions.groupby("variant", sort=True):
        groupings = [("overall", [], subset),
                     ("operational_h2_h5", [], subset.loc[subset["horizon_quarters"].between(2, 5)])]
        for dimension, column in (("horizon", "horizon_quarters"),
                                  ("country", "country_code"),
                                  ("product", "hs_code")):
            groupings.append((dimension, [column], subset))
        for dimension, column in (("country_horizon", "country_code"),
                                  ("product_horizon", "hs_code")):
            groupings.append((dimension, [column, "horizon_quarters"], subset))
        for dimension, keys, data in groupings:
            groups = [((), data)] if not keys else data.groupby(keys, dropna=False, sort=True)
            for key, group in groups:
                if not isinstance(key, tuple):
                    key = (key,)
                row = {"variant": variant, "dimension": dimension}
                row.update(dict(zip(keys, key)))
                row.update(metric_values(group[TARGET], group["prediction_usd"]))
                records.append(row)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/shared/v001")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--l2-leaf-reg", type=float)
    args = parser.parse_args()

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        print("CatBoost가 필요합니다. `pip install -e '.[models]'`로 설치하세요.", file=sys.stderr)
        return 2

    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_schemas = {Path(item["path"]).stem: item["schema"] for item in manifest["files"]}
    train, train_path = load_frame(data_dir, "train", file_schemas["train"])
    validation, validation_path = load_frame(data_dir, "validation", file_schemas["validation"])
    features = list(manifest["feature_columns"])
    contract = manifest["feature_contract"]
    categorical = list(contract["categorical"])
    missing_features = set(features) - set(train.columns) | (set(features) - set(validation.columns))
    if missing_features:
        raise ValueError(f"manifest feature_columns와 데이터 열이 맞지 않습니다: {sorted(missing_features)}")
    if TARGET != contract["target"]:
        raise ValueError(f"예상 타깃 불일치: {contract['target']}")
    if set(features) & set(contract["forbidden_feature_columns"]):
        raise ValueError("금지된 타깃·메타정보가 feature_columns에 포함돼 있습니다.")

    model_config = yaml.safe_load((ROOT / "configs/models/catboost.yaml").read_text(encoding="utf-8"))
    feature_config = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))
    macro_base = list(feature_config["supervised"]["macro"]["columns"])
    macro_columns = [name for name in features if name in macro_base or name.removesuffix("_missing") in macro_base]
    variants = {
        "with_macro": features,
        "without_macro": [name for name in features if name not in set(macro_columns)],
    }
    if args.iterations is not None:
        model_config["params"]["iterations"] = args.iterations
    if args.seed is not None:
        model_config["params"]["random_seed"] = args.seed
    if args.l2_leaf_reg is not None:
        if args.l2_leaf_reg <= 0:
            parser.error("--l2-leaf-reg은 0보다 큰 값이어야 합니다.")
        model_config["params"]["l2_leaf_reg"] = args.l2_leaf_reg
    params = dict(model_config["params"])
    params.setdefault("allow_writing_files", False)

    train_period = manifest["train_period"]
    validation_period = manifest["validation_period"]
    train = train.loc[train["split"].eq("train")].copy()
    validation = validation.loc[validation["split"].eq("validation")].copy()
    train["_target_q"] = quarter_key(train["target_quarter"])
    train["_origin_q"] = quarter_key(train["origin_quarter"])
    validation["_target_q"] = quarter_key(validation["target_quarter"])
    validation["_origin_q"] = quarter_key(validation["origin_quarter"])
    train_start, train_end = quarter_key(pd.Series([train_period["start"], train_period["end"]])).tolist()
    valid_start, valid_end = quarter_key(pd.Series([validation_period["start"], validation_period["end"]])).tolist()
    train = train.loc[train["_target_q"].between(train_start, train_end)]
    validation = validation.loc[validation["_target_q"].between(valid_start, valid_end)
                                & validation["horizon_quarters"].between(1, 5)].copy()
    if not validation["_origin_q"].between(train_start - 12, valid_end).all():
        raise ValueError("검증 기준 분기 범위가 예상과 다릅니다.")

    target_available = validation[TARGET].notna()
    if "target_available" in validation:
        target_available &= validation["target_available"].fillna(False).astype(bool)
    evaluation = validation.loc[target_available].copy()
    if evaluation.empty:
        raise ValueError("검증 가능한 정답 행이 없습니다.")

    run_id = datetime.now(timezone.utc).strftime("catboost_%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir.resolve() if args.output_dir else ROOT / "artifacts" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions = []
    fitted_models = []
    origins = sorted(evaluation["_origin_q"].unique())
    origin_labels = (evaluation[["_origin_q", "origin_quarter"]]
                     .drop_duplicates().set_index("_origin_q")["origin_quarter"].to_dict())

    for origin_index in origins:
        origin = origin_labels[origin_index]
        fit_mask = (train["target_export_value_usd"].notna()
                    & train["_target_q"].le(origin_index)
                    & train["_origin_q"].le(origin_index))
        fit = train.loc[fit_mask]
        score = evaluation.loc[evaluation["_origin_q"].eq(origin_index)]
        if fit.empty:
            raise ValueError(f"{origin}: 사용 가능한 학습 정답이 없습니다.")
        if (fit["_target_q"] > origin_index).any():
            raise AssertionError("기준 분기 이후 정답이 학습에 포함됐습니다.")
        if (fit["_origin_q"] > origin_index).any():
            raise AssertionError("기준 분기 이후 학습 사례가 포함됐습니다.")

        for variant, columns in variants.items():
            x_fit = build_features(fit, columns, [name for name in categorical if name in columns])
            x_score = build_features(score, columns, [name for name in categorical if name in columns])
            model = CatBoostRegressor(**params)
            model.fit(x_fit, fit[TARGET].astype(float), cat_features=[name for name in categorical if name in columns])
            predicted = np.maximum(model.predict(x_score), 0.0)
            file_stem = f"{variant}_{origin}"
            model_path = output_dir / f"{file_stem}.cbm"
            model.save_model(str(model_path))
            fitted_models.append({
                "variant": variant, "origin_quarter": origin, "model_file": model_path.name,
                "fit_rows": int(len(fit)), "fit_target_max_quarter": str(fit["target_quarter"].max()),
                "feature_columns": columns, "categorical_columns": [c for c in categorical if c in columns],
                "prediction_floor_usd": 0.0,
            })
            out = score[VALIDATION_COLUMNS].copy()
            out["variant"] = variant
            out["prediction_usd"] = predicted
            out["fit_origin_quarter"] = origin
            out["fit_rows"] = len(fit)
            predictions.append(out)
            print(f"{variant} {origin}: train={len(fit):,}, validation={len(score):,}", flush=True)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    scores = metric_table(prediction_frame)
    prediction_frame.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()
    metadata = {
        "run_id": run_id, "model_id": "catboost", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": manifest["dataset_version"], "dataset_manifest_sha256": sha256(manifest_path),
        "training_code_sha256": sha256(Path(__file__)),
        "config_sha256": {"catboost": sha256(ROOT / "configs/models/catboost.yaml"),
                          "features": sha256(ROOT / "configs/features.yaml")},
        "train_file": train_path, "train_sha256": sha256(Path(train_path)),
        "validation_file": validation_path, "validation_sha256": sha256(Path(validation_path)),
        "train_period": train_period, "validation_period": validation_period,
        "validation_method": "expanding_origin; fit train rows with target_quarter <= origin_quarter and row origin <= origin",
        "historical_asof_verified": bool(contract.get("historical_asof_use_verified", False)),
        "target": TARGET, "currency": "USD", "valuation": "FOB",
        "validation_candidate_rows": int(len(validation)), "scored_rows_per_variant": int(len(evaluation)),
        "missing_validation_targets_excluded": int(len(validation) - len(evaluation)),
        "horizons_evaluated": sorted(evaluation["horizon_quarters"].unique().astype(int).tolist()),
        "macro_columns_removed_for_without_macro": macro_columns,
        "variants_feature_columns": variants,
        "categorical_columns": categorical,
        "categorical_policy": f"Cast categories to strings; nulls use {NULL_CATEGORY!r}; CatBoost native categorical handling.",
        "numeric_missing_policy": "Keep numeric nulls as NaN; CatBoost native numeric missing handling; no imputer fit.",
        "negative_prediction_policy": "clip predictions to 0 USD before scoring and saving.",
        "params": params, "models": fitted_models,
        "versions": {"python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
                     "catboost": CatBoostRegressor.__module__},
        "git_commit": git_commit, "git_status_at_run": git_status,
        "interpretation_limit": "Current revised source snapshots; historical publication-time availability is unverified.",
    }
    import catboost
    metadata["versions"]["catboost"] = catboost.__version__
    (output_dir / "run_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"완료: {output_dir}", flush=True)
    print(scores.loc[scores.dimension.isin(["overall", "operational_h2_h5", "horizon"])].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
