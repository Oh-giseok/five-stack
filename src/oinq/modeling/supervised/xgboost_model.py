
"""XGBoost quarterly export forecasting: v002 validation."""

from pathlib import Path
import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


# ============================================================
# STEP 1. 경로 및 기본 설정
# ============================================================

ROOT = Path(__file__).resolve().parents[4]

DATA = ROOT / "data" / "shared" / "v002"
OUT = ROOT / "artifacts" / "xgboost" / "v002" / "baseline"

SEED = 42

MACRO = [
    "gdp_usd",
    "gdp_per_capita_usd",
    "population",
    "inflation_pct",
    "exchange_rate_lcu_per_usd",
    "internet_users_pct",
]

PARAMS = dict(
    n_estimators=250,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=SEED,
    n_jobs=4,
)


# ============================================================
# STEP 2. 분기를 숫자로 변환
# ============================================================

def quarter_number(series):
    s = series.astype("string")

    year = pd.to_numeric(
        s.str[:4], errors="raise"
    ).astype(int)

    quarter = pd.to_numeric(
        s.str[-1], errors="raise"
    ).astype(int)

    return year * 4 + quarter - 1


# ============================================================
# STEP 3. 전처리 및 XGBoost 모델
# ============================================================

def make_pipeline(columns, categorical, params=None, standard_scaler=False):

    numeric = [
        c for c in columns
        if c not in categorical
    ]

    transform = ColumnTransformer([
        (
            "cat",
            Pipeline([
                (
                    "fill",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                ),
            ]),
            categorical
        ),
        (
            "num",
            Pipeline([
                (
                    "fill",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0,
                        keep_empty_features=True
                    )
                ),
            ] + ([("scaler", StandardScaler())] if standard_scaler else [])),
            numeric
        ),
    ])

    return Pipeline([
        ("preprocessor", transform),
        ("model", XGBRegressor(**(PARAMS if params is None else params))),
    ])


# ============================================================
# STEP 4. 모델 입력 데이터 정리
# ============================================================

def inputs(frame, columns, categorical):

    x = frame[columns].copy()

    for c in categorical:
        x[c] = (
            x[c]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    for c in columns:
        if c not in categorical:
            x[c] = pd.to_numeric(
                x[c], errors="coerce"
            ).astype(float)

    return x


# ============================================================
# STEP 5. 평가 지표
# ============================================================

def metrics(frame):

    y = frame["actual_usd"].to_numpy(dtype=float)
    p = frame["predicted_usd"].to_numpy(dtype=float)

    e = np.abs(y - p)

    denom = np.abs(y).sum()
    sm_den = np.abs(y) + np.abs(p)

    return pd.Series({
        "rows": len(frame),

        "WAPE_pct": (
            100 * e.sum() / denom
            if denom else np.nan
        ),

        "MAE_usd": e.mean(),

        "RMSE_usd": np.sqrt(
            np.mean((y - p) ** 2)
        ),

        "sMAPE_pct": 100 * np.mean(
            np.divide(
                2 * e,
                sm_den,
                out=np.zeros_like(e),
                where=sm_den != 0
            )
        ),
    })


# ============================================================
# STEP 6. 메인 학습 및 검증
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="New artifacts directory; never overwritten")
    parser.add_argument("--experiment", choices=["baseline", "l1", "l2", "l1_l2", "standard_scaler"], default="baseline")
    parser.add_argument("--resume", action="store_true", help="Reuse already saved per-origin models from an incomplete run")
    args = parser.parse_args()
    params = PARAMS.copy()
    # Exact settings supplied in the team's regularization/scaler notebooks.
    params.update({"baseline": {"reg_alpha": 0.0, "reg_lambda": 1.0},
                   "l1": {"reg_alpha": 1.0, "reg_lambda": 1.0},
                   "l2": {"reg_alpha": 0.0, "reg_lambda": 5.0},
                   "l1_l2": {"reg_alpha": 1.0, "reg_lambda": 5.0},
                   "standard_scaler": {"reg_alpha": 0.0, "reg_lambda": 1.0}}[args.experiment])
    standard_scaler = args.experiment == "standard_scaler"
    run_id = args.run_id or ("xgboost_v002_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    if not run_id or not all(c.isascii() and (c.isalnum() or c in "_-") for c in run_id) or run_id == "production":
        raise ValueError("Invalid experimental run ID")
    out_dir = ROOT / "artifacts" / run_id
    if args.resume and (out_dir / "experiment_config.json").exists():
        raise FileExistsError("Run already complete; reuse its results without rerunning")
    # Reuse the existing validation protocol and input integrity checks.
    from oinq.modeling.supervised.lightgbm import load_inputs, frame_hash, sha256, KEY
    checked_train, checked_val, checked_manifest, *_ = load_inputs(ROOT)
    print("=== XGBoost v002 Baseline 시작 ===")

    manifest = json.loads(
        (DATA / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    if manifest["dataset_version"] != "v002":
        raise ValueError(
            "Unexpected dataset version"
        )

    train = pd.read_parquet(
        DATA / "train.parquet"
    )

    val = pd.read_parquet(
        DATA / "validation.parquet"
    )

    features = manifest["feature_columns"]

    target = manifest["feature_contract"]["target"]

    categorical = manifest[
        "feature_contract"
    ]["categorical"]

    print("데이터 버전:", manifest["dataset_version"])
    print("Train:", train.shape)
    print("Validation:", val.shape)
    print("전체 특성 수:", len(features))

    # 분기 변환 및 타깃 정리
    for frame in (train, val):

        frame["_target_qn"] = quarter_number(
            frame["target_quarter"]
        )

        frame["_origin_qn"] = quarter_number(
            frame["origin_quarter"]
        )

        frame[target] = pd.to_numeric(
            frame[target],
            errors="coerce"
        )

    # 시간 순서 확인
    assert train["target_quarter"].between(
        "2013Q1", "2021Q4"
    ).all()

    assert val["target_quarter"].between(
        "2022Q1", "2025Q4"
    ).all()

    assert (
        val["_origin_qn"] < val["_target_qn"]
    ).all()

    forbidden = manifest[
        "feature_contract"
    ]["forbidden_feature_columns"]

    assert not set(features) & set(forbidden)

    out_dir.mkdir(
        parents=True,
        exist_ok=args.resume
    )

    all_predictions = []
    configs = {}
    fit_audit = []
    model_files = []
    pool = pd.concat([train, val], ignore_index=True)
    if manifest != checked_manifest or len(train) != len(checked_train) or len(val) != len(checked_val):
        raise ValueError("Shared input check differs")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 거시지표와 해당 결측 여부 변수 함께 제외
    macro_exclude = MACRO + [
        f"{m}_missing" for m in MACRO
    ]

    features_no_macro = [
        c for c in features
        if c not in macro_exclude
    ]

    features_with_macro = features

    variants = [
        ("without_macro", features_no_macro),
        ("with_macro", features_with_macro),
    ]

    # ========================================================
    # STEP 7. 거시지표 제외 / 포함 모델 학습
    # ========================================================

    for name, cols in variants:

        configs[name] = cols

        print(
            f"\nRunning {name}: {len(cols)} features",
            flush=True
        )

        origins = sorted(
            val["origin_quarter"].dropna().unique(),
            key=lambda q: int(q[:4]) * 4 + int(q[-1])
        )

        for origin in origins:

            origin_qn = int(
                quarter_number(
                    pd.Series([origin])
                ).iloc[0]
            )

            # 예측 시점까지 사용 가능한 학습 정답만 사용
            fit = pool.loc[
                (pool["_target_qn"] <= origin_qn)
                & pool[target].notna()
                & pool["target_available"].fillna(False)
            ].sort_values(KEY).copy()

            # 검증 정답이 있는 행만 평가
            score = val.loc[
                (val["origin_quarter"] == origin)
                & val[target].notna()
                & val["target_available"].fillna(False)
            ].sort_values(KEY).copy()

            if fit.empty or score.empty:
                continue

            model_dir = out_dir / name / origin
            model_path = model_dir / "pipeline.joblib"
            resumed = bool(args.resume and model_path.exists())
            if resumed:
                bundle = joblib.load(model_path)
                if bundle["feature_columns"] != cols or bundle["origin_quarter"] != origin or bundle["trained_through_target_quarter"] != str(fit.target_quarter.max()):
                    raise ValueError("Saved checkpoint input/origin differs")
                pipeline = bundle["pipeline"]
                actual_params = pipeline.named_steps["model"].get_params()
                if any(actual_params[k] != value for k,value in params.items()):
                    raise ValueError("Saved checkpoint parameters differ")
                if ("scaler" in pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps) != standard_scaler:
                    raise ValueError("Saved checkpoint scaler differs")
                if (model_dir / "fit_metadata.json").exists():
                    prior = json.loads((model_dir / "fit_metadata.json").read_text(encoding="utf-8"))
                    if prior["fit_rows_sha256"] != frame_hash(fit, KEY):
                        raise ValueError("Saved checkpoint fit keys differ")
            else:
                pipeline = make_pipeline(cols, categorical, params=params, standard_scaler=standard_scaler)
                pipeline.fit(inputs(fit, cols, categorical), fit[target].astype(float).to_numpy())

            pred = np.maximum(
                pipeline.predict(
                    inputs(score, cols, categorical)
                ),
                0
            )
            if not np.isfinite(pred).all():
                raise ValueError("Non-finite prediction")
            model_dir.mkdir(parents=True, exist_ok=args.resume)
            if not resumed:
                joblib.dump({"pipeline": pipeline, "feature_columns": cols,
                         "categorical_columns": categorical, "dataset_version": "v002",
                         "origin_quarter": origin, "model_purpose": "validation",
                         "trained_through_target_quarter": str(fit["target_quarter"].max())},
                            model_path, compress=3)
            pipeline.named_steps["model"].save_model(model_dir / "model.ubj")
            replay = np.maximum(joblib.load(model_path)["pipeline"].predict(inputs(score, cols, categorical)), 0)
            if not np.array_equal(pred, replay):
                raise AssertionError("Saved XGBoost prediction replay differs")
            audit = {"variant": name, "origin_quarter": origin, "fit_rows": len(fit),
                     "past_train_rows": int(fit["split"].eq("train").sum()),
                     "past_validation_rows": int(fit["split"].eq("validation").sum()),
                     "max_fit_target": str(fit["target_quarter"].max()),
                     "fit_rows_sha256": frame_hash(fit, KEY), "score_rows": len(score),
                     "reload_max_abs_diff": float(np.max(np.abs(pred-replay))), "resumed_checkpoint": resumed}
            (model_dir / "fit_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
            fit_audit.append(audit)
            model_files.append({"file": model_path.relative_to(out_dir).as_posix(), "sha256": sha256(model_path),
                                "variant": name, "origin_quarter": origin})

            result = score[
                [
                    "country_code",
                    "hs_code",
                    "origin_quarter",
                    "target_quarter",
                    "horizon_quarters",
                ]
            ].copy()

            result["actual_usd"] = (
                score[target].to_numpy(dtype=float)
            )

            result["predicted_usd"] = pred
            result["model_variant"] = name
            result["fit_rows"] = len(fit)

            all_predictions.append(result)

            print(
                f"  {origin}: "
                f"train={len(fit)}, "
                f"validation={len(score)}",
                flush=True
            )

        print(
            f"{name} 모델 저장 완료",
            flush=True
        )

    # ========================================================
    # STEP 8. 예측 결과 저장
    # ========================================================

    if not all_predictions:
        raise RuntimeError(
            "No validation predictions: "
            "check origin and target availability"
        )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    predictions.to_csv(
        out_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # STEP 9. WAPE 및 세부 지표 계산
    # ========================================================

    grouped = []

    groupings = [
        ["model_variant"],
        ["model_variant", "horizon_quarters"],
        ["model_variant", "country_code"],
        ["model_variant", "hs_code"],
    ]

    for group_by in groupings:

        for key, subset in predictions.groupby(
            group_by,
            dropna=False
        ):

            key = (
                key if isinstance(key, tuple)
                else (key,)
            )

            row = dict(zip(group_by, key))

            row.update(
                metrics(subset).to_dict()
            )

            row["grouping"] = "+".join(group_by)

            grouped.append(row)

    metrics_df = pd.DataFrame(grouped)

    metrics_df.to_csv(
        out_dir / "metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # STEP 10. 실험 설정 저장
    # ========================================================

    config = {
        "dataset_version": "v002",
        "seed": SEED,
        "parameters": params,
        "experiment": args.experiment,
        "standard_scaler": standard_scaler,
        "source_notebook": ("notebooks/XGBoost_StandardScaler.ipynb" if standard_scaler else
                            "notebooks/XGBoost_L1_L2_Regularization.ipynb" if args.experiment != "baseline" else
                            "notebooks/XGBoost_Baseline.ipynb"),
        "feature_columns_by_variant": configs,
        "categorical_columns": categorical,
        "train_target_period": manifest["train_period"],
        "validation_target_period": manifest["validation_period"],
        "validation_method": "expanding_origin_train_plus_past_validation_labels",
        "metric_priority": "WAPE_pct",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "models": model_files,
        "fitted_pipelines": len(model_files),
        "reload_verified": True,
        "final_test_used": False,
        "input_checksums": {name: sha256(DATA / name) for name in ["manifest.json", "train.parquet", "validation.parquet"]},
        "limitations": manifest.get(
            "availability_limitation"
        ),
        "note": (
            "Only validation labels whose target <= current origin join the fit; "
            "validation is calendar-aligned, "
            "not vintage-verified."
        ),
    }

    pd.DataFrame(fit_audit).to_csv(out_dir / "fit_audit.csv", index=False, encoding="utf-8-sig")
    (out_dir / "experiment_config.json").write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # STEP 11. 최종 결과 출력
    # ========================================================

    print("\n=== XGBoost v002 Baseline 결과 ===")

    print(
        metrics_df.query(
            "grouping == 'model_variant'"
        ).to_string(index=False)
    )

    print("\n저장 위치:", out_dir)
    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()
