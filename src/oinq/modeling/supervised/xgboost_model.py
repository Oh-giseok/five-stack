
"""XGBoost quarterly export forecasting: v002 validation."""

from pathlib import Path
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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
    n_jobs=-1,
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

def make_pipeline(columns, categorical):

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
            ]),
            numeric
        ),
    ])

    return Pipeline([
        ("preprocessor", transform),
        ("model", XGBRegressor(**PARAMS)),
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

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    all_predictions = []
    configs = {}

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
        ("no_macro", features_no_macro),
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
            fit = train.loc[
                (train["_target_qn"] <= origin_qn)
                & train[target].notna()
                & train["target_available"].fillna(False)
            ].copy()

            # 검증 정답이 있는 행만 평가
            score = val.loc[
                (val["origin_quarter"] == origin)
                & val[target].notna()
                & val["target_available"].fillna(False)
            ].copy()

            if fit.empty or score.empty:
                continue

            pipeline = make_pipeline(
                cols, categorical
            )

            pipeline.fit(
                inputs(fit, cols, categorical),
                fit[target].astype(float).to_numpy()
            )

            pred = np.maximum(
                pipeline.predict(
                    inputs(score, cols, categorical)
                ),
                0
            )

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

        # 전체 Train 데이터로 재사용 모델 학습
        full_fit = train.loc[
            train[target].notna()
            & train["target_available"].fillna(False)
        ].copy()

        final_pipeline = make_pipeline(
            cols, categorical
        )

        final_pipeline.fit(
            inputs(full_fit, cols, categorical),
            full_fit[target].astype(float).to_numpy()
        )

        joblib.dump(
            {
                "pipeline": final_pipeline,
                "feature_columns": cols,
                "categorical_columns": categorical,
                "dataset_version": "v002",
                "trained_through_target_quarter": "2021Q4",
            },
            OUT / f"{name}_train_only.joblib"
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
        OUT / "validation_predictions.csv",
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
        OUT / "metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # STEP 10. 실험 설정 저장
    # ========================================================

    config = {
        "dataset_version": "v002",
        "seed": SEED,
        "parameters": PARAMS,
        "feature_columns_by_variant": configs,
        "categorical_columns": categorical,
        "train_target_period": manifest["train_period"],
        "validation_target_period": manifest["validation_period"],
        "validation_method": "expanding_origin_train_only_labels",
        "metric_priority": "WAPE_pct",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "limitations": manifest.get(
            "availability_limitation"
        ),
        "note": (
            "Validation labels are never used for fitting; "
            "validation is calendar-aligned, "
            "not vintage-verified."
        ),
    }

    (OUT / "experiment_config.json").write_text(
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

    print("\n저장 위치:", OUT)
    print("=== 학습 완료 ===")


if __name__ == "__main__":
    main()
