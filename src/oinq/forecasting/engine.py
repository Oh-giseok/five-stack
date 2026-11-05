"""Inference-only quarterly revenue engine using saved models and origin features."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import importlib.metadata

import numpy as np
import pandas as pd

from oinq.common.config import load_yaml
from oinq.features.prediction import build_prediction_inputs, prepare_model_inputs, reusable_origin_features
from oinq.modeling.loader import load_model, PROJECT_ROOT


def _path(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def annual_summary(quarterly):
    """Only label a sum as annual when all four forecasts are present."""
    keys = ["country_code", "country_name", "hs_code", "product_name", "origin_quarter", "target_year",
            "model_id", "model_variant", "run_id", "dataset_version", "currency", "valuation"]
    groups = quarterly.groupby(keys, sort=True, dropna=False)["predicted_export_value_usd"]
    # Use one grouped summation consistently, including large float64 revenues.
    result = groups.count().rename("available_quarters").to_frame()
    result["predicted_partial_sum_usd"] = groups.sum(min_count=1)
    result = result.reset_index()
    result["predicted_annual_export_value_usd"] = result.predicted_partial_sum_usd.where(result.available_quarters.eq(4))
    result["annual_status"] = np.where(result.available_quarters.eq(4), "complete", "incomplete")
    return result


def run_forecast(*, config_path=None, model_id=None, variant=None, origin_quarter=None,
                 features_path=None, output_dir=None, run_id=None, root=None):
    """Build inputs, load a fitted model, predict and persist a private run.

    No fit, API collection, missing-value carry-forward or model selection by test.
    """
    root = Path(root).resolve() if root is not None else PROJECT_ROOT
    config_path = _path(root, config_path or "configs/engine.yaml")
    settings = load_yaml(config_path)
    if settings.get("input_policy") != "current_revised_calendar_candidates":
        raise ValueError("현재 엔진은 학습과 동일한 기간 후보 입력 정책을 사용합니다.")
    paths = settings.get("paths")
    model_settings = settings.get("model")
    if not isinstance(paths, dict) or any(not isinstance(paths.get(k), str) or not paths[k].strip()
                                         for k in ["models_dir", "features_file", "dataset_manifest", "output_dir"]):
        raise ValueError("engine.yaml의 paths에 모델·입력·명세·출력 경로를 문자열로 지정하세요.")
    if not isinstance(model_settings, dict) or not all(k in model_settings for k in ["id", "variant"]):
        raise ValueError("engine.yaml에 model 설정이 필요합니다.")
    config = {name: load_yaml(root/"configs"/f"{name}.yaml") for name in ["countries", "products", "splits", "features"]}
    manifest_path = _path(root, settings["paths"]["dataset_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("dataset_release_approved"):
        raise ValueError("공유 승인된 데이터 명세가 필요합니다.")
    if ({c["iso3"] for c in config["countries"]["countries"]} != set(manifest["country_codes"])
            or {p["hs_code"] for p in config["products"]["products"]} != set(manifest["hs_codes"])):
        raise ValueError("국가·제품 설정과 공유 데이터 명세의 범위가 다릅니다.")
    model_id = model_id or settings["model"]["id"]
    variant = variant or settings["model"]["variant"]
    if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", model_id) or not isinstance(variant, str) or variant not in {"with_macro", "without_macro"}:
        raise ValueError("모델명 또는 거시 구성이 잘못됐습니다.")
    model_root = _path(root, settings["paths"]["models_dir"])
    model_folder = (model_root/model_id/variant).resolve()
    if not model_folder.is_relative_to(model_root):
        raise ValueError("모델 경로가 공유 모델 폴더 밖입니다.")
    candidates = [p for p in [model_folder/"pipeline.joblib", model_folder/"model.cbm"] if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"모델 파일은 구성별로 하나여야 합니다: {model_folder}")
    model_path = candidates[0]
    source_path = _path(root, features_path or settings["paths"]["features_file"])
    if not source_path.is_file():
        raise FileNotFoundError(f"최신 기준분기 입력 파일이 없습니다: {source_path}. --features로 준비한 origin_features.parquet 경로를 지정하세요.")
    origin_quarter = origin_quarter or config["splits"]["forecast"]["preferred_origin_quarter"]
    run_id = run_id or ("forecast_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id):
        raise ValueError("실행 ID에는 영문·숫자·밑줄·하이픈만 사용하세요.")
    destination = _path(root, output_dir or settings["paths"]["output_dir"])
    if not destination.is_relative_to(root) or destination.is_relative_to(model_root) or any(destination.is_relative_to(root/p) for p in ["data/shared", "data/processed", "data/raw"]):
        raise ValueError("결과는 프로젝트 내부의 별도 출력 폴더에 저장하세요.")
    out = destination/run_id
    if out.exists():
        raise FileExistsError(f"기존 예측 결과는 덮어쓰지 않습니다: {out}")
    protected = {p: _sha(p) for p in [config_path, manifest_path, model_path, source_path]}
    for name in config:
        path = root/"configs"/f"{name}.yaml"
        protected[path] = _sha(path)
    if source_path.suffix.lower() == ".parquet":
        origin_features = pd.read_parquet(source_path)
    elif source_path.suffix.lower() == ".csv":
        origin_features = pd.read_csv(source_path, encoding="utf-8-sig", dtype={"country_code":"string", "hs_code":"string", "origin_quarter":"string"},
                                      keep_default_na=False, na_values=[r"\N"])
    else:
        raise ValueError("입력은 Parquet 또는 CSV여야 합니다.")
    inputs = build_prediction_inputs(origin_features, config, manifest, origin_quarter)
    loaded = load_model(model_path)
    selected = loaded["feature_columns"]
    allowed = manifest["feature_columns"]
    if not set(selected).issubset(allowed) or set(selected).intersection(manifest["feature_contract"]["forbidden_feature_columns"]):
        raise ValueError("모델 입력에 미등록 변수 또는 타깃이 포함됐습니다.")
    macro = config["features"]["supervised"]["macro"]["columns"]
    removed = set(macro+[f"{c}_missing" for c in macro])
    if variant == "without_macro" and set(selected).intersection(removed):
        raise ValueError("거시 제외 모델에 거시 변수가 포함됐습니다.")
    if variant == "with_macro" and not set(macro).intersection(selected):
        raise ValueError("거시 포함 모델의 입력 구성이 맞지 않습니다.")
    available = inputs.source_row_available
    numeric_values = [c for c in selected if c in manifest["feature_contract"]["numeric_candidates"] and not c.endswith("_missing")]
    missing_count = inputs[numeric_values].isna().sum(axis=1)
    raw = pd.Series(np.nan, index=inputs.index, dtype=float)
    if available.any():
        prepared = prepare_model_inputs(inputs.loc[available], loaded)
        predicted = np.asarray(loaded["model"].predict(prepared))
        if predicted.ndim != 1 or len(predicted) != int(available.sum()) or not np.isfinite(predicted).all():
            raise ValueError("모델이 행별 유한한 수출액 예측을 반환하지 않았습니다.")
        raw.loc[available] = predicted
    columns = ["country_code", "country_name", "hs_code", "product_name", "origin_quarter", "origin_cutoff_at", "target_quarter",
               "target_year", "target_quarter_number", "horizon_quarters", "macro_candidate_year", "market_candidate_quarter", "historical_release_verified"]
    result = inputs[columns].copy()
    result["raw_predicted_export_value_usd"] = raw
    result["predicted_export_value_usd"] = raw.clip(lower=config["features"]["target"]["minimum_prediction"])
    result["prediction_status"] = np.where(~available, "unavailable_input", np.where(missing_count.gt(0), "predicted_with_missing_inputs", "predicted"))
    result["input_missing_feature_count"] = missing_count
    result["model_id"], result["model_variant"], result["run_id"] = model_id, variant, run_id
    result["dataset_version"] = manifest["dataset_version"]
    result["currency"], result["valuation"] = "USD", "FOB"
    annual = annual_summary(result)
    origin_snapshot = reusable_origin_features(inputs, manifest)
    if any(_sha(path) != digest for path,digest in protected.items()):
        raise RuntimeError("예측 실행 중 모델·원천·설정 파일이 변경됐습니다.")
    out.mkdir(parents=True, exist_ok=False)
    origin_snapshot.to_parquet(out/"origin_features.parquet", index=False)
    result.to_parquet(out/"quarterly_forecast.parquet", index=False)
    annual.to_parquet(out/"annual_forecast.parquet", index=False)
    metadata = {"status":"completed", "run_id":run_id, "generated_at_utc":datetime.now(timezone.utc).isoformat(),
                "model_id":model_id, "model_variant":variant, "model_path":str(model_path.relative_to(root)),
                "model_sha256":protected[model_path], "dataset_version":manifest["dataset_version"],
                "origin_quarter":origin_quarter, "target_quarters":sorted(result.target_quarter.unique().tolist()),
                "horizons":sorted(result.horizon_quarters.unique().tolist()), "origin_rows":len(origin_snapshot),
                "quarterly_rows":len(result), "annual_rows":len(annual), "predicted_rows":int(result.predicted_export_value_usd.notna().sum()),
                "missing_input_rows":int(missing_count.gt(0).sum()), "unavailable_source_rows":int((~available).sum()),
                "input_columns":selected, "input_policy":settings["input_policy"], "historical_release_verified":bool(result.historical_release_verified.all()),
                "refit_performed":False, "source_paths_sha256":{str(p):h for p,h in protected.items()},
                "outputs":{name:_sha(out/name) for name in ["origin_features.parquet","quarterly_forecast.parquet","annual_forecast.parquet"]}}
    estimator = list(loaded["model"].named_steps.values())[-1] if loaded["model_format"] == "joblib_pipeline" else loaded["model"]
    package = type(estimator).__module__.split(".")[0]
    packages = {"numpy", "pandas", "scikit-learn", "joblib", "scikit-learn" if package == "sklearn" else package}
    metadata["libraries"] = {}
    for package_name in packages:
        try:
            metadata["libraries"][package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            metadata["libraries"][package_name] = None
    (out/"run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {"quarterly":result, "annual":annual, "output_dir":out, "metadata":metadata}
