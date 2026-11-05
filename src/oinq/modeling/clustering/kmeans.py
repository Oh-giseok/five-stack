"""Validate the shared country/window data and manage retrospective K-means."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import pickle
import platform
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from oinq.common.config import load_yaml
from oinq.preprocessing.full_release import check_formats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
KEY = ["country_code", "window_id"]
FEATURES = ["gdp_per_capita_usd", "population", "internet_users_pct"]
FIT_SCOPE = "eligible_observed_periods_for_retrospective_comparison"


class ClusterError(ValueError):
    """The input, configuration, or saved cluster model violates its contract."""


def require(condition, message):
    if not condition:
        raise ClusterError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_features(frame, countries=None, windows=None):
    """Keep every row and null; only validate eligibility, keys and raw values."""
    required = KEY + FEATURES + ["eligible_for_fit"]
    require(frame.columns.is_unique, "군집 입력 열 이름이 중복됐습니다.")
    require(set(required).issubset(frame), f"군집 필수 열 누락: {sorted(set(required)-set(frame))}")
    result = frame.copy()
    require(not result.empty, "군집 입력이 비어 있습니다.")
    for key in KEY:
        require(result[key].notna().all() and result[key].map(lambda x: isinstance(x, str) and bool(x.strip())).all(),
                f"{key}는 결측 없는 문자열이어야 합니다.")
    require(not result.duplicated(KEY).any(), "국가·분석기간 키가 중복됐습니다.")
    flag = result.eligible_for_fit
    require(pd.api.types.is_bool_dtype(flag.dtype) and flag.notna().all(), "적합 여부는 결측 없는 boolean이어야 합니다.")
    for column in FEATURES:
        require(pd.api.types.is_numeric_dtype(result[column]) and not pd.api.types.is_bool_dtype(result[column]),
                f"군집 변수는 수치형이어야 합니다: {column}")
        values = result[column].to_numpy(dtype=float, na_value=np.nan)
        require(not np.isinf(values).any(), f"무한대 군집 입력: {column}")
        valid = values[np.isfinite(values)]
        require((valid >= 0).all(), f"음수 군집 입력: {column}")
        if column == "population":
            require((valid > 0).all(), "인구는 양수여야 합니다.")
        if column == "internet_users_pct":
            require((valid <= 100).all(), "인터넷 이용률은 0~100%여야 합니다.")
    complete = result[FEATURES].notna().all(axis=1)
    require((~flag | complete).all(), "적합 가능 행에 군집 입력 결측이 있습니다.")
    if countries is not None and windows is not None:
        expected = {(c, w) for c in countries for w in windows}
        actual = set(result[KEY].itertuples(index=False, name=None))
        require(actual == expected, "전체 국가×분석기간 조합에 누락 또는 추가 키가 있습니다.")
    return result.sort_values(KEY).reset_index(drop=True)


def merge_explanations(base, other, label):
    """A strict 1:1 join: reject missing keys and conflicting shared columns."""
    require(other.columns.is_unique and set(KEY).issubset(other), f"{label} 열 구성이 잘못됐습니다.")
    require(not other[KEY].isna().any().any() and not other.duplicated(KEY).any(), f"{label} 키 누락·중복입니다.")
    require(set(base[KEY].itertuples(index=False, name=None)) == set(other[KEY].itertuples(index=False, name=None)),
            f"{label} 국가·기간 범위가 입력과 다릅니다.")
    left = base.set_index(KEY).sort_index()
    right = other.set_index(KEY).sort_index()
    for column in left.columns.intersection(right.columns):
        equal = left[column].eq(right[column]) | (left[column].isna() & right[column].isna())
        require(equal.fillna(False).all(), f"{label}의 공통 열 값이 다릅니다: {column}")
    return base.merge(other[KEY + [c for c in other if c not in base]], on=KEY, how="left", validate="one_to_one")


def load_inputs(root=None):
    """Read only the approved shared release, never private preprocessing caches."""
    root = Path(root).resolve() if root is not None else PROJECT_ROOT
    folder = root / "data/shared/v002/unsupervised"
    parent_path, manifest_path = folder.parent / "manifest.json", folder / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(parent.get("dataset_release_approved") and parent.get("dataset_version") == "v002"
            and manifest.get("approved_for_shared_release") and manifest.get("shared_version") == "v002"
            and manifest.get("status") == "released", "승인된 공유 v002 비지도 명세가 필요합니다.")
    model = load_yaml(root / "configs/models/kmeans.yaml")
    feature_config = load_yaml(root / "configs/features.yaml")
    country_config = load_yaml(root / "configs/countries.yaml")
    clustering = feature_config["clustering"]
    require(model["model_id"] == "kmeans" and model["inputs"]["columns"] == FEATURES
            and clustering["columns"] == FEATURES and manifest["feature_columns"] == FEATURES
            and parent["unsupervised"]["feature_columns"] == FEATURES
            and model["inputs"]["row_key"] == KEY and manifest["feature_key"] == KEY, "군집 입력 3개와 행 키가 사양과 다릅니다.")
    require(model["inputs"]["fit_scope"] == FIT_SCOPE and clustering["fit_scope"] == FIT_SCOPE
            and model["inputs"]["shared_scaler_and_centroids_across_windows"] is True
            and clustering["shared_scaler_and_centroids_across_windows"] is True
            and manifest["shared_scaler_and_centroids_required"] is True
            and model["inputs"]["scaler"] == clustering["scaler"] == "StandardScaler"
            and model["inputs"]["historical_backtest_claim"] is False
            and manifest["historical_backtest_claim"] is False
            and model["inputs"]["missing"] == "leave_country_unclassified"
            and model["output"]["supervised_feature"] is False
            and model["output"]["cluster_id_is_rank"] is False, "공통 표준화·사후 비교·미분류 유지 정책을 확인하세요.")
    require(clustering["allow_mixed_years"] is False
            and model["inputs"]["reference_year"] == clustering["reference_year"] == "three_year_window"
            and model["inputs"]["aggregation"] == clustering["aggregation"], "구간 평균·동일 연도 사용 정책이 다릅니다.")
    countries = [c["iso3"] for c in country_config["countries"]]
    windows = clustering["context"]["trade"]["windows"]
    window_ids = [w["window_id"] for w in windows]
    require(len(set(countries)) == len(countries) and len(set(window_ids)) == len(window_ids), "국가·구간 설정이 중복됐습니다.")
    require(set(countries) == set(parent["country_codes"]) and len(countries) == manifest["countries"]
            and window_ids == [w["window_id"] for w in manifest["windows"]], "공유 명세와 국가·구간 설정이 다릅니다.")
    parent_files = {entry["path"]: entry for entry in parent["files"]}
    records = manifest["files"]
    require(len({r["path"] for r in records}) == len(records), "비지도 파일 명세가 중복됐습니다.")
    hashes = {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in [parent_path, manifest_path]}
    paths = {}
    for entry in records:
        path = (root / entry["path"]).resolve()
        require(path.is_relative_to(folder) and path.suffix in {".csv", ".parquet"}, "비지도 공유 폴더 밖의 파일은 읽지 않습니다.")
        digest = sha256(path)
        require(entry["path"] in parent_files and digest == entry["sha256"] == parent_files[entry["path"]]["sha256"],
                f"공유 파일 체크섬 불일치: {path.name}")
        hashes[entry["path"]] = digest
        paths[path.name] = (path, entry)
    tables = {}
    for name, (path, entry) in paths.items():
        if path.suffix != ".parquet":
            continue
        require(path.with_suffix(".csv").name in paths, f"CSV 쌍이 없습니다: {name}")
        try:
            frame = check_formats(path)
        except (AssertionError, ValueError, TypeError) as error:
            raise ClusterError(f"CSV·Parquet 자료 불일치: {name}") from error
        schema = [{"name": c, "dtype": str(frame[c].dtype)} for c in frame]
        require(len(frame) == entry["rows"] and schema == entry["schema"], f"행 수·스키마 불일치: {name}")
        tables[path.stem] = frame
    needed = {"country_window_features", "country_window_profile", "country_window_context"}
    require(needed.issubset(tables), "군집 입력·설명 표가 없습니다.")
    features = validate_features(tables["country_window_features"], countries, window_ids)
    merge_explanations(merge_explanations(features, tables["country_window_profile"], "profile"),
                       tables["country_window_context"], "context")
    for window in windows:
        block = features.loc[features.window_id.eq(window["window_id"])]
        start = pd.Period(window.get("start_month", f"{window['start_year']}-01"), freq="M")
        end = pd.Period(window.get("end_month", f"{window['end_year']}-12"), freq="M")
        require(block.start_month.eq(str(start)).all() and block.end_month.eq(str(end)).all()
                and block.period_month_count.eq(end.ordinal-start.ordinal+1).all()
                and block.is_short_period.eq(window.get("short_period", False)).all(), "분석기간·개월 수가 설정과 다릅니다.")
        shared = next(w for w in manifest["windows"] if w["window_id"] == window["window_id"])
        require(int(block.eligible_for_fit.sum()) == shared["fit_eligible"], "구간별 적합 가능 수가 명세와 다릅니다.")
    require(len(features) == manifest["feature_rows"] == parent["unsupervised"]["feature_rows"]
            and int(features.eligible_for_fit.sum()) == manifest["fit_eligible_rows"] == parent["unsupervised"]["fit_eligible_rows"],
            "전체 행·적합 가능 행 수가 명세와 다릅니다.")
    audit = {"dataset_version":"v002", "rows":len(features), "countries":len(countries), "windows":len(windows),
             "eligible_rows":int(features.eligible_for_fit.sum()), "ineligible_rows":int((~features.eligible_for_fit).sum()),
             "feature_columns":FEATURES, "feature_missing_counts":{c:int(features[c].isna().sum()) for c in FEATURES},
             "per_window":[{"window_id":w, "rows":int(features.window_id.eq(w).sum()),
                            "eligible_rows":int(features.loc[features.window_id.eq(w), "eligible_for_fit"].sum())} for w in window_ids],
             "verified_files":len(records), "csv_parquet_pairs_verified":len(tables),
             "fit_scope":FIT_SCOPE, "historical_release_verified":False, "historical_backtest_claim":False}
    audit["ineligible"] = features.loc[~features.eligible_for_fit, KEY+["eligibility_reason"]].to_dict("records")
    return {"root":root, "features":features, "profile":tables["country_window_profile"],
            "context":tables["country_window_context"], "manifest":manifest, "config":model,
            "audit":audit, "source_sha256":hashes}


def model_inputs(frame):
    """Numeric order is fixed; no imputation, scaling, or fitting here."""
    return frame[FEATURES].astype("float64")


def fit_model(data, n_clusters):
    frame = validate_features(data["features"])
    require(isinstance(n_clusters, int) and not isinstance(n_clusters, bool) and n_clusters >= 2, "군집 수는 2 이상의 정수여야 합니다.")
    fit = frame.loc[frame.eligible_for_fit]
    require(len(fit) > n_clusters and len(fit[FEATURES].drop_duplicates()) >= n_clusters, "군집 수에 비해 유효 표본·서로 다른 입력이 부족합니다.")
    params = data["config"]["params"].copy()
    params["n_clusters"] = n_clusters
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", KMeans(**params))])
    pipeline.fit(model_inputs(fit))
    labels = pipeline.predict(model_inputs(fit))
    require(len(np.unique(labels)) == n_clusters, "실제 생성된 군집 수가 요청과 다릅니다.")
    return {"pipeline":pipeline, "model_id":"kmeans", "task":"country_clustering", "feature_columns":FEATURES.copy(),
            "categorical_columns":[], "dataset_version":"v002", "fit_scope":FIT_SCOPE,
            "fit_rows":len(fit), "fit_keys":fit[KEY].to_dict("records"), "n_clusters":n_clusters,
            "params":params, "source_sha256":data.get("source_sha256", {}).copy(),
            "python_version":platform.python_version(),
            "libraries":{name:importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow"]},
            "window_ids":sorted(frame.window_id.unique().tolist()),
            "model_version":"kmeans_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
            "historical_backtest_claim":False, "historical_release_verified":False}


def validate_model(bundle, data=None):
    require(isinstance(bundle, dict) and bundle.get("task") == "country_clustering"
            and bundle.get("model_id") == "kmeans" and bundle.get("dataset_version") == "v002"
            and bundle.get("fit_scope") == FIT_SCOPE and bundle.get("feature_columns") == FEATURES,
            "공유 사양에 맞는 K-means 모델이 아닙니다.")
    required = {"pipeline", "fit_rows", "fit_keys", "n_clusters", "source_sha256", "model_version"}
    require(required.issubset(bundle), f"저장 군집 모델 정보 누락: {sorted(required-set(bundle))}")
    require(isinstance(bundle["n_clusters"], int) and not isinstance(bundle["n_clusters"], bool)
            and bundle["n_clusters"] >= 2 and isinstance(bundle["fit_rows"], int)
            and bundle["fit_rows"] > bundle["n_clusters"] and isinstance(bundle["source_sha256"], dict)
            and isinstance(bundle["model_version"], str) and bool(bundle["model_version"]), "저장 군집 모델 정보의 자료형·범위가 잘못됐습니다.")
    pipeline = bundle.get("pipeline")
    require(isinstance(pipeline, Pipeline) and list(pipeline.named_steps) == ["scaler", "model"], "표준화기와 K-means가 포함된 Pipeline이 필요합니다.")
    require(isinstance(pipeline.named_steps["scaler"], StandardScaler)
            and isinstance(pipeline.named_steps["model"], KMeans), "군집 모델의 전처리기·모델 종류가 다릅니다.")
    try:
        check_is_fitted(pipeline.named_steps["scaler"])
        check_is_fitted(pipeline.named_steps["model"])
    except ValueError as error:
        raise ClusterError("아직 학습되지 않은 군집 모델입니다.") from error
    require(list(pipeline.feature_names_in_) == FEATURES and pipeline.named_steps["model"].cluster_centers_.shape == (bundle["n_clusters"], len(FEATURES))
            and np.isfinite(pipeline.named_steps["model"].cluster_centers_).all(), "저장 모델 입력·중심 정보가 명세와 다릅니다.")
    require(pipeline.named_steps["scaler"].with_mean and pipeline.named_steps["scaler"].with_std
            and int(pipeline.named_steps["scaler"].n_samples_seen_) == bundle["fit_rows"], "공통 표준화기 설정·적합 행 수가 다릅니다.")
    if data is not None:
        require(bundle["source_sha256"] == data.get("source_sha256", {}), "저장 모델의 공유 데이터 체크섬이 현재 입력과 다릅니다.")
        frame = validate_features(data["features"])
        require(bundle["fit_keys"] == frame.loc[frame.eligible_for_fit, KEY].to_dict("records"), "저장 모델의 적합 국가·구간이 현재 자료와 다릅니다.")
    return pipeline


def load_cluster_model(path, data=None):
    path = Path(path)
    require(path.suffix == ".joblib", "군집 모델 파일은 .joblib이어야 합니다.")
    if not path.is_file():
        raise FileNotFoundError(f"군집 모델이 없습니다. 학습 실행을 먼저 완료하세요: {path}")
    try:
        bundle = joblib.load(path)
    except (ValueError, EOFError, OSError, pickle.UnpicklingError) as error:
        raise ClusterError(f"저장 군집 모델을 읽을 수 없습니다: {path}") from error
    validate_model(bundle, data)
    return bundle


def save_model(bundle, path, data):
    pipeline = validate_model(bundle, data)
    path = Path(path)
    require(path.suffix == ".joblib", "군집 모델 파일은 .joblib이어야 합니다.")
    path.parent.mkdir(parents=True, exist_ok=True)
    x = model_inputs(data["features"].loc[data["features"].eligible_for_fit])
    expected = pipeline.predict(x)
    with path.open("xb") as handle:
        joblib.dump(bundle, handle, compress=3)
    try:
        loaded = load_cluster_model(path, data)
        np.testing.assert_array_equal(expected, loaded["pipeline"].predict(x))
    except Exception:
        # Only this newly created file is removed; pre-existing files use xb.
        path.unlink()
        raise
    return path


def compare_candidates(data):
    frame = validate_features(data["features"])
    fit = frame.loc[frame.eligible_for_fit]
    candidates = data["config"]["selection"]["candidate_cluster_counts"]
    require(candidates and len(set(candidates)) == len(candidates), "군집 수 후보가 비었거나 중복됐습니다.")
    rows, profiles = [], []
    for k in candidates:
        bundle = fit_model(data, k)
        pipeline = bundle["pipeline"]
        x = model_inputs(fit)
        scaled = pipeline.named_steps["scaler"].transform(x)
        labels = pipeline.predict(x)
        sizes = pd.Series(labels).value_counts().sort_index()
        silhouette = float(silhouette_score(scaled, labels)) if 2 <= len(sizes) < len(fit) else None
        rows.append({"n_clusters":k, "fit_rows":len(fit), "silhouette":silhouette,
                     "inertia":float(pipeline.named_steps["model"].inertia_),
                     "smallest_cluster":int(sizes.min()), "largest_cluster":int(sizes.max()),
                     "cluster_sizes":json.dumps({str(i):int(n) for i,n in sizes.items()})})
        values = fit[FEATURES].copy()
        values["cluster_id"] = labels
        profile = values.groupby("cluster_id")[FEATURES].mean().reset_index()
        profile["rows"] = profile.cluster_id.map(sizes)
        profile["n_clusters"] = k
        profiles.append(profile)
    # Do not select by score alone or publish a candidate as the final model.
    return pd.DataFrame(rows), pd.concat(profiles, ignore_index=True)


def check_sources(data):
    require(all(sha256(data["root"] / path) == digest for path, digest in data["source_sha256"].items()),
            "실행 중 공유 데이터·명세가 변경됐습니다.")


def run_cluster(*, mode, run_id=None, n_clusters=None, model_path=None, root=None):
    """Audit, compare, explicitly train, or infer; results always stay private."""
    require(mode in {"audit", "compare", "train", "predict"}, "지원하지 않는 군집 실행 모드입니다.")
    require(mode == "train" or n_clusters is None, "군집 수 지정은 train 모드에서만 가능합니다.")
    require(mode != "train" or n_clusters is not None, "train 모드에는 검토 후 선택한 --n-clusters가 필요합니다.")
    data = load_inputs(root)
    root = data["root"]
    run_id = run_id or ("kmeans_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    require(isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id), "실행 ID에는 영문·숫자·밑줄·하이픈만 사용하세요.")
    out = root / ".private/kmeans" / run_id
    if out.exists():
        raise FileExistsError(f"기존 군집 결과를 덮어쓰지 않습니다: {out}")
    path = Path(model_path) if model_path is not None else Path("artifacts/shared_models/kmeans/pipeline.joblib")
    path = (path if path.is_absolute() else root/path).resolve()
    bundle, metrics, profiles, result = None, None, None, None
    if mode == "train":
        base = (root / "artifacts/shared_models").resolve()
        require(path.parent.parent == base and re.fullmatch(r"kmeans(?:_[A-Za-z0-9_-]+)?", path.parent.name)
                and path.name == "pipeline.joblib", "공유 군집 모델은 shared_models/kmeans[_버전]/pipeline.joblib에 저장하세요.")
        if path.exists():
            raise FileExistsError(f"기존 군집 모델을 덮어쓰지 않습니다: {path}")
        require(n_clusters in data["config"]["selection"]["candidate_cluster_counts"], "설정에 등록된 군집 수 후보를 선택하세요.")
        bundle = fit_model(data, n_clusters)
    elif mode == "compare":
        metrics, profiles = compare_candidates(data)
    elif mode == "predict":
        bundle = load_cluster_model(path, data)
    if bundle is not None:
        from oinq.modeling.clustering.results import build_results
        result = build_results(data, bundle, run_id)
    check_sources(data)
    if mode == "train":
        save_model(bundle, path, data)
    out.mkdir(parents=True, exist_ok=False)
    (out / "input_audit.json").write_text(json.dumps(data["audit"], ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if metrics is not None:
        metrics.to_csv(out / "candidate_metrics.csv", index=False, encoding="utf-8-sig")
        profiles.to_parquet(out / "candidate_profiles.parquet", index=False)
    if result is not None:
        result.to_parquet(out / "country_window_clusters.parquet", index=False)
    metadata = {"status":"completed", "mode":mode, "run_id":run_id, "dataset_version":"v002",
                "generated_at_utc":datetime.now(timezone.utc).isoformat(), "rows":len(data["features"]),
                "eligible_rows":data["audit"]["eligible_rows"], "model_fitted":mode in {"compare", "train"},
                "final_model_saved":mode == "train", "source_sha256":data["source_sha256"],
                "fit_scope":FIT_SCOPE, "historical_backtest_claim":False, "historical_release_verified":False}
    if bundle is not None:
        metadata.update({"model_path":str(path), "model_sha256":sha256(path), "model_version":bundle["model_version"],
                         "n_clusters":bundle["n_clusters"], "classified_rows":int(result.cluster_id.notna().sum()),
                         "unclassified_rows":int(result.cluster_id.isna().sum())})
    (out / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {"output_dir":out, "metadata":metadata, "audit":data["audit"], "results":result, "metrics":metrics}
