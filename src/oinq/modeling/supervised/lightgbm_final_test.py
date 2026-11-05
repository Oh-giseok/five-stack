"""One frozen-origin final evaluation of the validation-selected LightGBM.

Selection is written before final-test rows are opened. Test labels are loaded
only after the fitted model and predictions are saved. No tuning or production
retraining is performed by this entry point.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import importlib.metadata

import joblib
import numpy as np
import pandas as pd

from oinq.common.config import load_yaml, quarter_number
from oinq.modeling.supervised.lightgbm import (
    KEY, TARGET, PRED, sha256, write_json, frame_hash, training_rows,
    prepare_x, make_pipeline, metric_values, baseline_predictions, git_state,
)

SELECTION_RUN = "lightgbm_v001_l2_lambda1_20261104_01"
VARIANT = "with_macro"


def check_features(features, contract):
    expected = contract["categorical"] + contract["known_future"] + contract["numeric_candidates"]
    if features != expected or len(set(features)) != len(features):
        raise ValueError("Feature order/uniqueness differs from shared contract")
    if set(features).intersection(contract["forbidden_feature_columns"]):
        raise ValueError("Forbidden target metadata in features")
    if contract["target"] != TARGET:
        raise ValueError("Wrong target")


def check_test_inputs(df, spec, countries, products):
    if df[KEY].isna().any().any() or df.duplicated(KEY).any():
        raise ValueError("Missing or duplicated final-test keys")
    if not df.origin_quarter.eq(spec["origin_quarter"]).all():
        raise ValueError("Final-test origin must stay frozen")
    if not df.target_quarter.isin(spec["target_quarters"]).all():
        raise ValueError("Unexpected final-test target")
    h = df.target_quarter.map(quarter_number) - df.origin_quarter.map(quarter_number)
    if not h.eq(df.horizon_quarters).all() or not h.isin(spec["horizon_quarters"]).all():
        raise ValueError("Final-test horizon mismatch")
    expected = pd.MultiIndex.from_product([countries, products, spec["target_quarters"]],
                                         names=["country_code", "hs_code", "target_quarter"])
    actual = pd.MultiIndex.from_frame(df[["country_code", "hs_code", "target_quarter"]])
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError("Final-test country/product/quarter grid is incomplete")


def select_scored_rows(df):
    available = df["target_available"].fillna(False)
    if not available.eq(df[TARGET].notna()).all():
        raise ValueError("Target availability mismatch")
    return df.loc[available].copy(), df.loc[~available].copy()


def test_metrics(predictions, model_id, primary_horizons, variant=None):
    records = []
    for scope, block in [("h1_3", predictions),
                         ("h2_3", predictions[predictions.horizon_quarters.isin(primary_horizons)])]:
        identity = dict(model_id=model_id, variant=variant if variant is not None else (VARIANT if model_id == "lightgbm" else "baseline"), scope=scope)
        records.append(identity | dict(group_by="overall", group_value="all") |
                       metric_values(block[TARGET], block[PRED]))
        for col in ["country_code", "hs_code", "horizon_quarters"]:
            for value, part in block.groupby(col, sort=True):
                records.append(identity | dict(group_by=col, group_value=str(value)) |
                               metric_values(part[TARGET], part[PRED]))
    return pd.DataFrame(records)


def run_final_test(root, run_id):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or run_id == "production":
        raise ValueError("Invalid experimental run ID")
    out = root / "artifacts" / run_id
    if out.exists():
        raise FileExistsError("Existing final-test run is never overwritten")
    shared = root / "data/shared/v002"
    selected = root / "artifacts" / SELECTION_RUN
    manifest = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    selected_meta = json.loads((selected / "run_metadata.json").read_text(encoding="utf-8"))
    selected_manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    selected_metrics = pd.read_csv(selected / "metrics.csv")
    choice = selected_metrics.loc[selected_metrics.model_id.eq("lightgbm") &
                selected_metrics.variant.eq(VARIANT) & selected_metrics.scope.eq("h2_5") &
                selected_metrics.group_by.eq("overall")]
    if len(choice) != 1 or selected_meta["status"] != "completed":
        raise ValueError("Missing completed validation selection")
    params = selected_meta["model_params"].copy()
    if params.get("reg_alpha", 0) != 0 or params.get("reg_lambda") != 1:
        raise ValueError("Approved selection is L1=0, L2=1")
    features = manifest["feature_columns"]
    if features != selected_manifest["feature_columns"]:
        raise ValueError("Selected model feature contract changed")
    check_features(features, manifest["feature_contract"])
    split = load_yaml(root / "configs/splits.yaml")
    feature_cfg = load_yaml(root / "configs/features.yaml")
    if split != selected_meta["split_config"]:
        raise ValueError("Split configuration differs from selected validation")
    spec = split["final_test"]
    if spec["origin_quarter"] != "2025Q4" or spec["target_quarters"] != ["2026Q1","2026Q2","2026Q3"]:
        raise ValueError("Unexpected approved final-test contract")
    if manifest["dataset_version"] != "v002" or not manifest["dataset_release_approved"]:
        raise ValueError("Approved shared v002 is required")
    if not manifest["final_test_included"]:
        raise ValueError("Released final-test input is missing")
    # Hash bytes for integrity; do not inspect any test target values here.
    checksums = {}
    files = {item["file"]:item for item in manifest["files"]}
    for name in ["train.csv","train.parquet","validation.csv","validation.parquet",
                 "final_test.csv","final_test.parquet"]:
        path = shared / name
        if sha256(path) != files[name]["sha256"]:
            raise ValueError(f"Released checksum mismatch: {name}")
        checksums[str(path.relative_to(root)).replace("\\","/")] = sha256(path)
        if name.startswith(("train.","validation.")):
            original = next(item for item in selected_manifest["files"] if item["file"] == name)
            if files[name]["sha256"] != original["sha256"]:
                raise ValueError("Train/validation bytes differ from validation selection")
    for path in [shared/"manifest.json",root/"configs/splits.yaml",root/"configs/features.yaml",
                 root/"configs/models/lightgbm.yaml"]:
        checksums[str(path.relative_to(root)).replace("\\","/")] = sha256(path)
    prior_evaluations = []
    for path in (root/"artifacts").glob("*/selection_lock.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("test_dataset_sha256") == files["final_test.parquet"]["sha256"]:
            prior_evaluations.append(str(path.parent.name))
    out.mkdir()
    lock = {
        "run_id": run_id, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_source_run": SELECTION_RUN, "selection_source_metadata_sha256":sha256(selected/"run_metadata.json"),
        "selection_source_metrics_sha256":sha256(selected/"metrics.csv"),
        "selection_basis":"lowest observed validation h2_5 WAPE among completed no-reg/L1=1/L2=1 experiments",
        "validation_wape":float(choice.wape.iloc[0]), "variant":VARIANT,
        "model_params":params, "input_columns":features,
        "origin_quarter":spec["origin_quarter"], "target_quarters":spec["target_quarters"],
        "horizons":spec["horizon_quarters"], "negative_prediction_policy":"clip_to_zero",
        "test_dataset_sha256":files["final_test.parquet"]["sha256"],
        "protected_checksums":checksums, "prior_recorded_evaluations":prior_evaluations,
        "test_labels_team_shared":True,
        "holdout_claim":"first local recorded evaluation only if prior list empty; team-wide unseen status not established",
        "tuning_allowed":False,"post_test_parameter_changes":False,
        "historical_asof_verified":False,
        "availability_limitation":manifest["availability_limitation"],
        "git":git_state(root),
    }
    write_json(out/"selection_lock.json",lock)
    lock_hash = sha256(out/"selection_lock.json")
    print("Selection frozen: with_macro, L1=0, L2=1, origin=2025Q4",flush=True)
    write_json(out/"manifest.json",manifest)
    write_json(out/"model_params.json",params)
    versions = {name:importlib.metadata.version(name) for name in
                ["lightgbm","scikit-learn","numpy","pandas","pyarrow","scipy","joblib","PyYAML"]}
    installed=sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
                       if d.metadata['Name'] and d.metadata['Name'].lower() not in
                       {"pip","setuptools","oinq-export-forecast"}})
    (out/"requirements-lock.txt").write_text("\n".join(installed)+"\n",encoding="utf-8")
    code=out/"code_snapshot"
    code.mkdir()
    for path in [Path(__file__),root/"src/oinq/modeling/supervised/lightgbm.py",
                 root/"scripts/evaluate_lightgbm_final_test.py",root/"tests/test_lightgbm_final_test.py"]:
        (code/path.name).write_bytes(path.read_bytes())
    # Fit only the train and validation partitions, never final_test.
    frames=[]
    for name in ["train","validation"]:
        df=pd.read_parquet(shared/f"{name}.parquet")
        if len(df)!=manifest["row_counts"][name]["rows"] or df.duplicated(KEY).any():
            raise ValueError("Training partition count or keys mismatch")
        bounds=split["splits"][name]
        q=df.target_quarter.map(quarter_number)
        if not q.between(quarter_number(bounds["start"]),quarter_number(bounds["end"])).all():
            raise ValueError("Target partition changed")
        if not df.split.eq(name).all() or not df.target_available.eq(df[TARGET].notna()).all():
            raise ValueError("Training partition labels mismatch")
        df["_target_index"],df["_origin_index"]=q,df.origin_quarter.map(quarter_number)
        frames.append(df)
    pool=pd.concat(frames,ignore_index=True)
    if pool.duplicated(KEY).any():raise ValueError("Training keys duplicated")
    fit=training_rows(pool,spec["origin_quarter"])
    if fit.target_quarter.max()>spec["origin_quarter"] or fit.target_quarter.str.startswith("2026").any():
        raise ValueError("Test labels entered fitting pool")
    if not np.isfinite(fit[TARGET].to_numpy(dtype=float)).all() or fit[TARGET].lt(0).any():
        raise ValueError("Invalid training target")
    fit[KEY+["horizon_quarters","split"]].to_parquet(out/"fit_rows.parquet",index=False)
    categorical=manifest["feature_contract"]["categorical"]
    pipe,used,dropped=make_pipeline(fit,features,categorical,params)
    pipe.fit(prepare_x(fit,used,categorical),fit[TARGET].to_numpy(dtype=float))
    modeldir=out/"model"
    modeldir.mkdir()
    joblib.dump(pipe,modeldir/"pipeline.joblib",compress=3)
    (modeldir/"model.txt").write_text(pipe.named_steps["model"].booster_.model_to_string(),encoding="utf-8")
    native=pipe.named_steps["model"].booster_.params
    if float(native.get("reg_lambda",native.get("lambda_l2",0)))!=1:
        raise AssertionError("L2 selection mismatch")
    write_json(modeldir/"input_columns.json",{"manifest_selected":features,"raw_input_order":used,
             "categorical":categorical,"dropped_all_missing":dropped,
             "transformed_order":pipe.named_steps["transformer"].get_feature_names_out().tolist()})
    fit_completed=datetime.now(timezone.utc).isoformat()
    print(f"Fit completed: {len(fit):,} historical rows, latest target {fit.target_quarter.max()}",flush=True)
    # First reading of final-test rows: only input columns and identifiers.
    input_cols=list(dict.fromkeys(KEY+["horizon_quarters"]+features))
    xrows=pd.read_parquet(shared/"final_test.parquet",columns=input_cols)
    check_test_inputs(xrows,spec,manifest["country_codes"],manifest["hs_codes"])
    x=prepare_x(xrows,used,categorical)
    raw=pipe.predict(x)
    if not np.isfinite(raw).all():raise ValueError("Non-finite test predictions")
    pred=np.maximum(raw,feature_cfg["target"]["minimum_prediction"])
    replay=np.maximum(joblib.load(modeldir/"pipeline.joblib").predict(x),0)
    if not np.array_equal(pred,replay):raise AssertionError("Reload predictions differ")
    prediction=xrows[KEY+["horizon_quarters"]].copy()
    prediction[PRED],prediction["raw_predicted_usd"]=pred,raw
    prediction["model_id"],prediction["variant"]="lightgbm",VARIANT
    prediction["run_id"],prediction["dataset_version"]=run_id,"v002"
    prediction["origin_cutoff_at"]=None
    prediction.to_parquet(out/"predictions_before_labels.parquet",index=False)
    predictions_saved=datetime.now(timezone.utc).isoformat()
    # Only now load actual targets for scoring; no refit occurs below this line.
    test_labels_opened=datetime.now(timezone.utc).isoformat()
    labels=pd.read_parquet(shared/"final_test.parquet",columns=KEY+[TARGET,"target_available","split","currency","valuation"])
    if not labels.split.eq("final_test").all() or not labels.currency.eq("USD").all() or not labels.valuation.eq("FOB").all():
        raise ValueError("Test partition or units mismatch")
    result=prediction.merge(labels,on=KEY,validate="one_to_one")
    if len(result)!=manifest["row_counts"]["final_test"]["rows"]:
        raise ValueError("Final-test count mismatch")
    scored,missing=select_scored_rows(result)
    if not np.isfinite(scored[TARGET].to_numpy(dtype=float)).all() or scored[TARGET].lt(0).any():
        raise ValueError("Invalid test actuals")
    result["evaluation_status"]=np.where(result.target_available,"scored","missing_actual_target")
    result["error_usd"]=result[PRED]-result[TARGET]
    result.to_parquet(out/"final_test_predictions.parquet",index=False)
    result.to_csv(out/"final_test_predictions.csv",index=False,encoding="utf-8-sig",na_rep="\\N")
    missing[KEY+["horizon_quarters","target_available"]].assign(reason="missing_actual_target").to_parquet(
        out/"evaluation_exclusions.parquet",index=False)
    metrics=test_metrics(scored,"lightgbm",split["validation"]["primary_horizons"])
    metrics.to_csv(out/"metrics.csv",index=False,encoding="utf-8-sig")
    baseline=baseline_predictions(result, pool)
    baseline["evaluation_status"]=np.where(baseline[TARGET].isna(),"missing_actual_target",
                               np.where(baseline[PRED].isna(),"missing_baseline_history","scored"))
    baseline.to_parquet(out/"baseline_predictions.parquet",index=False)
    base_scored=baseline.loc[baseline[TARGET].notna()]
    test_metrics(base_scored,"seasonal_naive",split["validation"]["primary_horizons"]).to_csv(
        out/"baseline_metrics.csv",index=False,encoding="utf-8-sig")
    common=baseline.loc[baseline[TARGET].notna() & baseline[PRED].notna(),KEY]
    shared_model=scored.merge(common,on=KEY,validate="one_to_one")
    shared_base=baseline.merge(common,on=KEY,validate="one_to_one")
    pd.concat([test_metrics(shared_model,"lightgbm",split["validation"]["primary_horizons"]),
               test_metrics(shared_base,"seasonal_naive",split["validation"]["primary_horizons"])],ignore_index=True).to_csv(
               out/"metrics_baseline_common_rows.csv",index=False,encoding="utf-8-sig")
    if sha256(out/"selection_lock.json")!=lock_hash:
        raise AssertionError("Selection changed after test exposure")
    for path,expected in checksums.items():
        if sha256(root/path)!=expected:raise AssertionError(f"Protected input changed: {path}")
    metadata={"run_id":run_id,"status":"completed","dataset_version":"v002","variant":VARIANT,
              "model_params":params,"libraries":versions,"selection_lock_sha256":lock_hash,
              "origin_quarter":spec["origin_quarter"],"fit_rows":len(fit),
              "fit_min_target":fit.target_quarter.min(),"fit_max_target":fit.target_quarter.max(),
              "fit_rows_sha256":frame_hash(fit,KEY),"test_rows":len(result),"scored_rows":len(scored),
              "missing_actual_rows":len(missing),"fitted_at_utc":fit_completed,
              "predictions_saved_before_labels_at_utc":predictions_saved,
              "test_labels_first_loaded_at_utc":test_labels_opened,
              "reload_max_abs_diff":float(np.max(np.abs(pred-replay))),
              "protected_inputs_unchanged":True,"selection_unchanged":True,
              "historical_asof_verified":False,"origin_cutoff_at":None,
              "post_test_tuning":False,"production_retraining":False,
              "h4_h5_evaluated":False,"four_quarter_metrics_generated":False,
              "prior_recorded_test_evaluations":prior_evaluations}
    write_json(out/"run_metadata.json",metadata)
    overall=metrics.loc[metrics.group_by.eq("overall")]
    note=f'''# LightGBM 고정 설정 최종 테스트

선정: {SELECTION_RUN}, 거시 포함, L1=0, L2=1.0. 검증 h2~5 WAPE {lock['validation_wape']:.6f}%를 근거로 테스트 전 고정했습니다.
학습: train·validation의 2025Q4까지 유효 정답 {len(fit):,}행. 테스트: origin=2025Q4, 2026Q1~Q3 h1~3.
입력 {len(features)}개를 사용하며 전처리는 적합 자료에서만 학습했습니다. 테스트 입력·정답은 적합에 사용하지 않았습니다.
테스트 {len(result)}행 모두 예측했고 정답 결측 {len(missing)}행은 보존·평가 제외해 {len(scored)}행을 평가했습니다.

{overall[['scope','n','wape','mae','rmse','smape']].to_string(index=False)}

WAPE·sMAPE는 %, MAE·RMSE는 USD입니다. h1~3 테스트와 기존 h2~5 검증은 기간·거리가 달라 점수를 직접 성능 개선으로 해석하지 않습니다.
3분기 테스트이므로 h4~5·4분기 연간 성능을 산출하지 않았습니다. 과거 공개시점·빈티지가 미검증인 현재 수정치 기반 평가입니다.
테스트 자료는 팀에 공유된 상태로, 팀 전체에서 한 번도 본 적 없는 평가라고 주장하지 않습니다. 이 도구의 이전 평가 기록: {prior_evaluations}.

## 파일
- selection_lock.json: 테스트 전 선택 모델·파라미터·입력·데이터 해시 고정.
- fit_rows.parquet: 실제 적합한 과거 행과 원래 train/validation 구분.
- model/: 재적합 모델·변환기 파이프라인, native model, 입력 컬럼 순서.
- predictions_before_labels.parquet: 실제값 열기 전에 저장한 예측.
- final_test_predictions.csv/parquet: 실측·예측·거리·평가 상태·오차.
- metrics.csv: 전체·국가·HS·거리별 지표. h1_3 전체와 운영 거리 교집합 h2_3.
- baseline_predictions.parquet / baseline_metrics.csv: 기준선 결과.
- metrics_baseline_common_rows.csv: 기준선과 모델의 공통 행 비교.
- evaluation_exclusions.parquet: 정답 없는 행. README의 n은 고유 테스트 예측 수입니다.
- run_metadata.json / code_snapshot / requirements-lock.txt: 시점·설정·버전·코드 재현 정보.

## 재실행
`.venv/Scripts/python.exe scripts/evaluate_lightgbm_final_test.py --run-id NEW_UNIQUE_ID`
기존 결과는 덮어쓰지 않습니다. 반복 실행은 이전 테스트 사용 이력을 함께 기록합니다. 테스트를 보고 재설정하면 같은 결과를 미사용 최종 평가라고 부르지 않습니다.
공통 데이터·YAML·기존 검증 결과는 변경하지 않았습니다. 이번 실행에서 튜닝·운영 재학습·2027 예측은 하지 않았습니다.
'''
    (out/"README_KO.md").write_text(note,encoding="utf-8")
    write_json(out/"output_checksums.json",{str(p.relative_to(out)).replace("\\","/"):sha256(p)
                for p in out.rglob("*") if p.is_file() and p.name!="output_checksums.json"})
    print(overall[["scope","n","wape","mae","rmse","smape"]].to_string(index=False),flush=True)
    print(f"Saved: {out}",flush=True)
    return out
