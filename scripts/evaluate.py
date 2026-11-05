"""Compare existing training outputs on identical validation keys and fit pools."""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.modeling.supervised.lightgbm import (
    KEY, TARGET, PRED, frame_hash, sha256, write_json, grouped_metrics,
    annual_results, training_rows, quarter_number,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgboost-run", type=Path)
    parser.add_argument("--lightgbm-run", type=Path)
    parser.add_argument("--catboost-run", type=Path)
    parser.add_argument("--run", action="append", default=[], help="family:experiment:artifacts/run_folder; repeat for all supplied experiments")
    parser.add_argument("--final-test", action="store_true", help="Freeze all validated candidates, fit through 2025Q4 and evaluate 2026Q1-Q3")
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.final_test:
        if args.validation_report is None:
            parser.error("--final-test requires --validation-report")
        return run_all_final_tests(args.validation_report.resolve(), args.output_dir.resolve(),
                                   args.report_dir.resolve() if args.report_dir else None)
    shared = ROOT / "data/shared/v002"
    manifest = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    # No final-test files are opened here.
    train = pd.read_parquet(shared / "train.parquet")
    validation = pd.read_parquet(shared / "validation.parquet")
    expected = validation.loc[validation[TARGET].notna() & validation.target_available.fillna(False)].sort_values(KEY)
    pool = pd.concat([train, validation], ignore_index=True)
    pool["_target_index"] = pool.target_quarter.map(quarter_number)
    pool["_origin_index"] = pool.origin_quarter.map(quarter_number)
    expected_hash = frame_hash(expected, KEY)
    expected_fits = {}
    for origin in sorted(expected.origin_quarter.unique()):
        fit = training_rows(pool, origin)
        expected_fits[origin] = (len(fit), frame_hash(fit, KEY))
    if args.run:
        if any([args.xgboost_run, args.lightgbm_run, args.catboost_run]):
            parser.error("Use repeated --run OR the three legacy run arguments")
        folders = {}
        for entry in args.run:
            family, experiment, location = entry.split(":", 2)
            if family not in {"xgboost", "lightgbm", "catboost"}:
                parser.error(f"Unsupported family: {family}")
            model_id = family if experiment == "baseline" else f"{family}_{experiment}"
            if model_id in folders:
                parser.error(f"Duplicate experiment: {model_id}")
            folders[model_id] = (family, experiment, Path(location).resolve())
    else:
        if not all([args.xgboost_run, args.lightgbm_run, args.catboost_run]):
            parser.error("Supply all three model runs or repeated --run")
        folders = {family: (family, "baseline", location.resolve()) for family, location in
                   [("xgboost", args.xgboost_run), ("lightgbm", args.lightgbm_run), ("catboost", args.catboost_run)]}
    models, audits, sources, artifacts = [], [], {}, []
    baseline = None
    for model, (family, experiment, folder) in folders.items():
        path = folder / "validation_predictions.csv"
        pred = pd.read_csv(path, dtype={"country_code": "string", "hs_code": "string",
                                      "origin_quarter": "string", "target_quarter": "string"}, na_values=[r"\N"],
                           low_memory=False, float_precision="round_trip")
        if family == "xgboost":
            pred = pred.rename(columns={"actual_usd": TARGET, "predicted_usd": PRED, "model_variant": "variant"})
            meta = json.loads((folder / "experiment_config.json").read_text(encoding="utf-8"))
            metadata_file = folder / "experiment_config.json"
            for entry in meta["models"]:
                file = folder / entry["file"]
                if sha256(file) != entry["sha256"]:
                    raise ValueError(f"Changed saved model: {file}")
                artifacts.append({"model_id": model, "path": file.relative_to(ROOT).as_posix(), **entry})
        elif family == "catboost":
            pred = pred.rename(columns={"prediction_usd": PRED})
            meta = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            metadata_file = folder / "run_manifest.json"
            for entry in meta["models"]:
                file = folder / entry["model_file"]
                if sha256(file) != entry["sha256"]:
                    raise ValueError(f"Changed saved model: {file}")
                artifacts.append({"model_id": model, "path": file.relative_to(ROOT).as_posix(), **entry})
        else:
            meta = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
            metadata_file = folder / "run_metadata.json"
            candidate_baseline = pred.loc[pred.model_id.eq("seasonal_naive")].copy()
            if baseline is not None:
                a = baseline.sort_values(KEY)[PRED].to_numpy()
                b = candidate_baseline.sort_values(KEY)[PRED].to_numpy()
                if not np.array_equal(a, b, equal_nan=True):
                    raise ValueError("Seasonal baseline changed across runs")
            baseline = candidate_baseline
            pred = pred.loc[pred.model_id.eq("lightgbm")].copy()
            checksums = json.loads((folder / "output_checksums.json").read_text(encoding="utf-8"))
            for relative, checksum in checksums.items():
                if sha256(folder / relative) != checksum:
                    raise ValueError(f"Changed LightGBM output: {relative}")
                if relative.endswith("pipeline.joblib"):
                    artifacts.append({"model_id": model, "path": (folder / relative).relative_to(ROOT).as_posix(), "sha256": checksum})
        if meta.get("final_test_used") is not False or not meta.get("reload_verified"):
            raise ValueError(f"Invalid validation metadata: {model}")
        if set(pred.variant) != {"with_macro", "without_macro"}:
            raise ValueError(f"Missing macro pair: {model}")
        pred["model_id"] = model
        for variant, block in pred.groupby("variant"):
            if block.duplicated(KEY).any() or frame_hash(block, KEY) != expected_hash:
                raise ValueError(f"Evaluation keys differ: {model}/{variant}")
            if not np.isfinite(block[PRED]).all() or block[PRED].lt(0).any():
                raise ValueError(f"Invalid predictions: {model}/{variant}")
            merged = block.merge(expected[KEY + [TARGET, "horizon_quarters"]], on=KEY, suffixes=("", "_expected"), validate="one_to_one")
            if not np.array_equal(merged[TARGET].to_numpy(), merged[f"{TARGET}_expected"].to_numpy()) or not merged.horizon_quarters.eq(merged.horizon_quarters_expected).all():
                raise ValueError(f"Evaluation actuals/horizons differ: {model}/{variant}")
        audit = pd.read_csv(folder / "fit_audit.csv")
        if len(audit) != 2 * len(expected_fits) or audit.duplicated(["variant", "origin_quarter"]).any():
            raise ValueError(f"Incomplete fit audit: {model}")
        for row in audit.itertuples():
            if (row.fit_rows, row.fit_rows_sha256) != expected_fits[row.origin_quarter]:
                raise ValueError(f"Training pools differ: {model}/{row.origin_quarter}")
        audit["model_id"] = model
        audits.append(audit)
        models.append(pred[KEY + ["horizon_quarters", TARGET, PRED, "model_id", "variant"]])
        sources[model] = {"directory": folder.relative_to(ROOT).as_posix(), "predictions_sha256": sha256(path),
                          "family": family, "experiment": experiment,
                          "metadata_file": metadata_file.relative_to(ROOT).as_posix(),
                          "metadata_sha256": sha256(metadata_file)}
    learned = pd.concat(models, ignore_index=True)
    if baseline is None or frame_hash(baseline, KEY) != expected_hash:
        raise ValueError("Baseline evaluation key mismatch")
    all_predictions = pd.concat([learned, baseline[learned.columns]], ignore_index=True)
    metrics = grouped_metrics(all_predictions, [2, 3, 4, 5])
    summary = metrics.loc[metrics.scope.eq("h2_5") & metrics.group_by.eq("overall")].sort_values("wape")
    common_keys = baseline.loc[baseline[PRED].notna(), KEY]
    common_metrics = grouped_metrics(all_predictions.merge(common_keys, on=KEY, validate="many_to_one"), [2,3,4,5])
    sums, annual, excluded = annual_results(all_predictions)
    for entry in manifest["files"]:
        if Path(entry["path"]).name in {"train.csv", "train.parquet", "validation.csv", "validation.parquet"}:
            if sha256(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("Shared training/validation source changed")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "summary_h2_5.csv", index=False, encoding="utf-8-sig")
    common_metrics.to_csv(out / "metrics_baseline_common_rows.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(out / "four_quarter_metrics.csv", index=False, encoding="utf-8-sig")
    sums.to_parquet(out / "four_quarter_sums.parquet", index=False)
    excluded.to_parquet(out / "four_quarter_exclusions.parquet", index=False)
    pd.concat(audits, ignore_index=True).to_csv(out / "fit_audit.csv", index=False, encoding="utf-8-sig")
    selected = summary.loc[summary.model_id.ne("seasonal_naive")].iloc[0]
    details = {"status": "completed", "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
               "dataset_version": "v002", "target": TARGET, "currency": "USD", "valuation": "FOB",
               "validation_period": manifest["validation_period"], "evaluation_keys_sha256": expected_hash,
               "rows_per_variant": len(expected), "missing_validation_targets": len(validation)-len(expected),
               "origins": len(expected_fits), "countries": int(expected.country_code.nunique()),
               "products": int(expected.hs_code.nunique()), "model_variants": len(folders)*2,
               "primary_horizons": [2,3,4,5], "fit_pool": "train plus past validation labels with target <= origin",
               "same_evaluation_keys_verified": True, "same_training_keys_verified": True,
               "model_reload_verified": True, "final_test_used": False, "historical_asof_verified": False,
               "parameters_tuned_in_this_run": any(experiment != "baseline" for _, experiment, _ in folders.values()),
               "parameter_selection_split": "validation", "submitted_settings_only": True,
               "baseline_missing_predictions": int(baseline[PRED].isna().sum()),
               "best_validation": {k: selected[k] for k in ["model_id", "variant", "wape", "mae", "rmse", "smape"]},
               "sources": sources, "saved_models": artifacts,
               "training_code_sha256": {name: sha256(ROOT / relative) for name, relative in {
                   "xgboost": "src/oinq/modeling/supervised/xgboost_model.py",
                   "lightgbm": "src/oinq/modeling/supervised/lightgbm.py",
                   "catboost": "scripts/train_catboost.py"}.items()},
               "evaluation_code_sha256": sha256(Path(__file__)),
               "environment_lock_sha256": sha256(ROOT / "requirements-training-windows-py312.lock.txt"),
               "limitation": "Current revised data; historical release dates unverified. Validation selection is not final-test performance."}
    write_json(out / "manifest.json", details)
    lines = ["# 공통 학습·검증 결과 — 2026-11-05", "",
             f"팀원 기존 코드를 사용한 {len(folders)}개 실험 × 거시 포함/제외 {len(folders)*2}구성. 이 보고서는 검증 결과이며 테스트 전 선택 근거로 고정한다.", "",
             "- 공유 v002: 학습 2013Q1~2021Q4, 검증 2022Q1~2025Q4.",
             f"- 검증 정답 {len(expected):,}행/구성, 국가 20개·제품 8개·origin {len(expected_fits)}개. 정답 결측 {len(validation)-len(expected):,}행은 원본에 유지.",
             "- 각 origin까지의 train+과거 validation 정답만 사용. 모든 모델의 적합 키·평가 키 동일성 검산.",
             "- 모델별 입력 변환과 기존 기본 파라미터 유지. XGBoost 수치 결측 0, LightGBM 중앙값, CatBoost native NaN.",
             "- WAPE·sMAPE는 %, MAE·RMSE는 USD. 작은 값이 좋음. 주 비교 h=2~5.", "",
             "| 모델 | 구성 | 평가 행 | WAPE(%) | MAE(USD) | RMSE(USD) | sMAPE(%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.model_id} | {row.variant} | {row.n:,} | {row.wape:.4f} | {row.mae:,.2f} | {row.rmse:,.2f} | {row.smape:.4f} |")
    lines += ["", f"검증상 1위: **{selected.model_id} / {selected.variant}**, WAPE **{selected.wape:.4f}%**.", "",
              f"학습 모델 {len(artifacts)}개를 기존 artifacts 하위에 저장했다. 모든 모델은 재로드 예측 일치를 확인했다.",
              "manifest.json에 실제 모델 경로·체크섬과 선택 근거를 기록했다. 평가용 origin별 모델이며 최신 운영용 재학습 모델과 구분한다.", "",
              "현재 수정치 기반 검증이며 과거 최초 공개시점 재현은 미검증이다. 최종 테스트 성능·2027 운영 성능으로 해석하지 않는다.",
              "Seasonal Naive 결측은 0으로 채우지 않았다. 기준선과의 엄밀한 비교는 metrics_baseline_common_rows.csv의 동일 교집합 행을 사용한다.",
              "4분기 합계는 같은 국가·제품·origin에서 해당 4개 horizon이 모두 있는 경우만 평가하며 반드시 달력연간은 아니다.",
              "팀원 제출 설정을 비교했으며 별도 추가 탐색은 하지 않았다. 최종 테스트는 이 검증을 완료한 후 별도 실행한다. 최신 운영용 재학습·커밋·푸시는 포함하지 않았다."]
    (out / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(summary[["model_id", "variant", "n", "wape", "mae", "rmse", "smape"]].to_string(index=False))
    print(f"Saved models: {len(artifacts)}; fit pools/evaluation keys/reload verified; report: {out}")


def run_all_final_tests(validation_folder, out, report_dir=None):
    """Evaluate every supplied configuration; test labels never choose/refit a model."""
    import joblib
    from oinq.common.config import load_yaml
    from oinq.modeling.supervised.lightgbm import load_inputs, make_pipeline as light_pipeline, prepare_x, baseline_predictions, git_state
    from oinq.modeling.supervised.xgboost_model import make_pipeline as xgb_pipeline, inputs as xgb_inputs
    from oinq.modeling.supervised.lightgbm_final_test import check_features, check_test_inputs, select_scored_rows, test_metrics
    from catboost import CatBoostRegressor
    import importlib.util
    cat_spec = importlib.util.spec_from_file_location("catboost_training_existing", ROOT / "scripts/train_catboost.py")
    cat_module = importlib.util.module_from_spec(cat_spec)
    cat_spec.loader.exec_module(cat_module)
    validation_manifest = json.loads((validation_folder / "manifest.json").read_text(encoding="utf-8"))
    if validation_manifest["status"] != "completed" or validation_manifest["final_test_used"]:
        raise ValueError("A completed validation-only comparison is required")
    summary = pd.read_csv(validation_folder / "summary_h2_5.csv")
    eligible_summary = summary.loc[summary.model_id.ne("seasonal_naive")].sort_values("wape")
    chosen = eligible_summary.iloc[0]
    expected_choice = validation_manifest["best_validation"]
    if chosen.model_id != expected_choice["model_id"] or chosen.variant != expected_choice["variant"] or not np.isclose(chosen.wape, expected_choice["wape"]):
        raise ValueError("Validation-selected candidate changed")
    train, val, manifest, feature_config, split_config, _, _, protected = load_inputs(ROOT)
    check_features(manifest["feature_columns"], manifest["feature_contract"])
    spec = split_config["final_test"]
    if spec["origin_quarter"] != "2025Q4" or spec["target_quarters"] != ["2026Q1", "2026Q2", "2026Q3"]:
        raise ValueError("Final-test protocol changed")
    pool = pd.concat([train, val], ignore_index=True)
    fit = training_rows(pool, spec["origin_quarter"])
    if fit.target_quarter.max() > "2025Q4" or fit.target_quarter.str.startswith("2026").any():
        raise ValueError("Test labels leaked into fitting")
    categorical = manifest["feature_contract"]["categorical"]
    candidates = []
    for model_id, source in validation_manifest["sources"].items():
        metadata_path = ROOT / source["metadata_file"]
        if sha256(metadata_path) != source["metadata_sha256"]:
            raise ValueError("Validated configuration metadata changed")
        if sha256(ROOT / source["directory"] / "validation_predictions.csv") != source["predictions_sha256"]:
            raise ValueError("Validation predictions changed")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        family = source["family"]
        for variant in ["with_macro", "without_macro"]:
            if family == "xgboost":
                params = meta["parameters"]
                features = meta["feature_columns_by_variant"][variant]
                scaler = bool(meta.get("standard_scaler", False))
            elif family == "catboost":
                params, features, scaler = meta["params"], meta["variants_feature_columns"][variant], False
            else:
                params, scaler = meta["model_params"], False
                features = [c for c in manifest["feature_columns"] if variant == "with_macro" or c not in meta["macro_removed_columns"]]
            if set(features).intersection(manifest["feature_contract"]["forbidden_feature_columns"]):
                raise ValueError("Forbidden test input feature")
            macro_columns = feature_config["supervised"]["macro"]["columns"]
            removed = set(macro_columns + [f"{name}_missing" for name in macro_columns])
            expected_features = [name for name in manifest["feature_columns"] if variant == "with_macro" or name not in removed]
            if features != expected_features:
                raise ValueError("Validated feature order differs from the common contract")
            score = eligible_summary.loc[eligible_summary.model_id.eq(model_id) & eligible_summary.variant.eq(variant)]
            if len(score) != 1:
                raise ValueError("Missing validated candidate")
            candidates.append({"model_id": model_id, "family": family, "experiment": source["experiment"],
                               "variant": variant, "params": params, "feature_columns": features,
                               "standard_scaler": scaler, "validation_wape": float(score.wape.iloc[0]),
                               "validation_metadata_sha256": source["metadata_sha256"]})
    if len(candidates) != validation_manifest["model_variants"] or len({(c["model_id"], c["variant"]) for c in candidates}) != len(candidates):
        raise ValueError("Incomplete candidate inventory")
    files = {entry["file"]: entry for entry in manifest["files"]}
    # Hash the test bytes for identity, without loading their labels.
    test_path = ROOT / "data/shared/v002/final_test.parquet"
    if sha256(test_path) != files["final_test.parquet"]["sha256"]:
        raise ValueError("Released test checksum mismatch")
    protected["data/shared/v002/final_test.parquet"] = files["final_test.parquet"]["sha256"]
    previous_tests = [p.parent.name for p in (ROOT / "artifacts").glob("*/selection_lock.json")]
    for relative in ["scripts/evaluate.py", "scripts/train_catboost.py", "src/oinq/modeling/supervised/xgboost_model.py",
                     "src/oinq/modeling/supervised/lightgbm.py", "src/oinq/modeling/supervised/lightgbm_final_test.py",
                     "requirements-training-windows-py312.lock.txt"]:
        protected[relative] = sha256(ROOT / relative)
    out.mkdir(parents=True, exist_ok=False)
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "origin_quarter": spec["origin_quarter"],
              "target_quarters": spec["target_quarters"], "horizons": spec["horizon_quarters"],
              "selected_by_validation": expected_choice, "candidates": candidates,
              "validation_manifest_sha256": sha256(validation_folder / "manifest.json"),
              "validation_metrics_sha256": sha256(validation_folder / "summary_h2_5.csv"),
              "test_dataset_sha256": files["final_test.parquet"]["sha256"],
              "fit_rows_sha256": frame_hash(fit, KEY), "fit_rows": len(fit),
              "fit_max_target": fit.target_quarter.max(), "test_labels_used_for_fit": False,
              "post_test_tuning_allowed": False, "previous_local_tests": previous_tests,
              "test_labels_team_shared": True, "team_wide_unseen_status_established": False,
              "historical_asof_verified": False, "git": git_state(ROOT)}
    frozen["protected_checksums"] = protected
    write_json(out / "selection_lock.json", frozen)
    lock_hash = sha256(out / "selection_lock.json")
    fit[KEY + ["horizon_quarters", "split"]].to_parquet(out / "fit_rows.parquet", index=False)
    print(f"Frozen {len(candidates)} candidates. Selected by validation: {chosen.model_id}/{chosen.variant}; fit rows={len(fit)}", flush=True)
    input_columns = list(dict.fromkeys(KEY + ["horizon_quarters"] + manifest["feature_columns"]))
    test_inputs = pd.read_parquet(test_path, columns=input_columns)
    check_test_inputs(test_inputs, spec, manifest["country_codes"], manifest["hs_codes"])
    saved, predictions = [], []
    for index, candidate in enumerate(candidates, 1):
        family, model_id, variant = candidate["family"], candidate["model_id"], candidate["variant"]
        features, params = candidate["feature_columns"], candidate["params"]
        folder = out / model_id / variant
        folder.mkdir(parents=True)
        if family == "xgboost":
            model = xgb_pipeline(features, categorical, params=params, standard_scaler=candidate["standard_scaler"])
            xfit, xtest = xgb_inputs(fit, features, categorical), xgb_inputs(test_inputs, features, categorical)
            model.fit(xfit, fit[TARGET].to_numpy(dtype=float))
            raw = model.predict(xtest)
            path = folder / "pipeline.joblib"
            joblib.dump({"pipeline": model, "feature_columns": features, "categorical_columns": categorical,
                         "dataset_version": "v002", "origin_quarter": "2025Q4", "model_purpose": "final_test",
                         "trained_through_target_quarter": str(fit.target_quarter.max())}, path, compress=3)
            model.named_steps["model"].save_model(folder / "model.ubj")
            restored = joblib.load(path)["pipeline"]
            replay = restored.predict(xtest)
            columns = {"raw_input_order": features, "categorical": categorical, "standard_scaler": candidate["standard_scaler"]}
        elif family == "lightgbm":
            model, used, dropped = light_pipeline(fit, features, categorical, params)
            xfit, xtest = prepare_x(fit, used, categorical), prepare_x(test_inputs, used, categorical)
            model.fit(xfit, fit[TARGET].to_numpy(dtype=float))
            raw = model.predict(xtest)
            path = folder / "pipeline.joblib"
            joblib.dump(model, path, compress=3)
            (folder / "model.txt").write_text(model.named_steps["model"].booster_.model_to_string(), encoding="utf-8")
            replay = joblib.load(path).predict(xtest)
            columns = {"raw_input_order": used, "categorical": categorical, "dropped_all_missing": dropped,
                       "transformed_order": model.named_steps["transformer"].get_feature_names_out().tolist()}
        else:
            model = CatBoostRegressor(**params)
            xfit, xtest = cat_module.build_features(fit, features, categorical), cat_module.build_features(test_inputs, features, categorical)
            model.fit(xfit, fit[TARGET].to_numpy(dtype=float), cat_features=categorical)
            raw = model.predict(xtest)
            path = folder / "model.cbm"
            model.save_model(str(path))
            restored = CatBoostRegressor()
            restored.load_model(str(path))
            replay = restored.predict(xtest)
            columns = {"raw_input_order": features, "categorical": categorical, "missing_category": "__MISSING__"}
        if not np.isfinite(raw).all() or not np.array_equal(raw, replay):
            raise AssertionError(f"Model replay invalid: {model_id}/{variant}")
        predicted = np.maximum(raw, feature_config["target"]["minimum_prediction"])
        prediction = test_inputs[KEY + ["horizon_quarters"]].copy()
        prediction["model_id"], prediction["variant"] = model_id, variant
        prediction[PRED], prediction["raw_predicted_usd"] = predicted, raw
        prediction.to_parquet(folder / "predictions_before_labels.parquet", index=False)
        write_json(folder / "input_columns.json", columns)
        write_json(folder / "fit_metadata.json", candidate | {"fit_rows": len(fit), "fit_rows_sha256": frame_hash(fit, KEY),
                   "fit_max_target": fit.target_quarter.max(), "model_purpose": "final_test", "origin_quarter": "2025Q4",
                   "reload_max_abs_diff": 0.0, "test_rows_predicted": len(prediction)})
        predictions.append(prediction)
        saved.append({"model_id": model_id, "family": family, "variant": variant, "path": path.relative_to(ROOT).as_posix(),
                      "sha256": sha256(path), "reload_max_abs_diff": 0.0})
        print(f"[{index}/{len(candidates)}] {model_id}/{variant}: saved/reloaded, {len(prediction)} predictions before labels", flush=True)
    before_labels = pd.concat(predictions, ignore_index=True)
    before_labels.to_parquet(out / "predictions_before_labels.parquet", index=False)
    predictions_saved_at = datetime.now(timezone.utc).isoformat()
    # Open test targets only after every configuration and prediction is fixed.
    labels = pd.read_parquet(test_path, columns=KEY + [TARGET, "target_available", "split", "currency", "valuation"])
    labels_opened_at = datetime.now(timezone.utc).isoformat()
    if not labels.split.eq("final_test").all() or not labels.currency.eq("USD").all() or not labels.valuation.eq("FOB").all():
        raise ValueError("Test partition/units changed")
    if not np.isfinite(labels[TARGET].dropna().to_numpy(dtype=float)).all() or labels[TARGET].dropna().lt(0).any():
        raise ValueError("Non-finite or negative test actuals")
    result = before_labels.merge(labels, on=KEY, validate="many_to_one")
    result["evaluation_status"] = np.where(result.target_available, "scored", "missing_actual_target")
    result["error_usd"] = result[PRED] - result[TARGET]
    result.to_parquet(out / "final_test_predictions.parquet", index=False)
    result.to_csv(out / "final_test_predictions.csv", index=False, encoding="utf-8-sig", na_rep=r"\N")
    rows, evaluated_hash = [], None
    for (model_id, variant), block in result.groupby(["model_id", "variant"]):
        scored, excluded = select_scored_rows(block)
        key_hash = frame_hash(scored, KEY)
        if evaluated_hash is not None and evaluated_hash != key_hash:
            raise ValueError("Final-test scoring keys differ")
        evaluated_hash = key_hash
        if len(block) != len(labels):
            raise ValueError("Not every test candidate was predicted")
        rows.append(test_metrics(scored, model_id, [2,3,4,5], variant=variant))
    metrics = pd.concat(rows, ignore_index=True)
    baseline = baseline_predictions(labels.merge(test_inputs[KEY + ["horizon_quarters"]], on=KEY, validate="one_to_one"), pool)
    baseline.to_parquet(out / "baseline_predictions.parquet", index=False)
    baseline_metrics = test_metrics(baseline.loc[baseline[TARGET].notna()], "seasonal_naive", [2,3,4,5], variant="baseline")
    common_keys = baseline.loc[baseline[TARGET].notna() & baseline[PRED].notna(), KEY]
    common_rows = []
    for (model_id, variant), block in result.groupby(["model_id", "variant"]):
        common_rows.append(test_metrics(block.merge(common_keys, on=KEY, validate="one_to_one"), model_id, [2,3,4,5], variant=variant))
    common_rows.append(test_metrics(baseline.merge(common_keys, on=KEY, validate="one_to_one"), "seasonal_naive", [2,3,4,5], variant="baseline"))
    pd.concat(common_rows, ignore_index=True).to_csv(out / "metrics_baseline_common_rows.csv", index=False, encoding="utf-8-sig")
    baseline_metrics.to_csv(out / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    table = metrics.loc[metrics.scope.eq("h1_3") & metrics.group_by.eq("overall")].copy()
    table = table.merge(eligible_summary[["model_id", "variant", "wape"]].rename(columns={"wape": "validation_wape_h2_5"}), on=["model_id", "variant"], validate="one_to_one")
    table["selected_before_test"] = table.model_id.eq(chosen.model_id) & table.variant.eq(chosen.variant)
    table = table.sort_values("wape")
    table.to_csv(out / "summary_h1_3.csv", index=False, encoding="utf-8-sig")
    result.loc[~result.target_available, ["model_id", "variant"] + KEY].assign(reason="missing_actual_target").to_parquet(out / "evaluation_exclusions.parquet", index=False)
    if sha256(out / "selection_lock.json") != lock_hash:
        raise AssertionError("Candidate selection changed after test exposure")
    for relative, expected in protected.items():
        if sha256(ROOT / relative) != expected:
            raise AssertionError(f"Protected input changed: {relative}")
    selected_test = table.loc[table.selected_before_test].iloc[0]
    info = {"status": "completed", "artifact_directory": out.relative_to(ROOT).as_posix(),
            "origin_quarter": "2025Q4", "test_period": ["2026Q1", "2026Q2", "2026Q3"],
            "horizons": [1,2,3], "model_configurations": len(candidates), "saved_models": saved,
            "test_rows_per_configuration": len(labels), "scored_rows_per_configuration": int(labels[TARGET].notna().sum()),
            "missing_rows_per_configuration": int(labels[TARGET].isna().sum()), "fit_rows": len(fit),
            "fit_rows_sha256": frame_hash(fit, KEY), "fit_max_target": fit.target_quarter.max(),
            "same_test_keys_verified": True, "model_reload_verified": True, "post_test_tuning": False,
            "selection_lock_sha256": lock_hash, "selected_by_validation": expected_choice,
            "selected_candidate_test_wape": float(selected_test.wape), "predictions_saved_at_utc": predictions_saved_at,
            "test_labels_opened_at_utc": labels_opened_at, "protected_inputs_unchanged": True,
            "h4_h5_evaluated": False, "four_quarter_test_metrics_generated": False,
            "production_retraining": False, "historical_asof_verified": False,
            "baseline_missing_predictions": int(baseline[PRED].isna().sum()), "previous_local_tests": previous_tests,
            "limitation": "Already team-shared test; team-wide unseen status unverified. Current revised sources, no historical release-vintage proof. Test results did not alter configurations or the validation selection."}
    write_json(out / "run_metadata.json", info)
    lines = ["# 전체 모델 최종 테스트 — 2026-11-05", "",
             f"팀원 제출 설정 {len(candidates)}개 모두 테스트했다. 설정·검증 선택을 테스트 전에 고정했고 정답을 본 뒤 재조정하지 않았다.", "",
             f"- 적합: train+validation에서 2025Q4까지 정답 {len(fit):,}행. 테스트 origin=2025Q4, 2026Q1~Q3(h=1~3).",
             f"- 구성별 {len(labels)}행 모두 예측, 정답 {labels[TARGET].notna().sum()}행 평가, 결측 {labels[TARGET].isna().sum()}행 보존.",
             "- WAPE·sMAPE는 %, MAE·RMSE는 USD. 작은 값이 좋다.", "",
             "| 모델/실험 | 거시 구성 | 검증 WAPE(h2~5) | 테스트 WAPE(h1~3) | MAE(USD) | RMSE(USD) | 검증 선택 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for row in table.itertuples():
        lines.append(f"| {row.model_id} | {row.variant} | {row.validation_wape_h2_5:.4f} | {row.wape:.4f} | {row.mae:,.2f} | {row.rmse:,.2f} | {'선택' if row.selected_before_test else ''} |")
    lines += ["", f"테스트 전에 검증으로 선택한 후보: **{chosen.model_id}/{chosen.variant}**. 이 후보의 테스트 WAPE는 **{selected_test.wape:.4f}%**다.",
              "표는 테스트 결과를 관찰해 정렬한 것이며, 테스트에서 낮은 모델로 자동 재선택하지 않는다. 검증과 테스트는 기간·거리가 달라 점수 차이를 성능 개선으로 단정하지 않는다.",
              "모든 테스트 모델은 평가용 2025Q4 적합본으로 저장했다. 2027년용 최신 운영 재학습 모델은 아니다.",
              "테스트가 3분기이므로 h4~5·4분기 연간 성능을 주장하지 않는다. 해당 거리는 과거 16분기 검증 결과로 확인한다.",
              "공유된 테스트의 팀 전체 미열람 여부·과거 공개시점은 입증되지 않았다. 반복 실험 시 이 테스트 사용 이력을 유지한다.",
              "실제 저장 경로·체크섬은 run_metadata.json, 테스트 전 설정은 selection_lock.json, 전체 국가·품목·거리별 지표는 metrics.csv를 확인한다."]
    (out / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out / "output_checksums.json", {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*") if p.is_file() and p.name != "output_checksums.json"})
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=False)
        for name in ["comparison.md", "metrics.csv", "summary_h1_3.csv", "baseline_metrics.csv", "metrics_baseline_common_rows.csv", "run_metadata.json", "selection_lock.json", "output_checksums.json"]:
            (report_dir / name).write_bytes((out / name).read_bytes())
    print(table[["model_id", "variant", "n", "validation_wape_h2_5", "wape", "selected_before_test"]].to_string(index=False), flush=True)
    print(f"Final test completed: {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
