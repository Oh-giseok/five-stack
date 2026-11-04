"""Package approved supervised files without source internals or test labels."""

import json
from pathlib import Path

import pandas as pd

from oinq.collectors.raw import utc_now, write_json
from oinq.common.config import load_project_config, load_yaml, validate_project_config
from oinq.preprocessing.customs import sha256
from oinq.preprocessing.world_bank import save_table

RELEASE_METADATA = [
    "country_name", "product_name", "origin_quarter", "origin_year", "origin_quarter_number",
    "origin_period_end_at", "origin_cutoff_at", "origin_cutoff_status", "origin_in_collection_range",
    "origin_export_data_status", "target_quarter", "target_period_months", "target_month_start", "target_month_end",
    "target_export_value_usd", "target_export_net_weight_kg", "target_observed_months", "target_data_status",
    "target_available", "label_status", "split", "currency", "valuation", "market_candidate_quarter",
    "market_selection_status", "macro_candidate_year", "macro_selection_status", "macro_candidate_row_exists",
    "historical_release_verified", "feature_selection_status",
]


class ReleaseError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def release_columns(contract):
    model = contract["categorical"] + contract["known_future"] + contract["numeric_candidates"]
    require(len(set(model)) == len(model), "중복 모델 입력")
    require(not set(model).intersection(contract["forbidden_feature_columns"]), "정답·품질 정보가 모델 입력에 있습니다.")
    return list(dict.fromkeys([*model, *RELEASE_METADATA]))


def select_release(frame, contract, bounds, split):
    require(frame["split"].eq(split).all(), "공유 대상 분할 불일치")
    require(frame.target_quarter.between(bounds["start"], bounds["end"]).all(), "공유 기간 밖 대상 정답")
    require(frame.target_quarter.lt("2026Q1").all(), "최종 테스트 정답 공유 금지")
    require(not frame.duplicated(["country_code", "hs_code", "origin_quarter", "target_quarter"]).any(), "공유 표본 중복")
    result = frame[release_columns(contract)].copy()
    require(len(result) == len(frame), "공유 표본 누락")
    require(result.target_available.eq(result.target_export_value_usd.notna()).all(), "정답 결측 표시 오류")
    # Metadata columns are retained but the model input whitelist is unchanged.
    require(not any(name.endswith(("source_file", "collected_at")) for name in result), "원천 내부 경로 포함")
    return result


def verify_shared(root, version):
    folder = Path(root) / "data/shared" / version
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["status"] == "released" and manifest["dataset_release_approved"], "공유 승인 상태 불일치")
    require(manifest["final_test_included"] is False and manifest["clustering_included"] is False, "공유 범위 불일치")
    for record in manifest["files"]:
        file = folder / record["file"]
        require(file.is_file() and sha256(file) == record["sha256"], f"공유 체크섬 오류: {record['file']}")
        if record["format"] != "parquet":
            continue
        frame = pd.read_parquet(file)
        require(len(frame) == record["rows"], "공유 행 수 불일치")
        require(frame.target_quarter.lt("2026Q1").all(), "최종 테스트 정답 포함")
        require(set(frame.country_code) == set(manifest["country_codes"]), "공유 국가 누락")
        require(set(frame.hs_code) == set(manifest["hs_codes"]), "공유 제품 누락")
        for name in manifest["feature_columns"]:
            require(name in frame, f"입력 컬럼 누락: {name}")
    return manifest


def package_dataset(root):
    root = Path(root).resolve()
    config = load_project_config(root)
    validate_project_config(config)
    policy = load_yaml(root / "configs/release.yaml")
    require(policy["dataset_release_approved"] and policy["approval_basis"] == "user_instruction", "사용자 공유 승인 필요")
    require(not policy["include_final_test"] and not policy["include_clustering"] and policy["preserve_all_candidate_rows"], "공유 범위 또는 행 보존 기준 오류")
    source_version, shared_version = policy["source_processing_version"], policy["shared_version"]
    require(all(isinstance(v, str) and v.isalnum() for v in (source_version, shared_version)), "버전 형식 오류")
    source = root / "data/processed/supervised" / source_version
    folder = root / "data/shared" / shared_version
    existing = folder / "manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        require(previous["status"] == "not_generated" and previous["files"] == [], "기존 공유 버전은 덮어쓰지 않습니다.")
    require(not folder.exists() or not any(p.name != "manifest.json" for p in folder.iterdir()), "공유 버전에 기존 파일이 있습니다.")
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    require(source_manifest["splits_config"] == config["splits"], "현재 분할과 원천 후보 버전 불일치")
    input_records = {r["path"].split("/")[-1]: r for r in source_manifest["files"]}
    contract = source_manifest["feature_contract"]
    model_columns = contract["categorical"] + contract["known_future"] + contract["numeric_candidates"]
    files, counts, schemas = [], {}, {}
    for split in ("train", "validation"):
        path = source / f"{split}_candidates.parquet"
        require(sha256(path) == input_records[path.name]["sha256"], "후보 파일 체크섬 불일치")
        original = pd.read_parquet(path)
        frame = select_release(original, contract, source_manifest[f"{split}_period"], split)
        files.extend(save_table(frame, folder, split))
        counts[split] = {"rows": len(frame), "target_available": int(frame.target_available.sum()),
                         "target_missing": int((~frame.target_available).sum()), "target_quarters": int(frame.target_quarter.nunique())}
        schemas[split] = [{"name": c, "dtype": str(frame[c].dtype)} for c in frame]
    manifest = {
        "dataset_version": shared_version, "status": "released", "released_at": utc_now(),
        "dataset_release_approved": True, "approval_basis": "user_instruction", "source_processing_version": source_version,
        "source_manifest_sha256": sha256(source / "manifest.json"),
        "source_versions": source_manifest["input_versions"],
        "source_inputs": source_manifest["inputs"],
        "train_period": {**source_manifest["train_period"], "quarters": counts["train"]["target_quarters"]},
        "validation_period": {**source_manifest["validation_period"], "quarters": counts["validation"]["target_quarters"]},
        "final_test_period": {**source_manifest["final_test_period"], "quarters": 3},
        "split_by": "target_quarter", "validation_method": "expanding_origin",
        "final_test_included": False, "clustering_included": False,
        "forecast_period": {"start": "2027Q1", "end": "2027Q4"},
        "country_codes": [r["iso3"] for r in config["countries"]["countries"]],
        "hs_codes": [r["hs_code"] for r in config["products"]["products"]],
        "candidate_source_version": source_version,
        "candidate_manifest_path": f"data/processed/supervised/{source_version}/manifest.json",
        "feature_contract": {**contract, "dataset_release_approved": True,
                             "training_approved": True, "historical_asof_use_verified": False},
        "feature_columns": model_columns, "column_roles": {"features": model_columns,
            "target": contract["target"], "auxiliary_target": "target_export_net_weight_kg",
            "metadata": [c for c in release_columns(contract) if c not in model_columns and not c.startswith("target_export_")]},
        "row_counts": counts, "schemas": schemas,
        "missing_targets_retained": True, "nulls_imputed": False, "scaler_or_encoder_fitted": False,
        "historical_release_evidence_verified": False, "origin_cutoff_at_set": False,
        "evaluation_scope": policy["evaluation_scope"],
        "availability_limitation": "Current revised snapshots; market at origin quarter and macro at previous calendar year are period-aligned features. Original publication dates/vintages are unverified. Do not claim a historically reproducible as-of backtest.",
        "fit_labels_must_be_available_by_origin_cutoff": True,
        "fit_guidance": "Use only feature_columns. Fit models and transformations per origin on past labels only. Exclude rows with missing targets from fitting/scoring; retain them in the shared file. For exact as-of evaluation, resolve publication availability first.",
        "formats": ["csv", "parquet"], "csv_encoding": "utf-8-sig", "csv_null_value": "\\N", "csv_keep_default_na": False,
        "files": [{**r, "file": r["path"].name, "path": r["path"].relative_to(root).as_posix()} for r in files],
        "release_policy_sha256": sha256(root / "configs/release.yaml"),
    }
    write_json(existing, manifest)
    write_readme(folder / "README.md", manifest)
    verify_shared(root, shared_version)
    print(json.dumps({"dataset_version": shared_version, "source_processing_version": source_version,
                      "status": manifest["status"], "row_counts": counts,
                      "feature_columns": len(model_columns), "shared_columns": len(release_columns(contract)),
                      "final_test_included": False, "clustering_included": False}, ensure_ascii=False, indent=2))
    return manifest


def write_readme(path, manifest):
    lines = ["# 지도학습 공통 데이터 v001", "",
             "팀장 승인 공유본. 처리 원천은 지도학습 v002이며 학습 대상 2013Q1~2021Q4(36분기), 검증 대상 2022Q1~2025Q4(16분기)다.",
             "20개국·8개 HS6, 국가×제품×origin×target 단위다. 학습 28,800행·검증 12,800행을 모두 보존했고 정답 금액 확보는 각각 23,290행·11,455행이다.", "",
             "## 파일과 읽기", "",
             "train.parquet·validation.parquet를 우선 사용한다. CSV도 같은 값·null·스키마로 제공하며 null은 `\\N`, 인코딩은 UTF-8 BOM이다. 파일별 SHA-256과 dtype은 manifest.json에 있다.", "",
             "```python", "import json", "from pathlib import Path", "import pandas as pd", "",
             'folder = Path("data/shared/v001")', 'manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))',
             'train = pd.read_parquet(folder / "train.parquet")', 'validation = pd.read_parquet(folder / "validation.parquet")',
             'feature_columns = manifest["feature_columns"]', 'target = manifest["feature_contract"]["target"]',
             '# 현재 fit 기준시점에 사용할 수 있는 과거 행을 먼저 선택한다.', '# 정답 없는 행은 적합·정답 평가에서 제외하고, 입력 결측은 모델 파이프라인에서 처리한다.',
             'fit_origin = "2020Q4"  # 2022Q1을 h=5로 예측하는 기간 제한 예시',
             '# 실제 공개일은 미검증이므로 아래 조건은 달력상 미래 정답만 차단한다.',
             'available_train = train.loc[train["target_available"] & train["target_quarter"].le(fit_origin)].copy()',
             'X = available_train[feature_columns]', 'y = available_train[target]', "```", "",
             "## 공통 학습 규칙", "",
             "- feature_columns의 57개 열만 모델 입력이다. 타깃 금액·보조 타깃 중량·타깃 완전성·분할 정보는 입력하지 않는다. 국가·품목명은 표시용이다.",
             "- 국가·제품·연도·분기·분기 구성 월과 origin/target을 보존했다. 여러 horizon의 동일 타깃 금액을 더해 총매출로 계산하지 않는다.",
             "- 후보에 남은 정답 결측은 원천 미확보다. 0으로 대체하지 않는다. 시차 이력 부족과 독립변수 null도 보존했다.",
             "- 시간순 검증으로 각 origin/cutoff 이후 정답을 학습에 넣지 않는다. 2022Q1·h=5 origin은 2020Q4이므로 해당 fit에 2021년 정답을 넣지 않는다. 결측 대체·인코더 등도 fit 구간에서만 학습한다.",
             "- 모델 비교 기간 2022~2025년 16분기, h=1~5 중 h=2~5 우선 비교다. 거시 포함/제외는 같은 검증 행·origin으로 비교한다.", "",
             "## 자료의 확인 범위", "",
             "현재 수정치 스냅샷을 공유한다. 시장은 origin 동일 분기, 거시는 직전 달력연도 값이다. 과거 최초 공개일·빈티지·cutoff가 검증된 자료는 아니며 origin_cutoff_at은 null이다. 이 자료로 과거 공개시점까지 재현한 성능이라고 주장하지 않는다. 명시적 시점 가정 또는 근거 확보 후 해당 평가를 진행한다.",
             "공유 확정은 학습·검증 파일 배포 승인이다. 원천 공개일을 새로 확인했다는 뜻이 아니다. 전체 후보/원천 메타정보는 팀장 로컬에 보존했고 이 공유본에서 모델 입력·필수 행 메타정보만 정리했다.", "",
             "2026Q1~Q3 최종 테스트 정답과 비지도학습 자료는 포함하지 않았다. 최종 테스트는 팀장 로컬에서 모델 선택 완료 후 수행한다. 비지도 작업은 별도 진행한다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
