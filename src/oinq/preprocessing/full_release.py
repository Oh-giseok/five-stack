"""Publish approved train/validation/final-test and period-clustering tables together."""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pandas as pd

from oinq.collectors.raw import utc_now, write_json
from oinq.preprocessing.customs import sha256
from oinq.preprocessing.release import require, select_release
from oinq.preprocessing.world_bank import save_table


def read_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_formats(path):
    frame = pd.read_parquet(path)
    floats = {c: lambda s: pd.NA if s == "\\N" else float(s)
              for c in frame if pd.api.types.is_float_dtype(frame[c].dtype)}
    csv = pd.read_csv(path.with_suffix(".csv"), encoding="utf-8-sig", keep_default_na=False, na_values=["\\N"],
                      dtype={c: t for c, t in frame.dtypes.items() if c not in floats},
                      converters=floats, low_memory=False).astype(frame.dtypes.to_dict())
    pd.testing.assert_frame_equal(frame, csv, check_exact=True)
    return frame


def shared_record(root, folder, record, role, split=None):
    path = record["path"]
    return {**record, "path": path.relative_to(root).as_posix(), "file": path.relative_to(folder).as_posix(),
            "dataset_role": role, **({"split": split} if split else {})}


def package_full_dataset(root, config, policy):
    root = Path(root)
    version = policy["shared_version"]
    require(version.isalnum() and policy["clustering_processing_version"].isalnum(), "공유 버전 형식 오류")
    require(policy["final_test_release_approved"] and policy["preserve_all_candidate_rows"], "전체 공유 승인·행 보존 필요")
    folder = root / "data/shared" / version
    require(not folder.exists(), "기존 공유 버전은 덮어쓰지 않습니다.")
    previous_folder = root / "data/shared/v001"
    previous = read_manifest(previous_folder / "manifest.json")
    source = root / "data/processed/supervised" / policy["source_processing_version"]
    source_manifest = read_manifest(source / "manifest.json")
    require(source_manifest["splits_config"] == config["splits"], "분할 설정 불일치")
    records = {Path(r["path"]).name: r for r in source_manifest["files"]}
    contract = source_manifest["feature_contract"]
    files, counts, schemas = [], {}, {}
    for split in ["train", "validation", "final_test"]:
        if split != "final_test":
            original = pd.read_parquet(previous_folder / f"{split}.parquet")
            old_records = {r["file"]: r for r in previous["files"]}
            folder.mkdir(parents=True, exist_ok=True)
            for extension in ["csv", "parquet"]:
                old = previous_folder / f"{split}.{extension}"
                require(sha256(old) == old_records[old.name]["sha256"], "기존 공유 파일 체크섬 오류")
                destination = folder / old.name
                shutil.copy2(old, destination)
                require(sha256(destination) == sha256(old), "학습/검증 복사 오류")
                record = {**old_records[old.name], "path": destination}
                files.append(shared_record(root, folder, record, "supervised", split))
        else:
            record = records["final_test_candidates.parquet"]
            path = root / record["path"]
            require(sha256(path) == record["sha256"], "기존 최종 테스트 후보 체크섬 오류")
            candidate = pd.read_parquet(path)
            original = select_release(candidate, contract, source_manifest["final_test_period"],
                                      "final_test", allow_final_test=True)
            files.extend(shared_record(root, folder, r, "supervised", split)
                         for r in save_table(original, folder, split))
        counts[split] = {"rows": len(original), "target_available": int(original.target_available.sum()),
                         "target_missing": int(original.target_export_value_usd.isna().sum()),
                         "target_quarters": int(original.target_quarter.nunique())}
        schemas[split] = [{"name": c, "dtype": str(original[c].dtype)} for c in original]
    unsup_source = root / "data/processed/unsupervised" / policy["clustering_processing_version"]
    unsup = read_manifest(unsup_source / "manifest.json")
    unsup_files = []
    for record in unsup["files"]:
        old = root / record["path"]
        require(sha256(old) == record["sha256"], "비지도 처리본 체크섬 오류")
        if record["format"] != "parquet":
            continue
        frame = pd.read_parquet(old)
        if "disclosure_status" in frame:
            frame["disclosure_status"] = "team_shared_by_user_instruction"
            frame["disclosure_status"] = frame["disclosure_status"].astype("string")
        for r in save_table(frame, folder / "unsupervised", old.stem):
            unsup_files.append(shared_record(root, folder, r, "unsupervised"))
    files.extend(unsup_files)
    unsup_manifest = {**deepcopy(unsup), "status": "released", "shared_version": version,
                      "released_at": utc_now(), "approved_for_shared_release": True,
                      "approval_basis": "user_instruction", "contains_private_final_test_amounts": False,
                      "contains_shared_2026_actuals": True,
                      "release_policy": "all_preprocessed_tables_and_2026_actuals_shared_by_user_instruction",
                      "files": [{**r, "local_only": False} for r in unsup_files],
                      "source_processing_manifest_sha256": sha256(unsup_source / "manifest.json")}
    write_json(folder / "unsupervised/manifest.json", unsup_manifest)
    manifest = {**deepcopy(previous), "dataset_version": version, "released_at": utc_now(),
                "supersedes": "v001", "shared_policy": "team_shared_train_validation_final_test_unsupervised",
                "approval_basis": "user_instruction", "final_test_included": True, "clustering_included": True,
                "final_test_access": "team_shared_with_labels", "row_counts": counts, "schemas": schemas,
                "final_test_origin": "2025Q4", "final_test_horizons": [1, 2, 3],
                "unsupervised": {"directory": "unsupervised", "manifest": "unsupervised/manifest.json",
                    "processing_version": unsup["processing_version"], "spec_version": unsup["spec_version"],
                    "feature_file": "unsupervised/country_window_features.parquet",
                    "feature_columns": unsup["feature_columns"], "feature_key": unsup["feature_key"],
                    "feature_rows": unsup["feature_rows"], "fit_eligible_rows": unsup["fit_eligible_rows"],
                    "windows": unsup["windows"], "all_context_tables_included": True,
                    "contains_shared_2026_actuals": True},
                "files": files, "release_policy_sha256": sha256(root / "configs/release.yaml"),
                "testing_policy": "Files and labels are team-shared. Preserve chronological splits and report whether final_test was used for model selection; repeated tuning on this split is exploratory, not an untouched final evaluation."}
    write_json(folder / "manifest.json", manifest)
    (folder / "README.md").write_text(readme(version), encoding="utf-8")
    (folder / "unsupervised/README.md").write_text(
        "# 기간별 비지도 공통 자료\n\n전체 10개 표를 CSV·Parquet로 공유한다. 2026Q3까지 수출금액도 사용자 지시로 포함했다.\n"
        "모델 입력은 country_window_features에서 eligible_for_fit=true인 행의 gdp_per_capita_usd·population·internet_users_pct 세 열이다.\n"
        "100행 모두 보존했으며 현재 82행은 세 지표가 있다. 최신 18행의 인터넷은 원천 결측이다. 국가·수출 자료는 삭제하지 않았다.\n"
        "과거 네 구간은 3년, 마지막은 2025-01~2026-09의 21개월이다. 최신 거시는 2025년 관측값이다.\n"
        "나머지 9개 표는 연간 원천·지표 품질·국가/제품 수출·변화 설명 자료이며 세 모델 입력에 자동 추가하지 않는다.\n"
        "표준화·군집은 아직 미적합이다. 공통 변환기·중심을 사용하고 실제 관측 연도를 표시한다.\n", encoding="utf-8")
    verify_full_shared(root, version)
    summary = {"dataset_version": version, "status": "released", "row_counts": counts,
               "unsupervised_tables": 10, "unsupervised_rows": unsup["feature_rows"],
               "unsupervised_fit_eligible": unsup["fit_eligible_rows"], "data_files": len(files),
               "final_test_included": True, "clustering_included": True,
               "previous_shared_v001_unchanged": True}
    write_json(root / f"reports/shared_{version}_release.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return manifest


def readme(version):
    return f'''# GEM 전체 공유 데이터셋 {version}

팀원은 이 폴더를 사용한다. 사용자 지시로 학습·검증·최종 테스트 정답·비지도 전체 10개 표를 공유한다. 이전 v001은 학습/검증만 포함한 이력이다.

| 파일 | 범위 | 행 수 | 금액 정답 확보 |
|---|---|---:|---:|
| train.parquet / csv | 2013Q1~2021Q4 | 28,800 | 23,290 |
| validation.parquet / csv | 2022Q1~2025Q4 | 12,800 | 11,455 |
| final_test.parquet / csv | 2026Q1~Q3, origin=2025Q4 | 480 | 447 |
| unsupervised/ | 10개 표, 과거 4구간+2025~2026Q3 | 군집 입력 100 | 세 지표 확보 82 |

## 지도학습 읽기

```python
import json
from pathlib import Path
import pandas as pd

folder = Path("data/shared/{version}")
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
train = pd.read_parquet(folder / "train.parquet")
validation = pd.read_parquet(folder / "validation.parquet")
final_test = pd.read_parquet(folder / "final_test.parquet")
features = manifest["feature_columns"]  # 57개; 정답·품질·분할 메타정보는 입력하지 않는다.
target = manifest["feature_contract"]["target"]
# 이동 검증의 각 origin까지 학습 가능한 과거 정답으로 먼저 제한한다.
```

- 세 파일은 동일한 87개 열·57개 입력이다. 국가·제품·연도·분기·origin·target·금액·중량·결측을 보존했다.
- 최종 테스트 480행도 정답 결측 33행을 삭제하지 않았다. 학습·정답 평가 시 target_available로 구분하며 0 대체하지 않는다.
- 최종 테스트를 팀원에게 배포한다. 학습·검증·테스트의 구분은 유지하며 파일 공유가 테스트 정답을 학습 입력에 넣으라는 뜻은 아니다.
- 새 모델도 같은 검증 조건으로 비교한다. 테스트 결과를 보고 모델을 수정했다면 해당 결과는 반복 실험 결과로 기록하고 미사용 최종 평가라고 표기하지 않는다.
- 현재 수정치·달력 기간 정렬 자료이며 과거 공개일/빈티지는 미검증이다. 해당 가정을 실험 결과에 명시한다.

## 비지도학습 읽기

```python
cluster = pd.read_parquet(folder / "unsupervised/country_window_features.parquet")
cluster_features = manifest["unsupervised"]["feature_columns"]  # 세 지표
cluster_input = cluster.loc[cluster["eligible_for_fit"], cluster_features]
```

연간 거시·구간 지표·국가/제품 수출·변화 맥락까지 전체 10개 표를 제공한다. 상세 스키마·키·체크섬은 manifest와 unsupervised/manifest에 있다. 2026년 수출 실적도 공유됐다.
GDP·수출·품질·국가 식별자·제품 반복 행을 세 군집 수치 입력에 자동 추가하지 않는다. 실제 결측은 보존하고 표준화·K-means는 팀원이 학습한다.
마지막은 2025-01~2026-09의 실제 21개월이다. 2027년 실적은 만들지 않았다. 최근 거시는 확보된 2025년 연간 값이며 18개국 인터넷 지표는 결측이다.

Parquet 사용을 권장한다. CSV는 UTF-8 BOM, null=\\N, keep_default_na=False와 manifest 스키마를 적용한다. country_code·hs_code는 문자열이다.
'''


def verify_full_shared(root, version):
    root = Path(root)
    folder = root / "data/shared" / version
    m = read_manifest(folder / "manifest.json")
    require(m["status"] == "released" and m["dataset_release_approved"]
            and m["final_test_included"] and m["clustering_included"], "전체 공유 승인·범위 불일치")
    require(len(m["files"]) == 26 and len(m["feature_columns"]) == 57, "전체 공유 파일/입력 수 불일치")
    for record in m["files"]:
        rel = Path(record["file"])
        require(not rel.is_absolute() and ".." not in rel.parts, "공유 파일 경로 이탈")
        path = folder / rel
        require(path.is_file() and sha256(path) == record["sha256"], "전체 공유 체크섬 오류: " + record["file"])
        if record["format"] != "parquet":
            continue
        frame = check_formats(path)
        require(len(frame) == record["rows"], "전체 공유 행 수 불일치")
        require(set(frame.country_code) == set(m["country_codes"]), "국가 누락")
        if "hs_code" in frame:
            require(set(frame.hs_code) == set(m["hs_codes"]), "제품 누락")
        if record["dataset_role"] == "supervised":
            split = record["split"]
            require(frame.split.eq(split).all() and frame.target_available.eq(frame.target_export_value_usd.notna()).all(), "분할/정답 결측 표시 오류")
            require(not frame.duplicated(["country_code", "hs_code", "origin_quarter", "target_quarter"]).any(), "공유 표본 중복")
            require(len(frame.columns) == 87 and set(m["feature_columns"]).issubset(frame), "지도학습 스키마 불일치")
            bounds = m[split + "_period"]
            require(frame.target_quarter.between(bounds["start"], bounds["end"]).all(), "시간 분할 혼재")
            if split == "final_test":
                require(len(frame) == 480 and frame.origin_quarter.eq("2025Q4").all()
                        and set(frame.horizon_quarters) == {1, 2, 3}, "최종 테스트 origin/거리/격자 오류")
                require(frame.target_available.sum() == 447, "최종 테스트 정답 손실")
        elif "disclosure_status" in frame:
            require(frame.disclosure_status.eq("team_shared_by_user_instruction").all(), "과거 로컬 전용 정책 잔존")
    features = pd.read_parquet(folder / m["unsupervised"]["feature_file"])
    require(len(features) == 100 and features.eligible_for_fit.sum() == 82
            and not features.duplicated(["country_code", "window_id"]).any(), "군집 입력 격자/가용성 오류")
    latest = features.loc[features.window_id.eq("2025_2026Q3")]
    require(len(latest) == 20 and latest.period_month_count.eq(21).all(), "최신 구간 오류")
    return m
