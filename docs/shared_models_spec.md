# 팀 공통 모델 저장·공유 사양서

기준일: 2026-11-05. 팀원은 학습을 완료한 모델을 아래 기준으로 저장한다.

## 1. 저장 위치

```text
artifacts/shared_models/
  <모델·실험명>/
    with_macro/       # 거시지표 포함
    without_macro/    # 거시지표 제외
```

모델·실험명은 `xgboost`, `xgboost_l1`, `xgboost_l2`, `xgboost_l1_l2`, `xgboost_standard_scaler`, `lightgbm`, `lightgbm_l1`, `lightgbm_l2`, `catboost`, `catboost_l2_1`처럼 구분한다.
실제로 학습한 구성만 저장한다. 같은 구성의 새 버전은 `xgboost_l2_v2`처럼 이름을 구분하고 기존 공유 모델을 덮어쓰지 않는다.

## 2. 파일 형식

| 모델 | 저장 파일 | 저장 대상 |
|---|---|---|
| XGBoost·LightGBM·그 밖의 joblib 지원 모델 | `pipeline.joblib` | 학습 완료 모델과 학습된 입력 변환기를 포함한 Pipeline |
| CatBoost | `model.cbm` | 학습 완료 CatBoost 모델 |

예: `artifacts/shared_models/xgboost_l2/with_macro/pipeline.joblib`

**shared_models에는 `.joblib`·`.cbm`만 넣는다.** 예측·평가 CSV/Parquet, 입력·실행 기록 JSON, 로그, 노트북, `.ubj`·`.txt` 중복 저장물은 넣지 않는다.
공통 사양서는 `docs/`에 두며, 실험 산출물은 각자의 로컬 실행 폴더에 보관한다. 이미 공유된 데이터·코드를 다시 복사할 필요는 없다.

모델 비교 화면의 공용 실행 데이터는 `data/shared/model_comparison/summary.json` 1개로 관리한다. 모델별 성능값·2027년 예상 수출액·평가 조건·모델 및 공통 입력 체크섬만 포함한다. 개인 실행 경로와 행별 실험 기록은 넣지 않는다. 이 파일은 화면 구동용이며, `shared_models/`에는 계속 모델 파일만 저장한다. ElasticNet의 2025Q4 테스트용 모델은 `elastic_net_test_2025Q4/`에서 운영용 `elastic_net/`과 구분한다.

일반 선형회귀·Ridge의 2025Q4 테스트용 모델은 `linear_regression_test_2025Q4/`, `ridge_test_2025Q4/`에 저장한다. 기존 `linear_regression/`, `ridge/`의 2026Q3 운영 모델은 보존하며, 운영 모델에 2026년 최종 테스트 점수를 붙이지 않는다. MLP의 두 은닉층 모델은 `mlp_64_32/`에 2025Q4 학습본을 저장한다. 각 경로 아래의 거시 포함·제외 구분은 동일하다.

## 3. joblib 저장 기준

- 새 모델은 학습 완료된 Pipeline을 저장한다. `fit()` 전 객체나 모델 클래스·학습 코드만 저장하지 않는다.
- 인코딩·결측 대체·StandardScaler를 학습했다면 해당 변환기를 Pipeline에 함께 넣는다. 추론할 때 새로 적합하지 않는다.
- 입력 변수의 선택과 순서를 저장한다. 모델 입력에 타깃·평가용 메타정보를 넣지 않는다.
- 새 공유본은 아래 dict 구조를 사용한다. 별도 JSON은 만들지 않는다.

```python
from pathlib import Path
import joblib

# 아래 변수는 해당 팀원의 학습 코드에서 결정된 실제 값을 사용한다.
model_path = Path("artifacts/shared_models") / model_id / variant / "pipeline.joblib"
if model_path.exists():
    raise FileExistsError("새 실험명을 사용하세요.")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "pipeline": trained_pipeline,
    "feature_columns": list(feature_columns),
    "categorical_columns": list(categorical_columns),
    "dataset_version": dataset_version,
    "trained_through_target_quarter": trained_through_target_quarter,
}, model_path, compress=3)
```

현재 공유된 XGBoost는 위 dict 형식이고, LightGBM은 Pipeline 본체 형식이다. 기존 LightGBM 파일은 그대로 사용하며 형식을 맞추려고 재학습·재저장하지 않는다.
로드할 때는 두 형식을 다음처럼 구분한다.

```python
saved = joblib.load(model_path)
if isinstance(saved, dict):
    model = saved["pipeline"]
    columns = saved["feature_columns"]
else:
    model = saved
    columns = list(model.feature_names_in_)

# X_ready는 기존 학습 코드와 동일한 자료형·결측 처리 규칙으로 준비한다.
prediction = model.predict(X_ready[columns])
```

입력 준비에는 프로젝트의 기존 입력 변환 함수를 사용한다. 저장된 변환기에는 `transform`·`predict`만 적용하며 다시 `fit`하지 않는다.

## 4. CatBoost 저장 기준

```python
from pathlib import Path
from catboost import CatBoostRegressor

model_path = Path("artifacts/shared_models") / model_id / variant / "model.cbm"
if model_path.exists():
    raise FileExistsError("새 실험명을 사용하세요.")
model_path.parent.mkdir(parents=True, exist_ok=True)
trained_model.save_model(str(model_path))

loaded_model = CatBoostRegressor()
loaded_model.load_model(str(model_path))
columns = loaded_model.feature_names_
# X_ready는 학습 때와 같은 문자열 범주형·수치 NaN 처리로 준비한다.
prediction = loaded_model.predict(X_ready[columns])
```

`.cbm`에 입력 이름·범주형 위치가 저장된다. 학습 시 DataFrame 열 이름을 유지하고, 추론에서도 같은 열 순서와 범주형 처리 규칙을 적용한다.

## 5. 저장 후 확인

1. 저장 전에 고정한 확인용 입력으로 예측값을 보관한다.
2. 저장 파일을 새 객체로 다시 불러온다. 재학습하지 않는다.
3. 같은 입력·같은 열 순서로 다시 예측한다.
4. 아래처럼 저장 전후 예측 일치를 확인한다. 수출액의 음수 제한 등 후처리도 양쪽에 동일하게 적용한다.

```python
import numpy as np
np.testing.assert_array_equal(prediction_before_save, prediction_after_reload)
```

공통 데이터 버전과 사용한 학습 기간을 확인한다. 2025Q4 기준 테스트용 적합본과 2026Q3까지 재학습한 운영용 모델을 구분한다.
수신자는 프로젝트의 동일한 라이브러리 버전과 기존 데이터·코드를 사용한다. 새 환경이 필요한 경우 버전을 별도로 안내하되 shared_models에 환경 파일을 넣지 않는다.

## 6. SourceTree 공유 기준

- 새 모델의 `.joblib`·`.cbm` 경로만 선택해 스테이징한다. 실험 폴더 전체 추가는 하지 않는다.
- 커밋 전에 모델·실험명, 거시 구성, 파일 형식, 저장 후 재로드 확인 결과를 점검한다.
- 원본 API 응답·인증키·`.env`·`.private`·가상환경은 공유하지 않는다.
- 현재 `.gitignore`는 shared_models 아래 `.joblib`·`.cbm`만 공유 대상으로 허용한다.
- 이 사양서는 공통 안내 문서이므로 모델 공유 시 `docs/shared_models_spec.md`를 함께 전달한다.

## 7. 공유 모델로 2027년 분기 예측 실행

공통 엔진·설정·테스트와 최신 입력은 모델 폴더와 별도 경로로 함께 공유한다.
`data/shared/v002/forecast/origin_features_2026Q3.parquet`는 20개국 × 8개 품목의 160행 입력이다. 기존 train·validation·final_test를 바꾸지 않는다.
설정은 `configs/engine.yaml`, 실행 명령은 `scripts/forecast.py`다. 원본 API 자료·인증키·로컬 processed 폴더 없이 공유본으로 실행한다.

Windows 64비트·Python 3.12 환경에서 프로젝트 폴더를 열고 다음 명령을 실행한다. 기존 가상환경이 있으면 생성 단계를 건너뛴다.
공유 모델을 저장한 환경의 버전을 `requirements-forecast-windows-py312.txt`에 기록했다. joblib 모델의 호환성을 위해 동일한 버전을 사용한다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-forecast-windows-py312.txt -e ".[models]"
.\.venv\Scripts\python.exe scripts/forecast.py
```

기본 구성은 검증 구간에서 선택한 `xgboost_l2/with_macro`다. 다음처럼 모델·거시 구성을 변경할 수 있다.

```powershell
.\.venv\Scripts\python.exe scripts/forecast.py --model lightgbm_l2 --variant without_macro
.\.venv\Scripts\python.exe scripts/forecast.py --model catboost --variant with_macro
```

2026Q3 기준으로 2027Q1~Q4를 h=2~5로 예측한다. 저장 모델과 입력 변환기를 다시 학습하지 않는다.
분기 640행·연간 160행과 재사용 가능한 입력·실행 정보는 각자 `.private/forecasts/<run_id>/`에 저장되며 Git에 올리지 않는다.
원천 결측은 보존하고 결과에 표시한다. 음수 원예측은 기존 평가 규칙에 따라 0으로 제한하며 원예측도 함께 남긴다.
640행 생성은 실제 2027년 예측 정확도 검증을 뜻하지 않는다. 과거 지표의 공개시점 확인도 미확정 상태다.
