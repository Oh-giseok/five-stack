# CatBoost 1차 지도학습 실행

이 실험은 `data/shared/v001/`의 고정 학습·검증 자료로 국가 20개와 HS6 제품 8개를 함께 학습하는 pooled CatBoost 회귀 모델을 실행한다. 검증은 각 `origin_quarter`마다 새 모델을 적합하는 expanding-origin 방식이다. 해당 기준 분기까지 실제 정답이 있는 train 행만 적합에 포함한다. 검증 정답은 뒤 기준 분기의 학습에 재사용하지 않는다.

## 실행

프로젝트 가상환경을 활성화한 뒤 저장소 루트에서 아래 명령을 실행한다.

```powershell
pip install -e ".[models]"
python scripts/train_catboost.py
```

Parquet 엔진이 설치되지 않은 경우 manifest의 자료형을 적용해 같은 데이터의 CSV 사본을 읽는다. 기본 설정은 `configs/models/catboost.yaml`에서 읽는다. `--iterations`, `--seed`, `--l2-leaf-reg`, `--data-dir`, `--output-dir` 인자로 실행별 값을 바꿀 수 있다. 예:

```powershell
python scripts/train_catboost.py --iterations 300 --seed 42
```

L2 leaf 정규화 계수를 1.0으로 실험하려면 `python scripts/train_catboost.py --l2-leaf-reg 1.0`을 실행한다. 기본 설정은 3.0이다.

## 산출물

기본 출력 위치는 `artifacts/catboost_<UTC 시각>/`이다. 각 origin·실험 구성별 `.cbm` 모델, 행별 `validation_predictions.csv`, `metrics.csv`, 설정 및 체크섬을 담은 `run_manifest.json`을 저장한다. 모델 입력 열은 manifest의 순서를 그대로 사용한다. `without_macro`는 GDP·인구·물가·환율·인터넷 지표 및 해당 `_missing` 열을 함께 제외한다.

범주형인 `country_code`, `region`, `hs_code`는 문자열로 바꿔 CatBoost 기본 범주형 처리를 이용한다. 범주형 결측은 `__MISSING__`으로 표기한다. 수치 결측은 NaN으로 유지해 CatBoost의 수치형 결측 처리를 사용한다. 음수 예측은 0 USD로 제한한 뒤 지표를 계산한다.

`metrics.csv`에는 전체·운영 우선 거리(h=2~5) 묶음·개별 예측 거리·국가·제품 및 국가×거리·제품×거리별 행 수, WAPE, MAE, RMSE, sMAPE가 있다. 두 구성은 정답이 있는 같은 검증 행과 같은 origin에서 비교한다. 검증 정답 결측 행은 점수 계산에서 제외하며 개수는 run manifest에 기록한다.

이 결과는 현재 수정치 자료를 사용한 1차 실험이다. 공유 데이터의 과거 공개시점은 검증되지 않았으므로 실제 과거 정보 공개 시점까지 재현한 성능으로 해석하지 않는다. 2026년 최종 테스트는 사용하지 않는다.
