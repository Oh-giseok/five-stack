# GEM 전체 공유 데이터셋 v002

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

folder = Path("data/shared/v002")
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

Parquet 사용을 권장한다. CSV는 UTF-8 BOM, null=\N, keep_default_na=False와 manifest 스키마를 적용한다. country_code·hs_code는 문자열이다.
