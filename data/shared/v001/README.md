# 지도학습 공통 데이터 v001

팀장 승인 공유본. 처리 원천은 지도학습 v002이며 학습 대상 2013Q1~2021Q4(36분기), 검증 대상 2022Q1~2025Q4(16분기)다.
20개국·8개 HS6, 국가×제품×origin×target 단위다. 학습 28,800행·검증 12,800행을 모두 보존했고 정답 금액 확보는 각각 23,290행·11,455행이다.

## 파일과 읽기

train.parquet·validation.parquet를 우선 사용한다. CSV도 같은 값·null·스키마로 제공하며 null은 `\N`, 인코딩은 UTF-8 BOM이다. 파일별 SHA-256과 dtype은 manifest.json에 있다.

```python
import json
from pathlib import Path
import pandas as pd

folder = Path("data/shared/v001")
manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
train = pd.read_parquet(folder / "train.parquet")
validation = pd.read_parquet(folder / "validation.parquet")
feature_columns = manifest["feature_columns"]
target = manifest["feature_contract"]["target"]
# 현재 fit 기준시점에 사용할 수 있는 과거 행을 먼저 선택한다.
# 정답 없는 행은 적합·정답 평가에서 제외하고, 입력 결측은 모델 파이프라인에서 처리한다.
fit_origin = "2020Q4"  # 2022Q1을 h=5로 예측하는 기간 제한 예시
# 실제 공개일은 미검증이므로 아래 조건은 달력상 미래 정답만 차단한다.
available_train = train.loc[train["target_available"] & train["target_quarter"].le(fit_origin)].copy()
X = available_train[feature_columns]
y = available_train[target]
```

## 공통 학습 규칙

- feature_columns의 57개 열만 모델 입력이다. 타깃 금액·보조 타깃 중량·타깃 완전성·분할 정보는 입력하지 않는다. 국가·품목명은 표시용이다.
- 국가·제품·연도·분기·분기 구성 월과 origin/target을 보존했다. 여러 horizon의 동일 타깃 금액을 더해 총매출로 계산하지 않는다.
- 후보에 남은 정답 결측은 원천 미확보다. 0으로 대체하지 않는다. 시차 이력 부족과 독립변수 null도 보존했다.
- 시간순 검증으로 각 origin/cutoff 이후 정답을 학습에 넣지 않는다. 2022Q1·h=5 origin은 2020Q4이므로 해당 fit에 2021년 정답을 넣지 않는다. 결측 대체·인코더 등도 fit 구간에서만 학습한다.
- 모델 비교 기간 2022~2025년 16분기, h=1~5 중 h=2~5 우선 비교다. 거시 포함/제외는 같은 검증 행·origin으로 비교한다.

## 자료의 확인 범위

현재 수정치 스냅샷을 공유한다. 시장은 origin 동일 분기, 거시는 직전 달력연도 값이다. 과거 최초 공개일·빈티지·cutoff가 검증된 자료는 아니며 origin_cutoff_at은 null이다. 이 자료로 과거 공개시점까지 재현한 성능이라고 주장하지 않는다. 명시적 시점 가정 또는 근거 확보 후 해당 평가를 진행한다.
공유 확정은 학습·검증 파일 배포 승인이다. 원천 공개일을 새로 확인했다는 뜻이 아니다. 전체 후보/원천 메타정보는 팀장 로컬에 보존했고 이 공유본에서 모델 입력·필수 행 메타정보만 정리했다.

2026Q1~Q3 최종 테스트 정답과 비지도학습 자료는 포함하지 않았다. 최종 테스트는 팀장 로컬에서 모델 선택 완료 후 수행한다. 비지도 작업은 별도 진행한다.
