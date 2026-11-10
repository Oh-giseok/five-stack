# GEM — Global Export Management

화장품 수출 예측 및 생산 계획 지원 서비스다.

20개국·8개 HS 제품의 한국 수출금액을 예측하고 2027Q1~Q4 매출·수요 비교를 제공한다.
DB는 사용하지 않으며 원본·정제·학습·예측 데이터를 파일로 관리한다.

## 현재 구현 상태

| 영역 | 상태 |
|---|---|
| 폴더 구조·공통 YAML·환경변수 예시 | 작성됨 |
| 설정 읽기·설정 정합성 점검 | 실행 가능 |
| 공통 데이터 규격·전처리·인계 안내 | 작성됨, 관세청 필드 매핑 검증 완료 |
| API 원본 수집·체크섬·확보 범위 점검 | 구현됨, 수집 결과는 별도 보고서 참고 |
| 관세청 HS6 월별·분기 전처리 | v003: 20개국·8개 제품 전수 대조, 분류 보류 0개, 미조회 별도 유지 |
| Comtrade 월별·분기 전처리·수입시장 3개 변수 | v001: 20개국·8개 제품·64,649행 전수 반영, 제외·분류 보류 0행 |
| World Bank 연간 거시지표·군집 후보 | v001: 20개국·6개 지표·1,560행 보존, 2024년 동일 연도 후보 20행 |
| 지도학습 통합·시차·분할 후보 | v002: 학습 36분기·검증 16분기·최종 테스트 3분기, 기존 v001 보존 |
| 전체 공유 데이터 | v002 확정: train·validation·final_test 정답·비지도 10개 표·명세 제공 |
| 화면 설계 | GEM v1.0 확정: 유스케이스·사용자 흐름·논리 ERD·8개 와이어프레임 이미지, 한 화면 결과 최대 2개 |
| 기간별 비지도 데이터 | v1.3 사양·로컬 v002 생성: 국가×5구간 100행, 최근 2025~2026Q3 포함; 현재 82행 세 지표 확보 |
| 모델·예측·화면 | 저장 모델 기반 2027년 예측·Streamlit UI·모델 비교 구현, 과거 공개시점 재현은 미검증 |

설정 파일이 준비됐다는 것이 API 인증·전 기간 데이터 확보·모델 성능 검증을 뜻하지는 않는다.

## 모델 비교 화면

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

사이드바의 **생산 시나리오 아래 모델 비교**에서 예상 수출액·검증 WAPE·테스트 WAPE·R²·MAE·RMSE를 선택한다.
가로 막대와 전체 성능표는 선택한 지표와 오름차순·내림차순에 맞춰 함께 정렬된다.

공용 실행 데이터는 `data/shared/model_comparison/summary.json`이다. 평가 완료 32개 구성과 운영용 2개 구성의 성능값·2027년 예상 수출액을 담으며, 개인 폴더 없이 화면을 조회할 수 있다. 모델 파일과 공통 입력의 체크섬이 맞아야 표시한다. 저장 모델을 다시 학습하거나 예측을 새로 생성할 필요가 없다.

## 공통 환경

Python 3.12.x를 기준으로 한다. 작성 환경은 3.12.10이며 의존성 범위는 `pyproject.toml`에 둔다.
아래는 Windows PowerShell에서 저장소 루트를 기준으로 실행하는 명령이다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[models,ui,dev]"
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
.\.venv\Scripts\python.exe scripts/check_config.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

`.env`가 이미 있으면 복사 명령을 건너뛴다. 실제 키는 로컬 `.env`에만 입력한다.
팀의 첫 전체 설치가 검증되면 실제 설치 버전을 기록한 잠금 파일을 배포한다. 현재 의존성 범위는 완전 고정된 설치 환경이 아니다.

## 공통 설정

| 경로 | 내용 |
|---|---|
| `configs/countries.yaml` | 20개국·지역·원천별 국가코드 |
| `configs/products.yaml` | 8개 HS 문자열·제품명 |
| `configs/splits.yaml` | 수집·학습·검증·테스트·2027년 예측 기간 |
| `configs/features.yaml` | 시계열·수량·단가·시장·거시지표와 누수 방지 |
| `configs/integration.yaml` | 통합 원천 버전·관측/후보 선택 기준·공개시점 미확정 상태 |
| `configs/release.yaml` | 지도학습 공유 승인·파일 범위·확인 한계 |
| `configs/sources.yaml` | API·응답 필드·환경변수 이름·저장 경로 |
| `configs/models/*.yaml` | 모델별 초기 실험 파라미터 |

YAML 들여쓰기는 공백을 사용한다. HS·날짜·분기 값은 따옴표로 감싼다. 설정 변경 후 점검 명령을 다시 실행한다.

## 데이터와 모델 흐름

```text
API → data/raw → data/interim → data/processed
    → 변수·시간 분할 → data/shared/v001 → 모델 학습·검증
    → 최종 테스트 → 운영용 재학습 → artifacts/production → Streamlit
```

- 원본·전체 정제 패널·모델 파일은 Git 기본 추적에서 제외한다.
- 사용자 승인 전체 공유 v002에는 학습·검증·최종 테스트 정답과 비지도 10개 표를 포함한다.
- 현재 전체 공유본은 `data/shared/v002/`이다. train 28,800행·validation 12,800행과 CSV/Parquet·실제 스키마·체크섬 manifest를 제공한다. 최종 테스트 480행과 비지도 전체 10개 표·2026년 수출 맥락도 포함한다.

## 실행 진입점

`scripts/check_config.py`는 설정 점검, `scripts/collect.py`는 원본 API 수집과 확보 범위 점검을 구현했다.
`scripts/prepare_data.py --source customs`는 관세청 원본을 월별·분기별 HS6 패널로 처리한다.
`scripts/prepare_data.py --source comtrade`는 수입시장·한국산 수입·수출 비교를 구분해 정제하고 수입시장 금액·한국산 금액·점유율을 생성한다.
`scripts/prepare_data.py --source world_bank`는 연간 거시지표 6개와 표준화 전 군집 후보 3개 지표를 분리해 정리한다.
`scripts/package_dataset.py`는 승인된 지도학습 공유본을 생성·검증한다. `train.py`, `cluster.py`, `evaluate.py`, `forecast.py`는 아직 미구현이다.
`app.py`도 비어 있으므로 지금 브라우저 대시보드가 표시되지는 않는다.

원본 수집은 로컬 `.env`의 인증키를 사용한다. 같은 snapshot의 성공 응답은 요청 조건과 SHA-256 확인 후 재사용한다.
서로 다른 원천 단계는 별도 프로세스로 실행할 수 있지만 같은 원천·snapshot을 중복 실행하지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts/collect.py --stage probe --snapshot collection_20261103
.\.venv\Scripts\python.exe scripts/collect.py --stage world_bank --snapshot collection_20261103
.\.venv\Scripts\python.exe scripts/collect.py --stage customs --snapshot collection_20261103
.\.venv\Scripts\python.exe scripts/collect.py --stage comtrade --snapshot collection_20261103
.\.venv\Scripts\python.exe scripts/collect.py --stage audit --snapshot collection_20261103
```

새 기준일로 재수집할 때는 새 snapshot 이름을 사용한다. `audit`는 원본 존재·체크섬·월/연도별 확보 범위만 확인한다.
관세청 HS6 조회 응답의 HS6·HS10 상세·합계 행은 원형대로 저장한다. 원본을 보존한 채 관세청 월별·분기별 집계를 별도 생성했다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --source customs --snapshot collection_20261103 --version v003
.\.venv\Scripts\python.exe scripts/audit_customs_classification.py
```

이미 생성된 버전은 덮어쓰지 않는다. 재처리 시 새 version을 사용한다. CSV·Parquet는 `data/processed/customs/<version>/`, 정규화 상세·검산 자료는 `data/interim/customs/<version>/`에 둔다.
현재 관세청 버전은 `v003`다. 모든 지정 국가·제품·기간에 공통 HS6 집계를 적용하고 명시적 총계를 제외한 조회 상세를 합산·검산했다. 코드 길이만을 이유로 보류하지 않으며 원본 코드·품명을 보존한다.
자료가 조회되지 않은 월은 null로 유지하고 부분 분기를 완전한 실적으로 대체하지 않는다. 관세청 v003은 사용자 확정본이다.

```powershell
.\.venv\Scripts\python.exe scripts/fetch_comtrade_references.py
.\.venv\Scripts\python.exe scripts/prepare_data.py --source comtrade --snapshot collection_20261103 --version v001
.\.venv\Scripts\python.exe scripts/audit_comtrade_preprocessing.py --version v001
```

Comtrade v001도 모든 국가·제품을 대조했다. 원본 64,649행 전체를 보존하고 세 용도별 월별 79,200행·분기별 26,400행, 수입시장 월별 26,400행·분기별 8,800행의 패널을 생성했다. 숫자는 결측을 포함한 격자 수다.
수입시장 점유율은 같은 대상국의 전체·한국산 수입금액으로 계산했다. 관세청 수출과 더하거나 자동 결측 보충하지 않았다. `data/interim/comtrade/v001/`와 `data/processed/comtrade/v001/`에 CSV·Parquet·명세를 보관한다.
단가·시차·통합 후보를 생성한 뒤 사용자 승인으로 아래 지도학습 공유본을 확정했다. 과거 공개시점·빈티지 검증은 별도다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --source world_bank --snapshot collection_20261103 --version v001
.\.venv\Scripts\python.exe scripts/audit_world_bank_preprocessing.py --version v001
```

World Bank v001은 `data/processed/world_bank/v001/`에 연간 long·wide 표와 당시 단일 연도 후보 `country_indicators_candidate`를 보존한다. 2024년 후보는 처리 이력이며, 현재 군집 학습 입력은 아래의 기간별 비지도 v002이다. 과거 공개 근거는 미검증이다.
World Bank CSV는 원본 빈 문자열을 보존하기 위해 null을 `\N`으로 저장한다. `keep_default_na=False, na_values=[r"\N"]`와 manifest 자료형으로 읽거나 Parquet를 사용한다. 후속 결합·분리·공유 확정 기준은 [참고 사양서](docs/dataset_release_reference.md)를 따른다.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_supervised.py --version v002
.\.venv\Scripts\python.exe scripts/audit_supervised_preprocessing.py --version v002
```

현재 지도학습 통합 파일은 `data/processed/supervised/v002/`의 `integrated_monthly`·`integrated_quarterly`, 기준분기별 변수는 `origin_feature_candidates`, 학습·검증 후보는 `train_candidates`·`validation_candidates`다. 모두 CSV·Parquet와 기간·행 수·체크섬 manifest를 제공한다. 기존 v001은 보존한다.
학습 대상은 2013Q1~2021Q4(36분기), 검증 대상은 2022Q1~2025Q4(16분기), 최종 테스트는 2026Q1~Q3(3분기)다. 학습 후보 28,800행·검증 후보 12,800행·테스트 후보 480행이며 정답 결측 후보도 유지한다. 실제 이동 검증에서 각 기준시점 이후 정답을 학습에 넣지 않는다.
월별 year/month/quarter와 분기별 구성 월 3개를 유지하며 20개국·8개 품목·원천 금액을 보존했다. 최종 테스트 후보는 `.private/evaluation/supervised/v002/`에 별도 저장했다.
후보 시장 변수는 기준분기와 같은 분기, 거시는 직전 달력연도다. 이 기간 선택은 공개일을 확인한 사용 기준이 아니다. 원본 처리 후보의 `origin_cutoff_at`은 null이고 승인 상태는 당시 이력으로 보존했다. 통합 관측표의 같은 연도 거시값을 모델 입력에 직접 넣지 않는다.
모델에는 공유 manifest `feature_columns`만 입력한다. 타깃·타깃 완전성·원천 경로 등은 입력에서 제외한다. 비지도학습 데이터·기존 원천은 보존했고, 공유 승인은 아래 v001 명세에 기록했다.

## 확정된 지도학습 공유본

사용자 승인으로 전체 공유 v002를 확정했다. 팀원은 `data/shared/v002/`의 train·validation·final_test와 unsupervised/를 사용한다. [파일 읽기·학습 기준](data/shared/v002/README.md)에 명시한 지도학습 입력 57개·전체 87개 열과 비지도 입력 세 지표를 구분한다. 모든 후보 행과 원래 값·null은 보존했다.

```powershell
.\.venv\Scripts\python.exe scripts/package_dataset.py --verify-only
```

현재 공유 승인과 과거 공개시점 입증은 구분한다. origin_cutoff_at은 null이고 원천 빈티지는 미확인이다. 현재 수정치·기간 기준 변수로 실험하며 정답 없는 행은 적합·정답 평가에서 제외한다. 최종 테스트 정답과 비지도 전체 자료도 공유됐으며 학습·평가에 사용한 범위와 튜닝 이력을 기록한다.

## 작업 기준과 읽을 문서

- [사양서](PROJECT_SPEC.md), [로드맵](PROJECT_ROADMAP.md)
- [GEM 전체 설계 확정안 PDF — 유스케이스·사용자 흐름·ERD·와이어프레임·전체 공유 정책 v1.4 17쪽](docs/GEM_전체설계_확정안_v1.4.pdf)
- [컨플루언스 GEM 설계 확정 문서 모음](https://kimdonggyu114.atlassian.net/wiki/spaces/VVhnfE1g5PNR/pages/2457601/GEM+v1.3)
- [협업 규칙](docs/collaboration.md), [데이터 명세](docs/data_dictionary.md)
- [전처리 기준](docs/data_processing.md), [모델 인계](docs/model_handoff.md)
- 유사 국가의 전체 기간 3년 구간·경제지표·결측·최신성 결합은 [자료 확정 참고 사양서 7절](docs/dataset_release_reference.md)을 따른다. PDF v1.0은 기존 배치 이력이며 v1.3 기간별 비교 보완 기준을 우선 적용한다.
- [원천·코드 확인 근거](docs/data_sources.md)
- [2026-11-03 원본 수집 결과·누락 범위](reports/collection_20261103.md)
- [관세청 v001 전처리 결과·보류 항목](reports/customs_v001_preprocessing.md)
- [관세청 v002 당시 립 제품 반영·보류 항목](reports/customs_v002_preprocessing.md)
- [현재 v003 전체 국가·제품 전수 대조 결과](reports/customs_v003_all_country_product_review.md)
- [Comtrade v001 전체 국가·제품 전수 대조 결과](reports/comtrade_v001_preprocessing.md)
- [World Bank v001 전체 국가·지표 전수 대조 결과](reports/world_bank_v001_preprocessing.md)
- [최종 자료 확정·결합·분리 참고 사양서](docs/dataset_release_reference.md)
- [현재 지도학습 후보 v002·검증 16분기 결과](reports/supervised_v002_integration.md)
- [이전 지도학습 통합 v001 결과](reports/supervised_v001_integration.md)

공통 문서·YAML이 다르면 구현 전에 맞춘다. 개인 문서는 공용 안내·배포에 포함하지 않는다.

## 기간별 비지도 데이터 전처리

```powershell
.\.venv\Scripts\python.exe scripts/prepare_unsupervised.py --version v002 --as-of-date 2026-11-04
.\.venv\Scripts\python.exe scripts/audit_unsupervised_preprocessing.py --version v002
```

기존 버전은 덮어쓰지 않는다. 결과는 `data/processed/unsupervised/v002/`의 10개 표(CSV·Parquet)와 manifest다. 모델에는 country_window_features의 eligible_for_fit=true 행에서 세 지표만 입력한다. 표준화·군집 적합은 후속이다.
사용자 추가 승인으로 전체 맥락·최종 테스트 입력과 정답을 공유 v002에 포함했다. [전처리 결과](reports/unsupervised_v002_preprocessing.md)와 [검산 결과](reports/unsupervised_v002_audit.json)를 확인한다.

## 전체 공유 정책 — 2026-11-04 사용자 지시

현재 배포는 [전체 공유 v002 안내](data/shared/v002/README.md)다. 학습 28,800행·검증 12,800행·최종 테스트 480행(정답 447/결측 33)과 비지도 전체 10개 표를 포함한다.
지도학습 입력 57개·전체 87개 열·시간 분할·원래 값·null을 유지한다. 비지도 입력은 100행 보존·82행 세 지표 확보이며 최신 구간은 2025-01~2026-09다.
팀장 전용 보관 제한을 해제하고 최종 테스트 정답·2026년 수출 맥락까지 공유한다. 원본 처리 캐시는 이동·삭제하지 않는다. 테스트 파일 공유와 학습 입력 사용은 구분하며 테스트로 모델을 조정했다면 해당 결과의 사용 이력을 기록한다.
과거 v001·로컬 전처리 보고서·Confluence v1.3의 비공개 정책은 당시 처리 이력이며 이 전체 공유 정책과 v002 manifest를 우선한다.
