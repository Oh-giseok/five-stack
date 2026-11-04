# 지도학습용 통합·전처리 결과

버전: `v001` | 관세청 v003 · Comtrade v001 · World Bank v001

사용자 지시에 따라 전체 통합자료와 지도학습 후보를 먼저 생성하고 공개시점·최종 공유 기준은 미확정으로 유지했다. 비지도학습 데이터는 변경하지 않았다.

## 국가·품목·금액·연월분기 보존

- 20개국×8개 품목×165개월 월별 26,400행, 55분기 분기별 8,800행을 보존했다. 국가·제품·기간 행 제외는 0개다.
- 월별에는 국가 코드/이름·지역·HS6/제품명·year·month·quarter·quarter_number가 있다. 분기에는 year·quarter_number·month_start/end·period_months(구성 3개월)가 있다.
- 원천 수출금액은 월별 23,348개·분기별 7,396개, 점유율은 월별 20,096개·분기별 5,991개를 그대로 유지했다. 없는 금액을 0으로 채우지 않았다.
- 금액은 한국 전체 수출금액 USD·FOB다. 기업 순이익이 아니며 Comtrade 수입액·비교용 수출액을 더하지 않는다. 중량·부분합·원천 상태·파일·수집시각도 보존했다.
- 원천별 공통 열 이름은 접두어로 구분하고 manifest의 column_lineage에 모든 입력 열의 연결을 남겼다. 거시지표는 같은 연도 관측 문맥으로 붙인 통합표이며 해당 분기의 모델 입력으로 바로 쓰지 않는다.

## 지도학습 후보

| 구분 | 후보 행 | 정답 금액 있음 | 정답 미확보 |
|---|---:|---:|---:|
| train | 35,200 | 28,930 | 6,270 |
| validation | 6,400 | 5,815 | 585 |
| final_test | 480 | 447 | 33 |

- 학습·검증은 각 대상 분기별 h=1~5 후보를 유지한다. 여러 행에 같은 대상 금액이 있는 것은 예측 기준분기·거리가 다른 표본이며 금액을 합산해 총매출로 해석하지 않는다.
- 최종 테스트는 2025Q4 origin으로 고정한 2026Q1~Q3·h=1~3 후보 480행이며 .private/evaluation/supervised 아래 별도 저장했다. 공개 품질표에는 테스트 금액을 넣지 않았다.
- 원천 수집 시작 이전 origin 후보도 결측 상태로 보존하고 origin_in_collection_range=false로 표시했다. 타깃 미확보 후보도 보존하며 실제 학습·정답 평가는 target_available 및 후속 공개시점 기준을 적용한다.
- 금액·중량·단가 lag1/2/4, 최근 4분기 sum/mean, 전년 대비 성장률, 변수별 결측 표시를 생성했다. 이동 통계는 4개 값이 모두 있을 때만 계산하며 부분합으로 대체하지 않았다.
- 후보 시장 변수는 origin의 같은 분기 값, 거시는 origin 직전 달력연도 값이다. 이는 관측 기간을 맞춘 후보 규칙이며 공개 지연을 가정하거나 과거 사용 가능성을 입증한 것이 아니다. 자동 전방 채우기·다른 연도 fallback은 하지 않았다.
- origin_period_end_at은 달력 분기 말이다. origin_cutoff_at은 null, historical_release_verified/training_approved는 false다. 변수·값을 삭제하지 않고 공개시점 확정 전 후보로 구분했다.
- feature_contract의 명시 목록만 모델 입력 후보로 사용한다. 타깃·타깃 중량·타깃 완전성·분할·원본 경로·동일 연도 관측 문맥은 입력 목록에 포함하지 않는다.

## 저장·검증

- 통합 월별/분기별·origin_feature_candidates·train_candidates·validation_candidates: `data/processed/supervised/v001/`.
- 최종 테스트 후보: `.private/evaluation/supervised/v001/`. shared 배포·Git 업로드·모델 학습은 실행하지 않았다.
- 모든 표는 CSV·Parquet이며 CSV null 표식은 `\N`이다. 빈 문자열과 null을 구분하고 manifest 스키마로 재조회한다.
- [160개 국가·제품 범위표](supervised_v001_country_product_coverage.csv) · [26,400개 월별 품질표](supervised_v001_month_quality.csv)
- 독립 검증: `python scripts/audit_supervised_preprocessing.py --version v001`, [검산 결과](supervised_v001_audit.json)

실행 검증은 통과했다. 전체 입력 열 연결 295개, 월별 26,400행·분기별 8,800행, 기준분기 변수 9,600행과 시차값 86,400개를 독립 검산했다. CSV·Parquet의 값·null·자료형과 후보별 정답·거리·시간 분리를 확인했다. 기존 원천·비지도 후보·shared 5,155개 파일의 무변경을 확인했고 자동 테스트 56개와 공통 설정 점검이 통과했다.

국가·품목·달력 식별정보 누락은 0개다. 원천에 수출금액이 없는 월 3,052개·완전 금액이 없는 분기 1,404개는 사유와 함께 유지했다. 이는 통합으로 생긴 누락이 아니며, 원천의 부분합을 완전 금액이나 정답으로 바꾸지 않았다.

## 후속 확정

[결합·분리 참고 사양서](../docs/dataset_release_reference.md)에 따라 원천 공개시점·빈티지 가정, 오래된 값 허용 기간, 최종 학습 사용 행·변수와 공유 버전을 확정한다. 이번 후보를 공개시점 검증을 마친 배포본으로 해석하지 않는다.
