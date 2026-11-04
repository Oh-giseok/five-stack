# 공통 데이터 명세

정규화 후 내부 컬럼명이다. 원천별 실제 필드와의 연결은 샘플 응답으로 검증하며, 원본의 이름·단위·HS 판본은 별도 보존한다.

## 월별 교역·분기 패널

| 컬럼 | 형식 | 의미 |
|---|---|---|
| `country_code` | string | 대상국 ISO alpha-3. 원천 코드와 구분 |
| `country_name`, `region` | string | 표시명·지역. 모델에는 이름 대신 공통 코드 사용 |
| `hs_code`, `product_name` | string | 6자리 HS·표시 제품명 |
| `period_month` | YYYY-MM string | 월별 자료 기간 |
| `quarter` | YYYYQn string | 분기 패널의 달력 분기 |
| `export_value_usd` | nullable float64 | 한국 신고 수출금액, USD·FOB |
| `export_net_weight_kg` | nullable float64 | 한국 신고 순중량 kg; 미신고를 0으로 대체하지 않음 |
| `export_unit_value_usd_per_kg` | nullable float64 | 금액/순중량. 중량 0·결측이면 null |
| `market_import_value_usd` | nullable float64 | 대상국의 전 세계 수입금액; 원천 평가 기준 보존 |
| `korea_import_value_usd` | nullable float64 | 대상국이 신고한 한국산 수입금액 |
| `korea_import_share` | nullable float64 | 한국산 수입/전체 수입, 비율 0~1 |
| `observed_months` | integer | 분기 내 확보 월 수, 완전 분기는 3 |
| `data_status` | enum string | actual / provisional / partial / estimated / missing |

월별 키는 `(country_code, hs_code, period_month)`, 분기 키는 `(country_code, hs_code, quarter)`다.
원천이 중복되는 단계에서는 `source`·거래 방향·신고국·상대국까지 키에 포함하고, 주 원천 선정 후 패널을 유일하게 만든다.

## 원천·시점 메타정보

`source`, `source_reporter_code`, `source_partner_code`, `hs_edition`, `valuation`, `collected_at`, `released_at`, `release_evidence`, `coverage_status`를 기록한다.
시간은 UTC ISO 8601로 보관하고 화면에서 한국 시간으로 표시할 수 있다. `collected_at`은 과거 공개일의 대체값이 아니다.

## 거시지표

`gdp_usd`, `gdp_per_capita_usd`, `population`, `inflation_pct`, `exchange_rate_lcu_per_usd`, `internet_users_pct`를 사용한다.
값과 `indicator_year`·공개일 근거를 함께 보관한다. 환율 단위는 현지통화/USD, 인터넷 이용률·물가는 퍼센트다.

K-means 입력은 국가×window_id 한 행에 세 지표의 과거는 완전 3개년 평균, 최근은 확보된 연간 평균을 저장한다. 모든 국가에 같은 구간을 사용한다. 표준화는 학습 시 공통 변환기로 수행하고 원래 단위를 보존한다. 마지막 기간은 2025-01~2026-09이며 수출 기대 월 수는 21이다.

### 기간별 비지도 데이터 — v1.3

현재 로컬 비지도 v002을 생성했다. 구간은 2013_2015, 2016_2018, 2019_2021, 2022_2024, 2025_2026Q3이다. 100행 중 82행 적합 가능, 실제 지표 결측 18행도 보존이다.

| 컬럼·표 | 의미 |
|---|---|
| country_window_features | 국가×구간의 세 지표와 적합 가능 여부; 세 수치 열만 모델 입력 |
| country_window_profile | GDP·세 지표의 완전 평균, 관측 부분 평균·연도·원천 |
| macro_annual | 기존 연간 6지표·관측 및 공개 상태 280행 보존 |
| macro_window_long | 국가×구간×4지표 400행, 지표별 품질 확인 |
| window_id, start_year, end_year | 분석기간 식별자·시작·끝 |
| value / gdp_usd·세 지표 | 과거 네 구간은 세 연도 완전 평균; 최근 구간은 확보 연간 값 평균과 관측 연도 표시 |
| observed_mean / *_observed_mean | 실제 관측 연도만의 부분 평균; 최신 짧은 구간은 실제 확보 연도를 표시해 사용; 과거 완결 구간 결측 대체 금지 |
| valid_year_count, observed_years | 관측 연도 수·JSON 배열 문자열 |
| missing_source_years, outside_source_scope_years | 원천 범위 내 값 누락 연도·범위 밖 연도 |
| future_years, in_progress_years | 기준일 이후 연도·진행 중 연도 |
| eligible_for_fit, eligibility_reason | 실제 세 지표 확보·관측 연도 집합 일치 여부; 최근 구간의 길이만으로 제외하지 않음 |
| start_month, end_month, period_month_count, is_short_period | 실제 선택 기간과 월 수; 마지막은 2025-01~2026-09, 21개월 |
| value_basis, aggregation | 완전 3년 평균 또는 최근 확보 연간 평균의 사용 기준 |
| export_value_usd, observed_export_value_usd | 완전 수출 합계·관측 부분합; USD·FOB |
| valid_value_cells, missing_collected_value_cells | 관측 금액 수·수집 범위 내 금액 결측 수 |
| expected_window_value_cells, expected_collected_value_cells | 완전 구간 기대 수·현재 수집 범위 기대 수 |
| outside_collection_value_cells | 수집 범위 밖 월별 셀 수 |
| future_value_cells, elapsed_outside_collection_value_cells, current_month_outside_collection_value_cells | 범위 밖 셀의 미래·과거·현재월 구분 |
| period_status, coverage_status | 실제 선택 기간 상태·그 범위의 관측 완전성 |
| observed_period_start/end, latest_observed_export_month | 금액 관측 첫/끝 월과 최신 월 |
| latest_month_observed_product_count | 최신 관측 월의 제품 수; 국가 전체 최대 8 |
| previous_window_id, *_change_pct, internet_users_pct_change_pp | 인접 이전 구간 식별자·증감률·인터넷 %p 변화 |
| source_files, collected_at, historical_release_verified | 원천 경로 JSON 배열·수집시각·과거 공개 검증 여부 |
| disclosure_status | 공유 v002에서는 team_shared_by_user_instruction; 기존 로컬 처리본의 상태는 원천 처리 이력 |

설명용 GDP·수출·품질은 세 군집 입력에 넣지 않는다. 명목 USD 1인당 GDP는 개인 소득·실질 구매력이 아니다. 표별 키·행 수·결합 규칙은 [참고 사양서 7절](dataset_release_reference.md)에 정의한다.

## 지도학습 입력과 정답

| 컬럼 | 의미 |
|---|---|
| `origin_quarter` | 마지막 실제 관측 분기 |
| `origin_cutoff_at` | 해당 예측 실행에서 정보 사용을 허용한 실제 공개 기준시각 |
| `target_quarter` | 예측 대상 분기 |
| `horizon_quarters` | 대상 분기와 관측 기준 분기의 거리 |
| `target_year`, `target_quarter_number`, `time_index` | 예측 시 미리 알 수 있는 달력 변수 |
| `target_export_value_usd` | 대상 분기 한국 수출금액, 학습·평가 정답 |

입력 행 키는 `(country_code, hs_code, origin_quarter, target_quarter)`이며 horizon은 1~5 중 실험 대상 거리다.
정답 이외의 입력에는 origin 이후 실제 수출·중량·단가·수입시장 값을 넣지 않는다. 분할은 대상 분기 기준이며 학습 정답도 적합 시점에 공개돼 있어야 한다.

`features.yaml`의 lag1은 origin 분기, lag2는 그 직전, lag4는 origin의 3분기 전 값이다. target 분기에서 무조건 shift(1)하는 방식과 구분한다.
최근 4분기 통계는 origin에서 끝나며 전년 동기 성장률은 origin과 4분기 전을 비교한다. 분모가 0이면 성장률은 null이다.

## 예측·군집 출력

- 매출: `country_code`, `hs_code`, `origin_quarter`, `origin_cutoff_at`, `target_quarter`, `horizon_quarters`, `predicted_export_value_usd`, `model_id`, `run_id`, `dataset_version`.
- 군집: `country_code`, `window_id`, `cluster_id`, 거시 구간 평균 3개, `classification_status`, `run_id`.
- 지표 부족으로 미분류이면 `cluster_id`는 null이다. 군집 번호는 순위가 아니다.
- 예측구간은 검증된 산출법이 구현된 경우에만 추가한다. 임의의 상·하한이나 신뢰도를 채우지 않는다.

CSV는 UTF-8 BOM, Parquet는 명시된 스키마로 저장한다. CSV를 읽을 때 국가·HS는 문자열 자료형을 지정한다.

## 관세청 1차 처리 파일

현재 `data/processed/customs/v003/`의 `customs_monthly_hs6`·`customs_quarterly_hs6`를 사용하고 v001·v002는 비교용으로 보존한다. 이 파일은 관측 패널이며 최종 지도학습 입력이 아니다. CSV·Parquet와 스키마·체크섬 manifest를 제공한다.
분기의 `export_value_usd`가 타깃의 원천이며, 예측 기준·대상 분기를 구성하는 후속 단계에서 `target_export_value_usd` 정답으로 연결한다. 현재 단가·시차 변수는 없다.

| 컬럼 | 의미 |
|---|---|
| `source_detail_count` | 집계에 대응하는 원본 비총계 행 수 |
| `observed_export_value_usd`, `observed_export_net_weight_kg` | 반환된 상세 행의 검산용 합계. 미완전·HS 혼재 구간의 확정 금액이 아님 |
| 월별 `record_observed` | 원본 상세가 한 행 이상 반환됐는지 여부 |
| 월별 `hs_level_review_required` | 원본 코드 혼재 중 아직 확정 집계를 보류한 월 |
| 분기 `valid_value_months`, `valid_weight_months` | 해당 측정값의 상세가 완전하고 HS 혼재 보류가 없는 월 수 |
| 분기 `target_value_complete` | 금액이 3개월 모두 확인됨. 과거 공개시점 검증·모델 사용 승인 의미는 아님 |
| `coverage_status`, `missing_reason` | 완전성 상태와 미반환·HS 혼재·구성값 누락 사유 |
| `source_file`, `collected_at` | 원본 요청 파일과 수집시각. 모든 월은 성공 요청과 연결됨 |
| `finality_status`, `snapshot_kind` | 확정 여부 미제공·현재 수정치 스냅샷임을 표시 |

일부 월만 조회되거나 모든 월이 조회되지 않은 분기의 `export_value_usd`·`export_net_weight_kg`는 측정값별 완전성 조건에 따라 null이다. `observed_*`로 이를 대체하지 않는다.
`data_status=actual`은 API 관측값이 갖춰진 상태를 뜻하며 최초 공개일·당시 빈티지가 입증됐다는 뜻이 아니다. `released_at`·`release_evidence`·`hs_edition`은 확인 전까지 null이다.
정규화 상세에는 `source_hs_code`와 `source_hs_digits`를 보존한다. 명시적 총계 행과 중복 키를 구분하고 지정 HS6 제품군에 속한 조회 상세를 원천 금액 검산 후 집계한다.

`hs_level_mixed`는 원본의 코드 길이 혼재 사실이다. 현재 v003은 공통 제품군 집계를 적용하므로 `hs_level_review_required`와 `hs_level_review_months`는 모두 0이다. `hs_level_user_approved`·`aggregation_rule_id`·`aggregation_decision_basis`로 공통 집계 적용 근거를 남긴다.
분기는 `hs_level_mixed_months`·`hs_level_user_approved_months`로 포함 월 수를 남긴다. 해당 상세가 포함된 완전 행의 v003 `coverage_status`는 `complete_with_configured_hs6_aggregation`이다. v002의 `complete_with_user_assumption`은 당시 처리 이력이다.
이 처리 근거 컬럼은 예측 독립변수가 아니다. v003의 공통 합산 지시는 지정된 모든 국가·8개 제품군에 적용한다. 6자리 원본에 임의의 10자리 하위코드를 부여하지 않는다.

## Comtrade v001 처리 파일

`data/interim/comtrade/v001/`에 원본 필드 전체와 정규화 상세 64,649행, 294개 요청 검산표, 관세청 v003과의 수출 비교를 보관한다. `source_file`·`source_row_number`로 원본 한 행과 연결한다. 원본을 삭제하거나 수정하지 않는다.

`data/processed/comtrade/v001/`의 `comtrade_monthly_hs6`·`comtrade_quarterly_hs6`는 세 용도의 long 패널이다. 키는 `(country_code, hs_code, source_role, period_month 또는 quarter)`다. 월별 79,200행·분기별 26,400행에는 미조회 격자도 포함된다.

| 컬럼 | 의미 |
|---|---|
| `source_role` | world_import=대상국 전체 수입, korea_import=대상국 한국산 수입, export_comparison=한국 신고 비교용 수출 |
| `trade_value_usd`, `net_weight_kg` | 해당 용도의 원본 primaryValue·netWgt. 수출·수입을 서로 합산하지 않음 |
| `valuation`, `hs_edition` | 원본 CIF/FOB 평가 기준과 H2~H6 판본. 분기는 포함된 코드 목록을 세미콜론으로 보존 |
| `isReported`, `isAggregate` | 국가 직접 신고·UNSD 집계 표시. 집계행을 삭제하는 조건이 아님 |
| `isNetWgtEstimated` | 월별 원본 중량 추정 여부. 금액 추정 여부와 다름 |
| `record_observed`, `weight_missing` | 원본 월 행 존재 여부와 순중량 결측 여부 |
| `valid_value_months`, `valid_weight_months` | 분기 내 금액·중량이 존재하는 월 수, 서로 독립적으로 판단 |
| `observed_trade_value_usd`, `observed_net_weight_kg` | 분기 관측 부분합. 3개월 미확보 시 완전 분기 값으로 대입하지 않음 |
| `value_complete`, `weight_complete` | 3개월 값 확보 여부. 과거 공개시점이나 학습 투입 승인과 다름 |
| `estimated_weight_months`, `unsd_aggregate_months`, `source_reported_months` | 분기 내 원본 표시별 월 수 |

`market_monthly_hs6`·`market_quarterly_hs6`는 동일 국가·제품·기간의 전체/한국산 수입을 나란히 둔 wide 패널이다. 기존 공통 명세의 `market_import_value_usd`, `korea_import_value_usd`, `korea_import_share`를 제공한다. 각 접두어별 중량·평가 기준·판본·수집시각·원본 파일·완전성 표시도 보존한다.

점유율은 비율 0~1이며 퍼센트 표시 시 화면에서 ×100한다. 두 수입금액의 값과 기준이 확인되고 분모가 양수일 때 계산한다. 분기는 금액 합계끼리 나눈다. 원본 한국산 행이 없으면 점유율 0으로 만들지 않는다.
`share_status`는 available / missing_component / zero_denominator / basis_or_edition_mismatch / korea_exceeds_world다. 문제 있는 점유율만 null로 표시하며 원본 금액을 삭제하지 않는다.

Comtrade의 `data_status=actual`은 금액 관측 상태이며, 중량 결측·추정 표시와 `finality_status=not_provided`를 함께 확인한다. `released_at`·`release_evidence`는 미확보 null이다. 관측 패널을 예측시점 독립변수로 바로 결합하지 않으며 공개 근거 또는 가정을 검토하는 후속 처리가 필요하다.
관세청 v003 수출이 최종 타깃 원천이다. 비교 파일의 `difference_usd`는 Comtrade−관세청이며, 비교 결과로 타깃을 수정하거나 두 수출을 더하지 않는다.

## World Bank v001 처리 파일

`data/interim/world_bank/v001/`에 정규화 원본 1,560행과 요청 검산 6행을 보존한다. 원본의 indicator·country 하위 필드는 평탄화하고 source_date·source_unit·source_obs_status·source_decimal 등 원래 메타정보를 유지한다.

| 파일 | 단위·행 수 | 용도 |
|---|---|---|
| `macro_annual_long` | 국가×지표×연도, 1,680행 | 2013~2026 후보 격자, 원본 null·행 없음 구분 |
| `macro_annual` | 국가×연도, 280행 | 지도학습 후보 6개 거시지표와 지표별 원천·단위·결측 |
| `country_indicators_candidate` | 동일 기준연도 국가당 한 행, 20행 | 2024년 1인당 GDP·인구·인터넷 이용률 원래 값 |

long의 키는 `(country_code, indicator_year, indicator_name)`, wide의 키는 `(country_code, indicator_year)`다. value는 nullable Float64, 연도는 Int64다. GDP는 명목 USD, 1인당 GDP는 명목 USD/명, 인구는 명, 환율은 현지통화/USD 연평균, 물가·인터넷은 퍼센트 원래 값이다.

`record_observed=false`는 원본에 행이 없음, true이면서 value가 null이면 원본에 행은 있으나 숫자가 없음이다. missing_reason으로 no_record_in_successful_response와 null_value_in_source를 구분한다. 실제 0·음의 물가·소수값은 수정하지 않는다. 범위 의문은 quality_issue로만 표시한다.
wide에는 지표별 `{indicator}_value_missing`, `_record_observed`, `_missing_reason`, `_quality_issue`, `_unit`, `_source_file`, `_collected_at`, `_source_lastupdated`, `_source_obs_status`를 둔다. 연간 관측값을 분기 독립변수로 자동 복제·결합하지 않는다.

군집 후보의 reference_year는 값 완전성에 따른 2024년이다. selection_basis=latest_common_observed_year, reference_year_status=candidate_release_evidence_pending, training_approved=false, scaled=false다. 이 파일은 World Bank v001 당시 단일 연도 후보 이력이며 현재 기간별 비지도 v002의 학습 입력을 대신하지 않는다. 국가를 제외하거나 연도를 섞지 않았다.
released_at·release_evidence는 null이며 source_lastupdated는 개별 지표 최초 공개일이 아니다. API source_decimal은 원본 표시 정보이며 값 자체를 반올림하는 지시로 쓰지 않는다.

World Bank CSV는 원본 빈 문자열을 보존하기 위해 null을 `\N`으로 저장한다. `pd.read_csv(path, keep_default_na=False, na_values=[r"\N"], dtype=manifest_schema)`로 읽는다. Parquet는 null·빈 문자열과 명시 자료형을 그대로 보존한다. [최종 결합·분리 참고 사양서](dataset_release_reference.md)를 따른다.

## 지도학습 통합 v002·후보 표본

`data/processed/supervised/v002/`에 월별 관측 26,400행·분기별 관측 8,800행과 origin 변수 후보 9,600행, train 후보 28,800행·validation 후보 12,800행을 저장한다. 학습 대상은 2013Q1~2021Q4의 36분기, 검증 대상은 2022Q1~2025Q4의 16분기다. final_test 후보 480행의 원본 캐시는 `.private/evaluation/supervised/v002/`에 보존하고, 정답을 포함한 공유 파일은 `data/shared/v002/final_test.parquet`다. 모든 표는 CSV·Parquet이며 manifest에 기간·스키마·체크섬·열 연결·입력 목록이 있다. 기존 v001은 보존한다.

| 표 | 키·달력 | 의미 |
|---|---|---|
| `integrated_monthly` | 국가·HS6·period_month, year/month/quarter/quarter_number | 세 원천 관측 문맥을 합친 월별 전체 격자 |
| `integrated_quarterly` | 국가·HS6·quarter, year/quarter_number/month_start/month_end/period_months | 분기 전체 격자, 구성 월 3개 보존 |
| `origin_feature_candidates` | 국가·HS6·origin_quarter | 수집 전 5개 origin 분기도 포함한 시차·이동통계·기간 기준 후보 |
| `train_candidates`, `validation_candidates` | 국가·HS6·origin_quarter·target_quarter | 대상 분기별 h=1~5, 정답 결측 행도 보존 |
| `final_test_candidates` | 같은 표본 키, origin=2025Q4 고정 | 2026Q1~Q3·h=1~3, 공유 v002 final_test의 원천 후보 |

국가 코드·이름·지역·HS6·제품명·금액·중량·원천 메타정보를 유지한다. 열 이름 충돌은 comtrade_·world_bank_ 접두어로 구분하며 `column_lineage`에서 각 입력 열을 찾는다. 전체 관측표의 거시값은 같은 달력연도 관측 문맥이며 현재 분기의 예측 입력으로 직접 사용하지 않는다.

단가는 금액/양의 중량이며 0·결측 중량에는 null과 사유를 남긴다. 후보 `*_lag1/2/4`는 origin/origin−1/origin−3 값, `*_rolling4_sum/mean`은 origin까지 4개 값이 모두 있는 경우만 계산한다. `*_rolling4_valid_quarters`는 품질용 확보 개수다. 전년 대비 성장률은 origin과 origin−4를 비교하고 분모 0·결측이면 null이다.

시장 후보의 기준은 `market_candidate_quarter=origin_quarter`, 거시 후보는 `macro_candidate_year=origin_year−1`이다. 원천 미조회·이전 연도 결측·수집 이전 연도에 다른 값을 자동 대입하지 않는다. 원래 열과 `*_missing`을 함께 제공한다. 이 기간 선택은 공개시점을 입증한 as-of 결합이 아니다.

`origin_period_end_at`은 달력 분기 말, `origin_cutoff_at`은 미설정 null이다. `historical_release_verified=false`, `training_approved=false`, `candidate_only=true`이며 공개시점 기준·최종 공유 확정 전 후보 상태다. source_target_missing과 origin_in_collection_range로 정답 미확보·수집 이전 이력을 구분한다.

`target_export_value_usd`·`target_export_net_weight_kg`는 각각 금액 정답·보조 중량이다. 같은 타깃을 여러 origin/horizon 후보에 연결하므로 후보 행의 금액을 합산해 총수출액을 계산하지 않는다. 실제 학습·정답 평가는 정답이 있는 행과 후속 시점 기준을 적용한다.
모델 입력은 `feature_contract`의 categorical·known_future·numeric_candidates 57개 목록으로 제한한다. 타깃·타깃 중량·타깃 완전성·분할·원천 경로를 입력하지 않는다. 거시 포함/제외 실험과 학습 구간에서만 수행할 결측 대체·인코딩·표준화는 후속이다.
CSV null 표식은 `\N`이며 manifest 자료형과 함께 읽는다. 비지도학습 데이터·원천 버전·이전 v001은 보존했다. 공유 준비 JSON의 기간·후보 참조는 v002 기준으로 갱신했으며 shared 학습 파일은 아직 배포하지 않았다.

## 확정 지도학습 공유 v001

`data/shared/v002/`의 train·validation CSV/Parquet는 처리 v002에서 모델 입력과 필수 행 메타정보를 선택한 공유본이다. 학습 28,800행·검증 12,800행, 모델 입력 57개·전체 87개 열을 보존했다. 금액 정답·보조 중량·국가/품목 표시명·연도/분기/구성 월·origin/target·결측 표시도 포함한다.
공유 manifest의 feature_columns만 모델 입력이다. target_export_value_usd와 target_export_net_weight_kg 및 타깃 완전성·기간 분할·기타 메타정보는 feature_columns에 없다. 각 파일의 dtype·행 수·체크섬은 schemas·files에 기록한다.
사용자 공유 승인은 dataset_release_approved=true, status=released다. historical_release_evidence_verified=false이며 실제 공개일·빈티지는 미검증이다. 원래 처리 후보 상태는 이력으로 보존한다. 공유 v002에는 최종 테스트 정답·비지도 전체 표를 포함하며 CSV null은 `\N`이다.
