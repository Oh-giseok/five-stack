# 데이터 원천·국가코드 확인 메모

확인일: 2026-11-03. 아래는 공식 문서·코드표 기준이다. 인증·샘플 응답 확인 후 설계 범위의 원본 수집을 실행했다.
국가·품목별 원본 확보 기간과 누락은 [수집 결과](../reports/collection_20261103.md), 최신 전처리 상태는 [관세청 v003](../reports/customs_v003_all_country_product_review.md)·[Comtrade v001](../reports/comtrade_v001_preprocessing.md)·[World Bank v001](../reports/world_bank_v001_preprocessing.md)을 따른다. 없는 원천 값을 만들어 채우지 않는다.

## 관세청 — 한국 수출 주 원천

- [품목별 국가별 수출입실적 API](https://www.data.go.kr/data/15100475/openapi.do): 금액·중량 월별 조회와 인증키 신청 규격을 확인한다.
- 국가 alpha-2, HS, 시작·종료 연월을 요청하며 설정은 `sources.yaml`의 `customs`에 둔다. 키는 `.env`의 `CUSTOMS_API_KEY`만 사용한다.
- 응답 오류·빈 결과·실제 0을 구분하고 HS 6자리 조회 결과와 단위를 샘플로 확인한다. 2026년 거시 총괄 발표가 개별 품목의 완전성을 보장하지 않는다.

## UN Comtrade — 장기 교역 비교·수입시장

- [API 안내](https://uncomtrade.org/docs/un-comtrade-api/), [자료 가용성](https://uncomtrade.org/docs/data-availability/), [국가코드 안내](https://uncomtrade.org/docs/country-codes/).
- 한국 신고 수출과 상대국 신고 수입은 방향·신고국·상대국을 구분한다. 수출과 수입 신고값을 같은 거래의 추가 금액으로 더하지 않는다.
- preview는 건수 제한이 있어 대량 수집 대용으로 쓰지 않는다. 실제 접근 권한·할당량과 누락·잘림 여부를 수집 시 확인한다.
- [금액 평가 기준](https://uncomtrade.org/docs/trade-valuation/): 수입 primaryValue는 CIF를 우선하나 국가별 FOB 자료도 있으므로 원본 cifvalue·fobvalue로 식별한다. 이번 원본에서 캐나다·남아공 수입은 FOB, 나머지 대상국은 CIF다.
- [공식 필드 정의](https://comtradeplus-test.azurewebsites.net/en/reference-tables?table=dataitem): isAggregate는 UNSD 집계, isReported는 국가 직접 신고, isNetWgtEstimated는 중량 추정 표시다. 유효한 HS6 집계행을 해당 플래그만으로 삭제하지 않는다.
- 공식 HS 사전 H2~H6를 `data/reference/comtrade/hs_20261103/`에 별도 저장했다. 8개 품목의 코드·상위 분류 40개 조합을 대조하고 원본별 판본을 유지했다. 판본 차이만으로 원본을 보류하지 않는다.

## 국가코드는 원천별로 다르다

공식 [신고국 코드표](https://comtradeapi.un.org/files/v1/app/reference/Reporters.json)와 [상대국 코드표](https://comtradeapi.un.org/files/v1/app/reference/partnerAreas.json)를 대조했다.
내부 ISO3를 유지하고 관세청·Comtrade 신고국/상대국·World Bank 매핑을 `countries.yaml`에서 별도로 읽는다.

| 국가 | 내부 | 관세청 매핑 | 현재 Comtrade 신고국 | 주의 |
|---|---|---|---|---|
| 한국 | KOR | KR | 410 | 한국 신고 수출의 reporter |
| 인도 | IND | IN | 699 | ISO 숫자 356을 현재 reporter에 대입하지 않음 |
| 프랑스 | FRA | FR | 251 | 250과 통계영역이 다름 |
| 미국 | USA | US | 842 | 840과 통계영역이 다름 |

국가별 대체 코드의 값을 합치면 영토 범위가 겹칠 수 있다. 관세청과 연결할 때 포함 영역·중복 여부를 확인한다.
현재 수집은 위 매핑으로 실행했다. 과거 브리핑의 잘못된 코드로 조회한 통계는 그대로 재사용하지 않으며, 이번 작업에서 월별 원본을 새로 확보했다.

## World Bank — 거시변수 공용 원천

- [API 호출 구조](https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-basic-call-structures)를 기준으로 국가·지표·연도·페이지를 요청한다. 별도 API 키를 두지 않는다.
- 6개 지표 코드는 `sources.yaml`, 지도학습·군집 입력은 `features.yaml`에 둔다. 연간 지표를 최신 공개 기준으로 결합하며 미래 연도 실측값은 제외한다.
- 현재 API의 연도와 `lastupdated`는 개별 값이 과거 어느 날 처음 공개됐는지를 증명하지 않는다. 과거 수정치·빈티지와 공개시점의 차이를 기록한다.
- 공개일 근거가 없으면 보수적 가정을 문서화한 제한적 분석 또는 변수 제외를 선택한다. 단순히 연도만 늦췄다고 누수 없는 검증으로 확정하지 않는다.
- v001은 20개국·6개 지표의 원본 1,560행을 전부 보존하고 실제 값 1,540개·결측 140개를 구분했다. 2026년 원본 행은 없으며 2025년 null 20개는 미국 물가·베트남 환율·18개국 인터넷이다. 2024년 20개국 세 지표를 군집 후보로 분리했지만 공개 근거·학습 적용을 자동 승인하지 않았다.

## 구현 참고

초기 모델 설정은 [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html), [LightGBM](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html), [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor), [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)의 파라미터 이름을 따른다.
YAML은 실험 시작값이다. 설치 환경·실제 데이터로 학습 성공이나 예측 정확도를 검증한 상태는 아니다.
