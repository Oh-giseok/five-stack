"""P05: enterprise production scenario inputs, pending UC06 calculation wiring."""

import streamlit as st

from oinq.ui.components import country_labels, onepage_section, product_labels, render_empty_result


def render(countries: list[dict], products: list[dict]) -> None:
    countries_by_code = country_labels(countries)
    products_by_code = product_labels(products)
    with onepage_section("p05_inputs"):
        st.title("생산 시나리오 | 기업 입력")
        st.caption("기업 가정값을 입력합니다. 입력값은 수출 예측의 사실값으로 취급하지 않습니다.")
        with st.form("p05_scenario_form"):
            left, middle, right = st.columns(3)
            with left:
                country = st.selectbox("국가", list(countries_by_code), format_func=countries_by_code.get, key="p05_country")
                hs_code = st.selectbox("제품", list(products_by_code), format_func=products_by_code.get, key="p05_product")
            with middle:
                target_share = st.number_input("목표 매출 비중 (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                unit_price = st.number_input("기업 단가 (USD/kg)", min_value=0.0, value=0.0, step=1.0, format="%.2f")
            with right:
                starting_stock = st.number_input("시작 재고 (kg)", min_value=0.0, value=0.0, step=1.0)
                safety_rate = st.number_input("안전 재고율 (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
            st.markdown("**분기별 입고량·생산능력 (kg)**")
            inbound_cols = st.columns(4)
            incoming = [col.number_input(f"Q{quarter} 입고량", min_value=0.0, value=0.0, step=1.0, key=f"p05_incoming_{quarter}")
                        for quarter, col in enumerate(inbound_cols, start=1)]
            capacity_cols = st.columns(4)
            capacity = [col.number_input(f"Q{quarter} 생산능력", min_value=0.0, value=0.0, step=1.0, key=f"p05_capacity_{quarter}")
                        for quarter, col in enumerate(capacity_cols, start=1)]
            st.caption("초기 0은 비어 있는 입력 필드의 표시값입니다. 실제 기업 단가·재고·능력을 입력하세요.")
            submitted = st.form_submit_button("입력 저장")

    with onepage_section("p05_results"):
        st.subheader("분기별 필요 생산량·생산능력 비교")
        if submitted:
            st.session_state["p05_scenario_inputs"] = {
                "country": country, "hs_code": hs_code, "target_share_pct": target_share,
                "unit_price_usd_per_kg": unit_price, "starting_stock_kg": starting_stock,
                "safety_stock_rate_pct": safety_rate, "incoming_kg": incoming, "capacity_kg": capacity,
            }
            st.success("기업 입력값을 저장했습니다.")
        render_empty_result(
            "시나리오 계산은 UC06 계산 모듈과 입출력 규격을 연결한 뒤 표시합니다.",
            detail="필요 생산량 차트와 분기별 생산 시나리오 표는 연결 전까지 임의 계산하지 않습니다.",
        )
