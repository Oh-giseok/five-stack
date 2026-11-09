"""Quarterly production planning from forecasts and editable company assumptions."""

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from oinq.ui.components import country_labels, product_labels, onepage_section, render_empty_result, render_run_caption
from oinq.ui.data import ForecastRun


def _calculate(inputs: dict, run: ForecastRun | None) -> pd.DataFrame:
    if run is None or inputs["unit_price_usd_per_kg"] <= 0:
        return pd.DataFrame()
    rows = run.quarterly.loc[run.quarterly.country_code.astype(str).eq(inputs["country"])
                             & run.quarterly.hs_code.astype(str).eq(inputs["hs_code"])]
    stock = float(inputs["starting_stock_kg"])
    output = []
    for index, quarter in enumerate(("2027Q1", "2027Q2", "2027Q3", "2027Q4")):
        match = rows.loc[rows.target_quarter.astype(str).eq(quarter)]
        revenue = float(match.iloc[0].predicted_export_value_usd) if not match.empty and pd.notna(match.iloc[0].predicted_export_value_usd) else math.nan
        incoming = float(inputs["incoming_kg"][index])
        capacity = float(inputs["capacity_kg"][index])
        demand = revenue * inputs["target_share_pct"] / 100 / inputs["unit_price_usd_per_kg"] if math.isfinite(revenue) else math.nan
        available = stock + incoming if math.isfinite(stock) else math.nan
        required = max(0.0, demand * (1 + inputs["safety_stock_rate_pct"] / 100) - available) if math.isfinite(demand) else math.nan
        ending = available + required - demand if math.isfinite(demand) else math.nan
        output.append({"분기":quarter,"예상 매출 (USD)":revenue,"예상 수요 (kg)":demand,"입고 (kg)":incoming,
                       "필요 생산량 (kg)":required,"생산능력 (kg)":capacity,"생산능력 초과 (kg)":max(0,required-capacity) if math.isfinite(required) else math.nan,
                       "기말 재고 (kg)":ending})
        stock = ending
    return pd.DataFrame(output)


def render(countries: list[dict], products: list[dict], run: ForecastRun | None = None) -> None:
    country_names = country_labels(countries)
    product_names = product_labels(products)
    with onepage_section("p05_inputs"):
        st.title("생산 시나리오 | 기업 가정 기반 계획")
        st.caption("기업 입력값과 2027년 전망을 사용한 시나리오입니다. 예측 사실값이나 확정 생산 지시가 아닙니다.")
        if run is not None:
            render_run_caption(run.metadata, run.grid_audit)
        with st.expander("기업 조건 입력", expanded="p05_scenario_inputs" not in st.session_state):
            with st.form("p05_scenario_form"):
                left, middle, right = st.columns(3)
                with left:
                    country = st.selectbox("국가", list(country_names), format_func=country_names.get, key="gem_selected_country")
                    hs_code = st.selectbox("품목", list(product_names), format_func=product_names.get, key="gem_selected_product")
                with middle:
                    target_share = st.number_input("목표 비중 (전망 수출액 대비, %)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="p05_share")
                    unit_price = st.number_input("기업 단가 (USD/kg)", min_value=0.01, value=10.0, step=1.0, format="%.2f", key="p05_price")
                with right:
                    starting_stock = st.number_input("시작 재고 (kg)", min_value=0.0, value=0.0, step=100.0, key="p05_stock")
                    safety_rate = st.number_input("안전재고율 (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="p05_safety")
                st.markdown("**분기별 입고량·생산능력 (kg)**")
                inbound_cols = st.columns(4)
                incoming = [col.number_input(f"Q{i} 입고", min_value=0.0, value=0.0, step=100.0, key=f"p05_incoming_{i}") for i, col in enumerate(inbound_cols, 1)]
                capacity_cols = st.columns(4)
                capacity = [col.number_input(f"Q{i} 생산능력", min_value=0.0, value=10000.0, step=100.0, key=f"p05_capacity_{i}") for i, col in enumerate(capacity_cols, 1)]
                submitted = st.form_submit_button("시나리오 계산", type="primary", key="p05_scenario_calculate")
        if submitted:
            st.session_state["p05_scenario_inputs"] = {"country":country,"hs_code":hs_code,"target_share_pct":target_share,
                "unit_price_usd_per_kg":unit_price,"starting_stock_kg":starting_stock,"safety_stock_rate_pct":safety_rate,
                "incoming_kg":incoming,"capacity_kg":capacity}
    with onepage_section("p05_results"):
        st.subheader("분기별 필요 생산량과 생산능력")
        inputs = st.session_state.get("p05_scenario_inputs")
        if inputs is None:
            render_empty_result("기업 조건을 입력하고 시나리오 계산을 실행하세요.")
            return
        result = _calculate(inputs, run)
        if result.empty:
            render_empty_result("저장된 예측 결과가 없거나 단가가 유효하지 않아 계산할 수 없습니다.", detail="사이드바 공통 예측 설정에서 저장 모델을 선택하고 기업 단가를 입력하세요.")
            return
        st.caption(f"{country_names[inputs['country']]} · {product_names[inputs['hs_code']]} · 목표 비중 {inputs['target_share_pct']:g}% · 단가 ${inputs['unit_price_usd_per_kg']:,.2f}/kg · 안전재고율 {inputs['safety_stock_rate_pct']:g}%")
        chart = go.Figure()
        chart.add_bar(x=result["분기"], y=result["필요 생산량 (kg)"], name="필요 생산량", marker_color="#65A30D")
        chart.add_scatter(x=result["분기"], y=result["생산능력 (kg)"], name="생산능력", mode="lines+markers", line={"color":"#d48a28", "width":3})
        chart.update_layout(height=450, barmode="group", yaxis_title="kg", xaxis_title="2027년 분기", legend={"orientation":"h", "y":1.12})
        display = result.copy()
        for col in display.columns[1:]:
            display[col] = display[col].map(lambda value: "미확보" if pd.isna(value) else f"{value:,.1f}")
        chart_col, table_col = st.columns([1.2, 1], gap="large")
        with chart_col:
            st.plotly_chart(chart, use_container_width=True)
        with table_col:
            st.dataframe(display, hide_index=True, use_container_width=True, height=450)
        st.caption("계산: 전망 수출액×목표 비중÷기업 단가로 수요를 환산하고, 안전재고 목표에서 시작 재고·입고를 차감합니다. 미확보 전망은 임의 보간하지 않습니다.")
