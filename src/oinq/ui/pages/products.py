"""P03: compare the eight configured products for one destination."""

import math

import pandas as pd
import plotly.express as px
import streamlit as st

from oinq.ui.components import country_labels, format_musd, onepage_section, render_empty_result, render_run_caption
from oinq.ui.data import ForecastRun


def render(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    labels = country_labels(countries)
    with onepage_section("p03_filters"):
        st.title("제품별 분석")
        st.caption("선택 국가의 8개 HS 품목을 비교합니다. 금액은 USD·FOB 기준입니다.")
        country = st.selectbox("국가", list(labels), format_func=labels.get, key="p03_country")
        period = st.selectbox("기간", ["연간", "Q1", "Q2", "Q3", "Q4"], key="p03_period")
        metric = "predicted_annual_export_value_usd" if period == "연간" else "predicted_export_value_usd"
        quarter = None if period == "연간" else f"2027{period}"
        st.caption(f"정렬: 선택 기간 예상 수출액 내림차순 · 단위: 백만 USD · 성장률 기준: 2025년 대비 2027년")
        if run is None:
            render_empty_result("저장된 예측 결과가 없어 제품 비교를 준비 중입니다.")
        else:
            render_run_caption(run.metadata, run.grid_audit)

    if run is None:
        with onepage_section("p03_chart"):
            st.subheader("제품별 예상 수출액 순위")
            render_empty_result("예측 파일이 연결되면 제품별 금액을 표시합니다.")
        with onepage_section("p03_table"):
            st.subheader("8개 제품 분기별 비교")
            render_empty_result("예측 파일이 연결되면 제품별 금액과 성장률을 표시합니다.")
        return

    q = run.quarterly.loc[
        run.quarterly.country_code.astype(str).eq(country)
        & run.quarterly.target_quarter.astype(str).eq(quarter)
    ] if quarter else run.quarterly.loc[run.quarterly.country_code.astype(str).eq(country)]
    a = run.annual.loc[run.annual.country_code.astype(str).eq(country)]
    records = []
    for product in products:
        code = product["hs_code"]
        qr = q.loc[q.hs_code.astype(str).eq(code)]
        ar = a.loc[a.hs_code.astype(str).eq(code)]
        annual_value = ar.iloc[0]["predicted_annual_export_value_usd"] if not ar.empty else math.nan
        selected_value = (annual_value if period == "연간" else
                          qr.iloc[0]["predicted_export_value_usd"] if not qr.empty else math.nan)
        row = {"코드": code, "제품": product["name_ko"], "선택 기간 금액": selected_value,
               "연간 예상 수출액": annual_value, "성장률": "계산 불가"}
        for q_number in range(1, 5):
            quarter_rows = run.quarterly.loc[
                run.quarterly.country_code.astype(str).eq(country)
                & run.quarterly.hs_code.astype(str).eq(code)
                & run.quarterly.target_quarter.astype(str).eq(f"2027Q{q_number}")
            ]
            row[f"Q{q_number}"] = quarter_rows.iloc[0]["predicted_export_value_usd"] if not quarter_rows.empty else math.nan
        records.append(row)
    result = pd.DataFrame(records).sort_values("선택 기간 금액", ascending=False, na_position="last")
    result["순위"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    available = result["선택 기간 금액"].notna()
    result.loc[available, "순위"] = range(1, int(available.sum()) + 1)

    with onepage_section("p03_chart"):
        st.subheader(f"{country} | {period} 제품별 예상 수출액 순위")
        chart = result.dropna(subset=["선택 기간 금액"]).sort_values("선택 기간 금액", ascending=True)
        if chart.empty:
            render_empty_result("선택한 기간의 예상 수출액이 미확보입니다.")
        else:
            chart["백만 USD"] = chart["선택 기간 금액"] / 1_000_000
            fig = px.bar(chart, x="백만 USD", y="제품", orientation="h", color_discrete_sequence=["#205f73"])
            fig.update_layout(height=520, margin={"l": 20, "r": 20, "t": 20, "b": 20},
                              xaxis_title="예상 수출액 (백만 USD)", yaxis_title="제품", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with onepage_section("p03_table"):
        st.subheader("8개 제품 분기·연간 금액·성장률·순위")
        display = result[["순위", "코드", "제품", "Q1", "Q2", "Q3", "Q4", "연간 예상 수출액", "성장률"]].copy()
        for col in ("Q1", "Q2", "Q3", "Q4", "연간 예상 수출액"):
            display[col] = display[col].map(format_musd)
        display = display.rename(columns={"연간 예상 수출액": "연간 (백만 USD)", "코드": "HS"})
        st.dataframe(display.reset_index(drop=True), hide_index=True, use_container_width=True, height=470)
        st.caption("성장률은 2025년 완전한 기준 실적이 연결되기 전까지 계산 불가로 표시합니다.")
