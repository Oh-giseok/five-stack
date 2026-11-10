"""Integrated product demand ranking and observed/forecast trend."""

from pathlib import Path
import math

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from oinq.ui.components import country_labels, product_labels, onepage_section, render_empty_result, render_run_caption
from oinq.ui.data import ForecastRun, load_historical_exports


@st.cache_data(show_spinner=False, max_entries=1)
def _history(root: str):
    return load_historical_exports(root)


def _open_production() -> None:
    st.session_state["gem_main_navigation"] = "생산 시나리오"


def render(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    countries_by_code = country_labels(countries)
    product_names = product_labels(products)
    with onepage_section("product_filters"):
        st.title("품목별 수요")
        st.caption("국가 내 8개 품목의 2027년 전망 순위와 선택 품목의 과거 실적·전망을 확인합니다.")
        left, right = st.columns(2)
        with left:
            country = st.selectbox("국가", list(countries_by_code), format_func=countries_by_code.get, key="gem_selected_country")
        with right:
            hs_code = st.selectbox("상세 품목", list(product_names), format_func=product_names.get, key="gem_selected_product")
        if run is None:
            st.info("저장된 예측 결과가 없습니다. 사이드바의 공통 예측 설정에서 모델을 선택하세요.")
        else:
            render_run_caption(run.metadata, run.grid_audit, show_context=False)

    if run is None:
        rank = pd.DataFrame()
    else:
        annual = run.annual.loc[run.annual.country_code.astype(str).eq(country)]
        rank_rows = []
        for product in products:
            matched = annual.loc[annual.hs_code.astype(str).eq(product["hs_code"])]
            value = matched.iloc[0]["predicted_annual_export_value_usd"] if not matched.empty else math.nan
            rank_rows.append({"품목": product["name_ko"], "예상 수출액 (백만 USD)": value / 1_000_000 if pd.notna(value) else math.nan})
        rank = pd.DataFrame(rank_rows).dropna().sort_values("예상 수출액 (백만 USD)")

    rank_col, trend_col = st.container(key="gem_equal_panels_products").columns(2, gap="large")
    with rank_col:
        st.subheader("2027년 품목별 예상 수출액")
        if rank.empty:
            render_empty_result("완전한 연간 예측값이 없어 품목 순위를 표시할 수 없습니다.")
        else:
            fig = px.bar(rank, x="예상 수출액 (백만 USD)", y="품목", orientation="h", color_discrete_sequence=["#EA580C"])
            fig.update_layout(height=440, margin={"l":20,"r":20,"t":55,"b":55},
                              xaxis_title="예상 수출액 (백만 USD)", yaxis_title="품목", showlegend=False)
            st.plotly_chart(fig, width="stretch", alt="선택 국가의 품목별 연간 예상 수출액")

    with trend_col:
        st.subheader("선택 품목의 실제 실적·전망")
        root = Path(__file__).resolve().parents[4]
        observed = _history(str(root))
        observed = observed.loc[observed.country_code.astype(str).eq(country) & observed.hs_code.astype(str).eq(hs_code)].copy()
        observed["분기"] = observed.target_quarter.astype(str)
        actual = observed.copy()
        valid = actual.target_data_status.astype(str).isin(["actual", "partial"])
        actual["chart_value"] = (actual.target_export_value_usd / 1_000_000).where(valid)
        fig = go.Figure()
        if not actual.empty and valid.any():
            fig.add_trace(go.Scatter(x=actual["분기"], y=actual.chart_value, name="실제 실적",
                mode="lines+markers", connectgaps=False, line={"color":"#205f73", "width":2},
                customdata=actual[["target_data_status", "target_observed_months"]],
                hovertemplate="%{x}<br>%{y:,.2f} 백만 USD<br>상태: %{customdata[0]}<br>관측 월: %{customdata[1]}<extra>실제 실적</extra>"))
        if run is not None:
            forecast = run.quarterly.loc[run.quarterly.country_code.astype(str).eq(country) & run.quarterly.hs_code.astype(str).eq(hs_code)].sort_values("target_quarter")
            if not forecast.empty:
                fig.add_trace(go.Scatter(x=forecast.target_quarter.astype(str), y=forecast.predicted_export_value_usd / 1_000_000,
                    name="2027 예측", mode="lines+markers", connectgaps=False, line={"color":"#d48a28", "width":3, "dash":"dash"}))
        if len(fig.data):
            fig.update_layout(height=440, margin={"l":20,"r":20,"t":55,"b":55},
                              xaxis_title="분기", yaxis_title="수출액 (백만 USD)", hovermode="x unified", legend={"orientation":"h", "y":1.12})
            st.plotly_chart(fig, width="stretch", alt="선택 품목의 과거 실적과 2027년 수출 전망")
        else:
            render_empty_result("선택한 국가·품목의 실적과 전망 자료가 없습니다.")
        st.caption("실적은 2013Q1–2026Q3이며 2026년은 1~9월 누적입니다. 실제 0과 결측·미확보는 구분하며, 2026Q4는 비워 둡니다.")
        st.button("선택 국가·품목으로 생산 시나리오 이동", key="product_to_production",
                  on_click=_open_production)
