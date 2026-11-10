"""Product demand cards with forecast ranking and observed/forecast timeline."""
from pathlib import Path
import math

import pandas as pd
import streamlit as st

from oinq.ui.components import (
    country_labels, product_labels, install_analysis_styles,
    render_empty_result, render_run_caption, render_export_timeline,
)
from oinq.ui.country_charts import forecast_series, history_series, ranking_figure
from oinq.ui.data import ForecastRun, load_monthly_exports


@st.cache_data(show_spinner=False, max_entries=2)
def _history(root: str, country_codes: tuple, hs_codes: tuple, signature: tuple):
    del signature
    return load_monthly_exports(root, country_codes, hs_codes)


def _open_production() -> None:
    st.session_state["gem_main_navigation"] = "생산 시나리오"


def _ranking_records(run, country, products, period):
    records = []
    frequency = "연간" if period == "2027년 연간" else "분기"
    for product in products:
        if run is None:
            row = {"value_usd": math.nan, "status": "forecast", "valid_count": 0,
                   "expected_count": 4 if frequency == "연간" else 1}
        else:
            values = forecast_series(run, country, product["hs_code"], frequency)
            selected = values.iloc[0] if frequency == "연간" else values.loc[values.period.eq(period)].iloc[0]
            row = selected.to_dict()
        records.append({**row, "code": product["hs_code"], "label": product["name_ko"]})
    return pd.DataFrame(records)


def render(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    country_names = country_labels(countries)
    product_names = product_labels(products)
    short_countries = {row["iso3"]: row["name_ko"] for row in countries}
    short_products = {row["hs_code"]: row["name_ko"] for row in products}
    install_analysis_styles("products")
    with st.container(key="gem_products_page"):
        st.title("품목별 수요", anchor=False)
        st.caption("어떤 품목을 어느 분기에 수출할지 비교합니다.")
        with st.container(border=True, key="gem_products_filters_selection"):
            a, b, c = st.columns([1.2, 1.8, 1])
            with a:
                country = st.selectbox("국가", list(country_names), format_func=country_names.get,
                                        key="gem_selected_country", persist_state="session")
            with b:
                hs_code = st.selectbox("품목", list(product_names), format_func=product_names.get,
                                        key="gem_selected_product", persist_state="session")
            with c:
                period = st.selectbox("품목 비교 기간", ["2027년 연간", "2027Q1", "2027Q2", "2027Q3", "2027Q4"],
                                       key="product_period")
        if run is None:
            st.info("저장된 예측 결과가 없습니다. 사이드바의 공통 예측 설정에서 모델을 선택하세요.")
        else:
            render_run_caption(run.metadata, run.grid_audit, show_context=False)
        root = Path(__file__).resolve().parents[4]
        monthly = None
        history_error = None
        try:
            path = root / "data/shared/history/customs_monthly_hs6.parquet"
            stat = path.stat()
            monthly = _history(str(root), tuple(country_names), tuple(product_names),
                               (stat.st_size, stat.st_mtime_ns))
        except (OSError, ValueError) as error:
            history_error = str(error)
        with st.container(border=True, key="gem_products_card_ranking"):
            st.subheader(f"{short_countries[country]} | 8개 품목 예상 수출액", anchor=False)
            st.plotly_chart(
                ranking_figure(_ranking_records(run, country, products, period), hs_code, predicted=True,
                               height=380, main_color="#EA580C", secondary_color="#F6D1BB"),
                width="stretch", theme=None, key="product_ranking",
                config={"displayModeBar": False}, alt="선택 국가의 8개 품목 예상 수출액과 미확보 품목",
            )
            if monthly is not None and run is not None:
                baseline = history_series(monthly, country, hs_code, "연간").set_index("period").loc["2025"]
                future = forecast_series(run, country, hs_code, "연간").iloc[0].value_usd
                if pd.notna(baseline.complete_value_usd) and baseline.complete_value_usd > 0 and pd.notna(future):
                    growth = (future / baseline.complete_value_usd - 1) * 100
                    st.caption(f"{short_products[hs_code]} · 2025년 대비 2027년 예상 증감률 {growth:+,.1f}% (2년 누적)")
                elif pd.isna(future):
                    st.caption("2027년 네 분기 예측이 모두 확보되지 않아 증감률을 계산하지 않습니다.")
                else:
                    st.caption("2025년의 완전한 양수 기준 실적이 없어 증감률을 계산하지 않습니다.")
        with st.container(border=True, key="gem_products_card_timeline"):
            st.subheader(f"{short_products[hs_code]} | 실적과 분기별 전망", anchor=False)
            if monthly is None:
                render_empty_result("공유된 실제 수출 자료를 확인하세요.", detail=history_error)
            else:
                render_export_timeline(monthly, country, hs_code, key="product_timeline", run=run,
                                       colors={"actual": "#205f73", "ytd": "#205f73", "forecast": "#d48a28"})
        st.button("이 품목으로 생산 시나리오 만들기", icon=":material/factory:", key="product_to_production",
                   on_click=_open_production)
