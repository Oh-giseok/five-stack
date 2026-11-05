"""P04: quarterly forecast for one country/product pair."""

import pandas as pd
import plotly.express as px
import streamlit as st

from oinq.ui.components import country_labels, format_musd, format_percent, onepage_section, render_empty_result, render_run_caption
from oinq.ui.data import ForecastRun


def render(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    country_names = country_labels(countries)
    product_names = {p["hs_code"]: f"{p['name_ko']} ({p['hs_code']})" for p in products}
    with onepage_section("p04_filters"):
        st.title("수요 예측 | 분기별 수출 전망")
        st.caption("실적과 2027년 예측을 함께 보는 화면입니다. 금액은 USD·FOB 기준입니다.")
        country = st.selectbox("국가", list(country_names), format_func=country_names.get, key="p04_country")
        hs_code = st.selectbox("제품", list(product_names), format_func=product_names.get, key="p04_product")
        quarter_choice = st.selectbox("예측 분기", ["전체 4분기", "Q1", "Q2", "Q3", "Q4"], key="p04_quarter")
        st.caption("기준분기 2026Q3 · 예측 대상 2027Q1~Q4 · 단위: 백만 USD")
        if run is None:
            render_empty_result("저장된 예측 결과가 없어 수요 예측을 준비 중입니다.")
        else:
            render_run_caption(run.metadata, run.grid_audit)

    if run is None:
        with onepage_section("p04_chart"):
            st.subheader("과거 실적과 2027년 예측 추이")
            render_empty_result("예측 파일이 연결되면 선택한 국가·제품의 분기 예측을 표시합니다.")
        with onepage_section("p04_table"):
            st.subheader("분기별 예상 수출액·성장률")
            render_empty_result("예측 파일이 연결되면 분기별 결과를 표시합니다.")
        return

    full_series = run.quarterly.loc[
        run.quarterly.country_code.astype(str).eq(country)
        & run.quarterly.hs_code.astype(str).eq(hs_code)
    ].copy()
    full_series = full_series.sort_values("target_quarter")
    full_series["qoq_growth_pct"] = full_series["predicted_export_value_usd"].pct_change(fill_method=None) * 100
    selected = full_series.copy()
    if quarter_choice != "전체 4분기":
        selected = selected.loc[selected.target_quarter.astype(str).eq(f"2027{quarter_choice}")]
    selected = selected.sort_values("target_quarter")
    selected["백만 USD"] = selected["predicted_export_value_usd"] / 1_000_000
    with onepage_section("p04_chart"):
        st.subheader(f"{country_names[country]} · {product_names[hs_code]} | 예측 수출액")
        st.info("과거 실적 자료가 아직 연결되지 않아 2027년 예측만 표시합니다. 2026Q4 실적은 생성하거나 보간하지 않습니다.")
        if selected["백만 USD"].notna().any():
            fig = px.line(selected, x="target_quarter", y="백만 USD", markers=True)
            fig.update_traces(line={"color": "#205f73", "width": 3, "dash": "dash"}, marker={"size": 9})
            fig.update_layout(height=520, margin={"l": 20, "r": 20, "t": 20, "b": 20},
                              xaxis_title="예측 대상 분기", yaxis_title="수출액 (백만 USD)", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_result("선택한 분기의 예측값이 미확보입니다.")
    with onepage_section("p04_table"):
        st.subheader("분기별 예상 수출액·전분기 대비 성장률")
        display = selected[["target_quarter", "predicted_export_value_usd", "qoq_growth_pct", "prediction_status",
                            "input_missing_feature_count"]].copy()
        display["predicted_export_value_usd"] = display["predicted_export_value_usd"].map(format_musd)
        display["qoq_growth_pct"] = display["qoq_growth_pct"].map(format_percent)
        display = display.rename(columns={
            "target_quarter": "분기", "predicted_export_value_usd": "예상 수출액 (백만 USD)",
            "qoq_growth_pct": "전분기 대비 성장률", "prediction_status": "예측 상태",
            "input_missing_feature_count": "입력 결측 변수 수",
        })
        if not display.empty and "2027Q1" in selected.target_quarter.astype(str).values:
            if selected.loc[selected.target_quarter.astype(str).eq("2027Q1")].shape[0]:
                display.loc[display["분기"].astype(str).eq("2027Q1"), "전분기 대비 성장률"] = "계산 불가 (2026Q4 미확보)"
        st.dataframe(display.reset_index(drop=True), hide_index=True, use_container_width=True)
