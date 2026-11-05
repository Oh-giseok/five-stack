"""P01: export overview for one destination country."""

import math

import pandas as pd
import plotly.express as px
import streamlit as st

from oinq.ui.components import (
    country_labels,
    format_musd,
    onepage_section,
    render_empty_result,
    render_run_caption,
)
from oinq.ui.data import ForecastRun, sum_if_complete


def _complete_quarter_totals(frame: pd.DataFrame, hs_codes: list[str], quarters: list[str]) -> pd.DataFrame:
    rows = []
    for quarter in quarters:
        group = frame.loc[frame.target_quarter.astype(str).eq(quarter)].set_index("hs_code")
        values = [group.loc[code, "predicted_export_value_usd"] if code in group.index else math.nan
                  for code in hs_codes]
        rows.append({"분기": quarter, "예상 수출액": sum_if_complete(values, len(hs_codes))})
    return pd.DataFrame(rows)


def render(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    labels = country_labels(countries)
    with onepage_section("p01_filters"):
        st.title("GEM | 화장품 수출 예측 대시보드")
        st.caption("한국 화장품 수출 수요를 국가별로 확인합니다. 금액은 USD·FOB 기준입니다.")
        selected_country = st.selectbox("국가", list(labels), format_func=labels.get, key="p01_country")
        st.caption("기간: 2027년 연간 예측 · 표시 단위: 백만 USD")
        if run is None:
            render_empty_result("저장된 예측 결과가 없어 화면을 준비 중입니다.")
        else:
            render_run_caption(run.metadata, run.grid_audit)

    quarters = [f"2027Q{q}" for q in range(1, 5)]
    if run is None:
        with onepage_section("p01_trend"):
            st.subheader("분기별 수출액 추이")
            render_empty_result("예측 파일이 연결되면 국가 선택에 따라 차트가 표시됩니다.")
        with onepage_section("p01_products"):
            st.subheader("예상 수출액 상위 3개 품목")
            render_empty_result("예측 파일이 연결되면 상위 품목을 표시합니다.")
        return

    selected = run.quarterly.loc[run.quarterly.country_code.astype(str).eq(selected_country)]
    chart_frame = _complete_quarter_totals(selected, [p["hs_code"] for p in products], quarters)
    with onepage_section("p01_trend"):
        st.subheader(f"{labels[selected_country]} | 2027 분기별 예상 수출액")
        st.info("과거 실적 자료가 아직 연결되지 않아 2027년 예측만 표시합니다.")
        if chart_frame["예상 수출액"].notna().any():
            chart_frame["백만 USD"] = chart_frame["예상 수출액"] / 1_000_000
            fig = px.line(chart_frame, x="분기", y="백만 USD", markers=True)
            fig.update_traces(line={"color": "#205f73", "width": 3}, marker={"size": 9})
            fig.update_layout(height=520, margin={"l": 20, "r": 20, "t": 25, "b": 20},
                              yaxis_title="수출액 (백만 USD)", xaxis_title="예측 대상 분기",
                              showlegend=False, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_result("선택한 국가의 분기 예측값이 모두 미확보입니다.")

    with onepage_section("p01_products"):
        st.subheader("예상 수출액 상위 3개 품목")
        annual = run.annual.loc[run.annual.country_code.astype(str).eq(selected_country)]
        rank_rows = []
        for product in products:
            item = annual.loc[annual.hs_code.astype(str).eq(product["hs_code"])]
            value = item.iloc[0]["predicted_annual_export_value_usd"] if not item.empty else math.nan
            rank_rows.append({"제품·HS": f"{product['name_ko']} · {product['hs_code']}", "금액": value})
        table = pd.DataFrame(rank_rows).dropna(subset=["금액"]).sort_values("금액", ascending=False).head(3)
        if table.empty:
            render_empty_result("완전한 연간 예측값이 없어 상위 품목을 계산할 수 없습니다.")
        else:
            table["예상 수출액 (백만 USD)"] = table.pop("금액").map(format_musd)
            st.dataframe(table.reset_index(drop=True), hide_index=True, use_container_width=True)
