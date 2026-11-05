"""P02A country comparison and P02B similar-country views."""

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


WINDOWS = ["2013_2015", "2016_2018", "2019_2021", "2022_2024", "2025_2026Q3"]
WINDOW_LABELS = {
    "2013_2015": "2013–2015", "2016_2018": "2016–2018", "2019_2021": "2019–2021",
    "2022_2024": "2022–2024", "2025_2026Q3": "2025–2026Q3",
}
INDICATORS = {
    "1인당 GDP": ("gdp_per_capita_usd", "USD/인", "1인당 GDP (USD/인)"),
    "인구": ("population", "명", "인구 (명)"),
    "인터넷 이용률": ("internet_users_pct", "%", "인터넷 이용률 (%)"),
}


def _country_quarter_summary(run: ForecastRun, country_codes: list[str], hs_codes: list[str]):
    quarters = list(run.metadata.get("target_quarters", sorted(run.quarterly.target_quarter.unique())))
    by_quarter = run.quarterly.loc[
        run.quarterly.country_code.astype(str).isin(country_codes)
        & run.quarterly.hs_code.astype(str).isin(hs_codes)
    ]
    annual = run.annual.loc[
        run.annual.country_code.astype(str).isin(country_codes)
        & run.annual.hs_code.astype(str).isin(hs_codes)
    ]
    rows = []
    for country in country_codes:
        row = {"국가": country}
        for quarter in quarters:
            group = by_quarter.loc[
                by_quarter.country_code.astype(str).eq(country)
                & by_quarter.target_quarter.astype(str).eq(str(quarter))
            ].set_index("hs_code")
            values = [group.loc[hs, "predicted_export_value_usd"] if hs in group.index else math.nan
                      for hs in hs_codes]
            row[quarter[-2:]] = sum_if_complete(values, len(hs_codes))
        group = annual.loc[annual.country_code.astype(str).eq(country)].set_index("hs_code")
        values = []
        for hs in hs_codes:
            if hs not in group.index or str(group.loc[hs, "annual_status"]) != "complete":
                values.append(math.nan)
            else:
                values.append(group.loc[hs, "predicted_annual_export_value_usd"])
        row["연간"] = sum_if_complete(values, len(hs_codes))
        rows.append(row)
    return pd.DataFrame(rows)


def render_comparison(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    labels = country_labels(countries)
    country_codes = list(labels)
    with onepage_section("p02a_filters"):
        st.title("국가별 분석 | 수출 비교")
        st.caption("대상국별 2027년 예상 수출액을 비교합니다. 금액은 USD·FOB 기준입니다.")
        compare_scope = st.radio("국가 범위", ["전체 20개국", "직접 선택"], horizontal=True, key="p02a_scope")
        selected_countries = country_codes
        if compare_scope == "직접 선택":
            selected_countries = st.multiselect(
                "비교 국가", country_codes, default=country_codes[:5],
                format_func=labels.get, key="p02a_countries",
            )
        hs_options = [row["hs_code"] for row in products]
        selected_product = st.selectbox(
            "품목", ["전체 8개 품목", *hs_options],
            format_func=lambda code: "전체 8개 품목" if code == "전체 8개 품목"
            else f"{next(p['name_ko'] for p in products if p['hs_code'] == code)} ({code})",
            key="p02a_product",
        )
        hs_codes = hs_options if selected_product == "전체 8개 품목" else [selected_product]
        st.caption("기간: 2027년 · 국가 차트 기준: 선택 품목의 연간 합계 · 단위: 백만 USD")
        if run is None:
            render_empty_result("저장된 예측 결과가 없어 국가 비교를 준비 중입니다.")
        else:
            render_run_caption(run.metadata, run.grid_audit)

    if run is None:
        with onepage_section("p02a_chart"):
            st.subheader("국가별 예상 수출액")
            render_empty_result("예측 파일이 연결되면 국가별 비교를 표시합니다.")
        with onepage_section("p02a_table"):
            st.subheader("국가별 분기·연간 예상 수출액")
            render_empty_result("예측 파일이 연결되면 국가별 금액을 표시합니다.")
        return
    if not selected_countries:
        with onepage_section("p02a_chart"):
            st.subheader("국가별 예상 수출액")
            render_empty_result("비교할 국가를 선택하세요.")
        with onepage_section("p02a_table"):
            st.subheader("국가별 분기·연간 예상 수출액")
            render_empty_result("국가를 선택하면 표가 표시됩니다.")
        return

    summary = _country_quarter_summary(run, selected_countries, hs_codes)
    summary["국가명"] = summary["국가"].map(labels)
    chart = summary.dropna(subset=["연간"]).sort_values("연간", ascending=True)
    with onepage_section("p02a_chart"):
        st.subheader("국가별 예상 수출액 | 2027 연간")
        if chart.empty:
            render_empty_result("완전한 연간 예상값이 없어 비교 차트를 표시할 수 없습니다.")
        else:
            chart["백만 USD"] = chart["연간"] / 1_000_000
            fig = px.bar(chart, x="백만 USD", y="국가명", orientation="h", color_discrete_sequence=["#205f73"])
            fig.update_layout(height=max(500, 34 * len(chart) + 100), margin={"l": 20, "r": 20, "t": 20, "b": 20},
                              xaxis_title="예상 수출액 (백만 USD)", yaxis_title="국가", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with onepage_section("p02a_table"):
        st.subheader("국가별 분기·연간 예상 수출액 (백만 USD)")
        display = summary[["국가명", "Q1", "Q2", "Q3", "Q4", "연간"]].copy()
        for column in ("Q1", "Q2", "Q3", "Q4", "연간"):
            display[column] = display[column].map(format_musd)
        display = display.rename(columns={"국가명": "국가"})
        st.dataframe(display, hide_index=True, use_container_width=True, height=min(650, 42 * len(display) + 45))
        if len(hs_codes) > 1:
            st.caption("분기·연간 금액은 선택된 품목 전부의 예측값이 있을 때만 합산됩니다.")


@st.cache_data(show_spinner=False)
def _load_context() -> tuple[pd.DataFrame, pd.DataFrame]:
    root = __import__("pathlib").Path(__file__).resolve().parents[4]
    folder = root / "data" / "shared" / "v002" / "unsupervised"
    return (
        pd.read_parquet(folder / "country_window_features.parquet"),
        pd.read_parquet(folder / "country_window_context.parquet"),
    )


def render_similar(countries: list[dict], products: list[dict]) -> None:
    labels = country_labels(countries)
    with onepage_section("p02b_filters"):
        st.title("국가별 분석 | 유사 국가")
        st.caption("기간별 거시지표 원래 단위와 관측 연도를 확인합니다.")
        view = st.radio("보기", ["군집 분포", "기간별 변화"], horizontal=True, key="p02b_view")
        try:
            features, context = _load_context()
            data_available = True
        except (OSError, ValueError):
            features, context, data_available = pd.DataFrame(), pd.DataFrame(), False
        if view == "기간별 변화":
            st.caption("분석기간: 2013–2015부터 2025–2026Q3까지")
            country = st.selectbox("국가", list(labels), format_func=labels.get, key="p02b_country")
            indicator_label = st.selectbox("지표", list(INDICATORS), key="p02b_indicator")
        else:
            window = st.selectbox("분석기간", WINDOWS, index=3, format_func=WINDOW_LABELS.get, key="p02b_window")
            country = None
            indicator_label = None
        if not data_available:
            render_empty_result("공유 비지도 원천 자료를 읽을 수 없어 화면을 준비 중입니다.")

    if not data_available:
        return
    if view == "군집 분포":
        selected = features.loc[features.window_id.astype(str).eq(window)].copy()
        label = WINDOW_LABELS[window]
        with onepage_section("p02b_distribution_chart"):
            st.subheader(f"{label} | 국가 특성·군집 분포")
            st.info("K-means 군집 결과가 아직 연결되지 않았습니다. 결과가 준비되면 GDP·인터넷 이용률·인구를 표시합니다.")
        with onepage_section("p02b_distribution_table"):
            st.subheader(f"{label} | 국가별 원천 지표")
            columns = ["country_code", "country_name", "gdp_per_capita_usd", "internet_users_pct", "population",
                       "gdp_per_capita_usd_observed_years", "internet_users_pct_observed_years", "eligible_for_fit"]
            display = selected[columns].rename(columns={
                "country_code": "국가 코드", "country_name": "국가", "gdp_per_capita_usd": "1인당 GDP (USD/인)",
                "internet_users_pct": "인터넷 이용률 (%)", "population": "인구 (명)",
                "gdp_per_capita_usd_observed_years": "GDP 관측 연도", "internet_users_pct_observed_years": "인터넷 관측 연도",
                "eligible_for_fit": "군집 입력 가능",
            })
            display["군집 상태"] = "준비 중"
            st.dataframe(display, hide_index=True, use_container_width=True, height=620)
            st.caption("인구는 점 크기 기준입니다. 군집 번호와 색상은 실제 학습 결과가 연결될 때 표시합니다.")
        return

    indicator, unit, axis_title = INDICATORS[indicator_label]
    values = context.loc[
        context.country_code.astype(str).eq(country)
        & context[f"{indicator}_observed_mean"].notna()
    ].copy()
    values["window_order"] = values.window_id.astype(str).map({key: idx for idx, key in enumerate(WINDOWS)})
    values = values.sort_values("window_order")
    values["기간"] = values.window_id.astype(str).map(WINDOW_LABELS)
    value_col = f"{indicator}_observed_mean"
    if value_col not in values:
        value_col = indicator
    with onepage_section("p02b_change_chart"):
        st.subheader(f"{labels[country]} | {axis_title} 기간별 변화")
        if values.empty:
            render_empty_result("선택한 지표의 관측값이 없습니다.")
        else:
            fig = px.line(values, x="기간", y=value_col, markers=True)
            fig.update_traces(line={"color": "#205f73", "width": 3}, marker={"size": 9})
            fig.update_layout(height=500, margin={"l": 20, "r": 20, "t": 20, "b": 20},
                              xaxis_title="분석기간", yaxis_title=f"{axis_title} · {unit}", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with onepage_section("p02b_change_table"):
        st.subheader("원래 단위와 실제 관측 연도")
        observed_years = f"{indicator}_observed_years"
        display = values[["기간", "start_month", "end_month", value_col, observed_years]].rename(columns={
            "start_month": "시작월", "end_month": "종료월", value_col: f"{axis_title} ({unit})",
            observed_years: "실제 관측 연도",
        })
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("마지막 기간은 2025-01~2026-09의 21개월이며, 확보된 실제 관측 연도만 반영합니다.")
