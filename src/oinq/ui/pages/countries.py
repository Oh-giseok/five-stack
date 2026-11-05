"""P02A country comparison and P02B similar-country views."""

import math
from pathlib import Path

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
from oinq.ui.clustering import (
    WINDOWS, WINDOW_LABELS, INDICATORS, cluster_source_signature, load_country_clusters,
    distribution_figure, distribution_table, country_history, history_figure,
)


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
            st.plotly_chart(fig, width="stretch", alt="국가별 연간 예상 수출액 비교")

    with onepage_section("p02a_table"):
        st.subheader("국가별 분기·연간 예상 수출액 (백만 USD)")
        display = summary[["국가명", "Q1", "Q2", "Q3", "Q4", "연간"]].copy()
        for column in ("Q1", "Q2", "Q3", "Q4", "연간"):
            display[column] = display[column].map(format_musd)
        display = display.rename(columns={"국가명": "국가"})
        st.dataframe(display, hide_index=True, width="stretch", height=min(650, 42 * len(display) + 45),
                     alt="국가별 분기 및 연간 예상 수출액")
        if len(hs_codes) > 1:
            st.caption("분기·연간 금액은 선택된 품목 전부의 예측값이 있을 때만 합산됩니다.")


@st.cache_data(show_spinner=False, max_entries=2)
def _read_clusters(root: str, signature: tuple):
    del signature  # Used as a cache key, alongside the project root.
    return load_country_clusters(root)


def render_similar(countries: list[dict], products: list[dict]) -> None:
    labels = country_labels(countries)
    with onepage_section("p02b_filters"):
        st.title("국가별 분석 | 유사 국가")
        st.caption("저장된 K-means 모델로 국가 특성과 기간별 군집 변화를 확인합니다.")
        view = st.radio("보기", ["군집 분포", "기간별 변화"], horizontal=True, key="p02b_view")
        if view == "기간별 변화":
            st.caption("분석기간: 2013–2015부터 2025–2026Q3까지")
            country = st.selectbox("국가", list(labels), format_func=labels.get, key="p02b_country")
            indicator_label = st.selectbox("지표", list(INDICATORS), key="p02b_indicator")
        else:
            window = st.selectbox("분석기간", WINDOWS, index=3, format_func=WINDOW_LABELS.get, key="p02b_window")
        root = Path(__file__).resolve().parents[4]
        try:
            clusters = _read_clusters(str(root), cluster_source_signature(root))
        except FileNotFoundError:
            render_empty_result("저장된 K-means 모델 또는 공유 자료가 없습니다.",
                                detail="artifacts/shared_models/kmeans/pipeline.joblib과 data/shared/v002를 확인하세요.")
            return
        except ImportError:
            st.error("K-means 조회에 필요한 scikit-learn·joblib 등 모델 실행 라이브러리가 설치되지 않았습니다.")
            return
        except (OSError, ValueError) as error:
            st.error(f"군집 모델·공유 데이터 확인에 실패했습니다: {error}")
            return
        results = clusters.results
        st.caption(f"전체 {len(results)}행 · 분류 {results.cluster_id.notna().sum()}행 · "
                   f"미분류 {results.cluster_id.isna().sum()}행 · K-means k={clusters.n_clusters}")
        st.caption("모든 기간에 같은 표준화기·군집 중심을 적용합니다. 군집 번호는 국가 특성 그룹이며 수출 유망도 순위가 아닙니다.")
        st.caption("관측자료를 함께 사용한 사후 비교입니다. 마지막 구간은 2025-01~2026-09의 21개월입니다.")

    if view == "군집 분포":
        selected = results.loc[results.window_id.eq(window)]
        label = WINDOW_LABELS[window]
        with onepage_section("p02b_distribution_chart"):
            st.subheader(f"{label} | 국가 특성·군집 분포")
            missing = int(selected.cluster_id.isna().sum())
            st.caption(f"전체 {len(selected)}개국 · 분류 {len(selected) - missing}개국 · 미분류 {missing}개국")
            st.plotly_chart(distribution_figure(results, window), width="stretch", theme=None,
                            key="p02b_distribution", alt="1인당 GDP와 인터넷 이용률에 따른 국가 군집, 점 면적은 인구")
            st.caption("X축=1인당 GDP · Y축=인터넷 이용률 · 점 면적=인구 · 색상=군집. 축 범위와 인구 크기 척도는 기간 간 동일합니다.")
            if missing:
                st.info(f"입력 결측으로 미분류된 {missing}개국은 아래 표에 유지합니다. 좌표가 없는 국가는 산점도에 표시하지 않습니다.")
        with onepage_section("p02b_distribution_table"):
            st.subheader(f"{label} | 전체 국가의 군집·경제지표·수출 맥락")
            st.dataframe(distribution_table(selected), hide_index=True, width="stretch", height=620,
                         alt="미분류를 포함한 20개국 군집과 실제 지표 관측 연도 및 관측 수출액")
            st.caption("관측 수출액은 확보된 월·품목의 합계입니다. 수출 집계 상태를 함께 확인하세요. 수출액은 군집 학습에 사용하지 않습니다.")
        return

    indicator, axis_title = INDICATORS[indicator_label]
    values = country_history(results, country)
    values["기간"] = values.window_id.astype(str).map(WINDOW_LABELS)
    value_col = f"{indicator}_observed_mean"
    with onepage_section("p02b_change_chart"):
        st.subheader(f"{labels[country]} | {axis_title} 기간별 변화")
        if values[value_col].notna().any():
            st.plotly_chart(history_figure(values, indicator, axis_title), width="stretch", theme=None,
                            key="p02b_history", alt="다섯 분석기간의 국가 경제지표 추이와 군집 변화")
        else:
            render_empty_result("선택한 지표의 관측값이 없습니다.")
        st.caption("색상은 해당 기간의 군집입니다. 선택한 지표가 있어도 다른 군집 입력이 결측이면 회색 미분류로 표시합니다. 지표 결측 구간은 선으로 연결하지 않습니다.")
    with onepage_section("p02b_change_table"):
        st.subheader("다섯 분석기간의 지표·군집·관측 연도")
        observed_years = f"{indicator}_observed_years"
        display = values[["기간", "start_month", "end_month", "군집", value_col, observed_years, "missing_features"]].rename(columns={
            "start_month": "시작월", "end_month": "종료월", value_col: axis_title,
            observed_years: "실제 관측 연도", "missing_features": "군집 입력 결측",
        })
        st.dataframe(display, hide_index=True, width="stretch", alt="결측과 미분류를 유지한 다섯 기간의 국가 지표 및 군집")
        st.caption("지표는 확보된 실제 관측 연도의 평균입니다. 결측은 0으로 대체하지 않습니다.")
