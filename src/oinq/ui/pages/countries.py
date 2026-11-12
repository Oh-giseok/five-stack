"""Country analysis using the reviewed demo layout and shared project data."""
from pathlib import Path
import math

import pandas as pd
import streamlit as st

from oinq.ui.components import country_labels, render_empty_result, render_run_caption, render_export_timeline
from oinq.ui.country_charts import ALL, history_series, ranking_figure, style_figure, timeline_figure
from oinq.ui.data import ForecastRun, load_monthly_exports, sum_if_complete
from oinq.ui.clustering import (
    WINDOWS, WINDOW_LABELS, INDICATORS, cluster_source_signature, load_country_clusters,
    distribution_figure, distribution_table, country_history, history_figure,
)


@st.cache_data(show_spinner=False, max_entries=2)
def _historical(root: str, countries: tuple, products: tuple, signature: tuple):
    del signature
    return load_monthly_exports(root, countries, products)


def _country_picker(labels):
    return st.selectbox("국가", list(labels), format_func=labels.get,
                        key="gem_selected_country", persist_state="session")


def _product_picker(products):
    labels = {row["hs_code"]: f"{row['name_ko']} ({row['hs_code']})" for row in products}
    return st.selectbox("품목", [ALL, *labels], key="gem_country_product", persist_state="session",
                        format_func=lambda code: "전체 8개 품목" if code == ALL else labels[code])


def _render_timeline(monthly, country, hs):
    render_export_timeline(monthly, country, hs, key="actual_timeline", height=450,
                           mode_key="actual_plot_kind")


def render_history(countries: list[dict], products: list[dict], run: ForecastRun | None = None) -> None:
    del run
    labels = country_labels(countries)
    names = {row["iso3"]: row["name_ko"] for row in countries}
    root = Path(__file__).resolve().parents[4]
    path = root / "data/shared/history/customs_monthly_hs6.parquet"
    try:
        stat = path.stat()
        monthly = _historical(str(root), tuple(labels), tuple(p["hs_code"] for p in products),
                              (stat.st_size, stat.st_mtime_ns))
    except (OSError, ValueError) as error:
        render_empty_result("공유된 실제 수출 자료를 확인하세요.", detail=str(error))
        return
    with st.container(border=True, key="gem_country_filters_history"):
        filters = st.columns([1.3, 1.8, 1, 1, 1])
        with filters[0]:
            country = _country_picker(labels)
        with filters[1]:
            hs = _product_picker(products)
        with filters[2]:
            frequency = st.selectbox("비교 집계", ["분기", "연간"], key="actual_grain")
        with filters[3]:
            year = st.selectbox("연도", list(range(2013, 2027)), index=11, key="actual_year")
        with filters[4]:
            choices = [1, 2, 3] if year == 2026 else [1, 2, 3, 4]
            quarter = st.selectbox("분기", choices, format_func=lambda q: f"Q{q}",
                                   key=f"actual_quarter_{len(choices)}", disabled=frequency == "연간")
    period = str(year) if frequency == "연간" else f"{year}Q{quarter}"
    if year == 2026 and frequency == "연간":
        st.caption("국가 비교의 2026년 금액은 모든 국가에 동일한 1–9월 범위를 적용합니다.")
    records = []
    for code in labels:
        row = history_series(monthly, code, hs, frequency).set_index("period").loc[period].to_dict()
        records.append({**row, "code": code, "label": names[code]})
    left, right = st.columns([.9, 1.7], gap="medium")
    with left.container(border=True, key="gem_country_card_history_ranking"):
        st.subheader(f"{period} | 국가별 실제 수출액", anchor=False)
        st.plotly_chart(ranking_figure(pd.DataFrame(records), country, height=635),
                        width="stretch", theme=None, key="actual_ranking_chart",
                        config={"displayModeBar": False}, alt="전체 20개국의 선택 기간 관측 수출액 비교")
        st.caption("적색은 관측 부분합입니다. 확보 범위가 다른 금액의 순위 해석에 주의하세요.")
    with right.container(border=True, key="gem_country_card_history_timeline"):
        st.subheader(f"{names[country]} | 과거 수출 실적", anchor=False)
        _render_timeline(monthly, country, hs)
    st.caption("관세청 v003의 국가·품목·월별 관측값을 한 번만 집계합니다. Comtrade·경제지표 금액을 더하지 않습니다.")


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


def _open_products(country, hs):
    st.session_state["gem_main_navigation"] = "품목별 수요"
    st.session_state["gem_selected_country"] = country
    if hs != ALL:
        st.session_state["gem_selected_product"] = hs


def render_comparison(countries: list[dict], products: list[dict], run: ForecastRun | None) -> None:
    labels = country_labels(countries)
    names = {row["iso3"]: row["name_ko"] for row in countries}
    with st.container(border=True, key="gem_country_filters_forecast"):
        cols = st.columns([1.2, 1.8, 1])
        with cols[0]:
            country = _country_picker(labels)
        with cols[1]:
            hs = _product_picker(products)
        with cols[2]:
            period = st.selectbox("예측 기간", ["2027년 연간", "2027Q1", "2027Q2", "2027Q3", "2027Q4"],
                                  key="p02a_period")
    if run is None:
        render_empty_result("저장된 예측 결과가 없어 국가 비교를 준비 중입니다.")
        return
    render_run_caption(run.metadata, run.grid_audit, show_context=False)
    codes = [row["hs_code"] for row in products] if hs == ALL else [hs]
    summary = _country_quarter_summary(run, list(labels), codes)
    column = "연간" if period == "2027년 연간" else period[-2:]
    records = pd.DataFrame({"code": summary["국가"], "label": summary["국가"].map(names),
                            "value_usd": summary[column], "status": "forecast",
                            "valid_count": summary[column].notna().astype(int), "expected_count": 1})
    with st.container(border=True, key="gem_country_card_forecast_ranking"):
        st.subheader(f"{period} | 국가별 예상 수출액 비교", anchor=False)
        st.plotly_chart(ranking_figure(records, country, predicted=True, height=540),
                        width="stretch", theme=None, key="p02a_ranking_chart",
                        config={"displayModeBar": False}, alt="전체 20개국의 선택 기간 예상 수출액 비교")
    row = summary.loc[summary["국가"].eq(country)].iloc[0]
    series = pd.DataFrame({"period": [f"2027Q{q}" for q in range(1, 5)],
                           "value_usd": [row[f"Q{q}"] for q in range(1, 5)], "status": "forecast",
                           "valid_count": [int(pd.notna(row[f"Q{q}"])) for q in range(1, 5)], "expected_count": 1})
    total = sum_if_complete(series.value_usd, 4)
    total_label = f"연간 {total / 1e6:,.2f}백만 USD" if pd.notna(total) else "연간 미확보"
    with st.container(border=True, key="gem_country_card_forecast_quarters"):
        st.subheader(f"{names[country]} | 2027년 분기별 전망 · {total_label}", anchor=False)
        figure = timeline_figure(series, "분기", "막대", height=300)
        figure.update_traces(marker_color="#205f73")
        st.plotly_chart(figure, width="stretch", theme=None, key="p02a_quarters_chart",
                        config={"displayModeBar": False}, alt="선택 국가·품목의 2027년 Q1부터 Q4까지 예상 수출액")
    st.button("선택 국가의 품목별 수요 보기", icon=":material/arrow_forward:", key="p02a_to_products",
               on_click=_open_products, args=(country, hs))


@st.cache_data(show_spinner=False, max_entries=2)
def _read_clusters(root: str, signature: tuple):
    del signature
    return load_country_clusters(root)


def render_similar(countries: list[dict], products: list[dict]) -> None:
    del products
    labels = country_labels(countries)
    names = {row["iso3"]: row["name_ko"] for row in countries}
    with st.container(border=True, key="gem_country_filters_clusters"):
        view = st.segmented_control("보기", ["군집 분포", "기간별 변화"], default="군집 분포", required=True,
                                     key="p02b_view", label_visibility="collapsed")
        if view == "군집 분포":
            window = st.selectbox("분석기간", WINDOWS, index=3, format_func=WINDOW_LABELS.get, key="p02b_window")
        else:
            left, right = st.columns(2)
            with left:
                country = _country_picker(labels)
            with right:
                indicator_label = st.selectbox("변화 지표", list(INDICATORS), key="p02b_indicator")
    root = Path(__file__).resolve().parents[4]
    try:
        with st.spinner("국가별 경제 지표와 K-means 군집 결과를 불러오고 있습니다…"):
            clusters = _read_clusters(str(root), cluster_source_signature(root))
    except FileNotFoundError:
        render_empty_result("저장된 K-means 모델 또는 공유 자료가 없습니다.")
        return
    except ImportError:
        st.error("K-means 조회에 필요한 모델 실행 라이브러리가 설치되지 않았습니다.")
        return
    except (OSError, ValueError) as error:
        st.error(f"군집 모델·공유 데이터 확인에 실패했습니다: {error}")
        return
    results = clusters.results
    left, right = st.columns([1.2, 1.15], gap="medium")
    if view == "군집 분포":
        selected = results.loc[results.window_id.eq(window)]
        with left.container(border=True, key="gem_country_card_clusters_distribution"):
            st.subheader(f"{WINDOW_LABELS[window]} | 국가 경제 특성", anchor=False)
            st.plotly_chart(style_figure(distribution_figure(results, window), 490), width="stretch", theme=None,
                            key="p02b_distribution", alt="1인당 GDP와 인터넷 이용률에 따른 국가 군집, 점 면적은 인구")
            st.caption("X축=1인당 GDP · Y축=인터넷 이용률 · 점 면적=인구 · 색상=군집")
        with right.container(border=True, key="gem_country_card_clusters_table"):
            st.subheader("국가별 군집·지표·관측 연도", anchor=False)
            st.dataframe(distribution_table(selected), hide_index=True, width="stretch", height=510,
                         column_config={"1인당 GDP (USD/인)": st.column_config.NumberColumn(format="%.0f"),
                                        "인터넷 이용률 (%)": st.column_config.NumberColumn(format="%.1f"),
                                        "인구 (명)": st.column_config.NumberColumn(format="%.0f"),
                                        "관측 수출액 (USD·FOB)": st.column_config.NumberColumn(format="%.0f")},
                         alt="미분류를 포함한 20개국의 군집·지표·관측 연도")
        missing = int(selected.cluster_id.isna().sum())
        if missing:
            st.info(f"선택 구간의 {missing}개국은 군집 입력 결측으로 미분류입니다. 해당 국가를 표에서 확인할 수 있습니다.")
    else:
        indicator, axis_title = INDICATORS[indicator_label]
        history = country_history(results, country)
        with left.container(border=True, key="gem_country_card_clusters_history"):
            st.subheader(f"{names[country]} | {indicator_label} 변화", anchor=False)
            st.plotly_chart(style_figure(history_figure(history, indicator, axis_title), 490),
                            width="stretch", theme=None, key="p02b_history",
                            alt="다섯 분석기간별 선택 국가의 경제지표와 군집 변화")
        with right.container(border=True, key="gem_country_card_clusters_history_table"):
            st.subheader("분석구간별 관측 연도·군집", anchor=False)
            display = history[["window_id", "군집", f"{indicator}_observed_mean",
                               f"{indicator}_observed_years", "missing_features"]].copy()
            display.columns = ["분석기간", "군집", axis_title, "실제 관측 연도", "입력 결측"]
            display["분석기간"] = display["분석기간"].map(WINDOW_LABELS)
            st.dataframe(display, hide_index=True, width="stretch", height=360,
                         alt="미분류와 결측을 유지한 다섯 분석구간의 지표·군집·관측 연도")
