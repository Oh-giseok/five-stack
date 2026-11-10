"""Metric and sort controls, horizontal bars and a full model performance table."""
from pathlib import Path
import logging

import plotly.graph_objects as go
import streamlit as st

from oinq.ui.components import install_comparison_styles
from oinq.ui.model_comparison import METRICS, comparison_source_signature, load_model_comparison

METRIC_KEY = "gem_comparison_metric"
ORDER_KEY = "gem_comparison_order"


@st.cache_data(show_spinner=False, max_entries=2, ttl=120)
def _comparison_data(root: str, signature: tuple):
    del signature
    return load_model_comparison(root)


def _reset_order():
    ascending = METRICS[st.session_state[METRIC_KEY]][2]
    st.session_state[ORDER_KEY] = "오름차순" if ascending else "내림차순"


def comparison_figure(selected, metric):
    available = selected.loc[selected.comparison_value.notna()].copy()
    unit = METRICS[metric][1]
    labels = [f"{row.model_name} · {row.experiment} · {'포함' if row.variant == 'with_macro' else '제외'}"
              for row in available.itertuples()]
    positions = [str(index) for index in range(len(available))]
    number_format = ".4f" if metric == "R²" else ",.2f"
    figure = go.Figure(go.Bar(
        x=available.comparison_value.tolist(), y=positions, orientation="h",
        marker_color="#009b8e", text=[format(value, number_format) for value in available.comparison_value],
        textposition="outside", cliponaxis=False, customdata=labels,
        hovertemplate=f"%{{customdata}}<br>{metric}: %{{x:{number_format}}} {unit}<extra></extra>",
    ))
    low, high = min(0., available.comparison_value.min()), max(0., available.comparison_value.max())
    span = high - low or 1.
    figure.update_layout(
        height=max(460, len(available) * 25 + 100), showlegend=False, bargap=.32,
        margin=dict(l=10, r=70, t=15, b=55), paper_bgcolor="#fff", plot_bgcolor="#fff",
        font=dict(color="#182c49", size=12), hoverlabel=dict(bgcolor="#fff"),
        xaxis=dict(title=f"{metric} ({unit})" if metric != "R²" else "테스트 R²",
                   range=[low - (span * .12 if low < 0 else 0), high + span * .18],
                   gridcolor="#e5ebf2", zeroline=True, zerolinecolor="#9bacc2",
                   tickformat=".2f" if metric == "R²" else ",.2f" if metric in {"MAE", "RMSE"} else ",.0f"),
        yaxis=dict(title=None, tickmode="array", tickvals=positions, ticktext=labels,
                   categoryorder="array", categoryarray=positions, autorange="reversed", automargin=True),
    )
    return figure


def render(root: str | Path) -> None:
    root = Path(root)
    install_comparison_styles()
    if METRIC_KEY not in st.session_state:
        st.session_state[METRIC_KEY] = "검증 WAPE"
    if ORDER_KEY not in st.session_state:
        _reset_order()

    with st.container(key="gem_comparison_page"):
        st.title("모델 비교")
        st.caption("지표와 정렬을 선택해 모델별 결과를 비교합니다.")
        with st.container(key="gem_comparison_controls"):
            metric_column, order_column = st.columns([2, 1], gap="large", vertical_alignment="bottom")
            with metric_column:
                metric = st.selectbox("세부지표", list(METRICS), key=METRIC_KEY, on_change=_reset_order)
            with order_column:
                order = st.segmented_control("정렬", ["오름차순", "내림차순"], required=True,
                                             key=ORDER_KEY, width="stretch")
        try:
            signature = comparison_source_signature(root)
            with st.spinner("저장된 비교 자료를 확인하고 있습니다."):
                data = _comparison_data(str(root), signature)
        except (OSError, ValueError, KeyError, TypeError):
            logging.getLogger(__name__).exception("Cannot load model comparison data")
            st.error("모델 비교 자료를 읽지 못했습니다. 연결된 평가·예측 자료의 상태를 확인해 주세요.")
            return

        ascending = order == "오름차순"
        selected = data.select_metric(metric, ascending=ascending)
        table = data.performance_table(metric, ascending=ascending)
        evaluated = int(data.models.validation_wape.notna().sum())
        operating = int(data.models.status.eq("운영용·비교 제외").sum())
        pending = len(data.models) - evaluated - operating
        with st.container(key="gem_comparison_chart"):
            title, direction = st.columns([3, 1], vertical_alignment="bottom")
            with title:
                st.subheader(f"{metric} 비교")
            with direction:
                st.caption(order)
            available = int(selected.comparison_value.notna().sum())
            if available:
                st.plotly_chart(comparison_figure(selected, metric), width="stretch", theme=None,
                                key="gem_comparison_bars", config={"displayModeBar": False, "scrollZoom": False},
                                alt=f"{metric} 기준 {order}으로 정렬한 {available}개 모델 구성의 가로 막대그래프")
            else:
                st.info("선택한 지표의 값이 있는 모델 구성이 없습니다.")
            if available < len(selected):
                st.caption(f"지표가 없는 {len(selected) - available}개 구성은 표에 —로 표시합니다.")

        with st.container(key="gem_comparison_table"):
            title, download = st.columns([3, 1], vertical_alignment="bottom")
            with title:
                st.subheader("전체 성능표")
            with download:
                st.download_button("CSV 다운로드", table.to_csv(index=False, na_rep="—").encode("utf-8-sig"),
                                   file_name="gem_model_comparison.csv", mime="text/csv", icon=":material/download:",
                                   on_click="ignore", key="gem_comparison_download")
            column_config = {
                "테스트 R² ↑": st.column_config.NumberColumn(format="%.4f"),
                "검증 WAPE (%) ↓": st.column_config.NumberColumn(format="%.2f"),
                "테스트 WAPE (%) ↓": st.column_config.NumberColumn(format="%.2f"),
                "테스트 MAE (백만 USD) ↓": st.column_config.NumberColumn(format="%.3f"),
                "테스트 RMSE (백만 USD) ↓": st.column_config.NumberColumn(format="%.3f"),
            }
            focus = {"검증 WAPE": "검증 WAPE (%) ↓", "테스트 WAPE": "테스트 WAPE (%) ↓", "R²": "테스트 R² ↑",
                     "MAE": "테스트 MAE (백만 USD) ↓", "RMSE": "테스트 RMSE (백만 USD) ↓"}.get(metric)
            display = table.style.set_properties(subset=[focus], **{"background-color": "#edf3fa"}) if focus else table
            st.dataframe(display, hide_index=True, width="stretch", height=420, row_height=34,
                         column_config=column_config, key="gem_comparison_scores", alt="전체 모델 구성의 성능표")
            st.caption(f"전체 {len(table)}개 구성 · 평가 완료 {evaluated}개 · 운영용 {operating}개 · 미평가 {pending}개")
