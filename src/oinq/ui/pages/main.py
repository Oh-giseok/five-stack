"""GEM landing page and analysis navigation."""

import streamlit as st

from oinq.ui.components import onepage_section


def _navigate(destination: str) -> None:
    st.session_state["gem_main_navigation"] = destination


def render(countries: list[dict], products: list[dict], run=None) -> None:
    del run
    with onepage_section("home_intro"):
        st.title("GEM | 글로벌 수출 관리")
        st.markdown("### 국가 비교 → 품목 선택 → 생산계획")
        st.caption(f"대상 국가 {len(countries)}개 · HS 품목 {len(products)}개 · 실적 2013Q1–2026Q3 · 전망 2027Q1–Q4")
        st.info("2026년은 1~9월 누적 실적입니다. 결측 구간은 0으로 대체하지 않으며, 예측은 실제 실적과 구분해 표시합니다.")
        st.subheader("분석 시작하기")
        a, b, c = st.columns(3)
        for column, title, description, destination in (
            (a, "국가별 분석", "수출 현황·전망과 유사 국가를 비교합니다.", "국가별 분석"),
            (b, "품목별 수요", "국가 안에서 8개 품목의 전망과 과거 흐름을 살펴봅니다.", "품목별 수요"),
            (c, "생산 시나리오", "기업의 목표 비중과 재고·능력 가정으로 계획을 확인합니다.", "생산 시나리오"),
        ):
            with column:
                st.markdown(f"#### {title}")
                st.write(description)
                st.button(f"{title} 열기", key=f"home_go_{destination}", use_container_width=True,
                          on_click=_navigate, args=(destination,))
