"""P06A/P06B: evaluation result screens with honest empty states."""

import streamlit as st

from oinq.ui.components import onepage_section, render_empty_result


def render_validation() -> None:
    with onepage_section("p06a_filters"):
        st.title("모델 성능 | 검증 비교")
        st.caption("동일한 v002 검증 행에서 모델·거시 구성별 WAPE를 비교합니다.")
        st.selectbox("검증 기간", ["2022Q1–2025Q4"], key="p06a_period")
        st.selectbox("예측 거리", ["2–5분기 앞"], key="p06a_horizon")
        st.selectbox("평가 범위", ["전체"], key="p06a_scope")
        st.info("v002 검증 점수 묶음이 아직 연결되지 않아 점수와 모델 목록을 표시하지 않습니다.")
    with onepage_section("p06a_chart"):
        st.subheader("모델별 WAPE 비교")
        render_empty_result("검증 결과가 전달되면 실제 제공된 모델·구성별 막대를 표시합니다.")
    with onepage_section("p06a_table"):
        st.subheader("동일 조건의 검증 지표")
        render_empty_result("MAE·RMSE·WAPE·sMAPE·평가 건수를 결과 파일에서 불러옵니다.")


def render_final_test() -> None:
    with onepage_section("p06b_filters"):
        st.title("모델 성능 | 최종 테스트")
        st.caption("고정된 2025Q4 기준의 2026Q1–Q3 평가 결과를 확인합니다.")
        st.selectbox("테스트 기간", ["2026Q1–Q3"], key="p06b_period")
        st.selectbox("예측 기준", ["2025Q4"], key="p06b_origin")
        st.selectbox("평가 범위", ["전체"], key="p06b_scope")
        st.info("확정 최종 테스트 평가 자료가 아직 연결되지 않아 점수를 표시하지 않습니다.")
    with onepage_section("p06b_chart"):
        st.subheader("예측 거리별 WAPE")
        render_empty_result("최종 테스트 결과가 전달되면 실제 거리별 WAPE를 표시합니다.")
    with onepage_section("p06b_table"):
        st.subheader("평가 범위별 지표·평가 건수")
        render_empty_result("평가 결과 파일의 전체·국가·제품 지표를 불러옵니다.")
