"""Streamlit entry point for GEM's eight result screens."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from oinq.common.config import load_yaml
from oinq.ui.components import install_styles
from oinq.ui.data import discover_forecast_runs, load_forecast_run
from oinq.ui.pages import countries, demand, main, performance, products, production


ROOT = Path(__file__).resolve().parent
COUNTRY_CONFIG = load_yaml(ROOT / "configs" / "countries.yaml")
PRODUCT_CONFIG = load_yaml(ROOT / "configs" / "products.yaml")
COUNTRIES = COUNTRY_CONFIG["countries"]
PRODUCTS = PRODUCT_CONFIG["products"]
COUNTRY_CODES = tuple(row["iso3"] for row in COUNTRIES)
HS_CODES = tuple(row["hs_code"] for row in PRODUCTS)


@st.cache_data(show_spinner=False)
def _read_run(folder: str, signature: tuple, country_codes: tuple, hs_codes: tuple):
    # Signature changes when any run file is replaced, while filter reruns reuse this result.
    del signature
    return load_forecast_run(folder, country_codes=country_codes, hs_codes=hs_codes)


def _signature(folder: Path) -> tuple:
    names = ("run.json", "quarterly_forecast.parquet", "annual_forecast.parquet")
    return tuple(
        (name, (folder / name).stat().st_size, (folder / name).stat().st_mtime_ns)
        for name in names
    )


def main_app() -> None:
    st.set_page_config(page_title="GEM | Global Export Management", page_icon="🌐", layout="wide")
    install_styles()

    st.sidebar.markdown("# GEM")
    st.sidebar.caption("Global Export Management")
    page_label = st.sidebar.radio(
        "메뉴",
        ["메인", "국가별 분석", "제품별 분석", "수요 예측", "생산 시나리오", "모델 성능"],
        label_visibility="collapsed",
        key="gem_main_navigation",
    )
    subpage = None
    if page_label == "국가별 분석":
        subpage = st.sidebar.radio("국가별 분석", ["수출 비교", "유사 국가"], key="gem_country_navigation")
    elif page_label == "모델 성능":
        subpage = st.sidebar.radio("모델 성능", ["검증 비교", "최종 테스트"], key="gem_performance_navigation")

    run = None
    run_ids = discover_forecast_runs(ROOT)
    if run_ids:
        default_index = run_ids.index("ui_demo") if "ui_demo" in run_ids else 0
        selected_run = st.sidebar.selectbox("예측 실행", run_ids, index=default_index, key="gem_forecast_run")
        run_dir = ROOT / ".private" / "forecasts" / selected_run
        try:
            run = _read_run(str(run_dir), _signature(run_dir), COUNTRY_CODES, HS_CODES)
        except (OSError, ValueError) as error:
            st.sidebar.error(f"예측 결과를 읽지 못했습니다: {error}")
    else:
        st.sidebar.caption("저장된 예측 실행이 없습니다.")

    if page_label == "메인":
        main.render(COUNTRIES, PRODUCTS, run)
    elif page_label == "국가별 분석" and subpage == "수출 비교":
        countries.render_comparison(COUNTRIES, PRODUCTS, run)
    elif page_label == "국가별 분석":
        countries.render_similar(COUNTRIES, PRODUCTS)
    elif page_label == "제품별 분석":
        products.render(COUNTRIES, PRODUCTS, run)
    elif page_label == "수요 예측":
        demand.render(COUNTRIES, PRODUCTS, run)
    elif page_label == "생산 시나리오":
        production.render(COUNTRIES, PRODUCTS)
    elif page_label == "모델 성능" and subpage == "검증 비교":
        performance.render_validation()
    else:
        performance.render_final_test()


if __name__ == "__main__":
    main_app()
