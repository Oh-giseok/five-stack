"""Streamlit entry point for GEM's analysis and model comparison pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from oinq.common.config import load_yaml
from oinq.ui.components import install_styles
from oinq.ui.data import load_forecast_run
from oinq.ui.forecasting import (
    MODEL_LABELS, VARIANT_LABELS, saved_forecast_models, matching_forecast_runs,
    create_forecast_run, saved_training_end,
)
from oinq.ui.pages import countries, demand, main, performance, products, production


ROOT = Path(__file__).resolve().parent
COUNTRY_CONFIG = load_yaml(ROOT / "configs" / "countries.yaml")
PRODUCT_CONFIG = load_yaml(ROOT / "configs" / "products.yaml")
COUNTRIES = COUNTRY_CONFIG["countries"]
PRODUCTS = PRODUCT_CONFIG["products"]
COUNTRY_CODES = tuple(row["iso3"] for row in COUNTRIES)
HS_CODES = tuple(row["hs_code"] for row in PRODUCTS)


@st.cache_data(show_spinner=False, max_entries=24)
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


@st.cache_data(show_spinner=False, max_entries=4)
def _training_end(path: str, size: int, mtime_ns: int):
    return saved_training_end(path)


def _select_main_navigation(page: str) -> None:
    st.session_state["gem_main_navigation"] = page


def _select_country_navigation(page: str) -> None:
    st.session_state["gem_country_navigation"] = page


def _select_forecast():
    models = saved_forecast_models(ROOT)
    if not models:
        st.info("공유 폴더에 저장된 수출액 예측 모델이 없습니다.")
        return None
    families = list(dict.fromkeys(key.split("/")[0] for key in models))
    default = "xgboost_l2" if "xgboost_l2" in families else families[0]
    model_id = st.selectbox("예측 모델", families, index=families.index(default),
                                    format_func=lambda value: MODEL_LABELS.get(value, value), key="gem_model")
    variants = [value for value in VARIANT_LABELS if f"{model_id}/{value}" in models]
    variant = st.selectbox("거시지표 구성", variants, format_func=VARIANT_LABELS.get,
                                   key=f"gem_variant_{model_id}")
    key = f"{model_id}/{variant}"
    path = models[key]
    if model_id in {"random_forest", "elastic_net"}:
        try:
            stat = path.stat()
            cutoff = _training_end(str(path), stat.st_size, stat.st_mtime_ns)
            st.caption(f"학습 정답 종료: {cutoff or '모델에 미기록'}")
        except (ImportError, OSError, ValueError) as error:
            st.error(f"저장 모델 정보 조회 실패: {error}")
    st.caption("2026Q3 기준 → 2027Q1~Q4 · 저장 모델 사용")
    try:
        run_ids = matching_forecast_runs(ROOT, key, path)
        if st.button("저장 모델로 예측 생성", key="gem_generate_forecast"):
            with st.spinner("저장 모델로 예측 중입니다. 재학습하지 않습니다."):
                generated = create_forecast_run(ROOT, key)
            run_ids = [generated, *run_ids]
            st.session_state[f"gem_forecast_run_{key}"] = generated
        if not run_ids:
            st.info("이 모델의 저장된 예측이 없습니다. 위 버튼으로 생성하세요.")
            return None
        selected = st.selectbox("예측 실행", run_ids, key=f"gem_forecast_run_{key}")
        folder = ROOT / ".private/forecasts" / selected
        return _read_run(str(folder), _signature(folder), COUNTRY_CODES, HS_CODES)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        st.error(f"예측 생성·조회 실패: {error}")
        return None


def main_app() -> None:
    st.set_page_config(page_title="GEM | Global Export Management", page_icon="🌐", layout="wide",
                       initial_sidebar_state="expanded")
    install_styles()

    st.sidebar.markdown("# GEM")
    st.sidebar.caption("Global Export Management")
    menu_items = [
        ("메인", ":material/home:"),
        ("국가별 분석", ":material/public:"),
        ("품목별 수요", ":material/query_stats:"),
        ("생산 시나리오", ":material/precision_manufacturing:"),
        ("모델 비교", ":material/leaderboard:"),
    ]
    if "gem_main_navigation" not in st.session_state:
        st.session_state["gem_main_navigation"] = "메인"
    page_label = st.session_state["gem_main_navigation"]
    for index, (item, icon) in enumerate(menu_items):
        st.sidebar.button(
            item, icon=icon, key=f"gem_nav_{index}", width="stretch",
            type="primary" if page_label == item else "secondary",
            on_click=_select_main_navigation, args=(item,),
        )
    subpage = None
    if page_label == "국가별 분석":
        st.title("국가별 분석")
        st.subheader(st.session_state.get("gem_country_navigation", "수출 현황"))
        country_items = ["수출 현황", "수출 전망", "유사 국가"]
        if "gem_country_navigation" not in st.session_state:
            st.session_state["gem_country_navigation"] = country_items[0]
        with st.container(key="gem_country_tabs"):
            country_cols = st.columns(len(country_items), gap="small")
            for index, (column, item) in enumerate(zip(country_cols, country_items)):
                with column:
                    st.button(
                        item, key=f"gem_country_tab_{index}", use_container_width=True,
                        type="primary" if st.session_state["gem_country_navigation"] == item else "secondary",
                        on_click=_select_country_navigation, args=(item,),
                    )
        subpage = st.session_state["gem_country_navigation"]
    uses_forecasts = page_label in {"품목별 수요", "생산 시나리오"} or (
        page_label == "국가별 분석" and subpage == "수출 전망"
    )
    run = None
    if uses_forecasts:
        with st.sidebar.expander("공통 예측 설정", expanded=False):
            run = _select_forecast()

    if page_label == "메인":
        main.render(COUNTRIES, PRODUCTS, run)
    elif page_label == "국가별 분석" and subpage == "수출 현황":
        countries.render_history(COUNTRIES, PRODUCTS, run)
    elif page_label == "국가별 분석" and subpage == "수출 전망":
        countries.render_comparison(COUNTRIES, PRODUCTS, run)
    elif page_label == "국가별 분석":
        countries.render_similar(COUNTRIES, PRODUCTS)
    elif page_label == "품목별 수요":
        products.render(COUNTRIES, PRODUCTS, run)
    elif page_label == "생산 시나리오":
        production.render(COUNTRIES, PRODUCTS, run)
    elif page_label == "모델 비교":
        performance.render(ROOT)
    else:
        production.render(COUNTRIES, PRODUCTS, run)


if __name__ == "__main__":
    main_app()
