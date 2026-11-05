"""Shared Streamlit layout, selector, and display helpers."""

from __future__ import annotations

from contextlib import contextmanager
from math import isfinite

import streamlit as st


def install_styles() -> None:
    """Apply GEM styling and one-section-per-scroll behavior."""
    st.markdown(
        """
        <style>
          :root { --gem-ink:#192230; --gem-muted:#667085; --gem-line:#d9dee7; --gem-accent:#205f73; }
          [data-testid="stAppViewContainer"] { background:#f6f7f9; }
          [data-testid="stSidebar"] { background:#eef0f3; border-right:1px solid #d9dee7; }
          [data-testid="stMain"] { scroll-snap-type:y mandatory; scroll-behavior:smooth; }
          [data-testid="stMainBlockContainer"] { padding-top:1.3rem; padding-bottom:0; }
          [class*="st-key-gem_section_"] {
            min-height:calc(100vh - 4.5rem); box-sizing:border-box;
            scroll-snap-align:start; scroll-snap-stop:always;
            display:flex; flex-direction:column; justify-content:center;
            padding:1rem 0 2rem;
          }
          [class*="st-key-gem_section_"] [data-testid="stVerticalBlock"] { gap:.8rem; }
          h1,h2,h3 { color:var(--gem-ink); letter-spacing:-.025em; }
          [data-testid="stCaptionContainer"] { color:var(--gem-muted); }
          div[data-testid="stPlotlyChart"] { border:1px solid var(--gem-line); border-radius:10px; background:#fff; padding:.2rem; }
          [data-testid="stDataFrame"] { border:1px solid var(--gem-line); border-radius:10px; background:#fff; }
          div[data-testid="stForm"] { border:1px solid var(--gem-line); border-radius:10px; background:#fff; padding:1rem; }
          @media (prefers-reduced-motion: reduce) { [data-testid="stMain"] { scroll-behavior:auto; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def onepage_section(section_id: str):
    """A full viewport section that the user reaches with one scroll step."""
    with st.container(key=f"gem_section_{section_id}"):
        yield


def country_labels(countries: list[dict]) -> dict[str, str]:
    return {row["iso3"]: f"{row['name_ko']} ({row['iso3']})" for row in countries}


def product_labels(products: list[dict]) -> dict[str, str]:
    return {row["hs_code"]: f"{row['name_ko']} ({row['hs_code']})" for row in products}


def format_musd(value: object, digits: int = 2) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "미확보"
    if not isfinite(amount):
        return "미확보"
    return f"{amount / 1_000_000:,.{digits}f}"


def format_percent(value: object, digits: int = 1) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "계산 불가"
    if not isfinite(amount):
        return "계산 불가"
    return f"{amount:,.{digits}f}%"


def render_run_caption(metadata: dict, audit=None) -> None:
    text = (
        f"기준분기 {metadata.get('origin_quarter', '미상')} · "
        f"모델 {metadata.get('model_id', '미상')} / {metadata.get('model_variant', '미상')} · "
        "예측 대상 2027Q1~Q4 · USD·FOB"
    )
    st.caption(text)
    if metadata.get("missing_input_rows", 0):
        st.info(
            f"입력 변수 결측이 포함된 예측 {metadata['missing_input_rows']:,}행이 있습니다. "
            "각 행의 결측 수를 확인하세요. 과거 공개시점 재현은 검증되지 않았습니다."
        )
    if audit and audit.missing_keys:
        st.warning(
            f"국가·품목·분기 예상 조합 {audit.expected_rows:,}개 중 "
            f"{len(audit.missing_keys):,}개 예측행이 없습니다. 미확보로 표시합니다."
        )


def render_empty_result(message: str, *, detail: str | None = None) -> None:
    st.info(message)
    if detail:
        st.caption(detail)


def dataframe_options() -> dict:
    return {"hide_index": True, "use_container_width": True}
