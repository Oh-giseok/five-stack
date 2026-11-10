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
          header, .stAppHeader, [data-testid="stHeader"] { display:flex !important; background:transparent !important; box-shadow:none !important; }
          [data-testid="stAppDeployButton"] { display:none !important; }
          [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
          [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] {
            display:flex !important; visibility:visible !important; opacity:1 !important;
          }
          @media (max-width: 768px) {
            [data-testid="stSidebarCollapseButton"],
            [data-testid="stExpandSidebarButton"],
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="collapsedControl"] {
              display:flex !important; visibility:visible !important; opacity:1 !important;
              position:relative !important; width:2.75rem !important; height:2.75rem !important;
              min-width:2.75rem !important; min-height:2.75rem !important;
              margin:0 !important; padding:0 !important; background-color:transparent !important;
            }
            [data-testid="stSidebarCollapseButton"] button,
            [data-testid="stExpandSidebarButton"] button,
            [data-testid="stSidebarCollapsedControl"] button,
            [data-testid="collapsedControl"] button {
              position:relative !important; display:flex !important; align-items:center !important;
              justify-content:center !important; box-sizing:border-box !important;
              width:100% !important; height:100% !important; min-width:2.75rem !important; min-height:2.75rem !important;
              margin:0 !important; padding:0 !important; color:transparent !important; font-size:0 !important;
              background-color:transparent !important; background-image:none !important;
              border:0 !important; box-shadow:none !important;
            }
            [data-testid="stSidebarCollapseButton"] svg,
            [data-testid="stExpandSidebarButton"] svg,
            [data-testid="stSidebarCollapsedControl"] svg,
            [data-testid="collapsedControl"] svg,
            [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
            [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] { opacity:0 !important; }
            [data-testid="stExpandSidebarButton"] button::after,
            [data-testid="stSidebarCollapsedControl"] button::after,
            [data-testid="collapsedControl"] button::after,
            [data-testid="stExpandSidebarButton"]::after,
            [data-testid="stSidebarCollapsedControl"]::after,
            [data-testid="collapsedControl"]::after {
              content:"" !important; position:absolute !important; z-index:5 !important;
              left:50% !important; top:50% !important; transform:translate(-50%,-50%);
              display:block !important; width:1.25rem !important; height:.875rem !important;
              pointer-events:none !important;
              background:linear-gradient(to bottom, #192230 0 2px, transparent 2px 6px, #192230 6px 8px, transparent 8px 12px, #192230 12px 14px) !important;
            }
            [data-testid="stSidebarCollapseButton"]::before,
            [data-testid="stSidebarCollapseButton"] button::before,
            [data-testid="stSidebarCollapseButton"]::after,
            [data-testid="stSidebarCollapseButton"] button::after {
              content:"" !important; position:absolute !important; z-index:5 !important;
              left:50% !important; top:50% !important; display:block !important;
              width:1.2rem !important; height:2px !important; border:0 !important;
              border-radius:2px !important; background:#192230 !important; pointer-events:none !important;
            }
            [data-testid="stSidebarHeader"] { position:relative !important; }
            [data-testid="stSidebarCollapseButton"] {
              position:absolute !important; right:.5rem !important; left:auto !important;
              top:50% !important; transform:translateY(-50%) !important; margin:0 !important; z-index:6 !important;
            }
            [data-testid="stSidebarCollapseButton"]::before,
            [data-testid="stSidebarCollapseButton"] button::before {
              transform:translate(-50%,-50%) rotate(45deg) !important;
            }
            [data-testid="stSidebarCollapseButton"]::after,
            [data-testid="stSidebarCollapseButton"] button::after {
              transform:translate(-50%,-50%) rotate(-45deg) !important;
            }
          }
          [data-testid="stMain"] { scroll-behavior:smooth; }
          [data-testid="stMainBlockContainer"] { padding-top:1rem; padding-bottom:1rem; }
          .st-emotion-cache-8ezv7j { margin-left:.5rem !important; }
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] button {
            width:100% !important; justify-content:flex-start !important; text-align:left !important;
          }
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] button > div {
            width:100%; display:flex; align-items:center; justify-content:flex-start !important; gap:.55rem;
          }
          [class*="st-key-gem_country_tab_"] button { width:100%; justify-content:center; }
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"],
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"] {
            background:#eef0f3 !important; border-color:transparent !important; color:var(--gem-ink) !important;
          }
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"]:hover,
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"]:hover {
            background:#e1e6eb !important; border-color:#d9dee7 !important;
          }
          @media (max-width: 768px) {
            .st-emotion-cache-1mr5ib5 { min-width:calc(100% - 10px) !important; }
            .st-key-gem_country_tabs [data-testid="stHorizontalBlock"] {
              display:grid !important; grid-template-columns:repeat(3, minmax(0, 1fr)) !important;
              width:100% !important; gap:.25rem !important; column-gap:.25rem !important;
            }
            .st-key-gem_country_tabs [data-testid="column"] {
              display:block !important; flex:none !important; width:100% !important; min-width:0 !important;
            }
            .st-key-gem_country_tabs [data-testid="column"] [class*="st-key-gem_country_tab_"] button {
              box-sizing:border-box !important; width:100% !important; min-width:0 !important;
              padding-left:.1rem !important; padding-right:.1rem !important;
              white-space:nowrap !important; font-size:.76rem !important; text-align:center !important;
            }
            .st-key-gem_country_tabs [data-testid="column"] [data-testid="stButton"],
            .st-key-gem_country_tabs [data-testid="column"] [data-testid="stButton"] > div,
            .st-key-gem_country_tabs [data-testid="column"] [class*="st-key-gem_country_tab_"] {
              box-sizing:border-box !important; display:block !important; width:100% !important; min-width:0 !important;
            }
            .st-key-gem_country_tabs [data-testid="column"] [data-testid="stButton"] button,
            .st-key-gem_country_tabs [data-testid="column"] [class*="st-key-gem_country_tab_"] button {
              box-sizing:border-box !important; display:flex !important; align-items:center !important;
              justify-content:center !important; width:100% !important; min-width:100% !important;
              max-width:100% !important; height:2.5rem !important; margin:0 !important;
            }
          }
          [data-testid="stBaseButton-primary"], [data-testid="baseButton-primary"], button[kind="primary"] {
            background:#205f73 !important; border:1px solid #205f73 !important;
            color:#fff !important; box-shadow:0 1px 2px rgba(25,34,48,.12) !important;
          }
          [data-testid="stBaseButton-primary"]:hover, [data-testid="baseButton-primary"]:hover,
          button[kind="primary"]:hover { background:#174a5a !important; border-color:#174a5a !important; }
          [data-testid="stForm"] [data-testid="stBaseButton-primary"],
          [data-testid="stForm"] [data-testid="baseButton-primary"],
          [data-testid="stForm"] button[kind="primary"] {
            background:#B45309 !important; border-color:#B45309 !important; color:#fff !important;
          }
          [data-testid="stForm"] [data-testid="stBaseButton-primary"]:hover,
          [data-testid="stForm"] [data-testid="baseButton-primary"]:hover,
          [data-testid="stForm"] button[kind="primary"]:hover {
            background:#92400E !important; border-color:#92400E !important;
          }
          .st-key-p05_scenario_calculate button,
          .st-key-p05_scenario_calculate [data-testid="stBaseButton-primaryFormSubmit"] {
            background:#B45309 !important; background-color:#B45309 !important;
            border-color:#B45309 !important; color:#fff !important;
          }
          .st-key-p05_scenario_calculate button:hover,
          .st-key-p05_scenario_calculate [data-testid="stBaseButton-primaryFormSubmit"]:hover {
            background:#92400E !important; background-color:#92400E !important; border-color:#92400E !important;
          }
          [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"],
          [class*="st-key-gem_country_tab_"] [data-testid="stBaseButton-secondary"],
          [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"],
          [class*="st-key-gem_country_tab_"] [data-testid="baseButton-secondary"] {
            background:#fff !important; border:1px solid var(--gem-line) !important; color:var(--gem-ink) !important;
          }
          [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"]:hover,
          [class*="st-key-gem_country_tab_"] [data-testid="stBaseButton-secondary"]:hover,
          [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"]:hover,
          [class*="st-key-gem_country_tab_"] [data-testid="baseButton-secondary"]:hover {
            background:#e8f0f2 !important; border-color:#9bb7bf !important;
          }
          [class*="st-key-gem_section_"] {
            box-sizing:border-box; padding:.25rem 0 .75rem;
          }
          [class*="st-key-gem_section_"] [data-testid="stVerticalBlock"] { gap:.8rem; }
          h1,h2,h3 { color:var(--gem-ink); letter-spacing:-.025em; }
          h3 { font-size:24px !important; }
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


def install_comparison_styles() -> None:
    """Apply the approved palette only to the comparison page."""
    st.html("""
    <style>
      .st-key-gem_comparison_page {
        background:#fff; color:#182c49; border:1px solid #d7e0eb;
        border-radius:12px; padding:1.2rem 1.35rem;
      }
      .st-key-gem_comparison_page h1,
      .st-key-gem_comparison_page h2,
      .st-key-gem_comparison_page h3 { color:#17365c; }
      .st-key-gem_comparison_controls {
        background:#f3f6fa; border-radius:10px; padding:.8rem 1rem;
        border:1px solid #d7e0eb;
      }
      .st-key-gem_comparison_page [data-testid="stCaptionContainer"] { color:#546780; }
      .st-key-gem_comparison_page [data-testid="stPlotlyChart"],
      .st-key-gem_comparison_page [data-testid="stDataFrame"] { border-color:#d7e0eb; }
      .st-key-gem_comparison_order button[aria-pressed="true"],
      .st-key-gem_comparison_order button[aria-checked="true"] {
        background:#17365c !important; border-color:#17365c !important; color:#fff !important;
      }
      @media (max-width:768px) {
        .st-key-gem_comparison_page { padding:.7rem; }
        .st-key-gem_comparison_controls { padding:.6rem; }
      }
    </style>
    """)


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
