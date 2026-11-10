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
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"],
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"] {
            background:#eef0f3 !important; border-color:transparent !important; color:var(--gem-ink) !important;
          }
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"]:hover,
          [data-testid="stSidebar"] [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"]:hover {
            background:#e1e6eb !important; border-color:#d9dee7 !important;
          }
          [data-testid="stMain"] [role="tablist"] {
            justify-content:flex-start; gap:.35rem; border-bottom:1px solid var(--gem-line);
          }
          [data-testid="stMain"] [role="tab"] {
            background:transparent !important; border:0 !important; border-radius:10px 10px 0 0;
            padding:.7rem 1.1rem !important; margin:0; color:var(--gem-muted) !important;
          }
          [data-testid="stMain"] [role="tab"][aria-selected="true"] {
            background:#e8f0f2 !important; color:#205f73 !important; font-weight:700;
            box-shadow:inset 0 -3px 0 #205f73;
          }
          [data-testid="stMain"] [data-baseweb="tab-highlight"],
          [data-testid="stMain"] [data-testid="stTabsActiveLine"] { display:none; }
          [class*="st-key-gem_home_icon_"] {
            width:76% !important; max-width:18rem; margin-inline:auto;
            aspect-ratio:1; flex-shrink:0;
          }
          [class*="st-key-gem_home_icon_"] [data-testid="stImageContainer"] img {
            height:auto !important; aspect-ratio:1;
            object-fit:contain;
          }
          [class*="st-key-gem_home_description_"] { min-height:4rem; text-align:center; }
          [class*="st-key-gem_home_description_"] p {
            text-align:center !important; word-break:keep-all; text-wrap:balance;
          }
          [class*="st-key-gem_equal_panels_"] [data-testid="stPlotlyChart"] {
            padding:0; border:0; box-shadow:inset 0 0 0 1px var(--gem-line);
          }
          @media (min-width:641px) {
            [class*="st-key-gem_equal_panels_"] h3 { min-height:4.5rem; }
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
          [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"] {
            background:#fff !important; border:1px solid var(--gem-line) !important; color:var(--gem-ink) !important;
          }
          [class*="st-key-gem_nav_"] [data-testid="stBaseButton-secondary"]:hover,
          [class*="st-key-gem_nav_"] [data-testid="baseButton-secondary"]:hover {
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


def install_analysis_styles(page: str = "country") -> None:
    """Use the reviewed demo card layout while isolating each analysis page."""
    if page not in {"country", "products"}:
        raise ValueError("지원하지 않는 분석 페이지입니다.")
    styles = """
    <style>
      .st-key-gem_country_page { gap:1rem; }
      .st-key-gem_country_page h1 { font-size:38px !important; font-weight:750; }
      .st-key-gem_country_page h3 { font-size:21px !important; line-height:1.35; }
      .st-key-gem_country_page [role="tablist"] { gap:0; }
      .st-key-gem_country_page [role="tab"] {
        padding:.45rem 0 .65rem !important; margin-right:1.1rem;
        border-radius:0; font-size:15px; color:var(--gem-ink) !important;
      }
      .st-key-gem_country_page [role="tab"][aria-selected="true"] {
        background:transparent !important; color:var(--gem-accent) !important;
        box-shadow:inset 0 -2px 0 var(--gem-accent); font-weight:600;
      }
      .st-key-gem_country_page [role="tab"] > .react-aria-SelectionIndicator { display:none; }
      .st-key-gem_country_page [role="tabpanel"] { padding-top:.75rem; }
      [class*="st-key-gem_country_filters_"],
      [class*="st-key-gem_country_card_"] {
        background:#fff; border-color:var(--gem-line) !important;
        border-radius:10px; padding:1rem;
      }
      [class*="st-key-gem_country_filters_"] { padding:.75rem 1rem; }
      .st-key-gem_country_page [data-testid="stPlotlyChart"] {
        border:0; border-radius:0; background:transparent; padding:0; box-shadow:none;
      }
      .st-key-gem_country_page [data-testid="stCaptionContainer"] { font-size:12px; }
      .st-key-gem_country_page [data-testid="stButtonGroup"] button {
        border-color:var(--gem-line); font-size:13px;
      }
      .st-key-gem_country_page [data-testid="stButtonGroup"] button[aria-checked="true"],
      .st-key-gem_country_page [data-testid="stButtonGroup"] button[aria-pressed="true"] {
        background:#e8f0f2 !important; color:var(--gem-accent) !important; border-color:var(--gem-accent) !important;
      }
      .st-key-gem_country_page [data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) {
        background:var(--gem-accent) !important;
      }
      .st-key-gem_country_page [data-testid="stSliderThumbValue"] { color:var(--gem-accent); }
      @media (min-width:641px) { .st-key-gem_country_page { margin-top:2.5rem; } }
      @media (max-width:640px) {
        [class*="st-key-gem_country_filters_"],
        [class*="st-key-gem_country_card_"] { padding:.7rem; }
        .st-key-gem_country_page h1 { font-size:30px !important; }
      }
    </style>
    """
    if page == "products":
        styles = styles.replace("gem_country_", "gem_products_")
    st.html(styles)


def install_country_styles() -> None:
    install_analysis_styles("country")


def render_export_timeline(monthly, country, hs, *, key, run=None, height=440, mode_key=None, colors=None):
    """Shared native timeline controls for the country and product pages."""
    from oinq.ui.country_charts import export_timeline, timeline_figure

    left, right = st.columns(2)
    with left:
        frequency = st.segmented_control("집계 단위", ["분기", "연간"], default="분기", required=True,
                                         key=f"{key}_frequency", label_visibility="collapsed")
    with right:
        mode = st.segmented_control("그래프 종류", ["막대", "선"], default="막대", required=True,
                                    key=mode_key or f"{key}_mode", label_visibility="collapsed")
    series = export_timeline(monthly, country, hs, frequency, run)
    n = len(series)
    if frequency == "분기":
        window_key = f"{key}_window"
        st.session_state.setdefault(window_key, 20)
        sizes = [("전체 보기", n), ("최근 5년", 20),
                 ("＋ 확대", max(8, st.session_state[window_key] // 2)),
                 ("－ 축소", min(n, st.session_state[window_key] * 2))]
        for i, (column, (label, size)) in enumerate(zip(st.columns(4), sizes)):
            if column.button(label, key=f"{key}_view_{i}", width="stretch"):
                st.session_state[window_key] = min(n, size)
        size = min(n, st.session_state[window_key])
        maximum = n - size
        offset = (st.slider("시계열 좌우 이동", 0, maximum, value=maximum,
                            key=f"{key}_offset_{n}_{size}", label_visibility="collapsed") if maximum else 0)
        if maximum:
            percent = 100 * offset / maximum
            st.html(f"""<style>
              .st-key-{key}_offset_{n}_{size} [data-testid="stSlider"]
              [role="group"] > [data-orientation="horizontal"] > div:first-child {{
                background:linear-gradient(to right, var(--gem-accent) {percent}%,
                                             var(--gem-line) {percent}%) !important;
              }}
            </style>""")
        visible = [offset - .5, offset + size - .5]
        st.caption(f"표시 구간 {series.iloc[offset].period}–{series.iloc[offset + size - 1].period} · "
                   f"전체 {series.iloc[0].period}–{series.iloc[-1].period}")
        if size > 32:
            st.caption("각 막대·점은 한 분기입니다. 전체 보기에서는 연도 눈금을 표시하며 확대하면 Q1~Q4가 표시됩니다.")
    else:
        visible = [-.5, n - .5]
        st.caption("2026년은 1–9월 누적이며 완전한 연간 실적과 구분합니다.")
    st.plotly_chart(timeline_figure(series, frequency, mode, visible, height=height, colors=colors),
                    width="stretch", theme=None, key=f"{key}_chart",
                    config={"displayModeBar": False, "scrollZoom": False},
                    alt="선택 국가·품목의 과거 실적과 관측 부분합 및 연결된 2027년 예측 시계열")
    st.caption("빗금=1–9월 누적 · 적색 빗금=관측 부분합. 결측은 0으로 채우지 않습니다.")


def render_navigation_icon(icon: str) -> None:
    """Render matching thin-line SVGs without a font-weight dependency."""
    drawings = {
        "public": ("지구 아이콘", '<circle cx="32" cy="32" r="24"/>'
                   '<path d="M9 25.5 24.8 41.2v2.5c0 3.2 2.6 5.8 5.8 5.8v6.3'
                   'M39.2 9.1v4.5c0 3.6-2.9 6.5-6.5 6.5h-3.2v6.4h-6.4V32h13.5'
                   'c2.1 0 3.9 1.8 3.9 3.9v8h3.1c2.7 0 5 1.8 5.9 4.6"/>'),
        "health_and_beauty": ("화장품 아이콘", '<path d="M8 56V34h18v22ZM11 34V18l12-6v22M8 43h18"/>'
                              '<rect x="34" y="24" width="22" height="32" rx="4"/>'
                              '<path d="M41 24v-9h8v9M37 15V9h20v5M40 38h10M40 43h10"/>'),
        "factory": ("공장 아이콘", '<path d="M8 56V30l16-9v9l16-9v13h16v22ZM42 34l2-26h8l2 26"/>'
                    '<path d="M14 40h7v9h-7ZM27 40h7v9h-7ZM40 40h7v9h-7Z"/>'),
    }
    label, drawing = drawings[icon]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64" '
           'fill="none" stroke="#205f73" stroke-width="1.6" stroke-linecap="round" '
           f'stroke-linejoin="round">{drawing}</svg>')
    st.image(svg, width="stretch", alt=label)


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


def render_run_caption(metadata: dict, audit=None, *, show_context: bool = True) -> None:
    text = (
        f"기준분기 {metadata.get('origin_quarter', '미상')} · "
        f"모델 {metadata.get('model_id', '미상')} / {metadata.get('model_variant', '미상')} · "
        "예측 대상 2027Q1~Q4 · USD·FOB"
    )
    if show_context:
        st.caption(text)
    if show_context and metadata.get("missing_input_rows", 0):
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
