"""Country chart presentation adapted from the approved private demo.

Consumes released observations and saved predictions; no fitting or collection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAL = "#0F766E"
FORECAST_COLOR = "#7C3AED"
NAVY = "#192230"
MUTED = "#667085"
RED = "#bc625b"
GRID = "#d9dee7"
ALL = "ALL"
KINDS = {"actual": "실적", "partial": "관측 부분합", "ytd": "1–9월 누적", "forecast": "2027년 예측"}


def history_series(frame: pd.DataFrame, country: str, hs: str, frequency="분기") -> pd.DataFrame:
    """Aggregate unique monthly observations once; expose coverage, never fill nulls."""
    selected = frame
    if country != ALL:
        selected = selected.loc[selected.country_code.eq(country)]
    if hs != ALL:
        selected = selected.loc[selected.hs_code.eq(hs)]
    pairs = selected[["country_code", "hs_code"]].drop_duplicates().shape[0]
    column = "year" if frequency == "연간" else "quarter"
    grouped = selected.groupby(column, sort=True).export_value_usd.agg(
        value_usd=lambda x: x.sum(min_count=1), valid_count="count",
    ).reset_index().rename(columns={column: "period"})
    grouped["period"] = grouped.period.astype(str)
    grouped["expected_count"] = grouped.period.map(
        lambda p: pairs * (9 if p == "2026" else 12) if frequency == "연간" else pairs * 3
    )
    grouped["status"] = np.where(grouped.valid_count.eq(0), "missing",
        np.where(grouped.valid_count.lt(grouped.expected_count), "partial", "actual"))
    if frequency == "연간":
        grouped.loc[grouped.period.eq("2026") & grouped.status.eq("actual"), "status"] = "ytd"
    grouped["complete_value_usd"] = grouped.value_usd.where(grouped.valid_count.eq(grouped.expected_count))
    return grouped

def style_figure(fig: go.Figure, height=420) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Malgun Gothic, Apple SD Gothic Neo, sans-serif", "color": NAVY, "size": 12},
        margin={"l": 12, "r": 28, "t": 20, "b": 42},
        legend={"orientation": "h", "y": 1.14, "x": 0},
        hoverlabel={"bgcolor": "white", "font_size": 12},
    )
    fig.update_xaxes(showgrid=False, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, rangemode="tozero")
    return fig

def forecast_series(run, country: str, hs: str, frequency="분기") -> pd.DataFrame:
    """Retain the four expected quarters for one country/product, including gaps."""
    periods = [f"2027Q{q}" for q in range(1, 5)]
    rows = run.quarterly.loc[run.quarterly.country_code.astype(str).eq(country)
                             & run.quarterly.hs_code.astype(str).eq(hs)]
    if rows.target_quarter.duplicated().any():
        raise ValueError("국가·품목별 예측 분기에 중복이 있습니다.")
    values = rows.set_index("target_quarter").predicted_export_value_usd.reindex(periods)
    result = pd.DataFrame({"period": periods, "value_usd": values.to_numpy(dtype=float, na_value=np.nan),
                           "valid_count": values.notna().astype(int).to_numpy(), "expected_count": 1})
    if frequency == "연간":
        annual = run.annual.loc[run.annual.country_code.astype(str).eq(country)
                                & run.annual.hs_code.astype(str).eq(hs)]
        count = int(values.notna().sum())
        complete = count == 4 and len(annual) == 1 and annual.iloc[0].annual_status == "complete"
        amount = annual.iloc[0].predicted_annual_export_value_usd if complete else np.nan
        result = pd.DataFrame([{"period": "2027", "value_usd": amount,
                                 "valid_count": count, "expected_count": 4}])
    result["status"] = "forecast"
    result["complete_value_usd"] = result.value_usd
    return result


def export_timeline(monthly, country, hs, frequency="분기", run=None):
    """Join observed history and forecasts on a complete calendar; never interpolate."""
    history = history_series(monthly, country, hs, frequency)
    if run is None:
        return history.reset_index(drop=True)
    predictions = forecast_series(run, country, hs, frequency)
    periods = ([str(y) for y in range(2013, 2028)] if frequency == "연간"
               else [f"{y}Q{q}" for y in range(2013, 2028) for q in range(1, 5)])
    result = pd.concat([history, predictions], ignore_index=True).set_index("period").reindex(periods)
    result.index.name = "period"
    result["status"] = result.status.fillna("missing")
    result[["valid_count", "expected_count"]] = result[["valid_count", "expected_count"]].fillna(0)
    return result.reset_index()


def timeline_figure(series, frequency, mode, visible=None, height=440, *, colors=None):
    fig = go.Figure()
    n = len(series)
    palette = {"actual": TEAL, "ytd": TEAL, "partial": RED, "forecast": FORECAST_COLOR}
    palette.update(colors or {})
    for kind, color in palette.items():
        mask = series.status.eq(kind)
        if not mask.any():
            continue
        y = series.value_usd.where(mask).astype(float) / 1_000_000
        detail = [f"{p}<br>{KINDS.get(s, '미확보')}<br>확보 월·품목 값 {int(v):,}/{int(e):,}"
                  for p, s, v, e in series[["period", "status", "valid_count", "expected_count"]].itertuples(index=False, name=None)]
        common = dict(x=list(range(n)), y=y.tolist(), name=KINDS[kind], customdata=detail,
                      hovertemplate="%{customdata}<br>%{y:,.3f} 백만 USD<extra></extra>")
        if mode == "막대":
            fig.add_trace(go.Bar(**common, marker={"color": color, "line_width": 0,
                "pattern_shape": "/" if kind in {"partial", "ytd"} else ""}))
        else:
            fig.add_trace(go.Scatter(**common, mode="lines+markers", connectgaps=False,
                line={"color": color, "width": 2.5, "dash": "dash" if kind == "forecast" else "solid"},
                marker={"color": color, "size": 6}))
    visible_count = (visible[1]-visible[0]) if visible is not None else n
    if frequency == "연간":
        tickvals = list(range(n))
        labels = series.period.tolist()
    elif visible_count > 32:
        # Keep each bar quarterly, but use annual axis ticks in a full-history overview.
        # Hover and zoom retain the individual quarter; do not print 60 overlapping labels.
        tickvals = [i for i, p in enumerate(series.period) if p.endswith("Q1")]
        labels = [series.iloc[i].period[:4] for i in tickvals]
    else:
        tickvals = list(range(n))
        labels = [f"{p[-2:]}<br>{p[:4]}" if p.endswith("Q2") or i == 0 else p[-2:]
                  for i, p in enumerate(series.period)]
    style_figure(fig, height)
    fig.update_layout(barmode="overlay", bargap=.23, hovermode="closest")
    fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=labels, tickangle=0,
                     tickfont_size=10 if visible_count > 24 else 12, title=None,
                     range=visible or [-.5, n-.5], fixedrange=True)
    fig.update_yaxes(title_text="수출액 (백만 USD · FOB)", fixedrange=True)
    if "2026Q4" in series.period.values:
        pos = series.index[series.period.eq("2026Q4")][0]
        fig.add_vrect(x0=pos-.5, x1=pos+.5, fillcolor="#e9edf2", opacity=.6, line_width=0,
                      annotation_text="미확보", annotation_position="top")
    if "2027Q1" in series.period.values:
        pos = series.index[series.period.eq("2027Q1")][0]
        fig.add_vline(x=pos-.5, line_dash="dot", line_color=MUTED)
    return fig

def ranking_figure(records: pd.DataFrame, selected: str, predicted=False, height=520,
                   *, main_color=None, secondary_color=None):
    """Keep missing countries/products on the axis, rather than silently dropping them."""
    rows = records.sort_values("value_usd", na_position="last", ascending=False).reset_index(drop=True)
    main = main_color or (FORECAST_COLOR if predicted else TEAL)
    secondary = secondary_color or ("#ddd0f5" if predicted else "#c9dbd8")
    colors = [main if key == selected else secondary for key in rows.code]
    partial = rows.status.eq("partial")
    colors = [RED if p else c for p, c in zip(partial, colors)]
    detail = [f"{label}<br>{'관측 부분합' if status == 'partial' else '예측' if predicted else '실적'}<br>확보 {int(v)}/{int(e)}"
              for label, status, v, e in rows[["label", "status", "valid_count", "expected_count"]].itertuples(index=False, name=None)]
    fig = go.Figure(go.Bar(
        x=(rows.value_usd.astype(float)/1_000_000).tolist(), y=rows.label.tolist(), orientation="h",
        marker={"color": colors, "pattern_shape": ["/" if p else "" for p in partial]},
        text=["" if pd.isna(v) else f"{float(v)/1_000_000:,.1f}" for v in rows.value_usd],
        textposition="outside", cliponaxis=False, customdata=detail,
        hovertemplate="%{customdata}<br>%{x:,.3f} 백만 USD<extra></extra>", showlegend=False,
    ))
    missing = rows.loc[rows.value_usd.isna()]
    if len(missing):
        fig.add_trace(go.Scatter(x=[0]*len(missing), y=missing.label.tolist(), mode="markers+text",
            marker={"color": MUTED, "symbol": "x"}, text=["미확보"]*len(missing),
            textposition="middle right", showlegend=False, hoverinfo="text"))
    style_figure(fig, height)
    fig.update_layout(margin={"l": 10, "r": 72, "t": 5, "b": 35}, bargap=.3)
    fig.update_xaxes(title_text="수출액 (백만 USD · FOB)", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=rows.label.tolist(),
                     showgrid=False, title=None)
    return fig
