"""Read saved country clusters and prepare comparable, inference-only charts."""

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


WINDOWS = ["2013_2015", "2016_2018", "2019_2021", "2022_2024", "2025_2026Q3"]
WINDOW_LABELS = {key: key.replace("_", "–") for key in WINDOWS}
CLUSTER_COLORS = {"군집 1": "#205f73", "군집 2": "#d48a28", "군집 3": "#7b61a8", "미분류": "#8a939f"}
INDICATORS = {
    "1인당 GDP": ("gdp_per_capita_usd", "1인당 GDP (USD/인)"),
    "인구": ("population", "인구 (명)"),
    "인터넷 이용률": ("internet_users_pct", "인터넷 이용률 (%)"),
}


@dataclass(frozen=True)
class ClusterView:
    results: pd.DataFrame
    model_version: str
    n_clusters: int


def cluster_source_signature(root: str | Path) -> tuple:
    """Invalidate the UI cache when a model, config or shared source changes."""
    root = Path(root)
    manifest = root / "data/shared/v002/unsupervised/manifest.json"
    records = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    paths = [
        "artifacts/shared_models/kmeans/pipeline.joblib",
        "configs/models/kmeans.yaml", "configs/features.yaml", "configs/countries.yaml",
        "data/shared/v002/manifest.json", "data/shared/v002/unsupervised/manifest.json",
        *[record["path"] for record in records],
    ]
    return tuple((name, (root / name).stat().st_size, (root / name).stat().st_mtime_ns)
                 for name in sorted(set(paths)))


def load_country_clusters(root: str | Path) -> ClusterView:
    """Use only the approved shared release and saved model; write nothing."""
    # UI-only environments can still import other pages without scikit-learn.
    from oinq.modeling.clustering.kmeans import load_inputs, load_cluster_model, check_sources
    from oinq.modeling.clustering.results import build_results

    root = Path(root).resolve()
    data = load_inputs(root)
    bundle = load_cluster_model(root / "artifacts/shared_models/kmeans/pipeline.joblib", data)
    results = build_results(data, bundle, "ui_saved_kmeans")
    check_sources(data)
    results["군집"] = results.cluster_id.map(
        lambda value: "미분류" if pd.isna(value) else f"군집 {int(value) + 1}"
    ).astype("string")
    return ClusterView(results, bundle["model_version"], bundle["n_clusters"])


def distribution_figure(results: pd.DataFrame, window: str) -> go.Figure:
    """Use raw units and one global axis, area scale and color mapping."""
    selected = results.loc[results.window_id.eq(window) & results.cluster_id.notna()]
    maximum_population = float(results.population.max())
    fig = go.Figure()
    for label, color in CLUSTER_COLORS.items():
        if label == "미분류":
            continue  # Missing coordinates stay in the table, never become zeros.
        group = selected.loc[selected["군집"].eq(label)]
        fig.add_trace(go.Scatter(
            x=group.gdp_per_capita_usd.astype(float).tolist(),
            y=group.internet_users_pct.astype(float).tolist(), mode="markers", name=label,
            marker={"color": color, "size": group.population.astype(float).tolist(),
                    "sizemode": "area", "sizeref": 2 * maximum_population / 45 ** 2,
                    "sizemin": 4, "opacity": .8, "line": {"color": "white", "width": 1}},
            customdata=group[["country_name", "population", "gdp_per_capita_usd_observed_years",
                              "internet_users_pct_observed_years", "population_observed_years"]].values.tolist(),
            hovertemplate=("%{customdata[0]}<br>1인당 GDP: %{x:,.0f} USD/인"
                           "<br>인터넷 이용률: %{y:.1f}%<br>인구: %{customdata[1]:,.0f}명"
                           "<br>GDP 관측 연도: %{customdata[2]}<br>인터넷 관측 연도: %{customdata[3]}"
                           "<br>인구 관측 연도: %{customdata[4]}<extra>%{fullData.name}</extra>"),
        ))
    fig.update_layout(
        height=520, margin={"l": 25, "r": 25, "t": 20, "b": 30},
        xaxis={"title": "1인당 GDP (USD/인)", "range": [0, float(results.gdp_per_capita_usd.max()) * 1.08]},
        yaxis={"title": "인터넷 이용률 (%)", "range": [0, 100]},
        legend={"orientation": "h", "y": 1.12},
    )
    return fig


def country_history(results: pd.DataFrame, country: str) -> pd.DataFrame:
    """Keep all five windows, including unclassified and missing observations."""
    frame = results.loc[results.country_code.eq(country)].copy()
    return frame.set_index("window_id").loc[WINDOWS].reset_index()


def history_figure(history: pd.DataFrame, indicator: str, axis_title: str) -> go.Figure:
    column = f"{indicator}_observed_mean"
    values = pd.to_numeric(history[column], errors="coerce").astype(float)
    periods = history.window_id.map(WINDOW_LABELS).tolist()
    fig = go.Figure(go.Scatter(
        x=periods, y=values.tolist(), mode="lines", connectgaps=False, showlegend=False,
        line={"color": "#b6bec8", "width": 2}, hoverinfo="skip",
    ))
    for label, color in CLUSTER_COLORS.items():
        mask = history["군집"].eq(label)
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=history.loc[mask, "window_id"].map(WINDOW_LABELS).tolist(),
            y=values.loc[mask].tolist(), mode="markers", name=label,
            marker={"color": color, "size": 11},
            customdata=history.loc[mask, [f"{indicator}_observed_years", "군집"]].values.tolist(),
            hovertemplate="%{x}<br>%{y:,.2f}<br>관측 연도: %{customdata[0]}<extra>%{customdata[1]}</extra>",
        ))
    fig.update_layout(
        height=480, margin={"l": 25, "r": 25, "t": 20, "b": 30},
        xaxis={"title": "분석기간", "categoryorder": "array", "categoryarray": list(WINDOW_LABELS.values())},
        yaxis_title=axis_title, legend={"orientation": "h", "y": 1.12},
    )
    if indicator == "internet_users_pct":
        fig.update_yaxes(range=[0, 100])
    return fig


def distribution_table(selected: pd.DataFrame) -> pd.DataFrame:
    return selected[[
        "country_code", "country_name", "군집", "gdp_per_capita_usd", "internet_users_pct", "population",
        "gdp_per_capita_usd_observed_years", "internet_users_pct_observed_years", "population_observed_years",
        "missing_features", "observed_export_value_usd", "coverage_status",
    ]].rename(columns={
        "country_code": "국가 코드", "country_name": "국가", "gdp_per_capita_usd": "1인당 GDP (USD/인)",
        "internet_users_pct": "인터넷 이용률 (%)", "population": "인구 (명)",
        "gdp_per_capita_usd_observed_years": "GDP 관측 연도", "internet_users_pct_observed_years": "인터넷 관측 연도",
        "population_observed_years": "인구 관측 연도", "missing_features": "군집 입력 결측",
        "observed_export_value_usd": "관측 수출액 (USD·FOB)", "coverage_status": "수출 집계 상태",
    })
