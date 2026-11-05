"""Comparable period charts must preserve missing observations and group colors."""

import json

import pandas as pd
import pytest

pytest.importorskip("plotly")

from oinq.ui.clustering import (
    CLUSTER_COLORS, WINDOWS, cluster_source_signature, country_history,
    distribution_figure, history_figure,
)


def results():
    records = []
    for country in ["AAA", "BBB"]:
        for i, window in enumerate(WINDOWS):
            missing = country == "AAA" and i in [2, 4]
            cluster_id = None if missing else (i % 3 if country == "AAA" else 2)
            records.append({
                "country_code": country, "country_name": country, "window_id": window,
                "gdp_per_capita_usd": 1000. * (i + 1), "population": 100000. if country == "AAA" else 1e9,
                "internet_users_pct": None if missing else 20. + i,
                "internet_users_pct_observed_mean": None if missing else 20. + i,
                "gdp_per_capita_usd_observed_years": "[2020]", "population_observed_years": "[2020]",
                "internet_users_pct_observed_years": "[]" if missing else "[2020]",
                "cluster_id": cluster_id, "군집": "미분류" if missing else f"군집 {cluster_id + 1}",
            })
    frame = pd.DataFrame(records)
    frame["cluster_id"] = frame.cluster_id.astype("Int64")
    return frame


def test_distribution_axes_area_scale_and_group_colors_are_identical_across_windows():
    frame = results()
    charts = [distribution_figure(frame, window) for window in WINDOWS]
    first = charts[0]
    for chart in charts:
        assert chart.layout.xaxis.range == first.layout.xaxis.range
        assert chart.layout.yaxis.range == (0, 100)
        assert [trace.marker.sizeref for trace in chart.data] == [trace.marker.sizeref for trace in first.data]
        assert {trace.name: trace.marker.color for trace in chart.data} == {
            label: color for label, color in CLUSTER_COLORS.items() if label != "미분류"
        }
    # Latest period has one classified country; the missing country isn't plotted at zero.
    assert sum(len(trace.x) for trace in charts[-1].data) == 1
    assert len(frame) == 10 and frame.cluster_id.isna().sum() == 2


def test_history_keeps_all_windows_and_gaps_and_retains_the_unclassified_label():
    history = country_history(results().sample(frac=1, random_state=1), "AAA")
    assert history.window_id.tolist() == WINDOWS
    assert history["군집"].eq("미분류").sum() == 2
    chart = history_figure(history, "internet_users_pct", "인터넷 이용률 (%)")
    assert len(chart.data[0].x) == 5
    assert chart.data[0].connectgaps is False
    assert pd.isna(chart.data[0].y[2]) and pd.isna(chart.data[0].y[4])
    assert chart.layout.yaxis.range == (0, 100)


def test_cluster_cache_signature_changes_when_model_or_source_changes(tmp_path):
    files = ["artifacts/shared_models/kmeans/pipeline.joblib", "configs/models/kmeans.yaml",
             "configs/features.yaml", "configs/countries.yaml", "data/shared/v002/manifest.json",
             "data/shared/v002/unsupervised/country_window_features.parquet"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original")
    manifest = tmp_path / "data/shared/v002/unsupervised/manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": files[-1]}]}), encoding="utf-8")
    previous = cluster_source_signature(tmp_path)
    for name in [files[0], files[-1]]:
        (tmp_path / name).write_bytes(b"a replacement file")
        current = cluster_source_signature(tmp_path)
        assert current != previous
        previous = current
