"""UI choices must order the real chart and table identically, without model filters."""
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from oinq.evaluation.comparison import COLUMNS
from oinq.ui.model_comparison import METRICS, ModelComparisonData
from oinq.ui.pages import performance


@pytest.fixture
def data():
    records, rows = [], []
    for name, r2, wape, amount in [("A", -.2, 120., 30e6), ("B", .9, 20., 10e6), ("C", None, None, 20e6)]:
        identity = dict(model_id=name, model_name=name, experiment="기본", variant="with_macro")
        status = "평가 완료" if r2 is not None else "운영용·비교 제외"
        score = dict(n=447, r2=r2, wape=wape, mae=3e6, rmse=12e6) if r2 is not None else None
        records.append({**identity, "status": status, "test": score,
                        "validation": dict(n=9164, wape=wape) if score else None})
        rows.append({**identity, "status": status, "validation_wape": wape, "test_wape": wape,
                     "test_r2": r2, "test_mae_usd": 3e6 if score else np.nan,
                     "test_rmse_usd": 12e6 if score else np.nan, "forecast_2027_usd": amount})
    return ModelComparisonData(records, pd.DataFrame(rows), {}, {})


@pytest.fixture
def page(monkeypatch, data):
    monkeypatch.setattr(performance, "_comparison_data", lambda *args: data)
    monkeypatch.setattr(performance, "comparison_source_signature", lambda *args: ())
    return AppTest.from_string("from oinq.ui.pages.performance import render\nrender('.')", default_timeout=15).run()


@pytest.mark.parametrize("metric", list(METRICS))
@pytest.mark.parametrize("ascending", [True, False])
def test_metric_and_direction_control_horizontal_chart_and_table(page, data, metric, ascending):
    page.selectbox(key=performance.METRIC_KEY).select(metric).run()
    page.get_by_key(performance.ORDER_KEY).set_value("오름차순" if ascending else "내림차순").run()
    assert not page.exception and not page.error
    assert len(page.selectbox) == 1 and page.selectbox[0].label == "세부지표"
    assert len(page.get("plotly_chart")) == len(page.dataframe) == 1
    expected = data.select_metric(metric, ascending=ascending)
    pd.testing.assert_frame_equal(page.dataframe[0].value, data.performance_table(metric, ascending=ascending))
    assert page.dataframe[0].value.columns.tolist() == COLUMNS
    spec = json.loads(page.get("plotly_chart")[0].proto.spec)
    assert spec["data"][0]["orientation"] == "h"
    assert spec["data"][0]["x"] == expected.comparison_value.dropna().tolist()
    assert len(spec["layout"]["yaxis"]["ticktext"]) == expected.comparison_value.notna().sum()
    if metric == "R²":
        assert spec["layout"]["xaxis"]["range"][0] < 0
    if metric == "검증 WAPE":
        assert max(spec["data"][0]["x"]) == 120.


def test_metric_changes_set_the_documented_default_direction(page):
    for metric, default in [("예상 수출액", "내림차순"), ("검증 WAPE", "오름차순"),
                            ("R²", "내림차순"), ("MAE", "오름차순")]:
        page.selectbox(key=performance.METRIC_KEY).select(metric).run()
        assert page.get_by_key(performance.ORDER_KEY).value == default
        assert not page.exception


def test_missing_connected_data_is_reported_without_fabricated_scores(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("not connected")
    monkeypatch.setattr(performance, "comparison_source_signature", missing)
    page = AppTest.from_string("from oinq.ui.pages.performance import render\nrender('.')").run()
    assert not page.exception and len(page.error) == 1
    assert not page.dataframe and not page.get("plotly_chart")
