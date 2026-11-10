"""Observed history must distinguish partial sums, missing values and real zero."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from oinq.ui.country_charts import history_series, ranking_figure, forecast_series, export_timeline, timeline_figure
from oinq.ui.data import ResultDataError, load_monthly_exports
from oinq.ui.pages.countries import _country_quarter_summary
from oinq.ui.pages.products import _ranking_records


def observations(year, values):
    months = pd.period_range(f"{year}-01", periods=len(values), freq="M")
    return pd.DataFrame({"country_code": "AAA", "hs_code": "330410", "period_month": months.astype(str),
                         "year": year, "quarter": months.asfreq("Q").astype(str),
                         "export_value_usd": values})


def test_partial_annual_is_labelled_and_missing_quarter_is_not_zero():
    source = observations(2024, [0, 10, None, *([None] * 3), *range(6)])
    annual = history_series(source, "AAA", "330410", "연간").iloc[0]
    assert annual.value_usd == 25 and annual.status == "partial"
    assert annual.valid_count == 8 and annual.expected_count == 12
    assert pd.isna(annual.complete_value_usd)
    quarters = history_series(source, "AAA", "330410").set_index("period")
    assert quarters.loc["2024Q1", "value_usd"] == 10
    assert quarters.loc["2024Q1", "status"] == "partial"
    assert pd.isna(quarters.loc["2024Q2", "value_usd"])
    assert quarters.loc["2024Q2", "status"] == "missing"
    zero = history_series(observations(2024, [0, 0, 0]), "AAA", "330410").iloc[0]
    assert zero.value_usd == 0 and zero.status == "actual"


def test_2026_annual_retains_nine_month_coverage_and_ytd_label():
    source = observations(2026, [1] * 9)
    row = history_series(source, "AAA", "330410", "연간").iloc[0]
    assert row.value_usd == 9 and row.status == "ytd" and row.expected_count == 9
    source.loc[0, "export_value_usd"] = np.nan
    assert history_series(source, "AAA", "330410", "연간").iloc[0].status == "partial"


def test_ranking_preserves_missing_country_without_creating_a_zero_export():
    frame = pd.DataFrame({"code": ["AAA", "BBB"], "label": ["A", "B"],
                          "value_usd": [0., np.nan], "status": ["actual", "missing"],
                          "valid_count": [3, 0], "expected_count": [3, 3]})
    figure = ranking_figure(frame, "AAA")
    assert len(figure.data[0].y) == 2
    assert figure.data[0].x[0] == 0 and pd.isna(figure.data[0].x[1])
    assert figure.data[1].text == ("미확보",)


def test_monthly_loader_rejects_duplicate_keys(tmp_path):
    months = pd.period_range("2013-01", "2026-09", freq="M").astype(str)
    frame = pd.DataFrame({"country_code": "AAA", "hs_code": "330410", "period_month": months,
                          "export_value_usd": 0., "currency": "USD", "valuation": "FOB"})
    path = tmp_path / "data/shared/history/customs_monthly_hs6.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    assert len(load_monthly_exports(tmp_path, ["AAA"], ["330410"])) == 165
    pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ResultDataError, match="중복"):
        load_monthly_exports(tmp_path, ["AAA"], ["330410"])


def test_forecast_summary_does_not_present_missing_product_as_complete():
    quarters = [f"2027Q{q}" for q in range(1, 5)]
    quarterly = pd.DataFrame({"country_code": "AAA", "hs_code": "330410",
                              "target_quarter": quarters, "predicted_export_value_usd": 10.})
    annual = pd.DataFrame({"country_code": ["AAA"], "hs_code": ["330410"],
                          "annual_status": ["complete"], "predicted_annual_export_value_usd": [40.]})
    run = SimpleNamespace(metadata={"target_quarters": quarters}, quarterly=quarterly, annual=annual)
    full = _country_quarter_summary(run, ["AAA"], ["330410"]).iloc[0]
    assert full["연간"] == 40 and full["Q1"] == 10
    missing = _country_quarter_summary(run, ["AAA"], ["330410", "330420"]).iloc[0]
    assert pd.isna(missing["연간"]) and pd.isna(missing["Q1"])


def saved_run():
    return SimpleNamespace(
        quarterly=pd.DataFrame({"country_code": "AAA", "hs_code": "330410",
                                 "target_quarter": [f"2027Q{q}" for q in range(1, 5)],
                                 "predicted_export_value_usd": [0., 10., 20., 30.]}),
        annual=pd.DataFrame({"country_code": ["AAA"], "hs_code": ["330410"],
                             "annual_status": ["complete"], "predicted_annual_export_value_usd": [60.]}),
    )


def test_product_forecast_keeps_missing_quarters_and_real_zero():
    run = saved_run()
    assert forecast_series(run, "AAA", "330410", "연간").iloc[0].value_usd == 60
    run.quarterly = run.quarterly.iloc[:3]
    series = forecast_series(run, "AAA", "330410")
    assert len(series) == 4 and series.iloc[0].value_usd == 0
    assert pd.isna(series.iloc[3].value_usd)
    assert pd.isna(forecast_series(run, "AAA", "330410", "연간").iloc[0].value_usd)


def test_product_ranking_retains_all_eight_products_including_missing():
    products = [{"hs_code": code, "name_ko": code} for code in
                ["330410", "330420", "330491", "330499", "330510", "330590", "330300", "340130"]]
    rows = _ranking_records(saved_run(), "AAA", products, "2027Q1")
    assert len(rows) == 8 and rows.iloc[0].value_usd == 0
    assert rows.value_usd.isna().sum() == 7
    missing = _ranking_records(None, "AAA", products, "2027년 연간")
    assert len(missing) == 8 and missing.value_usd.isna().all()


def test_product_timeline_has_explicit_2026q4_gap_and_separate_forecasts():
    monthly = observations(2026, [1.] * 9)
    series = export_timeline(monthly, "AAA", "330410", run=saved_run()).set_index("period")
    assert len(series) == 60 and pd.isna(series.loc["2026Q4", "value_usd"])
    assert series.loc["2026Q4", "status"] == "missing"
    assert series.loc["2027Q1", "value_usd"] == 0
    assert series.loc["2027Q4", "value_usd"] == 30
    figure = timeline_figure(series.reset_index(), "분기", "선")
    assert all(trace.connectgaps is False for trace in figure.data)
    assert any(shape.type == "rect" and shape.x0 == 54.5 for shape in figure.layout.shapes)
    yearly = export_timeline(monthly, "AAA", "330410", "연간", saved_run()).set_index("period")
    assert yearly.loc["2026", "status"] == "ytd" and yearly.loc["2027", "value_usd"] == 60
