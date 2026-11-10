"""Failures that could otherwise produce misleading model comparisons."""
import json

import numpy as np
import pandas as pd
import pytest

from oinq.evaluation.comparison import (
    COLUMNS, KEY, PRED, TARGET, ComparisonReport, RecordReader,
    check_predictions, metrics, performance_table, save_report, sha256,
)


@pytest.fixture
def expected():
    return pd.DataFrame({"country_code": ["CHN", "IND", "VNM"], "hs_code": ["330410"] * 3,
                         "origin_quarter": ["2025Q4"] * 3, "target_quarter": ["2026Q1"] * 3,
                         "horizon_quarters": [1] * 3, TARGET: [100.0, 200.0, np.nan]})


@pytest.fixture
def predictions(expected):
    return expected.assign(**{PRED: [90.0, 240.0, 25.0], "target_available": [True, True, False]})


def test_missing_actuals_kept_but_not_scored(expected, predictions):
    scored = check_predictions(predictions, expected, full_grid=True)
    score = metrics(scored[TARGET], scored[PRED])
    assert score["n"] == 2
    assert score["wape"] == pytest.approx(50 / 300 * 100)
    assert score["mae"] == 25
    assert score["rmse"] == pytest.approx(np.sqrt(850))
    assert score["r2"] == pytest.approx(0.66)
    assert len(predictions) == 3


@pytest.mark.parametrize("change, error", [
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.assign(country_code=["FRA", "IND", "VNM"]), "keys differ"),
    (lambda d: d.assign(**{TARGET: [101.0, 200.0, np.nan]}), "actuals differ"),
    (lambda d: d.assign(horizon_quarters=[2, 1, 1]), "horizons differ"),
    (lambda d: d.assign(**{PRED: [np.nan, 240.0, 25.0]}), "scored prediction"),
    (lambda d: d.assign(**{PRED: [90.0, 240.0, np.nan]}), "scored prediction"),
    (lambda d: d.assign(**{PRED: [-1.0, 240.0, 25.0]}), "scored prediction"),
    (lambda d: d.assign(currency="KRW"), "currency"),
    (lambda d: d.assign(valuation="CIF"), "valuation"),
    (lambda d: d.assign(dataset_version="v001"), "dataset_version"),
    (lambda d: d.assign(target_available=[True, True, True]), "availability"),
    (lambda d: d.assign(raw_predicted_usd=[91.0, 240.0, 25.0]), "postprocessing"),
    (lambda d: d.iloc[:2], "keys differ"),
])
def test_invalid_comparisons_fail(expected, predictions, change, error):
    with pytest.raises(ValueError, match=error):
        check_predictions(change(predictions), expected, full_grid=True)


def test_old_validation_may_omit_only_unlabelled_rows(expected, predictions):
    assert len(check_predictions(predictions.iloc[:2], expected, full_grid=False)) == 2
    with pytest.raises(ValueError, match="Scored evaluation keys"):
        check_predictions(predictions.iloc[[0, 2]], expected, full_grid=False)


def test_undefined_metrics_are_not_zero_or_accuracy():
    assert metrics([0, 0], [1, 2])["wape"] is None
    assert metrics([1, 1], [1, 1])["r2"] is None
    assert metrics([100, 200], [1000, 2000])["wape"] == 900
    assert metrics([100, 200], [1000, 2000])["r2"] < 0
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics([1, np.nan], [1, 2])


def test_changed_source_is_rejected(tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("value\n1\n", encoding="utf-8")
    reader = RecordReader(tmp_path, {path.name: sha256(path)})
    path.write_text("value\n2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        reader.frame({"path": path.name})


def test_display_units_leave_internal_usd_and_bad_scores_unchanged():
    score = {"n": 447, "wape": 118.2, "mae": 16928423.91, "rmse": 59753074.41, "r2": -0.0039}
    record = {"model_name": "ElasticNet", "experiment": "L1+L2", "variant": "with_macro",
              "status": "평가 완료", "validation": {"n": 9164, "wape": 124.94}, "test": score}
    table = performance_table([record])
    assert table.columns.tolist() == COLUMNS
    assert table.iloc[0]["테스트 MAE (백만 USD) ↓"] == pytest.approx(16.92842391)
    assert table.iloc[0]["테스트 RMSE (백만 USD) ↓"] == pytest.approx(59.75307441)
    assert table.iloc[0]["테스트 WAPE (%) ↓"] == 118.2
    assert table.iloc[0]["테스트 R² ↑"] == -0.0039
    assert table.iloc[0]["평가 건수"] == "9,164 / 447"
    assert score["mae"] == 16928423.91


def test_report_preserves_missing_scores_and_does_not_overwrite(tmp_path):
    row = dict(zip(COLUMNS, ["ElasticNet", "운영용", "포함", None, None, None, None, None, "—", "운영용·비교 제외"]))
    report = ComparisonReport(pd.DataFrame([row], columns=COLUMNS), [{"test": None}], [], {"training_performed": False})
    output = tmp_path / "result"
    save_report(report, output)
    frame = pd.read_csv(output / "comparison.csv", keep_default_na=False)
    assert frame.columns.tolist() == COLUMNS
    assert frame.iloc[0]["테스트 R² ↑"] == "—"
    data = json.loads((output / "comparison.json").read_text("utf-8"))
    assert data["models"][0]["test"] is None
    before = (output / "comparison.csv").read_bytes()
    with pytest.raises(FileExistsError):
        save_report(report, output)
    assert (output / "comparison.csv").read_bytes() == before
