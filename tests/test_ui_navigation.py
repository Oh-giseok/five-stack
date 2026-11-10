"""Native country tabs execute one view and preserve the selected view."""
import streamlit as st
from streamlit.testing.v1 import AppTest


def _app():
    from app import main_app
    main_app()


def test_native_country_tabs_are_lazy_and_survive_page_navigation(monkeypatch):
    import app

    calls = []

    def history(*args):
        calls.append("history")
        st.text("history-view")

    def forecast(*args):
        calls.append("forecast")
        st.text("forecast-view")

    def similar(*args):
        calls.append("similar")
        st.text("similar-view")

    monkeypatch.setattr(app.countries, "render_history", history)
    monkeypatch.setattr(app.countries, "render_comparison", forecast)
    monkeypatch.setattr(app.countries, "render_similar", similar)
    monkeypatch.setattr(app, "_select_forecast", lambda: None)
    page = AppTest.from_function(_app, default_timeout=20).run()
    assert not page.exception
    assert not page.info
    assert "분석 시작하기" not in [item.value for item in page.subheader]
    assert len(page.get("image")) == 3
    assert {item.proto.imgs[0].alt for item in page.get("image")} == {
        "지구 아이콘", "화장품 아이콘", "공장 아이콘",
    }
    page.button(key="home_go_국가별 분석").click().run()
    assert [tab.label for tab in page.tabs] == ["수출 현황", "수출 전망", "유사 국가"]
    assert calls == ["history"]
    assert [item.value for item in page.text] == ["history-view"]
    for label, name in [("수출 전망", "forecast"), ("유사 국가", "similar")]:
        before = len(calls)
        page.session_state["gem_country_tab_selection"] = label
        page.run()
        assert not page.exception
        assert len(calls) == before + 1 and calls[-1] == name
        assert [item.value for item in page.text] == [name+"-view"]
    page.button(key="gem_nav_0").click().run()
    page.button(key="gem_nav_1").click().run()
    assert page.session_state["gem_country_tab_selection"] == "유사 국가"
    assert [item.value for item in page.text] == ["similar-view"]
    assert not page.exception


def test_hiding_forecast_context_retains_real_missing_grid_warning():
    def caption():
        import streamlit as st
        from types import SimpleNamespace
        from oinq.ui.components import render_run_caption
        render_run_caption({"origin_quarter": "2026Q3", "missing_input_rows": 640},
                           SimpleNamespace(missing_keys=["missing-row"], expected_rows=640),
                           show_context=False)

    page = AppTest.from_function(caption).run()
    assert not page.exception and not page.caption and not page.info
    assert len(page.warning) == 1 and "1개 예측행" in page.warning[0].value
