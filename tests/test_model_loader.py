"""Stored model contracts and path handling; the loader must never fit/predict."""
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from oinq.modeling import loader


class SavedTransform(BaseEstimator):
    def __init__(self):
        self.feature_names_in_ = np.array(["country_code", "value"], dtype=object)
        self.transformers_ = [("categorical", object(), ["country_code"])]

    def fit(self, values, target=None):
        raise AssertionError("Loading must not fit transformers")

    def transform(self, values):
        raise AssertionError("Loading must not transform inputs")


class SavedRegressor(BaseEstimator):
    def fit(self, values, target=None):
        raise AssertionError("Loading must not fit models")

    def __sklearn_is_fitted__(self):
        return True

    def predict(self, values):
        raise AssertionError("Loading must not generate predictions")


def saved_pipeline(step="preprocessor"):
    return Pipeline([(step, SavedTransform()), ("model", SavedRegressor())])


class ModelLoaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.model_path = self.root / "pipeline.joblib"
        self.model_path.touch()

    def test_dict_bundle_returns_fitted_model_and_order_without_execution(self):
        model = saved_pipeline()
        saved = {"pipeline": model, "feature_columns": ["country_code", "value"],
                 "categorical_columns": ["country_code"]}
        with patch.object(loader.joblib, "load", return_value=saved):
            result = loader.load_model(self.model_path)
        self.assertIs(result["model"], model)
        self.assertEqual(result["model_format"], "joblib_pipeline")
        self.assertEqual(result["feature_columns"], ["country_code", "value"])
        self.assertEqual(result["categorical_columns"], ["country_code"])

    def test_bare_pipeline_extracts_categories_from_fitted_transformer(self):
        model = saved_pipeline("transformer")
        with patch.object(loader.joblib, "load", return_value=model):
            result = loader.load_model(self.model_path)
        self.assertIs(result["model"], model)
        self.assertEqual(result["categorical_columns"], ["country_code"])

    def test_relative_path_is_independent_of_current_directory(self):
        other = self.root / "notebooks"
        other.mkdir()
        original = Path.cwd()
        try:
            os.chdir(other)
            with patch.object(loader, "PROJECT_ROOT", self.root), patch.object(loader.joblib, "load", return_value=saved_pipeline()):
                result = loader.load_model("pipeline.joblib")
        finally:
            os.chdir(original)
        self.assertEqual(result["model_path"], self.model_path)

    def test_missing_file_is_reported_before_deserialization(self):
        with patch.object(loader.joblib, "load") as load:
            with self.assertRaises(FileNotFoundError):
                loader.load_model(self.root / "missing.joblib")
            load.assert_not_called()

    def test_unsupported_file_is_rejected_before_deserialization(self):
        path = self.root / "model.pkl"
        path.touch()
        with patch.object(loader.joblib, "load") as load:
            with self.assertRaisesRegex(ValueError, "지원하지 않는"):
                loader.load_model(path)
            load.assert_not_called()

    def test_malformed_bundle_is_reported(self):
        with patch.object(loader.joblib, "load", return_value={"model": saved_pipeline()}):
            with self.assertRaisesRegex(ValueError, "pipeline"):
                loader.load_model(self.model_path)

    def test_non_pipeline_and_unfitted_objects_are_rejected(self):
        for saved in [object(), Pipeline([("model", DummyRegressor())])]:
            with self.subTest(saved=type(saved).__name__):
                with patch.object(loader.joblib, "load", return_value=saved):
                    with self.assertRaises(ValueError):
                        loader.load_model(self.model_path)

    def test_inconsistent_feature_order_and_duplicate_names_are_rejected(self):
        for features in [["value", "country_code"], ["country_code", "value", "value"], "country_code"]:
            with self.subTest(features=features):
                with patch.object(loader.joblib, "load", return_value={"pipeline": saved_pipeline(), "feature_columns": features}):
                    with self.assertRaises(ValueError):
                        loader.load_model(self.model_path)

    def test_fitted_order_is_used_when_metadata_includes_dropped_columns(self):
        saved = {"pipeline": saved_pipeline(), "feature_columns": ["country_code", "empty_at_fit", "value"]}
        with patch.object(loader.joblib, "load", return_value=saved):
            result = loader.load_model(self.model_path)
        self.assertEqual(result["feature_columns"], ["country_code", "value"])

    def test_inconsistent_category_metadata_is_rejected(self):
        saved = {"pipeline": saved_pipeline(), "categorical_columns": ["value"]}
        with patch.object(loader.joblib, "load", return_value=saved):
            with self.assertRaisesRegex(ValueError, "범주형"):
                loader.load_model(self.model_path)

    def test_catboost_loads_native_features_without_fit_or_predict(self):
        path = self.root / "model.cbm"
        path.touch()
        model = Mock()
        model.is_fitted.return_value = True
        model.feature_names_ = ["country_code", "value"]
        model.get_cat_feature_indices.return_value = [0]
        with patch("catboost.CatBoostRegressor", return_value=model):
            result = loader.load_model(path)
        model.load_model.assert_called_once_with(str(path))
        model.fit.assert_not_called()
        model.predict.assert_not_called()
        self.assertEqual(result["model_format"], "catboost")
        self.assertEqual(result["categorical_columns"], ["country_code"])


if __name__ == "__main__":
    unittest.main()
