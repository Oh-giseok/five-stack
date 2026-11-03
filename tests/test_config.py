"""Guard cross-file contracts that can invalidate all team experiments."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import ConfigError, load_project_config, load_yaml, validate_project_config


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(ROOT)

    def test_shared_configs_agree(self):
        result = validate_project_config(self.config)
        self.assertEqual(result["quarters"], {"train": 36, "validation": 16, "final_test": 3})
        self.assertEqual(result["forecast_quarters"], 4)

    def test_invalid_cross_file_edits_are_rejected(self):
        edits = [
            (("countries", "countries", 0, "iso3"), "JPN"),
            (("countries", "countries", 0, "region"), "europe"),
            (("countries", "countries", 2, "comtrade_reporter"), 356),
            (("countries", "countries", 7, "comtrade_reporter"), 250),
            (("products", "products", 0, "hs_code"), 330410),
            (("products", "products", 0, "hs_code"), "330420"),
            (("splits", "splits", "validation", "start"), "2023Q4"),
            (("splits", "final_test", "origin_quarter"), "2025Q3"),
            (("splits", "forecast", "horizon_quarters"), [1, 2, 3, 4]),
            (("splits", "validation", "final_test_for_tuning"), True),
            (("splits", "forecast", "partial_quarter_as_actual"), True),
            (("features", "supervised", "macro", "future_actuals_allowed"), True),
            (("sources", "world_bank", "indicators", "gdp_usd"), None),
            (("features", "clustering", "cluster_label_as_supervised_feature"), True),
            (("sources", "storage", "final_test_dir"), "data/shared/test"),
            (("sources", "storage", "raw_dir"), "../outside"),
            (("sources", "storage", "raw_dir"), "D:\\outside"),
            (("sources", "comtrade", "preview_for_production_dataset"), True),
            (("models", "kmeans", "inputs", "columns"), ["population"]),
            (("models", "xgboost", "training", "selection_split"), "final_test"),
        ]
        for path, value in edits:
            with self.subTest(path=path):
                config = deepcopy(self.config)
                node = config
                for key in path[:-1]:
                    node = node[key]
                node[path[-1]] = value
                with self.assertRaises(ConfigError):
                    validate_project_config(config)

    def test_macro_indicator_missing_is_rejected(self):
        config = deepcopy(self.config)
        del config["sources"]["world_bank"]["indicators"]["gdp_usd"]
        with self.assertRaises(ConfigError):
            validate_project_config(config)

    def test_missing_required_field_is_config_error(self):
        config = deepcopy(self.config)
        del config["features"]["target"]
        with self.assertRaises(ConfigError):
            validate_project_config(config)

    def test_bad_yaml_documents_are_rejected(self):
        documents = [
            "schema_version: 1\na: 1\na: 2\n",
            "schema_version: 1\na:\n  b: 1\n  b: 2\n",
            "schema_version: 1\na: [\n",
            "- not-a-mapping\n", "", "schema_version: 2\n",
            "schema_version: 1\na: !!python/object:object {}\n",
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "invalid.yaml"
            for document in documents:
                with self.subTest(document=document):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaises(ConfigError):
                        load_yaml(path)


if __name__ == "__main__":
    unittest.main()
