"""Future input isolation, full-grid retention and inference-only execution."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor

from oinq.features.prediction import build_prediction_inputs, prepare_model_inputs, PredictionInputError
from oinq.forecasting.engine import run_forecast


def configuration():
    return {
        "countries": {"schema_version":1,"countries":[{"iso3":"CHN","name_ko":"중국","region":"asia"},{"iso3":"JPN","name_ko":"일본","region":"asia"}]},
        "products": {"schema_version":1,"products":[{"hs_code":"330410","name_ko":"립"},{"hs_code":"330420","name_ko":"아이"}]},
        "splits": {"schema_version":1,"collection":{"monthly_start":"2013-01"},"validation":{"horizon_quarters":[1,2,3,4,5]},
                   "forecast":{"start":"2027Q1","end":"2027Q4","preferred_origin_quarter":"2026Q3","horizon_quarters":[2,3,4,5],"updated_origin_quarter":"2026Q4","updated_horizon_quarters":[1,2,3,4]}},
        "features": {"schema_version":1,"target":{"minimum_prediction":0.0},"supervised":{"macro":{"columns":["gdp_usd"]}}},
    }


def dataset_manifest():
    categorical = ["country_code","region","hs_code"]
    calendar = ["target_year","target_quarter_number","horizon_quarters","time_index"]
    numeric = ["export_value_usd_lag1","gdp_usd","export_value_usd_lag1_missing","gdp_usd_missing"]
    columns = categorical+calendar+numeric
    return {"dataset_version":"v002","dataset_release_approved":True,"country_codes":["CHN","JPN"],"hs_codes":["330410","330420"],
            "feature_columns":columns,"feature_contract":{"categorical":categorical,"known_future":calendar,"numeric_candidates":numeric,"forbidden_feature_columns":["target_export_value_usd"]},
            "schemas":{"train":[{"name":c,"dtype":"string" if c in categorical else "Int64" if c in calendar else "boolean" if c.endswith("_missing") else "Float64"} for c in columns]}}


def origin_table():
    return pd.DataFrame([{"country_code":country,"hs_code":hs,"origin_quarter":"2026Q3", "export_value_usd_lag1":100.,"gdp_usd":1000.,
                          "export_value_usd_lag1_missing":False,"gdp_usd_missing":False,"macro_candidate_year":2025,"market_candidate_quarter":"2026Q3",
                          "origin_in_collection_range":True} for country in ["CHN","JPN"] for hs in ["330410","330420"]])


class InputTests(unittest.TestCase):
    def setUp(self):
        self.cfg,self.manifest,self.source=configuration(),dataset_manifest(),origin_table()

    def build(self, source=None, origin="2026Q3"):
        return build_prediction_inputs(self.source if source is None else source,self.cfg,self.manifest,origin)

    def test_future_calendar_and_history_are_separate(self):
        self.source["target_year"]=9999
        self.source["target_export_value_usd"]=999999.
        future=self.source.copy();future["origin_quarter"]="2026Q4";future["export_value_usd_lag1"]=999.
        data=self.build(pd.concat([self.source,future]))
        self.assertEqual(len(data),16)
        self.assertEqual(data.horizon_quarters.unique().tolist(),[2,3,4,5])
        self.assertEqual(data.time_index.unique().tolist(),[56,57,58,59])
        self.assertTrue(data.target_year.eq(2027).all())
        self.assertTrue(data.export_value_usd_lag1.eq(100).all())
        self.assertNotIn("target_export_value_usd",data)
        self.assertTrue(data.origin_cutoff_at.isna().all())
        self.assertFalse(data.historical_release_verified.any())

    def test_missing_pair_keeps_the_grid_and_missing_source_flag(self):
        data=self.build(self.source.iloc[1:])
        self.assertEqual(len(data),16)
        missing=data.loc[~data.source_row_available]
        self.assertEqual(len(missing),4)
        self.assertTrue(missing.export_value_usd_lag1.isna().all())
        self.assertTrue(missing.export_value_usd_lag1_missing.all())

    def test_existing_null_and_observed_zero_are_distinct(self):
        self.source.loc[0,"gdp_usd"]=np.nan
        self.source.loc[0,"gdp_usd_missing"]=True
        self.source.loc[1,"export_value_usd_lag1"]=0.
        data=self.build()
        self.assertEqual(data.gdp_usd.isna().sum(),4)
        self.assertEqual(data.export_value_usd_lag1.eq(0).sum(),4)
        self.assertFalse(data.loc[data.export_value_usd_lag1.eq(0),"export_value_usd_lag1_missing"].any())

    def test_bad_schema_duplicates_flags_and_alignment_are_rejected(self):
        cases=[]
        cases.append(self.source.drop(columns=["gdp_usd"]))
        cases.append(pd.concat([self.source,self.source.iloc[:1]]))
        bad=self.source.copy();bad.loc[0,"gdp_usd_missing"]=True;cases.append(bad)
        bad=self.source.copy();bad.loc[0,"macro_candidate_year"]=2024;cases.append(bad)
        bad=self.source.copy();bad.loc[0,"market_candidate_quarter"]="2026Q2";cases.append(bad)
        bad=self.source.copy();bad.loc[0,"country_code"]="USA";cases.append(bad)
        bad=self.source.copy();bad.loc[0,"gdp_usd"]=np.inf;cases.append(bad)
        for source in cases:
            with self.subTest(columns=source.columns.tolist(),rows=len(source)):
                with self.assertRaises(PredictionInputError):self.build(source)

    def test_absent_origin_and_out_of_training_horizons_are_rejected(self):
        with self.assertRaises(PredictionInputError):self.build(origin="2026Q4")
        old=self.source.copy();old.origin_quarter="2025Q4"
        with self.assertRaises(PredictionInputError):self.build(old,origin="2025Q4")

    def test_q4_origin_uses_updated_horizons(self):
        self.source.origin_quarter="2026Q4"
        self.source.market_candidate_quarter="2026Q4"
        self.assertEqual(self.build(origin="2026Q4").horizon_quarters.unique().tolist(),[1,2,3,4])

    def test_model_input_order_and_dtypes_are_restored_without_imputing(self):
        self.source.loc[0,"gdp_usd"]=np.nan;self.source.loc[0,"gdp_usd_missing"]=True
        data=self.build()
        loaded={"feature_columns":["gdp_usd","country_code"],"categorical_columns":["country_code"]}
        frame=prepare_model_inputs(data,loaded)
        self.assertEqual(frame.columns.tolist(),loaded["feature_columns"])
        self.assertEqual(frame.gdp_usd.dtype,np.dtype("float64"))
        self.assertEqual(frame.gdp_usd.isna().sum(),4)
        self.assertTrue(all(isinstance(v,str) for v in frame.country_code))


class PredictOnly:
    def __init__(self, nonfinite=False):
        self.named_steps={"model":DummyRegressor()}
        self.inputs=[]
        self.nonfinite=nonfinite

    def fit(self,*args,**kwargs):
        raise AssertionError("Engine must never fit")

    def predict(self,frame):
        self.inputs.append(frame.copy())
        result=np.full(len(frame),10.)
        result[0]=np.nan if self.nonfinite else -5.
        return result


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        (self.root/"configs").mkdir()
        self.cfg=configuration();self.manifest=dataset_manifest()
        for name,config in self.cfg.items():
            (self.root/"configs"/f"{name}.yaml").write_text(yaml.safe_dump(config,allow_unicode=True),encoding="utf-8")
        (self.root/"manifest.json").write_text(json.dumps(self.manifest),encoding="utf-8")
        self.model_path=self.root/"artifacts/shared_models/test_model/with_macro/pipeline.joblib"
        self.model_path.parent.mkdir(parents=True);self.model_path.write_bytes(b"fixture; loader mocked")
        self.source=origin_table();self.source.to_parquet(self.root/"origin.parquet",index=False)
        settings={"schema_version":1,"model":{"id":"test_model","variant":"with_macro"},
                  "paths":{"models_dir":"artifacts/shared_models","features_file":"origin.parquet","dataset_manifest":"manifest.json","output_dir":".private/forecasts"},
                  "input_policy":"current_revised_calendar_candidates"}
        (self.root/"configs/engine.yaml").write_text(yaml.safe_dump(settings),encoding="utf-8")
        self.model=PredictOnly()
        self.loaded={"model":self.model,"model_format":"joblib_pipeline","feature_columns":self.manifest["feature_columns"],"categorical_columns":["country_code","region","hs_code"]}

    def run_engine(self,**kwargs):
        with patch("oinq.forecasting.engine.load_model",return_value=self.loaded):
            return run_forecast(root=self.root,**kwargs)

    def test_full_inference_clips_negatives_and_persists_reusable_inputs(self):
        result=self.run_engine(run_id="one")
        q=result["quarterly"]
        self.assertEqual(len(q),16);self.assertEqual(len(result["annual"]),4)
        self.assertEqual(q.predicted_export_value_usd.min(),0.)
        self.assertTrue(result["annual"].annual_status.eq("complete").all())
        self.assertFalse(result["metadata"]["refit_performed"])
        snapshot=result["output_dir"]/"origin_features.parquet"
        self.assertEqual(len(pd.read_parquet(snapshot)),4)
        again=self.run_engine(run_id="two",features_path=snapshot)
        np.testing.assert_array_equal(q.predicted_export_value_usd,again["quarterly"].predicted_export_value_usd)

    def test_source_missing_rows_are_retained_and_not_given_fake_predictions(self):
        self.source.iloc[1:].to_parquet(self.root/"origin.parquet",index=False)
        result=self.run_engine(run_id="missing")
        q=result["quarterly"]
        self.assertEqual(len(q),16)
        self.assertEqual(q.prediction_status.eq("unavailable_input").sum(),4)
        self.assertEqual(q.predicted_export_value_usd.isna().sum(),4)
        annual=result["annual"]
        self.assertEqual(annual.predicted_annual_export_value_usd.isna().sum(),1)

    def test_duplicate_run_does_not_overwrite_results(self):
        result=self.run_engine(run_id="fixed")
        before=(result["output_dir"]/"quarterly_forecast.parquet").read_bytes()
        with self.assertRaises(FileExistsError):self.run_engine(run_id="fixed")
        self.assertEqual((result["output_dir"]/"quarterly_forecast.parquet").read_bytes(),before)

    def test_nonfinite_model_outputs_fail_before_writing_results(self):
        self.model.nonfinite=True
        with self.assertRaises(ValueError):self.run_engine(run_id="invalid")
        self.assertFalse((self.root/".private/forecasts/invalid").exists())

    def test_model_traversal_and_writing_into_model_folder_are_rejected(self):
        with self.assertRaises(ValueError):self.run_engine(model_id="../bad",run_id="bad")
        with self.assertRaises(ValueError):self.run_engine(output_dir="artifacts/shared_models/results",run_id="bad")

    def test_shared_release_without_latest_inputs_reports_an_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError,"--features"):
            self.run_engine(features_path="missing.parquet",run_id="no_input")

    def test_unknown_model_inputs_are_rejected(self):
        self.loaded["feature_columns"]=self.manifest["feature_columns"]+["target_export_value_usd"]
        with self.assertRaises(ValueError):self.run_engine(run_id="leak")

    def test_missing_engine_paths_are_reported_before_any_prediction(self):
        path=self.root/"configs/engine.yaml"
        settings=yaml.safe_load(path.read_text(encoding="utf-8"))
        del settings["paths"]["features_file"]
        path.write_text(yaml.safe_dump(settings),encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"paths"):
            self.run_engine(run_id="bad_config")
        self.assertEqual(self.model.inputs,[])


if __name__ == "__main__":
    unittest.main()
