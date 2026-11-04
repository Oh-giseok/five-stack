"""Behavioral checks for leakage, paired inputs, scoring and artifact replay."""
from pathlib import Path
import sys
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oinq.modeling.supervised.lightgbm import (
    training_rows, metric_values, make_pipeline, prepare_x, baseline_predictions,
    annual_results, quarter_number, resolve_model_params, TARGET, PRED,
)

class LightGBMValidationTests(unittest.TestCase):
    def test_l2_override_is_independent_and_preserves_configuration(self):
        config={"params":{"num_leaves":15,"random_state":42}}
        params=resolve_model_params(config,reg_alpha=0,reg_lambda=1)
        self.assertEqual(params["reg_alpha"],0)
        self.assertEqual(params["reg_lambda"],1)
        self.assertEqual(config["params"],{"num_leaves":15,"random_state":42})
        for value in [-1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError):resolve_model_params(config,reg_lambda=value)
    def test_l1_override_preserves_shared_configuration(self):
        config={"params":{"num_leaves":15,"random_state":42}}
        params=resolve_model_params(config,1.0)
        self.assertEqual(params["reg_alpha"],1.0)
        self.assertNotIn("reg_alpha",config["params"])
        self.assertEqual(resolve_model_params(config),config["params"])
        for value in [-1,float("nan"),float("inf")]:
            with self.assertRaises(ValueError):resolve_model_params(config,value)
    def test_origin_filters_future_and_missing_labels(self):
        df = pd.DataFrame({"country_code":["USA"]*3, "hs_code":["330410"]*3,
            "origin_quarter":["2020Q3","2020Q4","2020Q2"],
            "target_quarter":["2020Q4","2021Q1","2020Q3"],
            TARGET:[10.,99.,np.nan], "target_available":[True,True,False]})
        df["_target_index"] = df.target_quarter.map(quarter_number)
        df["_origin_index"] = df.origin_quarter.map(quarter_number)
        result = training_rows(df, "2020Q4")
        self.assertEqual(result.target_quarter.tolist(), ["2020Q4"])

    def test_metrics_weighted_wape_and_zero_cases(self):
        values = metric_values([100,1], [90,0])
        self.assertAlmostEqual(values["wape"], 100*11/101)
        self.assertEqual(metric_values([0],[0])["smape"], 0)
        self.assertIsNone(metric_values([0],[5])["wape"])

    def test_preprocessor_learns_only_fit_and_reloads(self):
        fit = pd.DataFrame({"country_code":["A","A","B","B"],
                            "value":[1.,3.,np.nan,2.], "empty":[np.nan]*4,
                            "value_missing":[0,0,1,0]})
        features = fit.columns.tolist()
        pipe, used, dropped = make_pipeline(fit, features, ["country_code"],
                    {"n_estimators":2,"num_leaves":2,"min_child_samples":1,
                     "verbosity":-1,"random_state":42,"n_jobs":1})
        pipe.fit(prepare_x(fit,used,["country_code"]),[1.,3.,2.,2.])
        test = pd.DataFrame({"country_code":["UNSEEN"],"value":[np.nan],"empty":[999.],"value_missing":[1]})
        self.assertEqual(dropped,["empty"])
        imputer = pipe.named_steps["transformer"].named_transformers_["numeric"]
        self.assertEqual(imputer.statistics_[0],2.)
        encoded = pipe.named_steps["transformer"].named_transformers_["categorical"]
        self.assertNotIn("UNSEEN",encoded.categories_[0])
        x = prepare_x(test,used,["country_code"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"pipe.joblib"
            joblib.dump(pipe,path)
            np.testing.assert_array_equal(pipe.predict(x), joblib.load(path).predict(x))

    def test_h5_baseline_cannot_read_target_minus_four(self):
        # 2022Q1 at origin 2020Q4: 2021Q1 is future, so use 2020Q1.
        history = pd.DataFrame({"country_code":["USA"]*2,"hs_code":["330410"]*2,
                        "target_quarter":["2020Q1","2021Q1"],TARGET:[10.,999.]})
        history["_target_index"] = history.target_quarter.map(quarter_number)
        evaluation = pd.DataFrame({"country_code":["USA"],"hs_code":["330410"],
                "origin_quarter":["2020Q4"],"target_quarter":["2022Q1"],
                "horizon_quarters":[5],TARGET:[15.]})
        result=baseline_predictions(evaluation,history)
        self.assertEqual(result[PRED].iloc[0],10.)
        self.assertEqual(result.baseline_source_quarter.iloc[0],"2020Q1")

    def test_four_quarters_do_not_mix_origins_or_accept_missing(self):
        df = pd.DataFrame({"model_id":["m"]*4,"variant":["v"]*4,
            "country_code":["USA"]*4,"hs_code":["330410"]*4,
            "origin_quarter":["2020Q4"]*4,"target_quarter":["2021Q1","2021Q2","2021Q3","2021Q4"],
            "horizon_quarters":[1,2,3,4],TARGET:[1.,2.,3.,4.],PRED:[1.,2.,3.,5.]})
        sums,metrics,_=annual_results(df)
        self.assertEqual(len(sums),1)
        self.assertEqual(sums.actual_sum_usd.iloc[0],10.)
        self.assertAlmostEqual(metrics.wape.iloc[0],10.)
        df.loc[3,"origin_quarter"]="2021Q1"
        self.assertTrue(annual_results(df)[0].empty)

    def test_missing_baseline_stays_missing_not_zero(self):
        history=pd.DataFrame({"country_code":["USA"],"hs_code":["330410"],
                  "target_quarter":["2020Q1"],TARGET:pd.Series([pd.NA],dtype="Float64")})
        history["_target_index"]=history.target_quarter.map(quarter_number)
        evaluation=pd.DataFrame({"country_code":["USA"],"hs_code":["330410"],
                "origin_quarter":["2020Q4"],"target_quarter":["2022Q1"],
                "horizon_quarters":[5],TARGET:[15.]})
        result=baseline_predictions(evaluation,history)
        self.assertTrue(pd.isna(result[PRED].iloc[0]))
        self.assertEqual(metric_values(result[TARGET],result[PRED])["n"],0)

if __name__ == "__main__":
    unittest.main()
