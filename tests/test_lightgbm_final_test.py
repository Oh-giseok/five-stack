"""Synthetic contract tests; no actual final-test file is opened."""
from pathlib import Path
import sys
import unittest
import pandas as pd
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from oinq.modeling.supervised.lightgbm_final_test import (
    check_features,check_test_inputs,select_scored_rows,test_metrics as calculate_test_metrics,TARGET,PRED)

class FrozenFinalTestTests(unittest.TestCase):
    def test_all_model_variant_labels_are_preserved(self):
        frame=pd.DataFrame({"country_code":["USA"]*3,"hs_code":["330410"]*3,
                            "horizon_quarters":[1,2,3],TARGET:[10.,20.,30.],PRED:[11.,19.,32.]})
        for model in ["xgboost_l1_l2","catboost_l2_1","lightgbm_l1"]:
            for variant in ["with_macro","without_macro"]:
                values=calculate_test_metrics(frame,model,[2,3,4,5],variant=variant)
                self.assertEqual(set(values.model_id),{model})
                self.assertEqual(set(values.variant),{variant})
                self.assertEqual(values.loc[values.group_by.eq("overall") & values.scope.eq("h1_3"),"n"].iloc[0],3)
    def test_target_metadata_cannot_be_input(self):
        contract={"categorical":["country_code"],"known_future":["horizon_quarters"],
                  "numeric_candidates":[TARGET],"forbidden_feature_columns":[TARGET],"target":TARGET}
        with self.assertRaises(ValueError):check_features(["country_code","horizon_quarters",TARGET],contract)

    def test_origin_and_complete_grid_are_required(self):
        spec={"origin_quarter":"2025Q4","target_quarters":["2026Q1","2026Q2","2026Q3"],"horizon_quarters":[1,2,3]}
        df=pd.DataFrame({"country_code":["USA"]*3,"hs_code":["330410"]*3,
                         "origin_quarter":["2025Q4"]*3,"target_quarter":spec["target_quarters"],"horizon_quarters":[1,2,3]})
        check_test_inputs(df,spec,["USA"],["330410"])
        with self.assertRaises(ValueError):check_test_inputs(df.iloc[:2],spec,["USA"],["330410"])
        df.loc[0,"origin_quarter"]="2026Q1"
        with self.assertRaises(ValueError):check_test_inputs(df,spec,["USA"],["330410"])

    def test_missing_actual_is_not_zero_and_can_still_have_prediction(self):
        df=pd.DataFrame({TARGET:[10.,np.nan,0.],PRED:[12.,5.,1.],"target_available":[True,False,True]})
        scored,excluded=select_scored_rows(df)
        self.assertEqual(len(scored),2)
        self.assertEqual(len(excluded),1)
        self.assertEqual(excluded[PRED].iloc[0],5.)
        self.assertTrue(pd.isna(excluded[TARGET].iloc[0]))

    def test_metrics_report_h1_3_and_h2_3_without_annual_claim(self):
        df=pd.DataFrame({"country_code":["USA"]*3,"hs_code":["330410"]*3,
                         "horizon_quarters":[1,2,3],TARGET:[10.,10.,10.],PRED:[11.,12.,13.]})
        result=calculate_test_metrics(df,"lightgbm",[2,3,4,5])
        overall=result[result.group_by.eq("overall")].set_index("scope")
        self.assertEqual(overall.loc["h1_3","n"],3)
        self.assertEqual(overall.loc["h2_3","n"],2)
        self.assertAlmostEqual(overall.loc["h1_3","wape"],20.)
        self.assertNotIn("annual",set(result.scope))

if __name__=="__main__":unittest.main()
