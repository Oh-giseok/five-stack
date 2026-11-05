"""Country/window scope, missing-data preservation and inference-only contracts."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from oinq.modeling.clustering import kmeans
from oinq.modeling.clustering.results import build_results


def frames():
    rows = []
    for i, country in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        for j, (window, start, end, months) in enumerate([
            ("2013_2015", "2013-01", "2015-12", 36),
            ("2025_2026Q3", "2025-01", "2026-09", 21),
        ]):
            eligible = not (j == 1 and i >= 2)
            rows.append({"country_code":country, "window_id":window, "country_name":country,
                         "start_month":start, "end_month":end, "period_month_count":months, "is_short_period":j == 1,
                         "gdp_per_capita_usd":1000.+i*3000+j*100, "population":100000.+i*900000+j*10000,
                         "internet_users_pct":10.+i*20+j if eligible else np.nan,
                         "eligible_for_fit":eligible, "eligibility_reason":"available" if eligible else "missing_internet"})
    f = pd.DataFrame(rows)
    for c in kmeans.KEY + ["country_name", "start_month", "end_month", "eligibility_reason"]:
        f[c] = f[c].astype("string")
    for c in kmeans.FEATURES:
        f[c] = f[c].astype("Float64")
    f["eligible_for_fit"] = f.eligible_for_fit.astype("boolean")
    profile = f.copy()
    profile["gdp_usd"] = pd.Series([1e8]*len(f), dtype="Float64")
    context = profile.copy()
    context["export_value_usd"] = pd.Series([0., pd.NA]+[100.]*6, dtype="Float64")
    context["observed_export_value_usd"] = pd.Series([0.]+[100.]*7, dtype="Float64")
    return f, profile, context


def configuration():
    return {"schema_version":1, "model_id":"kmeans", "params":{"n_clusters":3, "init":"k-means++", "n_init":2, "max_iter":30, "random_state":42},
            "inputs":{"columns":kmeans.FEATURES, "row_key":kmeans.KEY, "fit_scope":kmeans.FIT_SCOPE,
                      "shared_scaler_and_centroids_across_windows":True, "scaler":"StandardScaler",
                      "reference_year":"three_year_window", "aggregation":"period_mean",
                      "historical_backtest_claim":False, "missing":"leave_country_unclassified"},
            "output":{"supervised_feature":False, "cluster_id_is_rank":False},
            "selection":{"candidate_cluster_counts":[2,3,4]}}


def data_fixture():
    features, profile, context = frames()
    return {"features":features, "profile":profile, "context":context,
            "config":configuration(), "source_sha256":{}}


def release_fixture(root):
    """Small synthetic release; never fit the production country's data in tests."""
    d = data_fixture()
    folder = root / "data/shared/v002/unsupervised"
    folder.mkdir(parents=True)
    records = []
    for name, frame in [("country_window_features",d["features"]), ("country_window_profile",d["profile"]), ("country_window_context",d["context"])]:
        frame.to_parquet(folder/f"{name}.parquet",index=False)
        frame.to_csv(folder/f"{name}.csv",index=False,encoding="utf-8-sig",na_rep=r"\N")
        schema = [{"name":c,"dtype":str(frame[c].dtype)} for c in frame]
        for extension in ["parquet","csv"]:
            path = folder/f"{name}.{extension}"
            records.append({"path":path.relative_to(root).as_posix(), "rows":len(frame), "schema":schema,
                            "sha256":kmeans.sha256(path),"format":extension})
    windows = [{"window_id":"2013_2015","start_year":2013,"end_year":2015,"fit_eligible":4},
               {"window_id":"2025_2026Q3","start_year":2025,"end_year":2026,"start_month":"2025-01","end_month":"2026-09","short_period":True,"fit_eligible":2}]
    child = {"status":"released","shared_version":"v002","approved_for_shared_release":True,
             "feature_columns":kmeans.FEATURES,"feature_key":kmeans.KEY,"countries":4,"windows":windows,
             "shared_scaler_and_centroids_required":True,"historical_backtest_claim":False,
             "feature_rows":8,"fit_eligible_rows":6,"files":records}
    parent = {"dataset_release_approved":True,"dataset_version":"v002","country_codes":["AAA","BBB","CCC","DDD"],
              "files":records,"unsupervised":{"feature_columns":kmeans.FEATURES,"feature_rows":8,"fit_eligible_rows":6}}
    (folder/"manifest.json").write_text(json.dumps(child),encoding="utf-8")
    (folder.parent/"manifest.json").write_text(json.dumps(parent),encoding="utf-8")
    configs = root / "configs"
    (configs/"models").mkdir(parents=True)
    feature = {"schema_version":1,"clustering":{"columns":kmeans.FEATURES,"fit_scope":kmeans.FIT_SCOPE,
               "allow_mixed_years":False,"reference_year":"three_year_window","aggregation":"period_mean",
               "shared_scaler_and_centroids_across_windows":True,"scaler":"StandardScaler","context":{"trade":{"windows":windows}}}}
    for path, obj in [(configs/"models/kmeans.yaml",d["config"]), (configs/"features.yaml",feature),
                      (configs/"countries.yaml",{"schema_version":1,"countries":[{"iso3":c} for c in parent["country_codes"]]})]:
        path.write_text(yaml.safe_dump(obj),encoding="utf-8")
    return kmeans.load_inputs(root)


class InputTests(unittest.TestCase):
    def test_valid_values_and_missing_rows_are_preserved(self):
        frame, _, _ = frames()
        result = kmeans.validate_features(frame, ["AAA","BBB","CCC","DDD"], ["2013_2015","2025_2026Q3"])
        pd.testing.assert_frame_equal(result, frame)
        self.assertEqual(result.eligible_for_fit.sum(),6)
        self.assertEqual(result.internet_users_pct.isna().sum(),2)

    def test_invalid_scope_flags_values_and_missing_columns_are_rejected(self):
        original, _, _ = frames()
        cases = [original.iloc[:-1],pd.concat([original,original.iloc[:1]]),original.drop(columns=["population"])]
        for column, value in [("country_code",pd.NA),("internet_users_pct",101.),("population",0.),("gdp_per_capita_usd",np.inf)]:
            f=original.copy();f.loc[0,column]=value;cases.append(f)
        f=original.copy();f["eligible_for_fit"]=f.eligible_for_fit.astype("string");cases.append(f)
        f=original.copy();f.loc[0,"internet_users_pct"]=pd.NA;cases.append(f)
        for case in cases:
            with self.subTest(columns=list(case),rows=len(case)),self.assertRaises(kmeans.ClusterError):
                kmeans.validate_features(case,["AAA","BBB","CCC","DDD"],["2013_2015","2025_2026Q3"])

    def test_release_formats_schema_dates_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=release_fixture(root)
            self.assertEqual(data["audit"]["verified_files"],6)
            self.assertEqual(data["audit"]["csv_parquet_pairs_verified"],3)
            self.assertEqual(data["audit"]["eligible_rows"],6)
            path=root/"data/shared/v002/unsupervised/country_window_features.parquet"
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(kmeans.ClusterError,"체크섬"):
                kmeans.load_inputs(root)

    def test_explanation_join_rejects_duplicates_missing_keys_and_conflicts(self):
        f,p,_=frames()
        cases=[p.iloc[:-1],pd.concat([p,p.iloc[:1]])]
        changed=p.copy();changed.loc[0,"population"]+=1;cases.append(changed)
        for table in cases:
            with self.subTest(rows=len(table)), self.assertRaises(kmeans.ClusterError):
                kmeans.merge_explanations(f,table,"profile")


class ModelTests(unittest.TestCase):
    def test_full_country_window_grid_remains_100_rows_with_18_unclassified(self):
        rows=[]
        windows=["2013_2015","2016_2018","2019_2021","2022_2024","2025_2026Q3"]
        for i in range(20):
            for j,w in enumerate(windows):
                valid=not (j==4 and i>=2)
                rows.append({"country_code":f"C{i:02d}","window_id":w,"gdp_per_capita_usd":1000.+i*2500+j*100,
                             "population":100000.+i*900000+j*5000,"internet_users_pct":10.+i*4+j if valid else np.nan,
                             "eligible_for_fit":valid,"internet_users_pct_observed_years":"[2025]" if valid else "[]"})
        f=pd.DataFrame(rows);profile=f.copy();context=f.copy();context["observed_export_value_usd"]=10.
        data={"features":f,"profile":profile,"context":context,"config":configuration(),"source_sha256":{}}
        bundle=kmeans.fit_model(data,3)
        with patch.object(Pipeline,"fit",side_effect=AssertionError("fit forbidden")):
            result=build_results(data,bundle,"full_grid")
            for w in windows:
                eligible=result.loc[result.window_id.eq(w)&result.eligible_for_fit]
                np.testing.assert_array_equal(eligible.cluster_id.to_numpy(dtype=int),
                                              bundle["pipeline"].predict(kmeans.model_inputs(eligible)))
        self.assertEqual(len(result),100)
        self.assertEqual(result.cluster_id.notna().sum(),82)
        self.assertEqual(result.cluster_id.isna().sum(),18)
        self.assertFalse(result.duplicated(kmeans.KEY).any())
        self.assertTrue(result.loc[~result.eligible_for_fit,"internet_users_pct"].isna().all())
        self.assertTrue(result.loc[~result.eligible_for_fit,"internet_users_pct_observed_years"].eq("[]").all())
        self.assertEqual(bundle["python_version"],__import__('platform').python_version())
        self.assertEqual(bundle["libraries"]["pandas"],pd.__version__)

    def test_scaler_uses_only_eligible_rows_and_all_windows(self):
        data=data_fixture()
        data["features"].loc[~data["features"].eligible_for_fit,"population"]=1e15
        bundle=kmeans.fit_model(data,2)
        expected=data["features"].loc[data["features"].eligible_for_fit,kmeans.FEATURES].mean().to_numpy(dtype=float)
        np.testing.assert_allclose(bundle["pipeline"].named_steps["scaler"].mean_,expected)
        self.assertEqual(bundle["fit_rows"],6)
        self.assertEqual(len({r["window_id"] for r in bundle["fit_keys"]}),2)

    def test_candidate_comparison_returns_diagnostics_without_selecting_a_model(self):
        data=data_fixture();metrics,profiles=kmeans.compare_candidates(data)
        self.assertEqual(metrics.n_clusters.tolist(),[2,3,4])
        self.assertTrue(metrics.fit_rows.eq(6).all())
        self.assertTrue(metrics.silhouette.between(-1,1).all())
        self.assertEqual(profiles.groupby("n_clusters").rows.sum().tolist(),[6,6,6])
        self.assertTrue(set(kmeans.FEATURES).issubset(profiles))

    def test_invalid_cluster_counts_or_insufficient_distinct_rows_are_rejected(self):
        for k in [1,True,6,2.5]:
            with self.subTest(k=k),self.assertRaises(kmeans.ClusterError):
                kmeans.fit_model(data_fixture(),k)
        data=data_fixture()
        data["features"].loc[:,kmeans.FEATURES]=[10.,100.,20.]
        with self.assertRaises(kmeans.ClusterError):kmeans.fit_model(data,2)

    def test_full_results_preserve_missing_values_zero_partial_sums_and_labels(self):
        data=data_fixture();bundle=kmeans.fit_model(data,2)
        before=data["features"].copy(deep=True)
        with patch.object(Pipeline,"fit",side_effect=AssertionError("fit forbidden")), \
             patch.object(StandardScaler,"fit",side_effect=AssertionError("fit forbidden")), \
             patch.object(KMeans,"fit",side_effect=AssertionError("fit forbidden")):
            result=build_results(data,bundle,"query")
        self.assertEqual(len(result),8)
        self.assertEqual(result.cluster_id.notna().sum(),6)
        self.assertTrue(result.loc[~result.eligible_for_fit,"cluster_id"].isna().all())
        self.assertTrue(result.loc[~result.eligible_for_fit,"missing_features"].eq("internet_users_pct").all())
        self.assertTrue(pd.isna(result.loc[1,"export_value_usd"]))
        self.assertEqual(result.loc[0,"export_value_usd"],0.)
        self.assertEqual(result.loc[1,"observed_export_value_usd"],100.)
        pd.testing.assert_frame_equal(data["features"],before)

    def test_context_only_values_cannot_change_model_assignments(self):
        data=data_fixture();bundle=kmeans.fit_model(data,2)
        first=build_results(data,bundle,"before")
        data["context"]["export_value_usd"]=1e15
        second=build_results(data,bundle,"after")
        pd.testing.assert_series_equal(first.cluster_id,second.cluster_id)

    def test_saved_reload_is_exact_and_overwriting_is_rejected(self):
        data=data_fixture();bundle=kmeans.fit_model(data,2)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"pipeline.joblib";kmeans.save_model(bundle,file,data)
            loaded=kmeans.load_cluster_model(file,data)
            pd.testing.assert_frame_equal(build_results(data,bundle,"same"),build_results(data,loaded,"same"))
            digest=kmeans.sha256(file)
            with self.assertRaises(FileExistsError):kmeans.save_model(bundle,file,data)
            self.assertEqual(kmeans.sha256(file),digest)

    def test_wrong_model_type_order_and_data_identity_are_rejected(self):
        data=data_fixture();bundle=kmeans.fit_model(data,2)
        for field,value in [("task","supervised_regression"),("feature_columns",list(reversed(kmeans.FEATURES))),
                            ("source_sha256",{"other":"different"}),("n_clusters",5)]:
            changed=deepcopy(bundle);changed[field]=value
            with self.subTest(field=field),self.assertRaises(kmeans.ClusterError):kmeans.validate_model(changed,data)
        changed=deepcopy(bundle);changed["pipeline"]=Pipeline([("scaler",StandardScaler()),("model",KMeans(2))])
        with self.assertRaises(kmeans.ClusterError):kmeans.validate_model(changed,data)

    def test_missing_saved_metadata_and_unreadable_model_are_actionable(self):
        data=data_fixture();bundle=kmeans.fit_model(data,2)
        for key in ["model_version","source_sha256","fit_keys"]:
            changed=deepcopy(bundle);changed.pop(key)
            with self.subTest(key=key),self.assertRaisesRegex(kmeans.ClusterError,"정보 누락"):
                kmeans.validate_model(changed,data)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"pipeline.joblib";file.write_bytes(b"")
            with self.assertRaisesRegex(kmeans.ClusterError,"읽을 수 없습니다"):
                kmeans.load_cluster_model(file,data)

    def test_failed_reload_removes_only_the_newly_created_model(self):
        data=data_fixture();bundle=kmeans.fit_model(data,2)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"pipeline.joblib"
            with patch.object(kmeans,"load_cluster_model",side_effect=kmeans.ClusterError("reload failed")):
                with self.assertRaises(kmeans.ClusterError):kmeans.save_model(bundle,file,data)
            self.assertFalse(file.exists())


class ExecutionTests(unittest.TestCase):
    def test_audit_and_missing_model_never_fit_or_auto_create_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release_fixture(root)
            with patch.object(Pipeline,"fit",side_effect=AssertionError("fit forbidden")):
                result=kmeans.run_cluster(mode="audit",root=root,run_id="audit")
                self.assertFalse(result["metadata"]["model_fitted"])
                with self.assertRaises(FileNotFoundError):kmeans.run_cluster(mode="predict",root=root,run_id="missing")
            self.assertFalse((root/"artifacts").exists())
            self.assertFalse((root/".private/kmeans/missing").exists())

    def test_train_and_prediction_readback_and_private_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release_fixture(root)
            trained=kmeans.run_cluster(mode="train",root=root,run_id="train",n_clusters=2)
            model=root/"artifacts/shared_models/kmeans/pipeline.joblib"
            digest=kmeans.sha256(model)
            with patch.object(Pipeline,"fit",side_effect=AssertionError("fit forbidden")), \
                 patch.object(KMeans,"fit",side_effect=AssertionError("fit forbidden")):
                query=kmeans.run_cluster(mode="predict",root=root,run_id="query")
            self.assertFalse(query["metadata"]["model_fitted"])
            self.assertEqual(kmeans.sha256(model),digest)
            pd.testing.assert_frame_equal(trained["results"].drop(columns="run_id"),query["results"].drop(columns="run_id"))
            readback=pd.read_parquet(query["output_dir"]/"country_window_clusters.parquet")
            pd.testing.assert_frame_equal(readback,query["results"])
            self.assertEqual(sorted(p.name for p in model.parent.iterdir()),["pipeline.joblib"])
            with self.assertRaises(FileExistsError):kmeans.run_cluster(mode="predict",root=root,run_id="query")

    def test_invalid_modes_paths_and_unselected_training_are_rejected(self):
        for kwargs in [{"mode":"train"},{"mode":"predict","n_clusters":2},{"mode":"unknown"}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(kmeans.ClusterError):kmeans.run_cluster(**kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release_fixture(root)
            for kwargs in [{"mode":"audit","run_id":"../escape"},
                           {"mode":"train","run_id":"bad","n_clusters":2,"model_path":"data/shared/model.joblib"}]:
                with self.subTest(kwargs=kwargs),self.assertRaises(kmeans.ClusterError):kmeans.run_cluster(root=root,**kwargs)

    def test_modified_shared_sources_are_detected_after_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=release_fixture(root)
            p=root/"data/shared/v002/unsupervised/country_window_context.csv"
            p.write_bytes(p.read_bytes()+b"change")
            with self.assertRaises(kmeans.ClusterError):kmeans.check_sources(data)


if __name__ == "__main__":
    unittest.main()
