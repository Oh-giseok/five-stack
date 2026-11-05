"""Build full country/window results without fitting or imputing missing values."""
import numpy as np
import pandas as pd

from oinq.modeling.clustering.kmeans import (
    KEY, FEATURES, merge_explanations, model_inputs, require, validate_features, validate_model,
)


def build_results(data, bundle, run_id):
    pipeline = validate_model(bundle, data)
    frame = validate_features(data["features"])
    reserved = {"cluster_id", "cluster_status", "missing_features", "model_id", "model_version", "run_id", "dataset_version"}
    require(not reserved.intersection(set(frame) | set(data["profile"]) | set(data["context"])), "원천 자료에 군집 결과 전용 열이 포함됐습니다.")
    result = merge_explanations(merge_explanations(frame, data["profile"], "profile"), data["context"], "context")
    eligible = result.eligible_for_fit
    result["cluster_id"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if eligible.any():
        predicted = np.asarray(pipeline.predict(model_inputs(result.loc[eligible])))
        require(predicted.ndim == 1 and len(predicted) == int(eligible.sum()) and np.isfinite(predicted).all()
                and np.equal(predicted, np.floor(predicted)).all()
                and ((predicted >= 0) & (predicted < bundle["n_clusters"])).all(), "유효하지 않은 군집 ID가 반환됐습니다.")
        result.loc[eligible, "cluster_id"] = predicted.astype(int)
    missing = result[FEATURES].isna()
    result["missing_features"] = missing.apply(lambda row: ",".join(c for c in FEATURES if row[c]), axis=1).astype("string")
    result["cluster_status"] = pd.Series(np.where(eligible, "classified",
                                                    np.where(missing.any(axis=1), "unclassified_missing_features", "unclassified_ineligible")),
                                         index=result.index, dtype="string")
    result["model_id"] = "kmeans"
    result["model_version"] = bundle["model_version"]
    result["dataset_version"] = bundle["dataset_version"]
    result["run_id"] = run_id
    require(len(result) == len(frame) and not result.duplicated(KEY).any(), "군집 결과의 행·키가 원천과 다릅니다.")
    return result.sort_values(KEY).reset_index(drop=True)
