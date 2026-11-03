"""Prove that validation expansion only reassigns existing candidate rows."""

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.collectors.raw import utc_now, write_json
from oinq.preprocessing.customs import sha256


def audit():
    before = ROOT / "data/processed/supervised/v001"
    after = ROOT / "data/processed/supervised/v002"
    manifest = json.loads((after / "manifest.json").read_text(encoding="utf-8"))
    keys = ["country_code", "hs_code", "origin_quarter", "target_quarter"]
    original = {name: pd.read_parquet(before / f"{name}_candidates.parquet") for name in ("train", "validation")}
    revised = {name: pd.read_parquet(after / f"{name}_candidates.parquet") for name in ("train", "validation")}

    def observations(frame):
        return frame.drop(columns="split").sort_values(keys).reset_index(drop=True)

    old_all = pd.concat(original.values(), ignore_index=True)
    new_all = pd.concat(revised.values(), ignore_index=True)
    assert len(old_all) == len(new_all) == 41600
    assert not old_all.duplicated(keys).any() and not new_all.duplicated(keys).any()
    pd.testing.assert_frame_equal(observations(old_all), observations(new_all), check_exact=True)
    moved = original["train"].loc[original["train"].target_quarter.between("2022Q1", "2023Q4")]
    current_moved = revised["validation"].loc[revised["validation"].target_quarter.between("2022Q1", "2023Q4")]
    pd.testing.assert_frame_equal(observations(moved), observations(current_moved), check_exact=True)
    assert len(moved) == 6400
    pd.testing.assert_frame_equal(observations(original["train"].loc[original["train"].target_quarter.le("2021Q4")]),
                                  observations(revised["train"]), check_exact=True)
    pd.testing.assert_frame_equal(observations(original["validation"]),
                                  observations(revised["validation"].loc[revised["validation"].target_quarter.ge("2024Q1")]), check_exact=True)
    unchanged = []
    for name in ("integrated_monthly", "integrated_quarterly", "origin_feature_candidates"):
        for extension in ("csv", "parquet"):
            filename = f"{name}.{extension}"
            assert sha256(before / filename) == sha256(after / filename), f"Unintended change: {filename}"
            unchanged.append(filename)
    for extension in ("csv", "parquet"):
        filename = f"final_test_candidates.{extension}"
        assert sha256(ROOT / ".private/evaluation/supervised/v001" / filename) == sha256(ROOT / ".private/evaluation/supervised/v002" / filename)
        unchanged.append(filename)
    shared = json.loads((ROOT / "data/shared/v001/manifest.json").read_text(encoding="utf-8"))
    for name in ("train", "validation", "final_test"):
        assert {k: shared[f"{name}_period"][k] for k in ("start", "end")} == manifest[f"{name}_period"]
    assert shared["train_period"]["quarters"] == 36 and shared["validation_period"]["quarters"] == 16
    assert shared["candidate_source_version"] == "v002"
    assert shared["candidate_manifest_path"] == (after / "manifest.json").relative_to(ROOT).as_posix()
    shared_released = shared["status"] == "released"
    if shared_released:
        from oinq.preprocessing.release import verify_shared
        verify_shared(ROOT, shared["dataset_version"])
        assert shared["dataset_release_approved"] is True
    else:
        assert shared["status"] == "not_generated" and shared["files"] == []
        assert shared["shared_release_approved"] is False
    assert shared["final_test_included"] is False
    result = {"status": "passed", "audited_at": utc_now(), "previous_version": "v001", "current_version": "v002",
              "old_train_rows": len(original["train"]), "new_train_rows": len(revised["train"]),
              "old_validation_rows": len(original["validation"]), "new_validation_rows": len(revised["validation"]),
              "moved_from_train_to_validation": len(moved), "combined_candidate_rows_before": len(old_all),
              "combined_candidate_rows_after": len(new_all), "dropped_or_added_sample_keys": 0,
              "candidate_values_unchanged_except_split": True, "unchanged_file_checksums": unchanged,
              "shared_json_periods_match_actual_candidates": True, "shared_data_released": shared_released,
              "private_final_test_unchanged": True}
    write_json(ROOT / "reports/supervised_v002_split_revision_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    audit()
