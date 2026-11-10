"""Compare recorded predictions without importing or fitting any model.

Source locations and legacy column mappings belong to the caller's JSON config.
The public module contains no machine-specific evaluation locations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["country_code", "hs_code", "origin_quarter", "target_quarter"]
TARGET = "target_export_value_usd"
PRED = "predicted_export_value_usd"
COLUMNS = ["모델명", "실험 구성", "거시지표", "테스트 R² ↑", "검증 WAPE (%) ↓",
           "테스트 WAPE (%) ↓", "테스트 MAE (백만 USD) ↓", "테스트 RMSE (백만 USD) ↓",
           "평가 건수", "평가 상태"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    text = frame[KEY].sort_values(KEY).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def quarter_index(values: pd.Series) -> pd.Series:
    parts = values.astype("string").str.extract(r"^(\d{4})Q([1-4])$")
    if parts.isna().any().any():
        raise ValueError("Invalid quarter label")
    return parts[0].astype(int) * 4 + parts[1].astype(int) - 1


def metrics(actual, prediction) -> dict:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if actual.shape != prediction.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("Metrics require nonempty, equal-length vectors")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite metric input")
    error = prediction - actual
    denominator = np.abs(actual).sum()
    variance = np.square(actual - actual.mean()).sum()
    return {"n": len(actual), "wape": float(np.abs(error).sum() / denominator * 100) if denominator else None,
            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": float(1 - np.square(error).sum() / variance) if len(actual) > 1 and variance else None}


def check_predictions(frame: pd.DataFrame, expected: pd.DataFrame, *, full_grid: bool) -> pd.DataFrame:
    """Require identical scored keys/actuals; test records must also keep missing labels."""
    needed = KEY + ["horizon_quarters", TARGET, PRED]
    if not set(needed).issubset(frame.columns):
        raise ValueError(f"Missing prediction columns: {set(needed) - set(frame.columns)}")
    frame = frame.copy()
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError("Null or duplicate prediction keys")
    for col in [TARGET, PRED, "horizon_quarters"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[PRED]).all() or frame[PRED].lt(0).any():
        raise ValueError("Missing, nonfinite or negative scored prediction")
    for col, value in [("currency", "USD"), ("valuation", "FOB"), ("dataset_version", "v002")]:
        if col in frame and not frame[col].eq(value).all():
            raise ValueError(f"Different {col}")
    for raw in ["raw_predicted_usd", "raw_predicted_export_value_usd"]:
        if raw in frame and not np.array_equal(np.maximum(frame[raw].to_numpy(float), 0), frame[PRED].to_numpy(float)):
            raise ValueError("Prediction postprocessing differs from nonnegative clipping")
    joined = frame.merge(expected[KEY + [TARGET, "horizon_quarters"]], on=KEY,
                         how="outer" if full_grid else "left", suffixes=("", "_reference"),
                         validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Evaluation keys differ from common data")
    if not np.array_equal(joined[TARGET].to_numpy(float), joined[TARGET + "_reference"].to_numpy(float), equal_nan=True):
        raise ValueError("Evaluation actuals differ from common data")
    if not joined.horizon_quarters.eq(joined.horizon_quarters_reference).all():
        raise ValueError("Evaluation horizons differ from common data")
    available = frame[TARGET].notna()
    if "target_available" in frame:
        flags = frame.target_available.map({True: True, False: False, "True": True, "False": False})
        if flags.isna().any() or not flags.eq(available).all():
            raise ValueError("Label availability differs from common data")
    scored = frame.loc[available].sort_values(KEY).reset_index(drop=True)
    reference = expected.loc[expected[TARGET].notna()].sort_values(KEY).reset_index(drop=True)
    if frame_hash(scored) != frame_hash(reference):
        raise ValueError("Scored evaluation keys differ from common data")
    return scored


@dataclass
class ComparisonReport:
    table: pd.DataFrame
    records: list[dict]
    supplementary: list[dict]
    metadata: dict


def performance_table(records: list[dict]) -> pd.DataFrame:
    """Format ordered evaluation records; keep full USD precision in records."""
    rows = []
    for record in records:
        val, score = record["validation"], record["test"]
        values = [record["model_name"], record["experiment"], "포함" if record["variant"] == "with_macro" else "제외",
                  score["r2"] if score else None, val["wape"] if val else None,
                  score["wape"] if score else None, score["mae"] / 1e6 if score else None,
                  score["rmse"] / 1e6 if score else None,
                  f"{val['n']:,} / {score['n']:,}" if score else "—", record["status"]]
        rows.append(dict(zip(COLUMNS, values)))
    return pd.DataFrame(rows, columns=COLUMNS)


class RecordReader:
    def __init__(self, root: Path, checksums: dict):
        self.root = root
        self.checksums = checksums
        self.frames: dict[str, pd.DataFrame] = {}
        self.used: dict[str, str] = {}

    def path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if name not in self.checksums:
            raise ValueError(f"Source checksum not configured: {name}")
        if name not in self.used:
            digest = sha256(path)
            if digest != self.checksums[name]:
                raise ValueError(f"Source checksum changed: {name}")
            self.used[name] = digest
        return path

    def json(self, name: str):
        return json.loads(self.path(name).read_text(encoding="utf-8-sig"))

    def frame(self, source: dict) -> pd.DataFrame:
        name = source["path"]
        if name not in self.frames:
            path = self.path(name)
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
            elif path.suffix == ".csv":
                frame = pd.read_csv(path, dtype={key: "string" for key in KEY},
                                    na_values=[r"\N"], keep_default_na=True, low_memory=False)
            else:
                raise ValueError(f"Unsupported record type: {path.suffix}")
            self.frames[name] = frame
        frame = self.frames[name]
        for col, value in source.get("select", {}).items():
            frame = frame.loc[frame[col].eq(value)]
        frame = frame.rename(columns=source.get("rename", {})).copy()
        for key in KEY:
            if key in frame:
                frame[key] = frame[key].astype("string")
        return frame


def _fit_pool(pool, origin):
    valid = pool[TARGET].notna() & pool.target_available.fillna(False)
    return pool.loc[valid & quarter_index(pool.target_quarter).le(quarter_index(pd.Series([origin])).iloc[0])]


def _audit(frame, fits):
    if frame.origin_quarter.isna().any() or frame.origin_quarter.duplicated().any() or set(frame.origin_quarter) != set(fits):
        raise ValueError("Missing or duplicate validation fit audit")
    for row in frame.itertuples():
        rows, digest = fits[row.origin_quarter]
        if row.fit_rows != rows or row.fit_rows_sha256 != digest or row.max_fit_target > row.origin_quarter:
            raise ValueError("Validation fit rows differ from common data")
        for col in ["fit_labels_after_origin", "future_labels_in_fit"]:
            if hasattr(row, col) and getattr(row, col) != 0:
                raise ValueError("Future labels in validation fit")


def _check_recorded(reader, spec, calculated):
    recorded = reader.frame(spec)
    if len(recorded) != 1:
        raise ValueError("Recorded metric selector must identify exactly one row")
    for col in ["n", "wape", "mae", "rmse", "r2"]:
        if col in recorded:
            value = recorded.iloc[0][col]
            expected = calculated[col]
            if expected is None:
                if not pd.isna(value):
                    raise ValueError(f"Recorded metric mismatch: {col}")
            elif not np.isclose(float(value), expected, rtol=1e-10, atol=1e-6):
                raise ValueError(f"Recorded metric mismatch: {col}")


def build_comparison(root: Path | str, config_path: Path | str) -> ComparisonReport:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1 or config.get("dataset_version") != "v002":
        raise ValueError("Unsupported comparison configuration")
    if "clustering" in config:
        raise ValueError("Clustering is outside the export-value model comparison scope")
    reader = RecordReader(root, config["source_checksums"])
    folder = config["dataset_dir"].rstrip("/")
    manifest = reader.json(folder + "/manifest.json")
    if manifest["dataset_version"] != "v002":
        raise ValueError("Different common data version")
    frames = {split: reader.frame({"path": f"{folder}/{split}.parquet"})
              for split in ["train", "validation", "final_test"]}
    for split, frame in frames.items():
        if frame.duplicated(KEY).any() or frame[KEY].isna().any().any():
            raise ValueError(f"Invalid common data keys: {split}")
        if not frame.target_available.eq(frame[TARGET].notna()).all():
            raise ValueError(f"Invalid common label availability: {split}")
        distance = quarter_index(frame.target_quarter) - quarter_index(frame.origin_quarter)
        if not distance.eq(frame.horizon_quarters).all():
            raise ValueError(f"Invalid common horizons: {split}")
        if "currency" in frame and not frame.currency.eq("USD").all():
            raise ValueError("Common currency differs")
    validation, test = frames["validation"], frames["final_test"]
    if (frames["train"].target_quarter.min(), frames["train"].target_quarter.max()) != ("2013Q1", "2021Q4"):
        raise ValueError("Different training period")
    if (validation.target_quarter.min(), validation.target_quarter.max()) != ("2022Q1", "2025Q4"):
        raise ValueError("Different validation period")
    if set(test.origin_quarter) != {"2025Q4"} or set(test.target_quarter) != {"2026Q1", "2026Q2", "2026Q3"}:
        raise ValueError("Different final-test period")
    if set(validation.horizon_quarters) != {1, 2, 3, 4, 5} or len(test) != 480 or test[TARGET].notna().sum() != 447:
        raise ValueError("Different evaluation coverage")
    primary = validation.loc[validation.horizon_quarters.isin([2, 3, 4, 5])]
    if primary[TARGET].notna().sum() != 9164:
        raise ValueError("Different primary validation coverage")
    pool = pd.concat([frames["train"], validation], ignore_index=True)
    if pool.duplicated(KEY).any():
        raise ValueError("Training and validation keys overlap")
    fits = {}
    for origin in sorted(validation.origin_quarter.unique()):
        fit = _fit_pool(pool, origin)
        fits[origin] = (len(fit), frame_hash(fit))
    final_fit = _fit_pool(pool, "2025Q4")
    seen, records = set(), []
    for model in config["models"]:
        identity = (model["model_id"], model["experiment"], model["variant"])
        if identity in seen or model["variant"] not in {"with_macro", "without_macro"}:
            raise ValueError(f"Duplicate or invalid model entry: {identity}")
        seen.add(identity)
        result = {key: model[key] for key in ["model_id", "model_name", "experiment", "variant"]}
        result.update(status=model["status"], validation=None, test=None,
                      detail=model.get("detail", ""), model_path=model.get("model_path"),
                      model_sha256=None, trained_through_target_quarter=None)
        if model["status"] == "평가 완료":
            _audit(reader.frame(model["fit_audit"]), fits)
            meta = reader.json(model["test_metadata"]["path"])
            keys = model["test_metadata"]
            if meta[keys["fit_max_target"]] != "2025Q4" or meta[keys["fit_rows"]] != len(final_fit) or meta[keys["fit_hash"]] != frame_hash(final_fit):
                raise ValueError("Final-test model fit differs from frozen 2025Q4 pool")
            val = reader.frame(model["validation_predictions"])
            # Legacy files may contain all horizons; detect duplicates before selecting h=2~5.
            if val.duplicated(KEY).any():
                raise ValueError("Duplicate validation predictions")
            val = val.loc[val.horizon_quarters.isin([2, 3, 4, 5])]
            val = check_predictions(val, primary, full_grid=False)
            raw_test = reader.frame(model["test_predictions"])
            scored_test = check_predictions(raw_test, test, full_grid=True)
            result["validation"] = metrics(val[TARGET], val[PRED])
            result["test"] = metrics(scored_test[TARGET], scored_test[PRED])
            result["test_prediction_rows"] = len(raw_test)
            result["test_missing_targets"] = len(raw_test) - len(scored_test)
            result["validation_keys_sha256"] = frame_hash(val)
            result["test_keys_sha256"] = frame_hash(scored_test)
            _check_recorded(reader, model["validation_metrics"], result["validation"])
            _check_recorded(reader, model["test_metrics"], result["test"])
            reader.path(model["model_path"])
            result["trained_through_target_quarter"] = "2025Q4"
        elif model["status"] == "운영용·비교 제외":
            reader.path(model["model_path"])
            result["trained_through_target_quarter"] = model.get("trained_through_target_quarter")
        elif model["status"] != "미평가":
            raise ValueError(f"Unsupported model status: {model['status']}")
        if model.get("model_path"):
            reader.path(model["model_path"])
            result["model_sha256"] = reader.used[model["model_path"]]
        records.append(result)
    records.sort(key=lambda row: (row["validation"] is None,
                                 row["validation"]["wape"] if row["validation"] and row["validation"]["wape"] is not None else float("inf"),
                                 row["model_id"], row["experiment"], row["variant"]))
    supplementary = []
    if "baseline" in config:
        baseline = config["baseline"]
        result = {"model_name": "Seasonal Naive", "status": "비교 조건 다름"}
        for split, expected in [("validation", primary), ("test", test)]:
            frame = reader.frame(baseline[split])
            frame = frame.loc[frame.horizon_quarters.isin([2, 3, 4, 5] if split == "validation" else [1, 2, 3])].copy()
            if frame.duplicated(KEY).any():
                raise ValueError("Duplicate baseline keys")
            # Check the baseline grid/actuals independently of its missing predictions.
            grid = frame.drop(columns=["raw_predicted_usd", "raw_predicted_export_value_usd"], errors="ignore")
            # The legacy baseline has an entirely empty availability column;
            # its actuals and scored keys are still checked against common data.
            if "target_available" in grid and grid.target_available.isna().all():
                grid = grid.drop(columns="target_available")
            check_predictions(grid.assign(**{PRED: 0.0}), expected, full_grid=split == "test")
            usable = frame[TARGET].notna() & frame[PRED].notna()
            if (frame.loc[usable, PRED] < 0).any():
                raise ValueError("Negative baseline prediction")
            result[split] = metrics(frame.loc[usable, TARGET], frame.loc[usable, PRED])
            result[split + "_excluded_missing_predictions"] = int((frame[TARGET].notna() & frame[PRED].isna()).sum())
        supplementary.append(result)
    # Reject source changes during evaluation as well as changes before loading.
    for name, digest in reader.used.items():
        if sha256((root / name).resolve()) != digest:
            raise ValueError(f"Source changed during comparison: {name}")
    metadata = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_version": "v002", "validation_period": ["2022Q1", "2025Q4"], "primary_horizons": [2, 3, 4, 5],
                "test_origin": "2025Q4", "test_period": ["2026Q1", "2026Q3"], "test_horizons": [1, 2, 3],
                "internal_currency": "USD", "valuation": "FOB", "display_error_unit": "million USD",
                "sort_by": "validation WAPE ascending", "historical_asof_verified": False,
                "test_labels_previously_shared": True, "training_performed": False,
                "model_selection_changed": False, "sources_unchanged": True,
                "limitations": ["현재 수정치 기반으로 과거 공개시점은 미검증", "테스트 정답은 이전에 공유·평가됨",
                                "검증 h=2~5와 최종 테스트 h=1~3은 구분", "거시지표 포함·제외 차이는 인과효과가 아님"],
                "experiment_notes": ["XGBoost 기본 및 L1 구성은 L2=1을 포함", "CatBoost 기본은 L2=3"],
                "evaluated_configurations": sum(row["validation"] is not None for row in records),
                "listed_configurations": len(records), "source_checksums": reader.used,
                "config_path": str(config_path), "config_sha256": sha256(config_path)}
    return ComparisonReport(performance_table(records), records, supplementary, metadata)


def save_report(report: ComparisonReport, output_dir: Path | str) -> None:
    """Create a new directory only: never overwrite previous evaluation results."""
    output_dir = Path(output_dir)
    payload = {"metadata": report.metadata, "models": report.records, "supplementary": report.supplementary}
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    csv = report.table.to_csv(index=False, na_rep="—", lineterminator="\n")
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.csv").write_text(csv, encoding="utf-8-sig")
    (output_dir / "comparison.json").write_text(encoded, encoding="utf-8")
