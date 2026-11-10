"""Discover saved regressors and reuse or create private inference runs."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from oinq.common.config import load_yaml
from oinq.ui.data import discover_forecast_runs


MODEL_LABELS = {"random_forest": "Random Forest", "elastic_net": "ElasticNet",
                "xgboost_l2": "XGBoost L2", "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost",
                "linear_regression": "Linear Regression (2026Q3 학습)",
                "linear_regression_test_2025Q4": "Linear Regression (2025Q4 학습)",
                "ridge": "Ridge (2026Q3 학습)", "ridge_test_2025Q4": "Ridge (2025Q4 학습)",
                "mlp_64_32": "MLP (64, 32)"}
VARIANT_LABELS = {"with_macro": "거시지표 포함", "without_macro": "거시지표 제외"}


def saved_forecast_models(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    settings = load_yaml(root / "configs/engine.yaml")
    folder = root / settings["paths"]["models_dir"]
    models = {}
    for family in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not family.is_dir():
            continue
        for variant in VARIANT_LABELS:
            files = [family / variant / name for name in ("pipeline.joblib", "model.cbm")
                     if (family / variant / name).is_file()]
            if len(files) > 1:
                raise ValueError(f"구성별 모델 파일이 둘 이상입니다: {family.name}/{variant}")
            if files:
                models[f"{family.name}/{variant}"] = files[0]
    return models


def matching_forecast_runs(root: str | Path, model_key: str, model_path: Path) -> list[str]:
    """Never offer another model's run or an obsolete saved-model revision."""
    root = Path(root)
    model_id, variant = model_key.split("/")
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    runs = []
    for run_id in discover_forecast_runs(root):
        try:
            metadata = json.loads((root / ".private/forecasts" / run_id / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (metadata.get("status") == "completed" and metadata.get("model_id") == model_id
                and metadata.get("model_variant") == variant and metadata.get("model_sha256") == digest
                and metadata.get("dataset_version") == "v002" and metadata.get("origin_quarter") == "2026Q3"
                and metadata.get("target_quarters") == [f"2027Q{i}" for i in range(1, 5)]):
            runs.append((str(metadata.get("generated_at_utc", "")), run_id))
    return [run_id for _, run_id in sorted(runs, reverse=True)]


def create_forecast_run(root: str | Path, model_key: str) -> str:
    """Run the existing saved-model engine on demand; never import training code."""
    from oinq.forecasting.engine import run_forecast

    root = Path(root)
    if model_key not in saved_forecast_models(root):
        raise ValueError("공유 폴더에서 선택한 저장 모델을 찾을 수 없습니다.")
    model_id, variant = model_key.split("/")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = run_forecast(root=root, model_id=model_id, variant=variant, origin_quarter="2026Q3",
                          output_dir=root / ".private/forecasts", run_id=f"ui_{model_id}_{variant}_{timestamp}")
    return result["metadata"]["run_id"]


def saved_training_end(model_path: str | Path) -> str | None:
    """New teammates' joblib bundles embed their actual training cutoff."""
    import joblib

    saved = joblib.load(model_path)
    return saved.get("trained_through_target_quarter") if isinstance(saved, dict) else None
