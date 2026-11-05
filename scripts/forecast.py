"""Run saved-model quarterly forecasts without retraining."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/engine.yaml")
    parser.add_argument("--model", help="Model/experiment folder below artifacts/shared_models")
    parser.add_argument("--variant", choices=["with_macro", "without_macro"])
    parser.add_argument("--origin", help="Latest observation quarter, e.g. 2026Q3")
    parser.add_argument("--features", help="Origin-feature Parquet/CSV, including a prior run's origin_features.parquet")
    parser.add_argument("--output-dir", help="Project-local output directory; defaults to .private/forecasts")
    parser.add_argument("--run-id", help="Unique output subdirectory; existing runs are never overwritten")
    args = parser.parse_args()
    from oinq.forecasting.engine import run_forecast
    try:
        run = run_forecast(config_path=args.config, model_id=args.model, variant=args.variant,
                           origin_quarter=args.origin, features_path=args.features, output_dir=args.output_dir,
                           run_id=args.run_id, root=ROOT)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"예측 실행 실패: {error}\n")
    info = run["metadata"]
    print(f"완료: {info['model_id']}/{info['model_variant']} | {info['origin_quarter']} → {', '.join(info['target_quarters'])}")
    print(f"분기 {info['quarterly_rows']}행 / 연간 {info['annual_rows']}행 / 예측 {info['predicted_rows']}행 / 입력 결측 포함 {info['missing_input_rows']}행")
    print(f"저장: {run['output_dir']} (재학습 없음)")


if __name__ == "__main__":
    main()
