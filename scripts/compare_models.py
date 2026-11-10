"""Generate a common performance table from recorded predictions; never train."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.evaluation.comparison import build_comparison, save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="JSON source mappings and checksums")
    parser.add_argument("--output-dir", type=Path, required=True, help="New result directory")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")
    report = build_comparison(ROOT, args.config)
    save_report(report, args.output_dir)
    print(f"평가 완료 {report.metadata['evaluated_configurations']}개 / 전체 목록 {len(report.table)}개")
    print(report.table.to_string(index=False, na_rep="—", float_format=lambda number: f"{number:.4f}"))
    print(f"결과: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
