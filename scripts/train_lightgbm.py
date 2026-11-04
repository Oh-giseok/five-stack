"""Run the LightGBM-only documented initial experiment."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Unique artifacts/<run_id> directory; never overwritten")
    parser.add_argument("--max-origins", type=int, help="Smoke run only; first N validation origins")
    parser.add_argument("--reg-alpha", type=float, help="Run-local L1 regularization strength; shared YAML unchanged")
    parser.add_argument("--reg-lambda", type=float, help="Run-local L2 regularization strength; shared YAML unchanged")
    args = parser.parse_args()
    from oinq.modeling.supervised.lightgbm import run_validation
    run_validation(ROOT, run_id=args.run_id, max_origins=args.max_origins,
                   reg_alpha=args.reg_alpha, reg_lambda=args.reg_lambda)

if __name__ == "__main__":
    main()
