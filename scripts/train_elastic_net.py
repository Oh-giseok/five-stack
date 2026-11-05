"""Train ElasticNet using released v002 data and expanding-origin validation."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Unique .private/elastic_net/<run_id> folder")
    parser.add_argument("--max-origins", type=int, help="Smoke validation only; does not test/publish")
    args = parser.parse_args()
    from oinq.modeling.supervised.elastic_net import run_training
    out = run_training(ROOT, run_id=args.run_id, max_origins=args.max_origins)
    print(f"Private reports: {out}")


if __name__ == "__main__":
    main()
