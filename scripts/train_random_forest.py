"""Train and evaluate Random Forest on the shared v002 partitions."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.modeling.supervised.random_forest import run_training


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Unique name for private evaluation outputs")
    parser.add_argument("--no-publish", action="store_true",
                        help="Keep fitted pipelines in the private run directory")
    args = parser.parse_args()
    print(run_training(ROOT, args.run_id, publish=not args.no_publish))
