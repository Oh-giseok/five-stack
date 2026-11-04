"""Build three-year country features and local historical context without training."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config, validate_project_config
from oinq.preprocessing.unsupervised import prepare_unsupervised, UnsupervisedDataError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--as-of-date", required=True, help="Client date YYYY-MM-DD, recorded for future/ongoing status")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.version):
        parser.error("version must be a simple directory name")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.as_of_date):
        parser.error("as-of-date must be YYYY-MM-DD")
    config = load_project_config(ROOT)
    validate_project_config(config)
    try:
        prepare_unsupervised(ROOT, config, args.version, args.as_of_date)
    except (UnsupervisedDataError, OSError, ValueError) as exc:
        print(f"비지도학습 전처리 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
