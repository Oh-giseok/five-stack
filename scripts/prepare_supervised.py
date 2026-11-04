"""Integrate source panels and prepare supervised candidates; no model fitting."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.common.config import load_project_config, validate_project_config
from oinq.preprocessing.supervised import SupervisedDataError, prepare_supervised


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.version):
        parser.error("version must be a simple directory name")
    config = load_project_config(ROOT)
    validate_project_config(config)
    try:
        prepare_supervised(ROOT, config, args.version)
    except (SupervisedDataError, OSError) as exc:
        print(f"지도학습 통합 실패: {exc}", file=sys.stderr)
        raise SystemExit(1)
