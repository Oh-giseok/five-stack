"""Prepare the explicitly selected source only; no cross-source joins or training."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config, validate_project_config
from oinq.preprocessing.customs import CustomsDataError, prepare_customs
from oinq.preprocessing.comtrade import ComtradeDataError, prepare_comtrade
from oinq.preprocessing.world_bank import WorldBankDataError, prepare_world_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["customs", "comtrade", "world_bank"], required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    for value in (args.snapshot, args.version):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            parser.error("snapshot/version must be a simple directory name")
    config = load_project_config(ROOT)
    validate_project_config(config)
    try:
        prepare = {"customs": prepare_customs, "comtrade": prepare_comtrade, "world_bank": prepare_world_bank}[args.source]
        prepare(ROOT, config, args.snapshot, args.version)
    except (CustomsDataError, ComtradeDataError, WorldBankDataError, OSError) as exc:
        print(f"{args.source} 전처리 실패: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
