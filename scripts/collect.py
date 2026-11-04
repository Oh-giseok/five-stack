"""Fetch raw source responses only; never normalize or train here."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oinq.common.config import load_project_config, validate_project_config
from oinq.collectors.raw import AccessError, RawClient, collect_comtrade, collect_customs, collect_world_bank, probe
from oinq.collectors.audit import audit_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["probe", "world_bank", "customs", "comtrade", "audit"], default="probe")
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.snapshot):
        parser.error("snapshot must be a simple directory name")
    config = load_project_config(ROOT)
    validate_project_config(config)
    if args.stage == "audit":
        summary = audit_raw(ROOT, config, args.snapshot)
        return int(summary["status"] not in {"audited", "audited_with_gaps"})
    client = RawClient(ROOT, config, args.snapshot)
    try:
        if args.stage == "probe":
            results = probe(client)
            return int(any(row["status"] != "success" for row in results.values()))
        {"world_bank": collect_world_bank, "customs": collect_customs, "comtrade": collect_comtrade}[args.stage](client)
        return 0
    except AccessError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
