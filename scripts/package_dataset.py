"""Package the user-approved supervised dataset or verify the shared files."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oinq.preprocessing.release import ReleaseError, package_dataset, verify_shared
from oinq.common.config import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--version", help="Verification version; defaults to configs/release.yaml")
    args = parser.parse_args()
    try:
        if args.verify_only:
            version = args.version or load_yaml(ROOT / "configs/release.yaml")["shared_version"]
            if not version.isalnum():
                parser.error("version must be alphanumeric")
            result = verify_shared(ROOT, version)
            print(f"공유 데이터 검증 통과: {result['dataset_version']} / {result['row_counts']}")
        else:
            package_dataset(ROOT)
    except (ReleaseError, OSError) as exc:
        print(f"공유 데이터 처리 실패: {exc}", file=sys.stderr)
        raise SystemExit(1)
