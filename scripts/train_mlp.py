"""MLP smoke checks only. Full validation/test/forecast are not enabled yet."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true', help='Synthetic rows, temporary model only')
    args = parser.parse_args()
    if not args.smoke:
        parser.error('Only --smoke is authorized; full validation/test/forecast are disabled')
    from oinq.modeling.supervised.mlp import smoke_check
    try:
        result = smoke_check(ROOT)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as error:
        parser.exit(1, f'MLP smoke failed: {error}\n')
    if result['status'] != 'passed':
        parser.exit(2, 'MLP did not converge; no completed model was published.\n')


if __name__ == '__main__':
    main()
