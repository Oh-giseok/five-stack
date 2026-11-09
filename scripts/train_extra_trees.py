"""Run immutable-contract Extra Trees experiments without changing common code."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=['all', 'depth', 'leaf', 'features', 'pruned'], default='all')
    parser.add_argument('--run-id')
    parser.add_argument('--no-publish', action='store_true')
    parser.add_argument('--no-forecast', action='store_true')
    parser.add_argument('--no-notebooks', action='store_true')
    args = parser.parse_args()
    from oinq.modeling.supervised.extra_trees import EXPERIMENTS, run_training
    try:
        out = run_training(ROOT, args.run_id, EXPERIMENTS if args.experiment=='all' else (args.experiment,),
            publish=not args.no_publish, forecast=not args.no_forecast, notebooks=not args.no_notebooks)
    except Exception as error:
        parser.exit(1, f'Extra Trees 실행 실패: {error}\n상세 오류: .private/extra_trees/<run-id>/error.log\n')
    print(f'완료: {out}', flush=True)


if __name__ == '__main__':
    main()
