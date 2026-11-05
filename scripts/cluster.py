"""Audit, compare, train, or query the shared country-window clustering model."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["audit", "compare", "train", "predict"])
    parser.add_argument("--run-id", help="Unique .private/kmeans/<run_id> folder; never overwritten")
    parser.add_argument("--n-clusters", type=int, help="Required explicit choice in train mode, after candidate review")
    parser.add_argument("--model-path", help="Saved cluster Pipeline; defaults to artifacts/shared_models/kmeans/pipeline.joblib")
    args = parser.parse_args(argv)
    from oinq.modeling.clustering.kmeans import run_cluster
    try:
        run = run_cluster(mode=args.mode, run_id=args.run_id, n_clusters=args.n_clusters,
                          model_path=args.model_path, root=ROOT)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.exit(2, f"군집 실행 실패: {error}\n")
    info = run["metadata"]
    print(f"완료: {info['mode']} | 전체 {info['rows']}행 / 적합 가능 {info['eligible_rows']}행")
    if args.mode == "compare":
        print(run["metrics"].to_string(index=False))
        print("후보 비교 결과입니다. 최종 군집 수는 크기·해석 가능성까지 검토해 선택하세요.")
    elif args.mode in {"train", "predict"}:
        print(f"분류 {info['classified_rows']}행 / 미분류 {info['unclassified_rows']}행 / k={info['n_clusters']}")
    print(f"학습 실행: {info['model_fitted']} | 최종 모델 저장: {info['final_model_saved']}")
    print(f"저장: {run['output_dir']}")


if __name__ == "__main__":
    main()
