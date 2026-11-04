"""Check the repository's shared settings without fetching or changing data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    try:
        from oinq.common.config import ConfigError, load_project_config, validate_project_config
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        print("설정 점검에 PyYAML이 필요합니다. README의 환경 설치 명령을 실행하세요.", file=sys.stderr)
        return 1
    try:
        result = validate_project_config(load_project_config(ROOT))
    except ConfigError as exc:
        print(f"설정 점검 실패: {exc}", file=sys.stderr)
        return 1
    print(f"설정 점검 통과: {result['countries']}개국 / {result['products']}개 제품")
    print("분기 수: " + ", ".join(f"{key}={value}" for key, value in result["quarters"].items()))
    print("모델: " + ", ".join(result["models"]))
    print("설정 정합성만 확인했습니다. API 인증·데이터 확보·학습 성능은 별도 점검 대상입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
