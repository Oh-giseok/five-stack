"""저장 모델을 공통 형식으로 불러온다. 학습·변환·예측은 실행하지 않는다."""

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _column_names(values, label, allow_empty=False):
    """순서를 유지하며 입력 명세의 중복·잘못된 이름을 확인한다."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label}는 변수 이름 목록이어야 합니다.")
    try:
        columns = list(values)
    except TypeError as error:
        raise ValueError(f"{label}는 변수 이름 목록이어야 합니다.") from error
    if (not columns and not allow_empty) or any(not isinstance(c, str) or not c for c in columns):
        raise ValueError(f"{label}에 유효한 변수 이름이 필요합니다.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"{label}에 중복된 변수가 있습니다.")
    return columns


def load_model(model_path):
    """프로젝트 루트 기준 경로 또는 절대 경로로 저장 모델을 불러온다.

    반환값의 feature_columns는 실제 적합된 모델의 입력 순서다.
    JSON·CSV 부가 파일을 읽거나 저장된 모델을 다시 적합하지 않는다.
    """

    path = Path(model_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}")

    suffix = path.suffix.lower()

    # XGBoost·LightGBM과 같은 방식으로 저장한 학습 완료 Pipeline
    if suffix == ".joblib":
        saved = joblib.load(path)

        # XGBoost는 dict, 현재 LightGBM은 Pipeline 본체로 저장됨
        if isinstance(saved, dict):
            if "pipeline" not in saved:
                raise ValueError("joblib dict에 'pipeline' 항목이 없습니다.")
            model = saved["pipeline"]
        else:
            model = saved

        if not isinstance(model, Pipeline):
            raise ValueError("joblib에는 학습 완료된 Pipeline을 저장해야 합니다.")
        if not callable(getattr(model, "predict", None)):
            raise ValueError("저장된 Pipeline은 예측 기능을 제공하지 않습니다.")
        try:
            check_is_fitted(model)
        except ValueError as error:
            raise ValueError("아직 학습되지 않은 Pipeline입니다.") from error

        columns = getattr(model, "feature_names_in_", None)

        if columns is None:
            raise ValueError("저장 모델에서 입력 변수 정보를 찾을 수 없습니다.")

        feature_columns = _column_names(columns, "입력 변수")

        # 실제 적합 때 제외한 전체 결측 열이 있을 수 있으므로 적합 순서가 우선이다.
        if isinstance(saved, dict) and "feature_columns" in saved:
            declared = _column_names(saved["feature_columns"], "저장된 입력 명세")
            if feature_columns != [c for c in declared if c in feature_columns]:
                raise ValueError("저장된 입력 명세와 실제 모델의 변수·순서가 다릅니다.")

        # 저장된 입력 변환기에서 범주형 변수 목록 확인
        steps = model.named_steps

        processor = steps.get("preprocessor", steps.get("transformer"))
        categorical_columns = []
        for name, transformer, columns in getattr(processor, "transformers_", []):
            if name in ("cat", "categorical"):
                categorical_columns = _column_names(columns, "범주형 변수", allow_empty=True)
                break

        if isinstance(saved, dict) and "categorical_columns" in saved:
            declared = _column_names(saved["categorical_columns"], "저장된 범주형 명세", allow_empty=True)
            if categorical_columns and categorical_columns != declared:
                raise ValueError("저장된 범주형 명세와 입력 변환기의 변수가 다릅니다.")
            categorical_columns = declared
        if not set(categorical_columns).issubset(feature_columns):
            raise ValueError("범주형 변수가 모델 입력에 포함돼 있지 않습니다.")

        model_format = "joblib_pipeline"

    # CatBoost
    elif suffix == ".cbm":
        # joblib 모델만 사용하는 환경에서는 CatBoost를 미리 불러오지 않는다.
        from catboost import CatBoostRegressor

        model = CatBoostRegressor()
        model.load_model(str(path))
        if not model.is_fitted():
            raise ValueError("아직 학습되지 않은 CatBoost 모델입니다.")

        feature_columns = _column_names(model.feature_names_, "CatBoost 입력 변수")
        categorical_columns = [
            feature_columns[index]
            for index in model.get_cat_feature_indices()
        ]

        model_format = "catboost"

    else:
        raise ValueError(f"지원하지 않는 모델 형식입니다: {path.suffix}")

    return {
        "model": model,
        "model_format": model_format,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "model_path": path,
    }
