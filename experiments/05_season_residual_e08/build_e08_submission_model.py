"""
E08 최종 제출 모델 생성
=======================

검증에서 선택한 E05 36.8353% + season-residual 63.1647%를 전체
2019~2024 train으로 학습해 하나의 weighted ensemble artifact로 저장한다.

대회 규칙
---------
- residual target과 2025 prior는 train 2019~2024에서만 계산한다.
- 2025 prior는 전체 과거 시즌 성공률의 선형 추세로 미리 고정한다.
- 추론에서는 test 평균/분포/다른 행을 사용하지 않는다.

사용법
------
    python scripts/build_e08_submission_model.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from build_ensemble_model import (  # noqa: E402
    BASIC_CAT_COLS,
    BASE_PARAMS,
    DOMAIN_FEATURES,
    encode,
    make_base,
    make_cat_maps,
    make_feature_spec,
)
from evaluate_residual_context import (  # noqa: E402
    TARGET,
    estimate_next_prior,
    residual_target,
    season_rates,
)
from script import predict_from_artifact  # noqa: E402

DATA_DIR = REPO / "data"
MODEL_DIR = REPO / "models"
RESULTS_DIR = REPO / "results"
E05_MODEL_PATH = MODEL_DIR / "lgbm_ensemble.joblib"
E08_SCREEN_PATH = RESULTS_DIR / "e08_screen_metrics.json"
E08_INTEGRATION_PATH = RESULTS_DIR / "e08_e05_integration_metrics.json"
OUTPUT_MODEL_PATH = MODEL_DIR / "e08_season_residual_ensemble.joblib"
OUTPUT_METRICS_PATH = RESULTS_DIR / "e08_submission_model_metrics.json"


def load_inputs() -> tuple[pd.DataFrame, list[str], dict, dict, dict]:
    required = [E05_MODEL_PATH, E08_SCREEN_PATH, E08_INTEGRATION_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"필수 E08/E05 파일이 없습니다: {missing}")

    test_columns = pd.read_csv(
        DATA_DIR / "test.csv", nrows=0, encoding="utf-8-sig"
    ).columns.tolist()
    raw_features = [column for column in test_columns if column != "row_id"]
    train = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=raw_features + [TARGET],
        encoding="utf-8-sig",
    )
    e05 = joblib.load(E05_MODEL_PATH)
    screen = json.loads(E08_SCREEN_PATH.read_text(encoding="utf-8"))
    integration = json.loads(E08_INTEGRATION_PATH.read_text(encoding="utf-8"))
    if e05.get("artifact_type") != "weighted_ensemble":
        raise SystemExit("E05 artifact가 weighted_ensemble 형식이 아닙니다.")
    e05_meta = e05.get("meta", {})
    if e05_meta.get("experiment") != "E05":
        raise SystemExit("E05 artifact의 experiment 메타데이터가 E05가 아닙니다.")
    if e05_meta.get("validation_scheme") != integration.get("validation"):
        raise SystemExit("E05 artifact와 E08 integration의 검증 구간이 다릅니다.")
    return train, raw_features, e05, screen, integration


def make_environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lightgbm": lgb.__version__,
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def main() -> None:
    started = time.time()
    print("#" * 74)
    print("#  E08 - 최종 season residual ensemble 생성")
    print("#" * 74)
    train, raw_features, e05, screen, integration = load_inputs()
    environment = make_environment()

    selected = integration["optimized_ensemble"]
    selected_weights = selected["weights"]
    if selected.get("reference") != "e05":
        raise RuntimeError("E08 integration 기준 모델이 E05가 아닙니다.")
    if set(selected_weights) != {"e05", "residual"}:
        raise RuntimeError("E08 최종 가중치가 E05와 residual 두 모델 조합이 아닙니다.")
    if integration.get("rules", {}).get(
        "context_excluded_by_rolling_nested_validation"
    ) is not True:
        raise RuntimeError("rolling nested 검증에서 context 제외가 확정되지 않았습니다.")
    e05_weight = float(selected_weights["e05"])
    residual_weight = float(selected_weights["residual"])
    if not np.isclose(e05_weight + residual_weight, 1.0):
        raise RuntimeError("E08 E05/residual validation weight 합이 1이 아닙니다.")
    if screen["best_residual_method"] != "linear_all":
        raise RuntimeError("E08 prior 방식이 사전 고정 linear_all이 아닙니다.")
    best_iteration = int(screen["residual_fit"]["best_iteration"])

    print(f"train rows={len(train):,}")
    print(
        f"validation weights: E05={e05_weight:.6f}, "
        f"residual={residual_weight:.6f}"
    )
    print(f"residual best_iteration={best_iteration}")

    final_spec = make_feature_spec(train, extrapolate_to=(2025, 2026))
    base = make_base(train, raw_features, True, final_spec)
    features = raw_features + DOMAIN_FEATURES
    cat_maps = make_cat_maps(base, BASIC_CAT_COLS)
    X_all = encode(base, features, cat_maps)
    y_residual = residual_target(train)

    rates = season_rates(train)
    prior_2025 = estimate_next_prior(rates, "linear_all")
    if not 0.0 <= prior_2025 <= 1.0:
        raise RuntimeError(f"2025 prior가 확률 범위를 벗어났습니다: {prior_2025}")
    print(f"2025 fixed prior={prior_2025:.9f}")

    params = dict(BASE_PARAMS)
    params.update(
        {
            "n_estimators": best_iteration,
            "random_state": 314,
        }
    )
    model = lgb.LGBMRegressor(objective="regression", **params)
    fit_started = time.time()
    model.fit(X_all, y_residual)
    fit_sec = time.time() - fit_started
    print(f"residual 전체 재학습 완료: {fit_sec:.1f}s")

    residual_member = {
        "name": "season_residual",
        "features": features,
        "cat_cols": BASIC_CAT_COLS,
        "cat_maps": cat_maps,
        "model": model,
        "kind": "residual_offset",
        "prediction_offset": prior_2025,
        "impute_median": None,
        "feature_spec": final_spec,
        "meta": {
            "description": "season-centered residual LGBM + fixed 2025 train prior",
            "prior_method": "linear_all",
            "prior_2025": prior_2025,
            "season_rates": rates,
            "best_iteration": best_iteration,
            "n_estimators": best_iteration,
            "params": params,
            "environment": environment,
        },
    }

    # E05 모델은 이미 전체 train으로 재학습된 artifact이므로 그대로 재사용한다.
    members = list(e05["members"]) + [residual_member]
    weights = [e05_weight * float(weight) for weight in e05["weights"]]
    weights.append(residual_weight)
    weights = [float(weight / sum(weights)) for weight in weights]

    artifact = {
        "artifact_type": "weighted_ensemble",
        "members": members,
        "weights": weights,
        "meta": {
            "experiment": "E08",
            "selected": "E05 + season residual ensemble",
            "validation_scheme": "2019~2023 -> 2024",
            "valid_score": float(selected["score"]),
            "valid_brier": float(selected["brier"]),
            "e05_weight": e05_weight,
            "residual_weight": residual_weight,
            "prior_method": "linear_all",
            "prior_2025": prior_2025,
            "environment": environment,
            "built_by": "scripts/build_e08_submission_model.py",
        },
    }

    # 저장 전에 소량 행으로 artifact 전체 경로가 정상인지 확인한다.
    sample = pd.read_csv(DATA_DIR / "test.csv", nrows=100, encoding="utf-8-sig")
    sample_prediction = predict_from_artifact(sample, artifact)
    if len(sample_prediction) != len(sample):
        raise RuntimeError("E08 artifact 샘플 예측 길이가 입력과 다릅니다.")
    if not np.isfinite(sample_prediction).all():
        raise RuntimeError("E08 artifact 샘플 예측에 비정상 값이 있습니다.")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, OUTPUT_MODEL_PATH, compress=3)
    print(
        f"저장: {OUTPUT_MODEL_PATH} "
        f"({OUTPUT_MODEL_PATH.stat().st_size / 1e6:.2f} MB)"
    )

    metrics = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "experiment": "E08-final-artifact",
        "branch": "exp/baek-32/e08-residual-context",
        "model_path": str(OUTPUT_MODEL_PATH.relative_to(REPO)),
        "model_size_mb": OUTPUT_MODEL_PATH.stat().st_size / 1e6,
        "validation": {
            "scheme": "2019~2023 -> 2024",
            "score": float(selected["score"]),
            "brier": float(selected["brier"]),
            "e05_weight": e05_weight,
            "residual_weight": residual_weight,
        },
        "residual": {
            "best_iteration": best_iteration,
            "prior_method": "linear_all",
            "prior_2025": prior_2025,
            "season_rates": rates,
            "fit_sec": fit_sec,
        },
        "members": [member.get("name") for member in members],
        "weights": weights,
        "rules": {
            "uses_test_rows_for_statistics": False,
            "uses_test_prediction_distribution": False,
            "prediction_offset_is_train_only_constant": True,
        },
        "environment": environment,
        "sample_prediction": {
            "rows": int(len(sample_prediction)),
            "min": float(sample_prediction.min()),
            "max": float(sample_prediction.max()),
            "mean": float(sample_prediction.mean()),
        },
        "total_sec": time.time() - started,
    }
    OUTPUT_METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"metrics: {OUTPUT_METRICS_PATH}")
    print(f"총 시간: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
