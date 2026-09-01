"""E08 2단계: 캐시된 residual/context 예측을 E05 또는 E06과 결합한다."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from build_ensemble_model import (  # noqa: E402
    BASIC_CAT_COLS,
    DOMAIN_FEATURES,
    encode,
    make_base,
    make_cat_maps,
    make_feature_spec,
    optimize_weights,
)
from evaluate_residual_context import (  # noqa: E402
    TARGET,
    VALID_SEASON,
    load_train,
    monthly_deltas,
    prediction_metrics,
)
from evaluate_trackman_features import reproduce_e05  # noqa: E402
from screen_residual_context import PREDICTION_CACHE_VERSION  # noqa: E402

MODEL_DIR = REPO / "models"
RESULTS_DIR = REPO / "results"
PREDICTION_CACHE = MODEL_DIR / "e08_screen_predictions.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-catboost", action="store_true")
    parser.add_argument(
        "--exclude-context",
        action="store_true",
        help="롤링 nested 검증에서 weight=0이 된 context 모델을 제외한다.",
    )
    args = parser.parse_args()
    started = time.time()
    if not PREDICTION_CACHE.exists():
        raise SystemExit(
            "E08 screen 예측이 없습니다. 먼저 "
            "python scripts/screen_residual_context.py 를 실행하세요."
        )

    train, raw_features = load_train()
    train_part = train[train["season"] < VALID_SEASON]
    valid_part = train[train["season"] == VALID_SEASON]
    y_train = train_part[TARGET].to_numpy()
    y_valid = valid_part[TARGET].to_numpy()
    cached = np.load(PREDICTION_CACHE)
    if str(cached["cache_version"]) != PREDICTION_CACHE_VERSION:
        raise RuntimeError("E08 prediction cache 버전이 현재 코드와 다릅니다.")
    if str(cached["prior_method"]) != "linear_all":
        raise RuntimeError("E08 prediction cache가 사전 고정 linear_all prior가 아닙니다.")
    if not np.array_equal(cached["target"], y_valid):
        raise RuntimeError("E08 prediction cache의 validation target이 현재 데이터와 다릅니다.")
    baseline_prediction = cached["baseline"]
    residual_prediction = cached["residual"]
    context_prediction = cached["context"]

    feature_spec = make_feature_spec(train_part, extrapolate_to=(VALID_SEASON,))
    base_train = make_base(train_part, raw_features, True, feature_spec)
    base_valid = make_base(valid_part, raw_features, True, feature_spec)
    base_features = raw_features + DOMAIN_FEATURES
    category_maps = make_cat_maps(base_train, BASIC_CAT_COLS)
    X_train = encode(base_train, base_features, category_maps)
    X_valid = encode(base_valid, base_features, category_maps)

    print("#" * 72, flush=True)
    print("#  E08 integration - E05/E06 + residual/context", flush=True)
    print("#" * 72, flush=True)
    print("\nE05 재현", flush=True)
    e05_prediction, e05_members = reproduce_e05(
        baseline_prediction, X_train, y_train, X_valid, y_valid
    )
    e05_result = prediction_metrics(e05_prediction, y_valid)
    print(f"  score={e05_result['score']:.2f}", flush=True)

    reference_name = "e05"
    reference_prediction = e05_prediction
    e06_result = None
    catboost_result = None
    if args.with_catboost:
        from build_catboost_ensemble import fit_catboost_validation

        print("\nE06 CatBoost 재현", flush=True)
        catboost_result, catboost_prediction = fit_catboost_validation(
            train_part, valid_part, raw_features, feature_spec
        )
        stored = json.loads(
            (RESULTS_DIR / "e06_catboost_metrics.json").read_text(encoding="utf-8")
        )["blended"]
        catboost_weight = float(stored["catboost_weight"])
        reference_prediction = (
            (1.0 - catboost_weight) * e05_prediction
            + catboost_weight * catboost_prediction
        )
        reference_name = "e06"
        e06_result = prediction_metrics(reference_prediction, y_valid)
        print(f"  score={e06_result['score']:.2f}", flush=True)

    if args.exclude_context:
        names = [reference_name, "residual"]
        matrix = np.column_stack([reference_prediction, residual_prediction])
    else:
        names = [reference_name, "residual", "context"]
        matrix = np.column_stack(
            [reference_prediction, residual_prediction, context_prediction]
        )
    weights = optimize_weights(matrix, y_valid)
    prediction = np.clip(matrix @ weights, 0.0, 1.0)
    result = prediction_metrics(prediction, y_valid)
    reference_result = prediction_metrics(reference_prediction, y_valid)
    result["weights"] = {
        name: float(weight) for name, weight in zip(names, weights)
    }
    result["reference"] = reference_name
    result["delta_score"] = result["score"] - reference_result["score"]
    result["delta_brier"] = result["brier"] - reference_result["brier"]
    result["monthly_delta"] = monthly_deltas(
        reference_prediction, prediction, valid_part
    )
    correlation = pd.DataFrame(matrix, columns=names).corr()

    print(
        f"\n{reference_name.upper()}+E08 score={result['score']:.2f} "
        f"({result['delta_score']:+.2f})",
        flush=True,
    )
    print(f"weights={result['weights']}", flush=True)
    print(f"monthly={result['monthly_delta']}", flush=True)

    payload = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "experiment": "E08-integration",
        "branch": "exp/baek-32/e08-residual-context",
        "validation": "2019~2023 -> 2024",
        "rules": {
            "uses_test_rows": False,
            "uses_test_prediction_distribution": False,
            "context_excluded_by_rolling_nested_validation": args.exclude_context,
        },
        "e05": e05_result,
        "e05_members": e05_members,
        "e06": e06_result,
        "catboost": catboost_result,
        "optimized_ensemble": result,
        "prediction_correlation": correlation.to_dict(),
        "total_sec": time.time() - started,
    }
    output = RESULTS_DIR / (
        "e08_e06_integration_metrics.json"
        if args.with_catboost
        else "e08_e05_integration_metrics.json"
    )
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {output}", flush=True)
    print(f"elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
