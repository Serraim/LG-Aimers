"""E08 안정성 진단: 25% 고정 표본으로 2022/2023/2024 pseudo-fold 비교."""

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
    estimate_next_prior,
    fit_classifier,
    fit_residual_regressor,
    load_train,
    prediction_metrics,
)

MODEL_DIR = REPO / "models"
RESULTS_DIR = REPO / "results"
CONTEXT_CACHE = MODEL_DIR / "e08_context_features.pkl"


def stratified_sample(df: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if fraction >= 1.0:
        return df
    return (
        df.groupby("season", group_keys=False)
        .sample(frac=fraction, random_state=2026)
        .sort_index()
    )


def run_fold(
    full: pd.DataFrame,
    sampled: pd.DataFrame,
    context: pd.DataFrame,
    raw_features: list[str],
    valid_season: int,
) -> tuple[dict, dict]:
    train_full = full[full["season"] < valid_season]
    train_part = sampled[sampled["season"] < valid_season]
    valid_part = sampled[sampled["season"] == valid_season]
    y_train = train_part[TARGET].to_numpy()
    y_valid = valid_part[TARGET].to_numpy()

    feature_spec = make_feature_spec(train_full, extrapolate_to=(valid_season,))
    base_train = make_base(train_part, raw_features, True, feature_spec)
    base_valid = make_base(valid_part, raw_features, True, feature_spec)
    features = raw_features + DOMAIN_FEATURES
    maps = make_cat_maps(base_train, BASIC_CAT_COLS)
    X_train = encode(base_train, features, maps)
    X_valid = encode(base_valid, features, maps)

    baseline_fit, baseline_prediction = fit_classifier(
        X_train, y_train, X_valid, y_valid
    )
    season_mean_all = full.groupby("season")[TARGET].mean()
    residual_train = (
        train_part[TARGET] - train_part["season"].map(season_mean_all)
    ).to_numpy()
    rates = {
        int(season): float(value)
        for season, value in season_mean_all[season_mean_all.index < valid_season].items()
    }
    fit_prior = estimate_next_prior(rates, "linear_all")
    # 실제 validation 평균을 빼지 않는다. 추론 시 사용 가능한 prior로 Brier를 평가한다.
    residual_valid = y_valid - fit_prior
    residual_fit, residual_value = fit_residual_regressor(
        X_train, residual_train, X_valid, residual_valid
    )

    X_train_context = pd.concat([X_train, context.loc[train_part.index]], axis=1)
    X_valid_context = pd.concat([X_valid, context.loc[valid_part.index]], axis=1)
    context_fit, context_prediction = fit_classifier(
        X_train_context, y_train, X_valid_context, y_valid
    )

    baseline_result = prediction_metrics(baseline_prediction, y_valid)
    context_result = prediction_metrics(context_prediction, y_valid)
    methods = {}
    prediction_matrices = {}
    for method in ("last", "linear_all", "linear_last3"):
        prior = estimate_next_prior(rates, method)
        residual_prediction = np.clip(prior + residual_value, 0.0, 1.0)
        residual_result = prediction_metrics(residual_prediction, y_valid)
        matrix = np.column_stack(
            [baseline_prediction, residual_prediction, context_prediction]
        )
        prediction_matrices[method] = matrix
        weights = optimize_weights(matrix, y_valid)
        ensemble_prediction = np.clip(matrix @ weights, 0.0, 1.0)
        ensemble_result = prediction_metrics(ensemble_prediction, y_valid)
        methods[method] = {
            "estimated_prior": prior,
            "prior_error": prior - float(y_valid.mean()),
            "residual": residual_result,
            "ensemble": ensemble_result,
            "weights": [float(value) for value in weights],
            "delta_vs_baseline": ensemble_result["score"] - baseline_result["score"],
        }

    result = {
        "valid_season": valid_season,
        "n_train": int(len(train_part)),
        "n_valid": int(len(valid_part)),
        "valid_rate": float(y_valid.mean()),
        "baseline": baseline_result,
        "baseline_fit": baseline_fit,
        "context": context_result,
        "context_fit": context_fit,
        "residual_fit": residual_fit,
        "methods": methods,
    }
    cache = {
        "target": y_valid,
        "baseline": baseline_prediction,
        "matrices": prediction_matrices,
    }
    return result, cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.sample <= 1:
        raise SystemExit("--sample은 0 초과 1 이하여야 합니다.")
    if not CONTEXT_CACHE.exists():
        raise SystemExit("먼저 python scripts/screen_residual_context.py 를 실행하세요.")

    started = time.time()
    full, raw_features = load_train()
    sampled = stratified_sample(full, args.sample)
    context = pd.read_pickle(CONTEXT_CACHE)
    folds = []
    fold_predictions = []
    for valid_season in (2022, 2023, 2024):
        print(f"\nfold {valid_season}", flush=True)
        fold, predictions = run_fold(
            full, sampled, context, raw_features, valid_season
        )
        folds.append(fold)
        fold_predictions.append(predictions)
        print(
            f"  baseline={fold['baseline']['score']:.2f}, "
            + ", ".join(
                f"{name}={value['ensemble']['score']:.2f}"
                f"({value['delta_vs_baseline']:+.2f})"
                for name, value in fold["methods"].items()
            ),
            flush=True,
        )

    summary = {}
    for method in ("last", "linear_all", "linear_last3"):
        deltas = [fold["methods"][method]["delta_vs_baseline"] for fold in folds]
        summary[method] = {
            "mean_delta": float(np.mean(deltas)),
            "min_delta": float(np.min(deltas)),
            "improving_folds": int(sum(delta > 0 for delta in deltas)),
            "deltas": deltas,
            "mean_weights": np.mean(
                [fold["methods"][method]["weights"] for fold in folds], axis=0
            ).tolist(),
        }

    # 2022·2023에서만 weight를 정하고, 보지 않은 2024에 그대로 적용한다.
    nested = {}
    for method in ("last", "linear_all", "linear_last3"):
        selection_matrix = np.vstack(
            [item["matrices"][method] for item in fold_predictions[:2]]
        )
        selection_target = np.concatenate(
            [item["target"] for item in fold_predictions[:2]]
        )
        weights = optimize_weights(selection_matrix, selection_target)
        holdout_matrix = fold_predictions[2]["matrices"][method]
        holdout_target = fold_predictions[2]["target"]
        prediction = np.clip(holdout_matrix @ weights, 0.0, 1.0)
        holdout_result = prediction_metrics(prediction, holdout_target)
        baseline_result = folds[2]["baseline"]
        nested[method] = {
            "weights_selected_on": "2022+2023",
            "weights": [float(value) for value in weights],
            "evaluated_on": 2024,
            "score": holdout_result["score"],
            "brier": holdout_result["brier"],
            "delta_vs_baseline": holdout_result["score"] - baseline_result["score"],
        }

    payload = {
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "experiment": "E08-rolling-diagnostic",
        "branch": "exp/baek-32/e08-residual-context",
        "sample_fraction": args.sample,
        "folds": folds,
        "summary": summary,
        "nested_2022_2023_to_2024": nested,
        "rules": {
            "context_uses_strictly_earlier_seasons": True,
            "uses_test_rows": False,
            "uses_test_prediction_distribution": False,
        },
        "total_sec": time.time() - started,
    }
    output = RESULTS_DIR / "e08_rolling_metrics.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nsummary", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("\nnested 2022+2023 -> 2024", flush=True)
    print(json.dumps(nested, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {output}", flush=True)
    print(f"elapsed: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
