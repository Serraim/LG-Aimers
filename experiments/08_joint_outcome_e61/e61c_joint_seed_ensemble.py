"""E61-C: three-seed ensemble of the E61 joint-outcome LightGBM.

Seed 42 OOF predictions from E61-A are reused. Only seeds 2024 and 3407 are
trained, then all three are equally averaged. The binary/joint member blend
alpha is fitted once on 2022 raw Brier and frozen for 2023/2024.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

try:
    from scripts import ab_feature as legacy
    from scripts.e13_feature_groups import (
        FIXED_TREES,
        PRED_DIR,
        TARGET,
        centered_metrics,
        make_temporal_features,
    )
    from scripts.e13_rolling_validation import _prepare_fold
    from scripts.e41_robust_temporal_weights import MODEL_NAMES, load_predictions
    from scripts.e52_e42_slope_calibration import (
        centered_loss_difference,
        cluster_bootstrap_mean_ci,
    )
    from scripts.e61_joint_outcome import (
        JOINT_CODES,
        LGBM_INDEX,
        encode_joint_codes,
        feature_contract,
        load_label_frame,
        reconstruct_joint_codes,
        replace_lgbm_member,
        success_probability,
    )
    from scripts.e61b_joint_blend import blend_members, fit_raw_brier_alpha
except ModuleNotFoundError:
    import ab_feature as legacy
    from e13_feature_groups import FIXED_TREES, PRED_DIR, TARGET, centered_metrics, make_temporal_features
    from e13_rolling_validation import _prepare_fold
    from e41_robust_temporal_weights import MODEL_NAMES, load_predictions
    from e52_e42_slope_calibration import centered_loss_difference, cluster_bootstrap_mean_ci
    from e61_joint_outcome import (
        JOINT_CODES,
        LGBM_INDEX,
        encode_joint_codes,
        feature_contract,
        load_label_frame,
        reconstruct_joint_codes,
        replace_lgbm_member,
        success_probability,
    )
    from e61b_joint_blend import blend_members, fit_raw_brier_alpha


REPO = Path(__file__).resolve().parent.parent
E42_RESULT_PATH = REPO / "results" / "e42_frozen_robust_weights.json"
E61_RESULT_PATH = REPO / "results" / "e61_joint_outcome.json"
RESULT_PATH = REPO / "results" / "e61c_joint_seed_ensemble.json"
VALID_SEASONS = (2022, 2023, 2024)
SELECTION_SEASON = 2022
STRESS_SEASON = 2023
EVALUATION_SEASON = 2024
DEFAULT_SEEDS = (42, 2024, 3407)
TRAIN_SEEDS = DEFAULT_SEEDS[1:]


def validate_sources(e42: dict, e61: dict) -> None:
    if e42.get("experiment_id") != "E42" or e42.get("weight_source") != "2021 OOF only":
        raise RuntimeError("Frozen train-OOF E42 result is required")
    if e61.get("experiment_id") != "E61-A" or e61.get("status") != "complete":
        raise RuntimeError("Completed E61-A result is required")
    if e61.get("seed") != 42 or e61.get("trees") != FIXED_TREES:
        raise RuntimeError("E61-A seed/tree contract changed")
    if e61.get("leaderboard_submissions") != 0:
        raise RuntimeError("E61-A must not use leaderboard selection")


def seed_average(predictions: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(prediction, dtype="float64") for prediction in predictions]
    if len(arrays) != len(DEFAULT_SEEDS) or len({array.shape for array in arrays}) != 1:
        raise ValueError("Exactly three aligned seed predictions are required")
    return np.mean(np.column_stack(arrays), axis=1)


def seed_prediction_path(season: int, seed: int) -> Path:
    if seed == 42:
        return PRED_DIR / f"e61_{season}_joint_lgbm.npy"
    if seed not in TRAIN_SEEDS:
        raise ValueError(f"Unexpected E61-C seed: {seed}")
    return PRED_DIR / f"e61c_{season}_joint_seed{seed}.npy"


def _load_prediction(path: Path, expected_length: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    prediction = np.load(path).astype("float64")
    if prediction.shape != (expected_length,):
        raise ValueError(f"Invalid prediction shape: {path}")
    return prediction


def _train_joint_seed(frame, train_mask, valid_mask, codes, features, seed: int):
    eligible_mask = train_mask & codes.ne(-1)
    labels = encode_joint_codes(codes.loc[eligible_mask])
    if set(np.unique(labels)) != set(range(len(JOINT_CODES))):
        raise RuntimeError("A training fold is missing one or more fixed joint classes")
    params = dict(legacy.PARAMS)
    params.update(
        objective="multiclass",
        num_class=len(JOINT_CODES),
        n_estimators=FIXED_TREES,
        random_state=int(seed),
    )
    model = lgb.LGBMClassifier(**params)
    started = time.time()
    model.fit(frame.loc[eligible_mask, features], labels)
    probabilities = model.predict_proba(frame.loc[valid_mask, features])
    prediction = success_probability(probabilities, model.classes_)
    fit_sec = time.time() - started
    del model, probabilities
    gc.collect()
    return prediction, fit_sec, int(eligible_mask.sum())


def selection_checks(alpha: float, folds: dict[str, dict], bootstrap: dict, e42: dict) -> dict:
    evaluation = folds[str(EVALUATION_SEASON)]
    baseline_ok = all(
        np.isclose(
            folds[str(season)]["e42"]["centered_brier"],
            e42["folds"][str(season)]["e42"]["centered_brier"],
            rtol=0.0,
            atol=1e-10,
        )
        for season in VALID_SEASONS
    )
    return {
        "three_fixed_seeds_equal_average": True,
        "seed42_reused_from_e61a": True,
        "alpha_fitted_only_on_2022_raw_brier": evaluation["alpha_source_season"]
        == SELECTION_SEASON,
        "alpha_bounded_zero_to_one": 0.0 <= alpha <= 1.0,
        "stress_2023_excluded_from_selection": STRESS_SEASON
        not in (SELECTION_SEASON, EVALUATION_SEASON),
        "e42_oof_baseline_reproduced": bool(baseline_ok),
        "raw_brier_improved_2024": evaluation["delta_brier"] < 0,
        "centered_brier_improved_2024": evaluation["delta_centered_brier"] < 0,
        "pitcher_bootstrap_upper_below_zero": float(bootstrap["ci_upper_95"]) < 0,
    }


def _save(payload: dict) -> None:
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(n_bootstrap: int = 2000, reuse_existing: bool = False) -> dict:
    started = time.time()
    e42 = json.loads(E42_RESULT_PATH.read_text(encoding="utf-8"))
    e61 = json.loads(E61_RESULT_PATH.read_text(encoding="utf-8"))
    validate_sources(e42, e61)
    weights = np.asarray([e42["weights"][name] for name in MODEL_NAMES], dtype="float64")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Invalid E42 weights")

    df, original_features = legacy.load_data()
    label_frame = load_label_frame()
    if len(label_frame) != len(df):
        raise RuntimeError("Feature and joint-label rows are not aligned")
    temporal = make_temporal_features(df)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    joint_averages: dict[int, np.ndarray] = {}
    training_records: dict[str, dict] = {}
    payload = {
        "experiment_id": "E61-C",
        "description": "Three-seed E61 joint model with 2022-fitted binary/joint blend",
        "seeds": list(DEFAULT_SEEDS),
        "seed42_source": "E61-A saved OOF predictions",
        "newly_trained_seeds": list(TRAIN_SEEDS),
        "seed_aggregation": "fixed equal arithmetic mean; no seed selection",
        "trees": FIXED_TREES,
        "test_data_used": False,
        "leaderboard_submissions": 0,
        "training_records": training_records,
        "status": "running",
    }
    _save(payload)

    for season in VALID_SEASONS:
        frame, train_mask, valid_mask, base_features = _prepare_fold(
            df, original_features, temporal, season
        )
        features = feature_contract(base_features)
        codes, _ = reconstruct_joint_codes(label_frame, season)
        n_valid = int(valid_mask.sum())
        predictions = [_load_prediction(seed_prediction_path(season, 42), n_valid)]
        seed_records = [{"seed": 42, "source": "reused_e61a", "fit_sec": 0.0}]
        for seed in TRAIN_SEEDS:
            path = seed_prediction_path(season, seed)
            if reuse_existing and path.exists():
                prediction = _load_prediction(path, n_valid)
                fit_sec = 0.0
                n_train_joint = int((train_mask & codes.ne(-1)).sum())
                source = "reused_e61c"
            else:
                prediction, fit_sec, n_train_joint = _train_joint_seed(
                    frame, train_mask, valid_mask, codes, features, seed
                )
                np.save(path, prediction.astype("float32"))
                source = "trained"
            predictions.append(prediction)
            seed_records.append(
                {
                    "seed": int(seed),
                    "source": source,
                    "fit_sec": fit_sec,
                    "n_train_joint": n_train_joint,
                }
            )
            training_records[str(season)] = {
                "n_valid": n_valid,
                "n_features": len(features),
                "seed_records": seed_records,
            }
            payload.update(training_records=training_records, elapsed_sec=time.time() - started)
            _save(payload)
            print(f"{season} seed={seed} {source} fit={fit_sec:.1f}s", flush=True)

        average = seed_average(predictions)
        joint_averages[season] = average
        np.save(PRED_DIR / f"e61c_{season}_joint_seed_average.npy", average.astype("float32"))
        del frame, codes, predictions
        gc.collect()

    matrices = {season: load_predictions(season) for season in VALID_SEASONS}
    targets = {
        season: df.loc[df["season"].eq(season), TARGET].to_numpy(dtype="float64")
        for season in VALID_SEASONS
    }
    selection_matrix = matrices[SELECTION_SEASON]
    alpha = fit_raw_brier_alpha(
        selection_matrix @ weights,
        selection_matrix[:, LGBM_INDEX],
        joint_averages[SELECTION_SEASON],
        targets[SELECTION_SEASON],
        weights[LGBM_INDEX],
    )

    folds: dict[str, dict] = {}
    evaluation: dict[str, np.ndarray] = {}
    for season in VALID_SEASONS:
        matrix = matrices[season]
        target = targets[season]
        baseline = matrix @ weights
        blended_member = blend_members(matrix[:, LGBM_INDEX], joint_averages[season], alpha)
        candidate = np.clip(replace_lgbm_member(matrix, blended_member) @ weights, 0.0, 1.0)
        np.save(PRED_DIR / f"e61c_{season}_ensemble.npy", candidate.astype("float32"))
        baseline_metrics = centered_metrics(baseline, target)
        candidate_metrics = centered_metrics(candidate, target)
        folds[str(season)] = {
            "fold_role": (
                "selection"
                if season == SELECTION_SEASON
                else "stress" if season == STRESS_SEASON else "evaluation"
            ),
            "alpha": alpha,
            "alpha_source_season": SELECTION_SEASON,
            "e42": baseline_metrics,
            "e61c": candidate_metrics,
            "delta_brier": candidate_metrics["brier"] - baseline_metrics["brier"],
            "delta_score": candidate_metrics["score"] - baseline_metrics["score"],
            "delta_centered_brier": candidate_metrics["centered_brier"]
            - baseline_metrics["centered_brier"],
            "delta_centered_score": candidate_metrics["centered_score"]
            - baseline_metrics["centered_score"],
        }
        fold = folds[str(season)]
        print(
            f"{season} E61-C alpha={alpha:.6f} delta_brier={fold['delta_brier']:+.8f} "
            f"delta_centered_score={fold['delta_centered_score']:+.2f}",
            flush=True,
        )
        if season == EVALUATION_SEASON:
            mask = df["season"].eq(season)
            evaluation = {
                "baseline": baseline,
                "candidate": candidate,
                "target": target,
                "groups": df.loc[mask, "pitcher_id"].to_numpy(),
            }

    differences = centered_loss_difference(
        evaluation["baseline"], evaluation["candidate"], evaluation["target"]
    )
    bootstrap = cluster_bootstrap_mean_ci(
        differences, evaluation["groups"], n_bootstrap=n_bootstrap, seed=6122
    )
    checks = selection_checks(alpha, folds, bootstrap, e42)
    adopted = all(checks.values())
    payload.update(
        alpha=alpha,
        alpha_definition="member=(1-alpha)*binary_lgbm + alpha*three_seed_joint_lgbm",
        alpha_fit="closed-form raw Brier minimizer on 2022 OOF, clipped to [0,1]",
        alpha_source_season=SELECTION_SEASON,
        folds=folds,
        bootstrap_2024_pitcher_cluster=bootstrap,
        selection_checks=checks,
        decision={"adopted": adopted, "status": "adopted" if adopted else "rejected"},
        total_sec=time.time() - started,
        status="complete",
    )
    _save(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    result = run(args.bootstrap_samples, args.reuse_existing)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
