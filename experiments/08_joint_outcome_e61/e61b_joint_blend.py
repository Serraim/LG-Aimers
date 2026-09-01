"""E61-B: blend the E42 binary and E61-A joint LightGBM members.

혼합 비율은 2022 OOF raw Brier에서 한 번만 계산하고 2023/2024에 그대로 적용한다.
모델 재학습, test 데이터, leaderboard 결과는 사용하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    from scripts import ab_feature as legacy
    from scripts.e13_feature_groups import PRED_DIR, TARGET, centered_metrics
    from scripts.e41_robust_temporal_weights import MODEL_NAMES, load_predictions
    from scripts.e52_e42_slope_calibration import (
        centered_loss_difference,
        cluster_bootstrap_mean_ci,
    )
    from scripts.e61_joint_outcome import LGBM_INDEX, replace_lgbm_member
except ModuleNotFoundError:
    import ab_feature as legacy
    from e13_feature_groups import PRED_DIR, TARGET, centered_metrics
    from e41_robust_temporal_weights import MODEL_NAMES, load_predictions
    from e52_e42_slope_calibration import centered_loss_difference, cluster_bootstrap_mean_ci
    from e61_joint_outcome import LGBM_INDEX, replace_lgbm_member


REPO = Path(__file__).resolve().parent.parent
E42_RESULT_PATH = REPO / "results" / "e42_frozen_robust_weights.json"
E61_RESULT_PATH = REPO / "results" / "e61_joint_outcome.json"
RESULT_PATH = REPO / "results" / "e61b_joint_blend.json"
VALID_SEASONS = (2022, 2023, 2024)
SELECTION_SEASON = 2022
STRESS_SEASON = 2023
EVALUATION_SEASON = 2024


def validate_sources(e42: dict, e61: dict) -> None:
    if e42.get("experiment_id") != "E42":
        raise RuntimeError("Frozen E42 result is required")
    if e42.get("weight_source") != "2021 OOF only":
        raise RuntimeError("E42 weights must remain train-OOF based")
    if e61.get("experiment_id") != "E61-A" or e61.get("status") != "complete":
        raise RuntimeError("Completed E61-A result is required")
    if e61.get("leaderboard_submissions") != 0:
        raise RuntimeError("E61-A must not use leaderboard selection")


def fit_raw_brier_alpha(
    baseline_prediction: np.ndarray,
    binary_member: np.ndarray,
    joint_member: np.ndarray,
    target: np.ndarray,
    lgbm_weight: float,
) -> float:
    """baseline + alpha * weight * (joint-binary)의 최소제곱 해를 [0, 1]로 제한한다."""
    baseline = np.asarray(baseline_prediction, dtype="float64")
    binary = np.asarray(binary_member, dtype="float64")
    joint = np.asarray(joint_member, dtype="float64")
    y = np.asarray(target, dtype="float64")
    if not (baseline.shape == binary.shape == joint.shape == y.shape):
        raise ValueError("Alpha-fit arrays are not aligned")
    direction = float(lgbm_weight) * (joint - binary)
    denominator = float(np.dot(direction, direction))
    if denominator <= 0:
        raise ValueError("Binary and joint predictions do not define a blend direction")
    unconstrained = float(np.dot(y - baseline, direction) / denominator)
    return float(np.clip(unconstrained, 0.0, 1.0))


def blend_members(binary_member: np.ndarray, joint_member: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    binary = np.asarray(binary_member, dtype="float64")
    joint = np.asarray(joint_member, dtype="float64")
    if binary.shape != joint.shape:
        raise ValueError("Member predictions are not aligned")
    return (1.0 - alpha) * binary + alpha * joint


def baseline_reproduced(folds: dict[str, dict], e42: dict) -> bool:
    for season in VALID_SEASONS:
        reference = e42.get("folds", {}).get(str(season), {}).get("e42")
        current = folds.get(str(season), {}).get("e42")
        if reference is None or current is None:
            return False
        if not np.isclose(
            reference["centered_brier"], current["centered_brier"], rtol=0.0, atol=1e-10
        ):
            return False
    return True


def selection_checks(alpha: float, folds: dict[str, dict], bootstrap: dict, e42: dict) -> dict:
    evaluation = folds[str(EVALUATION_SEASON)]
    return {
        "alpha_fitted_only_on_2022_raw_brier": evaluation["alpha_source_season"]
        == SELECTION_SEASON,
        "alpha_bounded_zero_to_one": 0.0 <= alpha <= 1.0,
        "stress_2023_excluded_from_selection": STRESS_SEASON
        not in (SELECTION_SEASON, EVALUATION_SEASON),
        "e42_oof_baseline_reproduced": baseline_reproduced(folds, e42),
        "raw_brier_improved_2024": evaluation["delta_brier"] < 0,
        "centered_brier_improved_2024": evaluation["delta_centered_brier"] < 0,
        "pitcher_bootstrap_upper_below_zero": float(bootstrap["ci_upper_95"]) < 0,
    }


def _load_joint_prediction(season: int, expected_length: int) -> np.ndarray:
    path = PRED_DIR / f"e61_{season}_joint_lgbm.npy"
    if not path.exists():
        raise FileNotFoundError(f"Run E61-A first: {path}")
    prediction = np.load(path).astype("float64")
    if prediction.shape != (expected_length,):
        raise ValueError(f"Invalid E61-A prediction shape: {path}")
    return prediction


def run(n_bootstrap: int = 2000) -> dict:
    started = time.time()
    e42 = json.loads(E42_RESULT_PATH.read_text(encoding="utf-8"))
    e61 = json.loads(E61_RESULT_PATH.read_text(encoding="utf-8"))
    validate_sources(e42, e61)
    weights = np.asarray([e42["weights"][name] for name in MODEL_NAMES], dtype="float64")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Invalid E42 weights")

    df, _ = legacy.load_data()
    matrices: dict[int, np.ndarray] = {}
    joint_predictions: dict[int, np.ndarray] = {}
    targets: dict[int, np.ndarray] = {}
    for season in VALID_SEASONS:
        target = df.loc[df["season"].eq(season), TARGET].to_numpy(dtype="float64")
        matrix = load_predictions(season)
        if matrix.shape[0] != len(target):
            raise ValueError(f"{season} E42 predictions are not aligned")
        matrices[season] = matrix
        targets[season] = target
        joint_predictions[season] = _load_joint_prediction(season, len(target))

    selection_matrix = matrices[SELECTION_SEASON]
    selection_baseline = selection_matrix @ weights
    alpha = fit_raw_brier_alpha(
        selection_baseline,
        selection_matrix[:, LGBM_INDEX],
        joint_predictions[SELECTION_SEASON],
        targets[SELECTION_SEASON],
        weights[LGBM_INDEX],
    )

    folds: dict[str, dict] = {}
    evaluation: dict[str, np.ndarray] = {}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for season in VALID_SEASONS:
        matrix = matrices[season]
        target = targets[season]
        baseline = matrix @ weights
        blended_member = blend_members(
            matrix[:, LGBM_INDEX], joint_predictions[season], alpha
        )
        candidate = np.clip(replace_lgbm_member(matrix, blended_member) @ weights, 0.0, 1.0)
        np.save(PRED_DIR / f"e61b_{season}_ensemble.npy", candidate.astype("float32"))
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
            "n_rows": int(len(target)),
            "e42": baseline_metrics,
            "e61b": candidate_metrics,
            "delta_brier": candidate_metrics["brier"] - baseline_metrics["brier"],
            "delta_score": candidate_metrics["score"] - baseline_metrics["score"],
            "delta_centered_brier": candidate_metrics["centered_brier"]
            - baseline_metrics["centered_brier"],
            "delta_centered_score": candidate_metrics["centered_score"]
            - baseline_metrics["centered_score"],
        }
        fold = folds[str(season)]
        print(
            f"{season} E61-B alpha={alpha:.6f} delta_brier={fold['delta_brier']:+.8f} "
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
        differences, evaluation["groups"], n_bootstrap=n_bootstrap, seed=6112
    )
    checks = selection_checks(alpha, folds, bootstrap, e42)
    adopted = all(checks.values())
    payload = {
        "experiment_id": "E61-B",
        "description": "2022-fitted blend of E42 binary and E61-A joint LGBM members",
        "alpha": alpha,
        "alpha_definition": "member=(1-alpha)*binary_lgbm + alpha*joint_lgbm",
        "alpha_fit": "closed-form raw Brier minimizer on 2022 OOF, clipped to [0,1]",
        "alpha_source_season": SELECTION_SEASON,
        "models_retrained": False,
        "test_data_used": False,
        "leaderboard_submissions": 0,
        "folds": folds,
        "bootstrap_2024_pitcher_cluster": bootstrap,
        "selection_checks": checks,
        "decision": {"adopted": adopted, "status": "adopted" if adopted else "rejected"},
        "total_sec": time.time() - started,
        "status": "complete",
    }
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    result = run(args.bootstrap_samples)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
