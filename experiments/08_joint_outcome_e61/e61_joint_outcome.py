"""E61: train-only 15-state joint-outcome LightGBM pilot.

현재 행 다음의 같은 투수 ``asof_pitcher_*`` 누적값은 train에서만 보조 결과를
복원하는 데 사용한다. validation 시즌으로 넘어가는 pair는 학습 라벨에서 제외하며,
test 추론에는 현재 행의 기존 72개 W/WB feature만 필요하다.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from scripts import ab_feature as legacy
    from scripts.e13_feature_groups import (
        FIXED_TREES,
        GROUP_COLS,
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
except ModuleNotFoundError:
    import ab_feature as legacy
    from e13_feature_groups import (
        FIXED_TREES,
        GROUP_COLS,
        PRED_DIR,
        TARGET,
        centered_metrics,
        make_temporal_features,
    )
    from e13_rolling_validation import _prepare_fold
    from e41_robust_temporal_weights import MODEL_NAMES, load_predictions
    from e52_e42_slope_calibration import centered_loss_difference, cluster_bootstrap_mean_ci


REPO = Path(__file__).resolve().parent.parent
RESULT_PATH = REPO / "results" / "e61_joint_outcome.json"
E42_RESULT_PATH = REPO / "results" / "e42_frozen_robust_weights.json"
VALID_SEASONS = (2022, 2023, 2024)
NORMAL_SEASONS = (2022, 2024)
SEED = 42
LGBM_INDEX = MODEL_NAMES.index("wwb_lgbm")

# bit: success=1, reverse=2, middle=4, ball=8, strike=16
# 전체 train에서 실제로 관측되는 15개 상태를 사전에 고정한다.
JOINT_CODES = (0, 1, 2, 4, 6, 8, 9, 10, 12, 14, 16, 17, 18, 20, 22)
SUCCESS_LABELS = tuple(index for index, code in enumerate(JOINT_CODES) if code & 1)
AUXILIARY_RATES = (
    ("reverse", "asof_pitcher_reverse_rate", 2),
    ("middle", "asof_pitcher_middle_rate", 4),
    ("ball", "asof_pitcher_ball_rate", 8),
    ("strike", "asof_pitcher_strike_rate", 16),
)
LABEL_COLUMNS = (
    "pitcher_id",
    "season",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    TARGET,
)


def load_label_frame() -> pd.DataFrame:
    """누적률 반올림 오판을 피하려고 라벨 복원 열만 원본 float64로 읽는다."""
    return pd.read_csv(
        REPO / "data" / "train.csv",
        encoding="utf-8-sig",
        usecols=list(LABEL_COLUMNS),
        dtype={
            "pitcher_id": "int32",
            "season": "int16",
            "asof_pitcher_n": "int32",
            TARGET: "int8",
        },
    )


def feature_contract(base_features: list[str]) -> list[str]:
    features = list(base_features) + list(GROUP_COLS["W_WB"])
    if len(features) != 72 or len(set(features)) != 72:
        raise ValueError("E61 requires the unchanged 72-feature W/WB set")
    return features


def _next_by_pitcher(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("pitcher_id", sort=False)[column].shift(-1)


def reconstructed_event(
    current_n: pd.Series,
    current_rate: pd.Series,
    next_n: pd.Series,
    next_rate: pd.Series,
) -> pd.Series:
    """두 누적 스냅샷 사이에 해당 이진 사건이 발생했는지 복원한다."""
    # n=0인 첫 행의 공식 누적률은 NaN이다. 누적 건수는 정의상 0이므로 0으로 둔다.
    increment = next_rate.astype("float64").fillna(0.0) * next_n.astype("float64")
    increment -= current_rate.astype("float64").fillna(0.0) * current_n.astype("float64")
    return increment.gt(0.5)


def reconstruct_joint_codes(
    df: pd.DataFrame, valid_season: int
) -> tuple[pd.Series, pd.Series]:
    """validation 경계를 넘지 않는 train pair에서 15-state code를 만든다."""
    next_n = _next_by_pitcher(df, "asof_pitcher_n")
    next_season = _next_by_pitcher(df, "season")
    eligible = (
        df["season"].lt(valid_season)
        & next_season.lt(valid_season)
        & next_n.eq(df["asof_pitcher_n"].astype("float64") + 1.0)
        & df[TARGET].notna()
    )

    # 공식 target을 success bit로 쓰고, 나머지 네 bit만 다음 train 스냅샷에서 복원한다.
    codes = df[TARGET].fillna(0).astype("int16")
    for _, rate_column, bit in AUXILIARY_RATES:
        event = reconstructed_event(
            df["asof_pitcher_n"],
            df[rate_column],
            next_n,
            _next_by_pitcher(df, rate_column),
        )
        codes = codes + event.astype("int16") * bit
    codes = codes.where(eligible, -1).astype("int16")

    unknown = set(codes.loc[eligible].unique()).difference(JOINT_CODES)
    if unknown:
        raise ValueError(f"Unexpected joint outcome codes: {sorted(unknown)}")
    return codes, eligible


def encode_joint_codes(codes: pd.Series) -> np.ndarray:
    mapping = {code: index for index, code in enumerate(JOINT_CODES)}
    encoded = codes.map(mapping)
    if encoded.isna().any():
        raise ValueError("Joint code mapping is incomplete")
    return encoded.to_numpy(dtype="int16")


def success_probability(probabilities: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """15-class 확률 중 success bit가 켜진 상태(1, 9, 17)를 합친다."""
    values = np.asarray(probabilities, dtype="float64")
    labels = np.asarray(classes, dtype="int64")
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("Probability matrix and classes are not aligned")
    selected = [index for index, label in enumerate(labels) if int(label) in SUCCESS_LABELS]
    if set(labels[selected]) != set(SUCCESS_LABELS):
        raise ValueError("Model does not contain all success joint classes")
    return values[:, selected].sum(axis=1)


def replace_lgbm_member(matrix: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype="float64")
    pred = np.asarray(prediction, dtype="float64")
    if values.ndim != 2 or values.shape[1] != len(MODEL_NAMES):
        raise ValueError("E42 member matrix has invalid shape")
    if pred.shape != (len(values),):
        raise ValueError("Replacement prediction is not aligned")
    result = values.copy()
    result[:, LGBM_INDEX] = pred
    return result


def selection_checks(folds: dict[str, dict], bootstrap: dict) -> dict[str, bool]:
    normal = [folds[str(season)] for season in NORMAL_SEASONS]
    return {
        "single_seed_fixed_without_selection": True,
        "tree_count_fixed_from_e13": FIXED_TREES == 232,
        "stress_2023_excluded": True,
        "raw_brier_improved_2022_and_2024": all(fold["delta_brier"] < 0 for fold in normal),
        "centered_brier_improved_2022_and_2024": all(
            fold["delta_centered_brier"] < 0 for fold in normal
        ),
        "pitcher_bootstrap_upper_below_zero": float(bootstrap["ci_upper_95"]) < 0,
    }


def audit_labels(df: pd.DataFrame) -> dict:
    """긴 학습 전에 라벨 복원 정확도와 상태 분포를 확인한다."""
    boundary = int(df["season"].max()) + 1
    codes, eligible = reconstruct_joint_codes(df, boundary)
    next_n = _next_by_pitcher(df, "asof_pitcher_n")
    reconstructed_success = reconstructed_event(
        df["asof_pitcher_n"],
        df["asof_pitcher_success_rate"],
        next_n,
        _next_by_pitcher(df, "asof_pitcher_success_rate"),
    )
    target = df[TARGET].astype(bool)
    agreement = float((reconstructed_success.loc[eligible] == target.loc[eligible]).mean())
    counts = codes.loc[eligible].value_counts().sort_index()
    return {
        "n_rows": int(len(df)),
        "n_eligible": int(eligible.sum()),
        "n_unavailable": int((~eligible).sum()),
        "reconstructed_success_agreement": agreement,
        "observed_codes": [int(value) for value in counts.index],
        "state_counts": {str(int(code)): int(count) for code, count in counts.items()},
        "minimum_state_count": int(counts.min()),
    }


def _save(payload: dict) -> None:
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _train_joint_model(frame, train_mask, valid_mask, codes, features):
    eligible_mask = train_mask & codes.ne(-1)
    labels = encode_joint_codes(codes.loc[eligible_mask])
    if set(np.unique(labels)) != set(range(len(JOINT_CODES))):
        raise RuntimeError("A training fold is missing one or more fixed joint classes")
    params = dict(legacy.PARAMS)
    params.update(
        objective="multiclass",
        num_class=len(JOINT_CODES),
        n_estimators=FIXED_TREES,
        random_state=SEED,
    )
    model = lgb.LGBMClassifier(**params)
    started = time.time()
    model.fit(frame.loc[eligible_mask, features], labels)
    probabilities = model.predict_proba(frame.loc[valid_mask, features])
    prediction = success_probability(probabilities, model.classes_)
    fit_sec = time.time() - started
    del model, probabilities
    gc.collect()
    return prediction, fit_sec, int(eligible_mask.sum()), np.bincount(labels, minlength=15)


def run(n_bootstrap: int = 2000, reuse_existing: bool = False) -> dict:
    started = time.time()
    e42 = json.loads(E42_RESULT_PATH.read_text(encoding="utf-8"))
    weights = np.asarray([e42["weights"][name] for name in MODEL_NAMES], dtype="float64")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError("Invalid E42 weights")

    df, original_features = legacy.load_data()
    label_frame = load_label_frame()
    if len(label_frame) != len(df):
        raise RuntimeError("Feature and joint-label rows are not aligned")
    audit = audit_labels(label_frame)
    if audit["reconstructed_success_agreement"] != 1.0:
        raise RuntimeError("Success reconstruction did not exactly match the official target")
    if audit["observed_codes"] != list(JOINT_CODES):
        raise RuntimeError("Observed joint-state contract changed")

    temporal = make_temporal_features(df)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    folds: dict[str, dict] = {}
    evaluation: dict[str, np.ndarray] = {}
    payload = {
        "experiment_id": "E61-A",
        "description": "15-state train-only joint-outcome LightGBM replaces E42 LGBM member",
        "data_scope": "official train only; test is not read",
        "validation": "rolling 2022/2023/2024; pairs crossing validation boundary excluded",
        "features": "unchanged E42 W/WB 72 features",
        "joint_codes": list(JOINT_CODES),
        "success_codes": [code for code in JOINT_CODES if code & 1],
        "seed": SEED,
        "trees": FIXED_TREES,
        "audit": audit,
        "folds": folds,
        "leaderboard_submissions": 0,
        "status": "running",
    }
    _save(payload)

    for valid_season in VALID_SEASONS:
        frame, train_mask, valid_mask, base_features = _prepare_fold(
            df, original_features, temporal, valid_season
        )
        features = feature_contract(base_features)
        codes, eligible = reconstruct_joint_codes(label_frame, valid_season)
        pred_path = PRED_DIR / f"e61_{valid_season}_joint_lgbm.npy"
        if reuse_existing and pred_path.exists():
            joint_prediction = np.load(pred_path).astype("float64")
            if joint_prediction.shape != (int(valid_mask.sum()),):
                raise ValueError(f"Invalid cached prediction: {pred_path}")
            fit_sec = 0.0
            reused = True
            class_counts = np.bincount(
                encode_joint_codes(codes.loc[train_mask & eligible]), minlength=15
            )
            n_train_joint = int((train_mask & eligible).sum())
        else:
            joint_prediction, fit_sec, n_train_joint, class_counts = _train_joint_model(
                frame, train_mask, valid_mask, codes, features
            )
            np.save(pred_path, joint_prediction.astype("float32"))
            reused = False

        y_valid = frame.loc[valid_mask, TARGET].to_numpy(dtype="float64")
        matrix = load_predictions(valid_season)
        baseline_prediction = matrix @ weights
        candidate_prediction = np.clip(
            replace_lgbm_member(matrix, joint_prediction) @ weights, 0.0, 1.0
        )
        baseline_metrics = centered_metrics(baseline_prediction, y_valid)
        candidate_metrics = centered_metrics(candidate_prediction, y_valid)
        fold = {
            "fold_role": "stress" if valid_season == 2023 else "normal",
            "n_train_binary_available": int(train_mask.sum()),
            "n_train_joint": n_train_joint,
            "n_boundary_or_last_rows_excluded": int(train_mask.sum() - n_train_joint),
            "n_valid": int(valid_mask.sum()),
            "n_features": len(features),
            "class_counts": {str(code): int(class_counts[index]) for index, code in enumerate(JOINT_CODES)},
            "fit_sec": fit_sec,
            "reused_prediction": reused,
            "binary_lgbm": centered_metrics(matrix[:, LGBM_INDEX], y_valid),
            "joint_lgbm": centered_metrics(joint_prediction, y_valid),
            "e42": baseline_metrics,
            "e61": candidate_metrics,
            "delta_brier": candidate_metrics["brier"] - baseline_metrics["brier"],
            "delta_score": candidate_metrics["score"] - baseline_metrics["score"],
            "delta_centered_brier": candidate_metrics["centered_brier"]
            - baseline_metrics["centered_brier"],
            "delta_centered_score": candidate_metrics["centered_score"]
            - baseline_metrics["centered_score"],
        }
        folds[str(valid_season)] = fold
        print(
            f"{valid_season} E61 delta_brier={fold['delta_brier']:+.8f} "
            f"delta_centered_score={fold['delta_centered_score']:+.2f} "
            f"fit={fit_sec:.1f}s",
            flush=True,
        )
        if valid_season == 2024:
            evaluation = {
                "baseline": baseline_prediction,
                "candidate": candidate_prediction,
                "target": y_valid,
                "groups": df.loc[valid_mask, "pitcher_id"].to_numpy(),
            }
        payload.update(folds=folds, elapsed_sec=time.time() - started)
        _save(payload)
        del frame, codes, joint_prediction, matrix
        gc.collect()

    differences = centered_loss_difference(
        evaluation["baseline"], evaluation["candidate"], evaluation["target"]
    )
    bootstrap = cluster_bootstrap_mean_ci(
        differences, evaluation["groups"], n_bootstrap=n_bootstrap, seed=6102
    )
    checks = selection_checks(folds, bootstrap)
    adopted = all(checks.values())
    payload.update(
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
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit_labels(load_label_frame()), ensure_ascii=False, indent=2))
        return
    result = run(args.bootstrap_samples, args.reuse_existing)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
