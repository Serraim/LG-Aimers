"""Build the one-time E61-C model-candidate submission artifact."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

try:
    from scripts import ab_feature as legacy
    from scripts.build_e14_submission_model import _make_frame
    from scripts.e13_feature_groups import FIXED_TREES, TARGET, make_temporal_features
    from scripts.e61_joint_outcome import (
        JOINT_CODES,
        SUCCESS_LABELS,
        audit_labels,
        encode_joint_codes,
        load_label_frame,
        reconstruct_joint_codes,
    )
    from scripts.e61c_joint_seed_ensemble import DEFAULT_SEEDS
except ModuleNotFoundError:
    import ab_feature as legacy
    from build_e14_submission_model import _make_frame
    from e13_feature_groups import FIXED_TREES, TARGET, make_temporal_features
    from e61_joint_outcome import (
        JOINT_CODES,
        SUCCESS_LABELS,
        audit_labels,
        encode_joint_codes,
        load_label_frame,
        reconstruct_joint_codes,
    )
    from e61c_joint_seed_ensemble import DEFAULT_SEEDS


REPO = Path(__file__).resolve().parent.parent
VALIDATION_PATH = REPO / "results" / "e61c_joint_seed_ensemble.json"
E42_MODEL_PATH = REPO / "models" / "e42_frozen_robust.joblib"
MODEL_PATH = REPO / "models" / "e61c_joint_seed_ensemble.joblib"
RESULT_PATH = REPO / "results" / "e61c_submission_metrics.json"


def validate_candidate(payload: dict) -> tuple[list[int], float, float]:
    """Validate the explicit one-submission override and return frozen settings.

    E61-C failed only the conservative bootstrap-significance gate. Submission is
    still a legitimate structurally different model comparison because every
    rolling fold improved and no leaderboard value selected any setting.
    """
    if payload.get("experiment_id") != "E61-C" or payload.get("status") != "complete":
        raise RuntimeError("Completed E61-C validation is required")
    if payload.get("leaderboard_submissions") != 0 or payload.get("test_data_used") is not False:
        raise RuntimeError("Leaderboard/test-derived E61-C is forbidden")
    if payload.get("alpha_source_season") != 2022:
        raise RuntimeError("E61-C alpha must come only from 2022 OOF")
    seeds = [int(seed) for seed in payload.get("seeds", [])]
    if seeds != list(DEFAULT_SEEDS):
        raise RuntimeError("E61-C seed contract changed")
    if payload.get("trees") != FIXED_TREES:
        raise RuntimeError("E61-C tree count changed")

    checks = payload.get("selection_checks", {})
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed != ["pitcher_bootstrap_upper_below_zero"]:
        raise RuntimeError(f"Unexpected E61-C failed checks: {failed}")
    for season in (2022, 2023, 2024):
        fold = payload.get("folds", {}).get(str(season), {})
        if fold.get("delta_brier", 1.0) >= 0 or fold.get("delta_centered_brier", 1.0) >= 0:
            raise RuntimeError(f"E61-C did not improve raw/centered Brier in {season}")

    alpha = float(payload["alpha"])
    if not 0.0 <= alpha <= 1.0:
        raise RuntimeError("E61-C alpha is outside [0,1]")
    evaluation = payload["folds"]["2024"]
    if evaluation.get("fold_role") != "evaluation":
        raise RuntimeError("E61-C 2024 fold is not the fixed evaluation fold")
    prediction_shift = -float(evaluation["e61c"]["center_error"])
    if not np.isfinite(prediction_shift):
        raise RuntimeError("E61-C validation shift is invalid")
    return seeds, alpha, prediction_shift


def _environment() -> dict[str, str]:
    packages = ("lightgbm", "catboost", "xgboost", "scikit-learn", "pandas", "numpy")
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def build() -> dict:
    started = time.time()
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    seeds, alpha, prediction_shift = validate_candidate(validation)
    e42 = joblib.load(E42_MODEL_PATH)
    if e42.get("artifact_type") != "e42_frozen_robust_v1":
        raise RuntimeError("Frozen E42 artifact is invalid")

    df, original_features = legacy.load_data()
    label_frame = load_label_frame()
    if len(label_frame) != len(df):
        raise RuntimeError("Feature and joint-label rows are not aligned")
    label_audit = audit_labels(label_frame)
    if label_audit["reconstructed_success_agreement"] != 1.0:
        raise RuntimeError("Joint-label reconstruction audit failed")

    temporal = make_temporal_features(df)
    full_mask = pd.Series(True, index=df.index)
    frame, features, cat_maps, _ = _make_frame(df, original_features, temporal, full_mask)
    binary_member = e42["wwb_lgbm"]
    if features != binary_member["features"] or cat_maps != binary_member["cat_maps"]:
        raise RuntimeError("E61-C feature/category schema differs from frozen E42")
    if len(features) != 72 or len(set(features)) != 72:
        raise RuntimeError("E61-C final feature contract is invalid")

    boundary = int(label_frame["season"].max()) + 1
    codes, eligible = reconstruct_joint_codes(label_frame, boundary)
    labels = encode_joint_codes(codes.loc[eligible])
    if set(np.unique(labels)) != set(range(len(JOINT_CODES))):
        raise RuntimeError("Full training data is missing a fixed joint class")

    models = []
    seed_records = []
    for seed in seeds:
        params = dict(legacy.PARAMS)
        params.update(
            objective="multiclass",
            num_class=len(JOINT_CODES),
            n_estimators=FIXED_TREES,
            random_state=int(seed),
        )
        model = lgb.LGBMClassifier(**params)
        fit_started = time.time()
        model.fit(frame.loc[eligible, features], labels)
        fit_sec = time.time() - fit_started
        if list(map(int, model.classes_)) != list(range(len(JOINT_CODES))):
            raise RuntimeError("Final joint model class order changed")
        models.append(model)
        seed_records.append({"seed": int(seed), "fit_sec": fit_sec})
        print(f"trained final joint seed={seed} trees={FIXED_TREES} fit={fit_sec:.1f}s", flush=True)

    joint_member = {
        "models": models,
        "seeds": seeds,
        "features": features,
        "success_labels": list(SUCCESS_LABELS),
        "joint_codes": list(JOINT_CODES),
        "seed_aggregation": "fixed equal arithmetic mean",
        "n_train_joint": int(eligible.sum()),
    }
    artifact = {
        "artifact_type": "e61c_joint_seed_ensemble_v1",
        "e42": e42,
        "joint_member": joint_member,
        "alpha": alpha,
        "prediction_shift": prediction_shift,
        "meta": {
            "experiment": "E61-C",
            "submission_purpose": "one structurally different model-candidate comparison",
            "internal_gate_override": "all rolling folds improved; only 2024 bootstrap upper<0 failed narrowly",
            "alpha_source": "2022 OOF raw Brier only",
            "shift_source": "E61-C 2024 validation center error only",
            "test_other_rows_used": False,
            "leaderboard_submissions_before_build": 0,
            "environment": _environment(),
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)

    result = {
        "experiment_id": "E61-C-submission",
        "model_path": str(MODEL_PATH.relative_to(REPO)),
        "n_train": int(len(df)),
        "n_train_joint": int(eligible.sum()),
        "n_features": len(features),
        "seeds": seeds,
        "trees": FIXED_TREES,
        "alpha": alpha,
        "prediction_shift": prediction_shift,
        "shift_source": artifact["meta"]["shift_source"],
        "seed_records": seed_records,
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "leaderboard_submissions": 0,
        "total_sec": time.time() - started,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    build()
