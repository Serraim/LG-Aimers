"""Build E61-B submission artifact from the already trained E61-C artifact.

E61-B uses the single seed-42 joint member and the alpha/shift fixed by E61-B
validation. No retraining or leaderboard-derived value is used.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib


REPO = Path(__file__).resolve().parent.parent
SOURCE_PATH = REPO / "models" / "e61c_joint_seed_ensemble.joblib"
MODEL_PATH = REPO / "models" / "e61b_single_joint_ensemble.joblib"
RESULT_PATH = REPO / "results" / "e61b_submission_metrics.json"
ALPHA = 0.7074745127512206
PREDICTION_SHIFT = -0.010227805998461181


def build() -> dict:
    source = joblib.load(SOURCE_PATH)
    if source.get("artifact_type") != "e61c_joint_seed_ensemble_v1":
        raise RuntimeError("The verified E61-C source artifact is required")
    joint = source.get("joint_member", {})
    models = joint.get("models", [])
    if len(models) != 3 or list(joint.get("seeds", [])) != [42, 2024, 3407]:
        raise RuntimeError("Unexpected E61-C joint seed contract")

    artifact = copy.deepcopy(source)
    artifact["artifact_type"] = "e61b_single_joint_ensemble_v1"
    artifact["alpha"] = ALPHA
    artifact["prediction_shift"] = PREDICTION_SHIFT
    artifact["joint_member"]["models"] = [artifact["joint_member"]["models"][0]]
    artifact["joint_member"]["seeds"] = [42]
    artifact["joint_member"]["seed_aggregation"] = "single fixed seed 42"
    artifact["meta"].update(
        {
            "experiment": "E61-B",
            "submission_purpose": "one structurally different single-seed joint candidate",
            "alpha_source": "2022 OOF raw Brier only",
            "shift_source": "E61-B 2024 validation center error only",
        }
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH, compress=3)
    result = {
        "experiment_id": "E61-B-submission",
        "source_model": str(SOURCE_PATH.relative_to(REPO)),
        "model_path": str(MODEL_PATH.relative_to(REPO)),
        "seeds": [42],
        "alpha": ALPHA,
        "prediction_shift": PREDICTION_SHIFT,
        "leaderboard_submissions": 0,
        "model_size_bytes": MODEL_PATH.stat().st_size,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    build()
