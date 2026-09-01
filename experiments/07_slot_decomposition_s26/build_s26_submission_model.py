"""Build S26-A or S26-B final model and submission ZIP.

긴 학습은 사용자가 터미널에서 명시적으로 실행한다. 후보 선택, 구성원 가중치,
트리 수, seed, 중심 보정은 저장된 train/validation 결과에서만 읽는다.
"""

from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

try:
    from scripts import ab_feature as legacy
    from scripts.build_e14_submission_model import _make_frame
    from scripts.e13_feature_groups import FIXED_TREES, make_temporal_features
    from scripts.e61_joint_outcome import (
        JOINT_CODES, SUCCESS_LABELS, audit_labels, encode_joint_codes,
        load_label_frame, reconstruct_joint_codes,
    )
    from scripts.e62_core_outcome import (
        CORE_CODES, DEFAULT_SEEDS as CORE_SEEDS, SUCCESS_CORE_CODE,
        collapse_to_core_codes, encode_core_codes,
    )
    from scripts.e63_multilabel_catboost import LABEL_NAMES, multilabel_targets
    from scripts.e65_call_outcome import (
        CALL_CODES, DEFAULT_SEEDS as CALL_SEEDS, SUCCESS_CALL_CODES,
        collapse_to_call_codes, encode_call_codes,
    )
except ModuleNotFoundError:
    import ab_feature as legacy
    from build_e14_submission_model import _make_frame
    from e13_feature_groups import FIXED_TREES, make_temporal_features
    from e61_joint_outcome import JOINT_CODES, SUCCESS_LABELS, audit_labels, encode_joint_codes, load_label_frame, reconstruct_joint_codes
    from e62_core_outcome import CORE_CODES, DEFAULT_SEEDS as CORE_SEEDS, SUCCESS_CORE_CODE, collapse_to_core_codes, encode_core_codes
    from e63_multilabel_catboost import LABEL_NAMES, multilabel_targets
    from e65_call_outcome import CALL_CODES, DEFAULT_SEEDS as CALL_SEEDS, SUCCESS_CALL_CODES, collapse_to_call_codes, encode_call_codes


REPO = Path(__file__).resolve().parent.parent
S26_PATH = REPO / "results" / "s26_slot_combination.json"
E42_PATH = REPO / "models" / "e42_frozen_robust.joblib"
E63_PATH = REPO / "results" / "e63_multilabel_catboost.json"
E69_PATH = REPO / "results" / "e69_joint_catboost.json"
INFERENCE_PATH = REPO / "scripts" / "s26_script.py"
COMPONENT_DIR = REPO / "models" / "s26_components"


def validate_selection(payload: dict, candidate: str) -> tuple[dict, float]:
    candidate = candidate.upper()
    if payload.get("experiment_id") != "S26":
        raise RuntimeError("S26 validation result is required")
    if payload.get("eval_season_used_in_selection") is not False:
        raise RuntimeError("2024 must not be used in S26 candidate selection")
    if int(payload.get("search_space", {}).get("n_combinations", 0)) != 1260:
        raise RuntimeError("S26 search-space contract changed")
    if not payload.get("post_submission_rule"):
        raise RuntimeError("Post-submission no-retuning rule is missing")
    selection = payload.get("selections", {}).get(candidate)
    if selection is None:
        raise ValueError(f"Unknown S26 candidate: {candidate}")
    folds = selection["folds"]
    if float(folds["2022"]["delta_centered_vs_e61c"]) <= 0:
        raise RuntimeError("S26 candidate must improve 2022")
    if float(folds["2024"]["delta_centered_vs_e61c"]) <= 0:
        raise RuntimeError("S26 candidate must improve held-out 2024")
    if candidate == "A":
        if float(selection["bootstrap_2024_vs_e61c"]["ci_upper_95"]) >= 0:
            raise RuntimeError("S26-A must pass the fixed 2024 bootstrap gate")
    elif candidate == "B":
        if any(float(folds[str(year)]["delta_centered_vs_e61c"]) <= 0 for year in (2022, 2023, 2024)):
            raise RuntimeError("S26-B must improve all three folds")
    else:
        raise ValueError("Candidate must be A or B")
    shift = -float(folds["2024"]["center_error"])
    if not np.isfinite(shift):
        raise RuntimeError("Invalid validation-derived shift")
    return selection, shift


def _lgbm_models(frame, features, eligible, labels, n_classes, seeds, label_name):
    models, records = [], []
    for seed in seeds:
        params = dict(legacy.PARAMS)
        params.update(
            objective="multiclass", num_class=n_classes,
            n_estimators=FIXED_TREES, random_state=int(seed),
        )
        model = lgb.LGBMClassifier(**params)
        started = time.time()
        model.fit(frame.loc[eligible, features], labels)
        elapsed = time.time() - started
        models.append(model)
        records.append({"member": label_name, "seed": int(seed), "trees": FIXED_TREES, "fit_sec": elapsed})
        print(f"trained {label_name} seed={seed} trees={FIXED_TREES} fit={elapsed:.1f}s", flush=True)
    return models, records


def _catboost_model(frame, features, eligible, labels, iterations, loss, member_name):
    params = dict(legacy.CAT_PARAMS)
    params.update(
        iterations=int(iterations), loss_function=loss, eval_metric=loss,
        early_stopping_rounds=None, random_seed=42, allow_writing_files=False,
    )
    model = CatBoostClassifier(**params)
    started = time.time()
    model.fit(frame.loc[eligible, features], labels, verbose=False)
    elapsed = time.time() - started
    print(f"trained {member_name} seed=42 trees={iterations} fit={elapsed:.1f}s", flush=True)
    return model, {"member": member_name, "seed": 42, "trees": int(iterations), "fit_sec": elapsed}


def _catboost_iterations(result_path: Path) -> int:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records = payload["training_records"]["2024"]["seed_records"]
    if len(records) != 1 or int(records[0]["seed"]) != 42:
        raise RuntimeError(f"Unexpected CatBoost seed record: {result_path}")
    return max(1, int(records[0]["best_iteration"]))


def _package(candidate: str, model_path: Path) -> tuple[Path, str]:
    zip_path = REPO / "submissions" / f"s26{candidate.lower()}_slot_combo.zip"
    # E61-C와 동일한 검증 완료 추론 의존성만 명시한다.
    requirements = "\n".join([
        "# 평가 서버 기본 설치 패키지는 명시하지 않음",
        "catboost==1.2.7", "lightgbm==4.5.0",
        "scikit-learn==1.9.0", "xgboost==2.1.1", "",
    ])
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(model_path, f"model/{model_path.name}")
        archive.write(INFERENCE_PATH, "script.py")
        archive.writestr("requirements.txt", requirements)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Submission ZIP CRC failed")
        if set(archive.namelist()) != {f"model/{model_path.name}", "script.py", "requirements.txt"}:
            raise RuntimeError("Submission ZIP structure changed")
    import hashlib
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()
    return zip_path, digest


def _component_path(name: str) -> Path:
    return COMPONENT_DIR / f"{name}.joblib"


def _load_component(name: str, features: list[str], reuse_existing: bool):
    path = _component_path(name)
    if not reuse_existing or not path.exists():
        return None
    payload = joblib.load(path)
    if payload.get("component") != name or payload.get("features") != features:
        raise RuntimeError(f"Invalid S26 component checkpoint: {path}")
    print(f"reused checkpoint {path.relative_to(REPO)}", flush=True)
    return payload["member"]


def _save_component(name: str, features: list[str], member: dict) -> None:
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"component": name, "features": features, "member": member},
        _component_path(name), compress=3,
    )


def build(candidate: str, reuse_existing: bool = False) -> dict:
    candidate = candidate.upper()
    started = time.time()
    s26 = json.loads(S26_PATH.read_text(encoding="utf-8"))
    selection, prediction_shift = validate_selection(s26, candidate)
    e42 = joblib.load(E42_PATH)
    if e42.get("artifact_type") != "e42_frozen_robust_v1":
        raise RuntimeError("Frozen E42 artifact is invalid")
    df, original_features = legacy.load_data()
    label_frame = load_label_frame()
    if len(df) != len(label_frame) or audit_labels(label_frame)["reconstructed_success_agreement"] != 1.0:
        raise RuntimeError("Joint-label alignment/audit failed")
    temporal = make_temporal_features(df)
    full_mask = pd.Series(True, index=df.index)
    frame, features, cat_maps, _ = _make_frame(df, original_features, temporal, full_mask)
    binary = e42["wwb_lgbm"]
    if features != binary["features"] or cat_maps != binary["cat_maps"] or len(features) != 72:
        raise RuntimeError("S26/E42 feature schema mismatch")
    boundary = int(label_frame["season"].max()) + 1
    joint_codes, eligible = reconstruct_joint_codes(label_frame, boundary)
    members, records = {}, []

    core_member = _load_component("core5", features, reuse_existing)
    if core_member is None:
        core_codes = collapse_to_core_codes(joint_codes)
        core_labels = encode_core_codes(core_codes.loc[eligible])
        core_models, core_records = _lgbm_models(
            frame, features, eligible, core_labels, len(CORE_CODES), tuple(CORE_SEEDS), "cor5"
        )
        core_member = {
            "models": core_models, "seeds": list(CORE_SEEDS),
            "success_labels": [CORE_CODES.index(SUCCESS_CORE_CODE)],
        }
        _save_component("core5", features, core_member)
        records.extend(core_records)
    else:
        records.append({"member": "cor5", "source": "checkpoint"})
    members["core5"] = core_member

    if candidate == "B":
        call_member = _load_component("call6", features, reuse_existing)
        if call_member is None:
            call_codes = collapse_to_call_codes(joint_codes)
            call_labels = encode_call_codes(call_codes.loc[eligible])
            call_models, call_records = _lgbm_models(
                frame, features, eligible, call_labels, len(CALL_CODES), tuple(CALL_SEEDS), "cal6"
            )
            call_member = {
                "models": call_models, "seeds": list(CALL_SEEDS),
                "success_labels": [CALL_CODES.index(code) for code in SUCCESS_CALL_CODES],
            }
            _save_component("call6", features, call_member)
            records.extend(call_records)
        else:
            records.append({"member": "cal6", "source": "checkpoint"})
        members["call6"] = call_member
    else:
        mcb_member = _load_component("multilabel_catboost", features, reuse_existing)
        if mcb_member is None:
            mcb_iterations = _catboost_iterations(E63_PATH)
            mcb_model, mcb_record = _catboost_model(
                frame, features, eligible, multilabel_targets(joint_codes.loc[eligible]),
                mcb_iterations, "MultiLogloss", "mcb",
            )
            mcb_member = {"model": mcb_model, "labels": list(LABEL_NAMES), "iterations": mcb_iterations}
            _save_component("multilabel_catboost", features, mcb_member)
            records.append(mcb_record)
        else:
            records.append({"member": "mcb", "source": "checkpoint"})
        members["multilabel_catboost"] = mcb_member
        e69_member = _load_component("joint_catboost", features, reuse_existing)
        if e69_member is None:
            e69_iterations = _catboost_iterations(E69_PATH)
            e69_model, e69_record = _catboost_model(
                frame, features, eligible, encode_joint_codes(joint_codes.loc[eligible]),
                e69_iterations, "MultiClass", "e69",
            )
            e69_member = {
                "model": e69_model, "joint_codes": list(JOINT_CODES),
                "success_labels": list(SUCCESS_LABELS), "iterations": e69_iterations,
            }
            _save_component("joint_catboost", features, e69_member)
            records.append(e69_record)
        else:
            records.append({"member": "e69", "source": "checkpoint"})
        members["joint_catboost"] = e69_member

    weights = {name: float(value) for name, value in selection["final_member_weights"].items() if float(value) > 1e-12}
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-12) or any(value < 0 for value in weights.values()):
        raise RuntimeError("S26 final member weights are invalid")
    artifact = {
        "artifact_type": "s26_slot_combo_v1", "candidate": candidate,
        "e42": e42, "members": members, "member_weights": weights,
        "prediction_shift": prediction_shift,
        "meta": {
            "selection_source": "S26 rule A: 2022 only" if candidate == "A" else "S26 rule B: 2022+2023 only",
            "evaluation_2024_used_in_selection": False,
            "shift_source": f"S26-{candidate} 2024 validation center error only",
            "post_submission_rule": s26["post_submission_rule"],
            "test_other_rows_used": False,
        },
    }
    model_path = REPO / "models" / f"s26{candidate.lower()}_slot_combo.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    zip_path, digest = _package(candidate, model_path)
    result = {
        "experiment_id": f"S26-{candidate}-submission", "candidate": candidate,
        "model_path": str(model_path.relative_to(REPO)), "zip_path": str(zip_path.relative_to(REPO)),
        "member_weights": weights, "prediction_shift": prediction_shift,
        "shift_source": artifact["meta"]["shift_source"], "training_records": records,
        "model_size_bytes": model_path.stat().st_size, "zip_size_bytes": zip_path.stat().st_size,
        "zip_sha256": digest, "leaderboard_submissions": 0,
        "total_sec": time.time() - started,
    }
    result_path = REPO / "results" / f"s26{candidate.lower()}_submission_metrics.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("A", "B", "a", "b"), required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    build(args.candidate, args.reuse_existing)


if __name__ == "__main__":
    main()
