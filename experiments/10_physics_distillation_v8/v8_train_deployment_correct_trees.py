"""V8: V7의 트리 수 오류를 고친 배포용 물리 증류 cb18.

V7 실패 원인 (LB 1089.81 -> 1066.74, -23.07점):
  1. 트리 수를 1000으로 임의 고정했다. 공식 cb18 배포 모델은 547개다
     (build_e18_submission_model.py가 2024 폴드 early stopping 값을 쓴다).
     검증 단계의 distilled 모델은 early stopping이 걸려 훨씬 적은 트리로
     멈췄는데, 배포본만 1000개로 학습해서 **검증한 것과 다른 모델**을 냈다.
  2. 메타러너를 비음수로 재적합했다. 재적합은 T1/U3/V7에서 세 번 다 불안정했다.

V8은 두 오류를 모두 제거한다:
  - 트리 수를 공식 빌더와 **같은 절차**로 정한다: 2024 walk-forward 폴드에서
    early stopping을 걸어 best_iteration을 얻고, 그 값으로 전체 데이터 재학습.
  - 메타러너 계수는 **건드리지 않는다** (배포 계수 그대로).

즉 공식 cb18과 비교했을 때 **바뀌는 것은 학습 타깃 하나뿐**이다
(hard label -> 0.5*hard + 0.5*teacher_soft). 통제된 단일 변수 실험.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parent.parent
ARTIFACT = REPO / "models" / "s37_canonical.joblib"
OUT_MODEL = REPO / "results" / "deploy_distilled_cb18_v8.cbm"
OUT_META = REPO / "results" / "deploy_distilled_cb18_v8_meta.json"

PHYSICS_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
                 "extension", "rel_height", "rel_side"]
ALPHA = 0.5
N_FOLDS = 5
SEED = 42
TREE_CALIBRATION_SEASON = 2024

import sys  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
import ab_feature as legacy  # noqa: E402
from e13_feature_groups import TARGET, make_temporal_features, GROUP_COLS  # noqa: E402
from e13_rolling_validation import _prepare_fold  # noqa: E402
from build_e14_submission_model import _make_frame  # noqa: E402


def build_matched_physics() -> pd.DataFrame:
    plink = pd.read_csv(REPO / "results" / "pitcher_link_all.csv", encoding="utf-8-sig")[["season", "train_pid", "tm_pid"]]
    blink = pd.read_csv(REPO / "results" / "s51_batter_link_consensus.csv", encoding="utf-8-sig")[["train_batter_id", "consensus_tm_batter_id"]]
    cols = ["season", "pitcher_id", "batter_id", "game_month", "game_dayofweek", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before"]
    tr = pd.read_csv(REPO / "data" / "train.csv", encoding="utf-8-sig", usecols=cols)
    tr["row_idx"] = np.arange(len(tr))
    tr = tr.merge(plink, left_on=["season", "pitcher_id"], right_on=["season", "train_pid"], how="left")
    tr = tr.merge(blink, left_on="batter_id", right_on="train_batter_id", how="left")
    tm_cols = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
               "balls_before", "strikes_before", "outs_before",
               "pitcher_trackman_id", "batter_trackman_id"] + PHYSICS_COLS
    tm = pd.read_csv(REPO / "data" / "trackman_history.csv", encoding="utf-8-sig", usecols=tm_cols)
    tm["top_bottom"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B"})
    tm = tm.rename(columns={"pitcher_trackman_id": "tm_pid", "batter_trackman_id": "tm_bid"})
    keys = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
            "balls_before", "strikes_before", "outs_before"]
    sub = tr.dropna(subset=["tm_pid", "consensus_tm_batter_id"]).copy()
    sub["tm_pid"] = sub["tm_pid"].astype("int64")
    sub["tm_bid"] = sub["consensus_tm_batter_id"].astype("int64")
    joined = sub.merge(tm, on=keys + ["tm_pid", "tm_bid"], how="inner")
    return joined.groupby("row_idx")[PHYSICS_COLS].mean().reset_index()


def calibrate_tree_count(physics: pd.DataFrame) -> int:
    """공식 빌더와 같은 절차: 2024 walk-forward 폴드에서 early stopping으로
    best_iteration을 얻는다. 단, 학습 타깃은 증류 소프트 라벨을 쓴다."""
    frame_all, original_features = legacy.load_data()
    temporal = make_temporal_features(frame_all)
    frame, train_mask, valid_mask, base_features = _prepare_fold(
        frame_all, original_features, temporal, TREE_CALIBRATION_SEASON)
    frame = frame.reset_index(drop=True)
    target = frame[TARGET].to_numpy(dtype="float64")
    feats = base_features + GROUP_COLS["W_WB"]
    train_idx = np.flatnonzero(train_mask.to_numpy())
    valid_idx = np.flatnonzero(valid_mask.to_numpy())

    cache = REPO / "results" / f"v2_teacher_oof_cache_{TREE_CALIBRATION_SEASON}.npy"
    if not cache.exists():
        raise FileNotFoundError(f"teacher OOF 캐시 없음: {cache}")
    oof = np.load(cache)
    soft = ALPHA * target[train_idx] + (1 - ALPHA) * oof

    params = dict(legacy.CAT_PARAMS)
    params["loss_function"] = "CrossEntropy"
    params.pop("eval_metric", None)
    m = CatBoostClassifier(**params)
    m.fit(frame[feats].iloc[train_idx], soft,
          eval_set=(frame[feats].iloc[valid_idx], target[valid_idx]), verbose=False)
    best = int(m.get_best_iteration())
    print(f"  early stopping best_iteration = {best} (공식 cb18은 547)", flush=True)
    return max(1, best)


def main() -> None:
    started = time.time()
    print("[1/5] 물리 매칭...", flush=True)
    physics = build_matched_physics()

    print("[2/5] 트리 수 교정 (2024 폴드 early stopping)...", flush=True)
    n_trees = calibrate_tree_count(physics)

    print("[3/5] 공식 배포 경로로 전체 피처 구성...", flush=True)
    df, original_features = legacy.load_data()
    temporal = make_temporal_features(df)
    full_mask = pd.Series(True, index=df.index)
    frame, features, cat_maps, _ = _make_frame(df, original_features, temporal, full_mask)
    frame = frame.reset_index(drop=True)

    art = joblib.load(ARTIFACT)
    official = art["e42"]["catboost"]
    if list(features) != list(official["features"]):
        raise ValueError("피처 순서가 공식 cb18과 다릅니다")
    if cat_maps != official["cat_maps"]:
        raise ValueError("범주 매핑이 공식 cb18과 다릅니다")
    print(f"  피처 계약 일치 ({len(features)}개), 공식 트리수 {official['model'].tree_count_}", flush=True)

    frame["row_idx"] = np.arange(len(frame))
    frame = frame.merge(physics, on="row_idx", how="left")
    target = frame[TARGET].to_numpy(dtype="float64")

    params = dict(legacy.CAT_PARAMS)
    params.update({"iterations": n_trees, "early_stopping_rounds": None})

    print(f"[4/5] teacher {N_FOLDS}-fold OOF (전체 데이터, 물리 포함)...", flush=True)
    x_teacher = frame[features + PHYSICS_COLS]
    oof = np.zeros(len(frame), dtype="float64")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for i, (tr_i, va_i) in enumerate(kf.split(frame), 1):
        m = CatBoostClassifier(**params)
        m.fit(x_teacher.iloc[tr_i], target[tr_i], verbose=False)
        oof[va_i] = m.predict_proba(x_teacher.iloc[va_i])[:, 1]
        print(f"    fold {i}/{N_FOLDS}", flush=True)

    print(f"[5/5] student 학습 (물리 없음, {n_trees} 트리)...", flush=True)
    soft = ALPHA * target + (1 - ALPHA) * oof
    sp = dict(params)
    sp["loss_function"] = "CrossEntropy"
    sp.pop("eval_metric", None)
    student = CatBoostClassifier(**sp)
    student.fit(frame[features], soft, verbose=False)
    student.save_model(str(OUT_MODEL))

    sample = frame.sample(n=min(50000, len(frame)), random_state=0)
    p_new = student.predict_proba(sample[features])[:, 1]
    p_old = official["model"].predict_proba(sample[features])[:, 1]
    corr = float(np.corrcoef(p_new, p_old)[0, 1])
    print(f"  공식 cb18과 상관 {corr:.4f}, 트리수 {student.tree_count_}", flush=True)
    if corr < 0.90:
        raise ValueError(f"상관 {corr:.3f} 너무 낮음 -- 피처 계약 의심")

    meta = {
        "experiment_id": "V8-deployment",
        "description": "physics-distilled cb18 with tree count calibrated the same way the official builder does (2024 fold early stopping), meta coefficients left FROZEN",
        "fixes_vs_v7": ["tree count 1000 -> early-stopping-calibrated", "no meta-learner refit"],
        "calibrated_tree_count": int(n_trees),
        "official_tree_count": int(official["model"].tree_count_),
        "student_tree_count": int(student.tree_count_),
        "alpha": ALPHA,
        "n_rows": int(len(frame)),
        "corr_with_official_cb18": corr,
        "physics_match_rate": float(frame[PHYSICS_COLS[0]].notna().mean()),
        "model_path": str(OUT_MODEL.relative_to(REPO)),
        "elapsed_sec": time.time() - started,
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
