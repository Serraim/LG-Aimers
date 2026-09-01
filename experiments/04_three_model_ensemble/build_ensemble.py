"""
앙상블 제출 모델 생성 (LightGBM + CatBoost + ExtraTrees)
========================================================

무엇을 하나
-----------
같은 62개 피처로 서로 다른 알고리즘 3개를 학습해 하나의 artifact 로 묶는다.
추론 때 `script.py` 가 세 예측을 균등 평균한다.

근거 (scripts/ab_feature.py, 2019~2023 학습 -> 2024 채점 실측)
----------------------------------------------------------------
    모델          점수      LightGBM 과의 상관
    LightGBM    739.87            -
    CatBoost    728.35        0.9490
    ExtraTrees  694.57        0.9154
    LGBM-L2     732.47        0.9824   <- 너무 비슷해 오히려 손해
    MLP         263.24        0.7549   <- 너무 약해 평균을 끌어내림

    조합                    점수
    lgbm+cat+et           757.17   <- 채택
    lgbm+cat+l2+et        756.62
    lgbm+cat              752.01
    lgbm+cat+l2           751.75

제곱오차(=Brier)에서는 다음 항등식이 성립한다.

    앙상블 오차 = 개별 오차의 평균 - 모델들이 서로 다른 정도

빼는 항이 항상 0 이상이므로 **앙상블은 개별 평균보다 나쁠 수 없다.**
다만 "가장 좋은 개별 모델"보다 낫다는 보장은 아니므로, 위처럼 실측이 필요하다.

가중치
------
균등(1/3씩)을 쓴다. 2024 에 맞춘 최적 가중(0.2/0.5/0.3)은 755.49 로 오히려 낮았다.
어떤 데이터에도 맞추지 않은 균등이 2025 로 넘어갈 때 더 안전하다.

트리 수
-------
각 모델의 검증 단계 best_iteration 을 그대로 쓴다. 근거 없는 배수는 곱하지 않는다.
ExtraTrees 는 배깅이라 조기 종료 개념이 없어 300 으로 고정한다.

대회 규칙
---------
모든 파생 피처는 한 행 안에서 계산되며, 리그평균 등 상수는 학습 데이터에서만
계산해 artifact 에 저장한다. 세 모델 모두 같은 행만 보고 예측하므로
행 독립성이 유지된다. (AGENTS.md 2, 5, 8)

사용법
------
    python scripts/build_ensemble.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm as lgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from script import add_domain_features          # 추론과 같은 함수를 재사용

DATA_DIR = REPO / "data"
MODEL_DIR = REPO / "models"
RESULTS_DIR = REPO / "results"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
VALID_SEASON = 2024
SHRINK_K = 200.0          # ab_feature.py 실측 결과 (720.65 -> 739.87)
N_EVAL = 245789           # 평가 데이터 행 수. 추론 시간 측정용

LGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
    subsample=0.8, bootstrap_type="Bernoulli", rsm=0.8,
    loss_function="Logloss", eval_metric="Logloss",
    early_stopping_rounds=100, random_seed=42, thread_count=-1,
    allow_writing_files=False,
)
ET_PARAMS = dict(n_estimators=300, min_samples_leaf=200,
                 max_features=0.6, n_jobs=-1, random_state=42)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score(p, y):
    y = np.asarray(y)
    r = y.mean()
    return max(0.0, 100000.0 * (1.0 - brier(p, y) / (r * (1.0 - r))))


def make_league_avg(df, extrapolate_to=(2025, 2026)):
    """시즌별 리그 평균. 학습 데이터에만 기반한다."""
    avg = {int(s): float(v) for s, v in df.groupby("season")[TARGET].mean().items()}
    ss = sorted(avg)
    coef = np.polyfit(ss, [avg[s] for s in ss], 1)
    for s in extrapolate_to:
        if s not in avg:
            avg[s] = float(np.polyval(coef, s))
    return avg, float(np.mean(list(avg.values())))


def prepare(df, spec, cat_maps, feats):
    X = add_domain_features(df.copy(), spec)[feats].copy()
    for c in CAT_COLS:
        X[c] = X[c].map(cat_maps[c]).fillna(-1).astype("int32")
    return X


def main():
    t_all = time.time()
    print("#" * 72)
    print("#  앙상블 제출 모델 생성 (LightGBM + CatBoost + ExtraTrees)")
    print("#" * 72)

    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier

    t = time.time()
    test_cols = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_feats = [c for c in test_cols if c != ID]
    train = pd.read_csv(DATA_DIR / "train.csv", encoding="utf-8-sig",
                        usecols=raw_feats + [TARGET])
    print(f"  데이터 {train.shape}  ({time.time() - t:.1f}s)")

    # ---------- 1) 검증: 트리 수 결정 + 앙상블 점수 확인 ----------
    print("\n" + "=" * 72)
    print("1) 검증 (2019~2023 -> 2024)")
    print("=" * 72)

    tr = train[train["season"] < VALID_SEASON]
    va = train[train["season"] == VALID_SEASON]
    la_v, def_v = make_league_avg(tr, extrapolate_to=(VALID_SEASON,))
    spec_v = {"league_avg": la_v, "default_league_avg": def_v, "shrink_k": SHRINK_K}
    cat_maps_v = {c: {v: i for i, v in enumerate(sorted(tr[c].dropna().unique()))}
                  for c in CAT_COLS}
    feats = list(add_domain_features(tr.head(50).copy(), spec_v)
                 .drop(columns=[TARGET]).columns)

    X_tr = prepare(tr, spec_v, cat_maps_v, feats)
    X_va = prepare(va, spec_v, cat_maps_v, feats)
    y_tr, y_va = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
    print(f"  학습 {len(X_tr):,}행 x {len(feats)}피처   채점 {len(X_va):,}행")

    preds, iters, secs = {}, {}, {}

    t = time.time()
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    secs["lgbm"] = time.time() - t
    iters["lgbm"] = int(m.best_iteration_)
    preds["lgbm"] = m.predict_proba(X_va)[:, 1]
    print(f"  LightGBM    {score(preds['lgbm'], y_va):8.2f}  "
          f"트리 {iters['lgbm']:4d}  {secs['lgbm']:6.1f}s")

    t = time.time()
    mc = CatBoostClassifier(**CAT_PARAMS)
    mc.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
    secs["catboost"] = time.time() - t
    iters["catboost"] = int(mc.get_best_iteration())
    preds["catboost"] = mc.predict_proba(X_va)[:, 1]
    print(f"  CatBoost    {score(preds['catboost'], y_va):8.2f}  "
          f"트리 {iters['catboost']:4d}  {secs['catboost']:6.1f}s")

    med_v = X_tr.median(numeric_only=True)
    t = time.time()
    me = ExtraTreesClassifier(**ET_PARAMS)
    me.fit(X_tr.fillna(med_v), y_tr)
    secs["extratrees"] = time.time() - t
    iters["extratrees"] = ET_PARAMS["n_estimators"]
    preds["extratrees"] = me.predict_proba(X_va.fillna(med_v))[:, 1]
    print(f"  ExtraTrees  {score(preds['extratrees'], y_va):8.2f}  "
          f"트리 {iters['extratrees']:4d}  {secs['extratrees']:6.1f}s")

    ens = np.mean(list(preds.values()), axis=0)
    s_ens, b_ens = score(ens, y_va), brier(ens, y_va)
    ind = float(np.mean([brier(p, y_va) for p in preds.values()]))
    div = float(np.mean([np.mean((p - ens) ** 2) for p in preds.values()]))
    r = y_va.mean()
    print(f"\n  앙상블(균등)  {s_ens:8.2f}   Brier {b_ens:.6f}")
    print(f"    개별 Brier 평균 {ind:.6f} - 흩어짐 {div:.6f} = {ind - div:.6f}")
    print(f"    흩어짐 점수 환산 {div / (r * (1 - r)) * 100000:+.2f}")
    print(f"    개별 최고 {max(score(p, y_va) for p in preds.values()):.2f} "
          f"-> 앙상블 {s_ens:.2f}")
    print(f"\n  참고: 단일 LightGBM(E05) 검증 점수는 741.82 였다")

    # ---------- 2) 전체 데이터로 재학습 ----------
    print("\n" + "=" * 72)
    print("2) 전체 데이터(2019~2024) 재학습")
    print("=" * 72)

    la_all, def_all = make_league_avg(train, extrapolate_to=(2025, 2026))
    spec_all = {"league_avg": la_all, "default_league_avg": def_all,
                "shrink_k": SHRINK_K}
    print("  리그 평균 (2025~2026 은 추세 외삽):")
    for s in sorted(la_all):
        print(f"    {s}: {la_all[s]:.4f}{'  <- 외삽' if s > 2024 else ''}")

    cat_maps_all = {c: {v: i for i, v in enumerate(sorted(train[c].dropna().unique()))}
                    for c in CAT_COLS}
    X_all = prepare(train, spec_all, cat_maps_all, feats)
    y_all = train[TARGET].to_numpy()
    med_all = X_all.median(numeric_only=True)

    t = time.time()
    f_lgb = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": iters["lgbm"]})
    f_lgb.fit(X_all, y_all)
    print(f"\n  LightGBM   재학습 {time.time() - t:6.1f}s  (트리 {iters['lgbm']})")

    t = time.time()
    f_cat = CatBoostClassifier(**{**CAT_PARAMS, "iterations": iters["catboost"],
                                  "early_stopping_rounds": None})
    f_cat.fit(X_all, y_all, verbose=False)
    print(f"  CatBoost   재학습 {time.time() - t:6.1f}s  (트리 {iters['catboost']})")

    t = time.time()
    f_et = ExtraTreesClassifier(**ET_PARAMS)
    f_et.fit(X_all.fillna(med_all), y_all)
    print(f"  ExtraTrees 재학습 {time.time() - t:6.1f}s  (트리 300)")

    # ---------- 3) 저장 ----------
    env = {
        "python": platform.python_version(), "platform": platform.platform(),
        "lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    try:
        import catboost
        env["catboost"] = catboost.__version__
    except Exception:
        pass

    artifact = {
        "features": feats,
        "cat_cols": CAT_COLS,
        "cat_maps": cat_maps_all,
        "feature_spec": spec_all,
        "models": [
            {"name": "lightgbm", "model": f_lgb, "kind": "proba"},
            {"name": "catboost", "model": f_cat, "kind": "proba"},
            {"name": "extratrees", "model": f_et, "kind": "proba",
             "impute_median": {k: float(v) for k, v in med_all.items()}},
        ],
        "weights": [1.0, 1.0, 1.0],
        "meta": {
            "experiment": "E06-ensemble",
            "selected": "LightGBM + CatBoost + ExtraTrees 균등 평균",
            "valid_score": s_ens,
            "valid_brier": b_ens,
            "valid_individual": {k: score(v, y_va) for k, v in preds.items()},
            "diversity_term": div,
            "best_iteration": iters,
            "shrink_k": SHRINK_K,
            "n_features": len(feats),
            "environment": env,
            "built_by": "scripts/build_ensemble.py",
        },
    }
    MODEL_DIR.mkdir(exist_ok=True)
    out = MODEL_DIR / "ensemble.joblib"
    joblib.dump(artifact, out, compress=3)
    size_mb = out.stat().st_size / 1e6
    print(f"\n  저장: {out}  ({size_mb:.1f} MB)")

    # ---------- 4) 추론 시간 측정 (AGENTS.md 20, 21) ----------
    print("\n" + "=" * 72)
    print("4) 추론 시간 측정 (평가 서버 제한 600s / 28GB)")
    print("=" * 72)
    import script as S
    S._ARTIFACT_CACHE = artifact
    probe = train.head(N_EVAL).drop(columns=[TARGET]).copy()
    probe["season"] = 2025
    probe.insert(0, ID, np.arange(len(probe)))

    t = time.time()
    pp = S.predict_control_success(probe.copy())
    infer_sec = time.time() - t
    print(f"  {len(probe):,}행 추론 {infer_sec:.1f}s")
    print(f"  예측 범위 {pp.min():.4f} ~ {pp.max():.4f}  평균 {pp.mean():.4f}  "
          f"결측 {int(np.isnan(pp).sum())}")

    # 행 독립성 즉석 확인 (AGENTS.md 2)
    idx = np.random.default_rng(0).choice(len(probe), 200, replace=False)
    solo = np.array([S.predict_control_success(probe.iloc[[i]].copy())[0] for i in idx])
    gap = float(np.abs(solo - pp[idx]).max())
    # 판정 기준은 check_rules.py 와 같은 1e-9 를 쓴다.
    #
    # 앙상블은 모델 3개의 예측을 더해 평균내므로, 입력 행 수에 따라
    # 부동소수점 덧셈 순서가 미세하게 달라져 1e-16 수준의 차이가 생긴다.
    # 이는 다른 행을 참조해서 생긴 것이 아니라 반올림 오차이며,
    # 유효숫자 16번째 자리라 예측값에 아무 영향이 없다.
    ok = gap < 1e-9
    print(f"  행 독립성: 200행 개별 vs 전체 최대차 {gap:.3e}  "
          f"{'PASS' if ok else 'FAIL'}  (기준 1e-9)")
    if not ok:
        raise SystemExit("행 독립성 위반. 제출하면 안 된다.")
    if infer_sec > 300:
        print("  주의: 제한 시간의 절반을 넘었다")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "ensemble_metrics.json").write_text(json.dumps({
        "run_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "environment": env,
        "validation": {"scheme": "2019~2023 -> 2024",
                       "n_train": int(len(tr)), "n_valid": int(len(va)),
                       "ensemble_score": s_ens, "ensemble_brier": b_ens,
                       "individual": {k: score(v, y_va) for k, v in preds.items()},
                       "individual_brier": {k: brier(v, y_va) for k, v in preds.items()},
                       "diversity_term": div,
                       "best_iteration": iters, "fit_sec": secs},
        "timing": {"infer_sec": infer_sec, "infer_n_rows": int(len(probe)),
                   "model_file_mb": size_mb,
                   "limits": {"infer_sec": 600, "ram_gb": 28}},
        "row_independence_max_gap": gap,
        "shrink_k": SHRINK_K,
        "features": feats,
        "league_avg": la_all,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n총 {time.time() - t_all:.1f}s")
    print("\n제출 파일 만들기:")
    print("  python scripts/make_submission.py --name e06_ensemble "
          "--model models/ensemble.joblib --lightgbm")


if __name__ == "__main__":
    main()
