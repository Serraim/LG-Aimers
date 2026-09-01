"""
중심 오차 측정 (2021~2024)
==========================

무엇을 하나
-----------
"우리 모델은 다음 시즌을 예측할 때 평균을 얼마나 높게(낮게) 찍는가" 를
여러 해에 걸쳐 재고, 그 패턴으로 2025 를 추정한다.

왜 필요한가
-----------
Brier 는 다음과 같이 쪼개진다.

    Brier = 어쩔 수 없는 부분 + (예측평균 - 실제 r)^2

뒤쪽은 **모델 성능과 무관한 벌점**이다. 2024 실측으로는

    예측 평균 0.4961,  실제 r 0.4861  ->  차이 0.0100  ->  40.3점 손해

성공률이 매년 떨어지는데 나무는 학습에서 본 범위 밖으로 나가지 못해
끝까지 내려오지 못한다. 구조적이므로 매년 같은 방향으로 어긋날 것으로 본다.

읽는 법
-------
    일정하다      -> 2025 도 같은 값. 그만큼 빼면 그 벌점이 0 이 된다
    추세가 있다   -> 연장해서 2025 를 추정
    제멋대로다    -> 이 방법은 쓸 수 없다. 중단

주의
----
로컬 검증은 리더보드와 방향이 뒤집힌 적이 두 번 있다.
여기서 얻은 추정은 **리더보드 2회 제출(+-0.01)로 확인해야 한다.**

사용법
------
    python scripts/center_error.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from script import add_domain_features
from build_feat_model import make_league_avg, SHRINK_K

CAT = ["top_bottom", "game_type", "base_state"]
T = "control_success"
GROUPS = ["P", "B", "B2", "M", "S"]     # batter 확장 포함 (LB +5.10 로 검증됨)

PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63,
              min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0,
              n_jobs=-1, random_state=42, verbose=-1)


def main():
    cols = [c for c in pd.read_csv(REPO / "data/test.csv", encoding="utf-8-sig",
                                   nrows=0).columns if c != "row_id"]
    df = pd.read_csv(REPO / "data/train.csv", encoding="utf-8-sig",
                     usecols=cols + [T])

    print(f"{'채점':>6} {'학습':>11} {'실제 r':>8} {'예측평균':>9} {'중심오차':>9} "
          f"{'벌점':>7} {'점수':>8} {'1군오차':>8} {'2군오차':>8}")
    print("-" * 88)
    rows = []
    for V in (2021, 2022, 2023, 2024):
        tr = df[df.season < V]
        va = df[df.season == V]
        la, dv = make_league_avg(tr, extrapolate_to=(V,))
        spec = {"league_avg": la, "default_league_avg": dv, "shrink_k": SHRINK_K,
                "groups": GROUPS,
                "reverse_league_avg": float(tr.asof_pitcher_reverse_rate.mean()),
                "middle_league_avg": float(tr.asof_pitcher_middle_rate.mean()),
                "batter_middle_league_avg": float(tr.asof_batter_middle_rate.mean())}
        cm = {c: {v: i for i, v in enumerate(sorted(tr[c].dropna().unique()))}
              for c in CAT}
        fe = list(add_domain_features(tr.head(50).copy(), spec)
                  .drop(columns=[T]).columns)

        def prep(d):
            X = add_domain_features(d.copy(), spec)[fe].copy()
            for c in CAT:
                X[c] = X[c].map(cm[c]).fillna(-1).astype("int32")
            for c in X.columns:
                if X[c].dtype == "float64":
                    X[c] = X[c].astype("float32")
            return X

        Xt, Xv = prep(tr), prep(va)
        yt, yv = tr[T].to_numpy(), va[T].to_numpy()
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xt, yt, eval_set=[(Xv, yv)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict_proba(Xv)[:, 1]

        r = yv.mean(); C = r * (1 - r); A = p.mean() - r
        br = float(np.mean((p - yv) ** 2))
        sc = max(0.0, 100000 * (1 - br / C))
        grp = va["game_type"].to_numpy()
        eR = p[grp == "R"].mean() - yv[grp == "R"].mean()
        eF = (p[grp == "F"].mean() - yv[grp == "F"].mean()
              if (grp == "F").sum() > 1000 else np.nan)
        rows.append({"valid_season": int(V), "r": float(r),
                     "pred_mean": float(p.mean()), "center_error": float(A),
                     "penalty_pts": float(A * A / C * 100000),
                     "score": float(sc), "brier": br,
                     "center_error_R": float(eR),
                     "center_error_F": (float(eF) if eF == eF else None),
                     "best_iter": int(m.best_iteration_)})
        print(f"{V:>6} {f'2019~{V-1}':>11} {r:8.4f} {p.mean():9.4f} {A:+9.4f} "
              f"{A*A/C*100000:7.1f} {sc:8.2f} {eR:+8.4f} {eF:+8.4f}")

    a = [x["center_error"] for x in rows]
    print()
    print(f"중심 오차 4년치 : {[f'{x:+.4f}' for x in a]}")
    print(f"  평균 {np.mean(a):+.4f}   표준편차 {np.std(a):.4f}")
    co = np.polyfit(range(len(a)), a, 1)
    print(f"  직선 기울기 {co[0]:+.5f}/년  ->  2025 추정 {np.polyval(co, len(a)):+.4f}")
    print(f"  단순 평균 추정                  ->  2025 {np.mean(a):+.4f}")
    print()
    if np.std(a) < 0.003:
        print("  판정: 일정하다. 2025 에 그대로 적용할 수 있다.")
    elif abs(co[0]) > np.std(a):
        print("  판정: 추세가 있다. 연장값을 쓴다.")
    else:
        print("  판정: 흔들린다. 리더보드 확인 없이는 쓰지 말 것.")

    out = REPO / "results" / "center_error.json"
    out.write_text(json.dumps({
        "folds": rows,
        "mean": float(np.mean(a)), "std": float(np.std(a)),
        "slope": float(co[0]), "est_2025_trend": float(np.polyval(co, len(a))),
        "est_2025_mean": float(np.mean(a)),
        "groups": GROUPS, "params": PARAMS,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  저장: {out}")


if __name__ == "__main__":
    main()
