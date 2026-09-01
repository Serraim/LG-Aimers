"""
캘리브레이션 진단 (A2)
======================

롤링 검증에서 2023 fold 가 0점이 나왔다.
"예측 평균이 실제 평균과 어긋나서"라는 것은 추론이었고 측정한 값이 아니다.
이 스크립트는 그것을 실제로 측정한다.

측정 항목
---------
1. 예측 평균 vs 실제 평균 (얼마나 어긋났나)
2. Brier 분해
     Brier = r(1-r) + (중심오차)^2 + (변별력 항)
   중심오차가 점수를 얼마나 깎았는지 분리한다
3. 중심을 맞추면 점수가 얼마나 오르는가
     (a) oracle  : 실제 평균으로 맞춤 (상한선 확인용. 대회에서는 불가)
     (b) 추세 외삽: 학습 구간 시즌들의 추세로 다음 시즌 평균을 추정 (대회에서 사용 가능)
     (c) 직전 시즌: 학습 구간 마지막 시즌의 평균을 사용 (대회에서 사용 가능)

주의: (a) 는 정답을 보고 맞추는 것이므로 실제 제출에 쓸 수 없다.
   "중심만 맞추면 최대 얼마나 오르는가"의 상한을 재는 진단용이다.
   실제로 쓸 수 있는 것은 (b), (c) 처럼 **학습 구간만으로 정하는** 방법이다.

사용법
------
    python scripts/diag_calibration.py --season 2022
    python scripts/diag_calibration.py --season 2023
    python scripts/diag_calibration.py --season 2024
    python scripts/diag_calibration.py --summary        # 결과 종합
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/train40.pkl")          # 40% 샘플 캐시 (없으면 원본에서 생성)
OUT = Path("/tmp/diag_calibration.json")

TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=-1, random_state=42, verbose=-1,
)


def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def score(p, y):
    y = np.asarray(y)
    r = y.mean()
    return max(0.0, 100000.0 * (1.0 - brier(p, y) / (r * (1.0 - r))))


def load_train() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    tc = pd.read_csv(REPO / "data/test.csv", encoding="utf-8-sig", nrows=0).columns
    feats = [c for c in tc if c != "row_id"]
    df = pd.read_csv(REPO / "data/train.csv", encoding="utf-8-sig",
                     usecols=feats + [TARGET])
    return df


def recenter(p: np.ndarray, target_mean: float) -> np.ndarray:
    """예측값 전체를 평행이동해 평균을 target_mean 에 맞춘다."""
    return np.clip(p + (target_mean - p.mean()), 0.0, 1.0)


def estimate_next_rate(rates: dict, method: str) -> float:
    """학습 구간 시즌들의 성공률로 다음 시즌 성공률을 추정한다.

    rates: {season: rate} - 학습 구간만 포함해야 한다 (채점 시즌 제외)
    """
    seasons = sorted(rates)
    values = [rates[s] for s in seasons]
    nxt = seasons[-1] + 1

    if method == "last":                      # 직전 시즌 그대로
        return values[-1]
    if method == "linear":                    # 전체 시즌에 직선 적합 후 외삽
        coef = np.polyfit(seasons, values, 1)
        return float(np.polyval(coef, nxt))
    if method == "last2":                     # 마지막 두 시즌의 변화량만큼 연장
        if len(values) < 2:
            return values[-1]
        return float(values[-1] + (values[-1] - values[-2]))
    raise ValueError(method)


def run_fold(df: pd.DataFrame, valid_season: int) -> dict:
    feats = [c for c in df.columns if c != TARGET]

    tr = df[df["season"] < valid_season]
    va = df[df["season"] == valid_season]

    # 전처리는 학습 구간에서만
    cat_maps = {c: {v: i for i, v in enumerate(sorted(tr[c].dropna().unique()))}
                for c in CAT_COLS}

    def enc(x):
        X = x[feats].copy()
        for c in CAT_COLS:
            X[c] = X[c].map(cat_maps[c]).fillna(-1).astype("int32")
        return X

    X_tr, y_tr = enc(tr), tr[TARGET].to_numpy()
    X_va, y_va = enc(va), va[TARGET].to_numpy()

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    p = model.predict_proba(X_va)[:, 1]

    r_true = float(y_va.mean())
    p_mean = float(p.mean())
    gap = p_mean - r_true

    # 학습 구간의 시즌별 성공률 (추세 추정에 사용)
    train_rates = {int(s): float(v) for s, v in
                   tr.groupby("season")[TARGET].mean().items()}

    out = {
        "valid_season":   valid_season,
        "n_train":        int(len(tr)),
        "n_valid":        int(len(va)),
        "train_mean":     float(y_tr.mean()),
        "valid_mean":     r_true,
        "pred_mean":      p_mean,
        "center_gap":     gap,
        "best_iteration": int(model.best_iteration_),
        "train_rates":    train_rates,
        "brier":          brier(p, y_va),
        "score":          score(p, y_va),
        "pred_std":       float(p.std()),
    }

    # 중심 보정 효과
    out["variants"] = {}

    # (a) oracle - 상한선 (대회에서는 사용 불가)
    p_oracle = recenter(p, r_true)
    out["variants"]["oracle"] = {
        "target": r_true, "brier": brier(p_oracle, y_va),
        "score": score(p_oracle, y_va), "legal": False,
    }

    # (b), (c) 학습 구간만으로 추정 - 대회에서 사용 가능
    for method in ("last", "linear", "last2"):
        est = estimate_next_rate(train_rates, method)
        p_adj = recenter(p, est)
        out["variants"][method] = {
            "target": est, "brier": brier(p_adj, y_va),
            "score": score(p_adj, y_va), "legal": True,
            "est_error": est - r_true,
        }

    return out


def print_fold(d: dict):
    print("=" * 74)
    print(f"Fold {d['valid_season']}  (학습 {d['n_train']:,} -> 채점 {d['n_valid']:,})")
    print("=" * 74)
    print(f"  학습 구간 평균 성공률 : {d['train_mean']:.4f}")
    print(f"  채점 시즌 실제 성공률 : {d['valid_mean']:.4f}")
    print(f"  모델 예측 평균        : {d['pred_mean']:.4f}")
    print(f"  중심 오차             : {d['center_gap']:+.4f}   <- 이만큼 어긋남")
    print(f"  예측 표준편차         : {d['pred_std']:.4f}")
    print(f"  사용 트리             : {d['best_iteration']}")
    print()
    r = d["valid_mean"]
    base = r * (1 - r)
    gap2 = d["center_gap"] ** 2
    print(f"  Brier 분해")
    print(f"    기준선 r(1-r)       : {base:.6f}")
    print(f"    실제 Brier          : {d['brier']:.6f}")
    print(f"    중심오차 제곱       : {gap2:.6f}   "
          f"(점수 환산 {gap2 / base * 100000:.1f}점 손실)")
    print()
    print(f"  현재 점수             : {d['score']:8.2f}")
    print(f"  중심 보정 후")
    for k, v in d["variants"].items():
        tag = "" if v["legal"] else "  (대회 사용 불가, 상한선)"
        err = f"  추정오차 {v['est_error']:+.4f}" if "est_error" in v else ""
        print(f"    {k:8s} 목표 {v['target']:.4f} -> {v['score']:8.2f}  "
              f"({v['score'] - d['score']:+8.2f}){err}{tag}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    store = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    if args.season:
        df = load_train()
        d = run_fold(df, args.season)
        store[str(args.season)] = d
        OUT.write_text(json.dumps(store, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print_fold(d)
        print(f"저장: {OUT}")

    if args.summary or not args.season:
        if not store:
            print("측정된 fold 가 없습니다. --season 2022 부터 실행하세요.")
            return
        print("\n" + "#" * 74)
        print("#  종합")
        print("#" * 74 + "\n")
        print(f"{'fold':>6} {'실제':>8} {'예측':>8} {'중심오차':>9} "
              f"{'현재점수':>9} {'oracle':>9} {'linear':>9} {'last':>9}")
        for s in sorted(store, key=int):
            d = store[s]
            v = d["variants"]
            print(f"{s:>6} {d['valid_mean']:8.4f} {d['pred_mean']:8.4f} "
                  f"{d['center_gap']:+9.4f} {d['score']:9.2f} "
                  f"{v['oracle']['score']:9.2f} {v['linear']['score']:9.2f} "
                  f"{v['last']['score']:9.2f}")


if __name__ == "__main__":
    main()
