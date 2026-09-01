"""S26: E42 슬롯별 라벨-분해 멤버 조합 탐색.

배경
----
지금까지 라벨 분해 실험(E61, E62, E63, E65, E67, E68)은 전부 E42 의
``wwb_lgbm`` 슬롯 하나만 교체했다. 그 슬롯의 가중치는 0.294 다.

    catboost         0.5356   <- 제일 큰 슬롯. 한 번도 안 바뀜
    wwb_lgbm         0.2945   <- 60번 넘게 교체된 슬롯
    season_residual  0.1377
    trackman_moe     0.0309
    xgboost          0.0013

그리고 catboost 슬롯의 기존 멤버(e18 이진 CatBoost)가 다른 후보들과
가장 많이 닮았다. 2024 상관은 wml 0.9757, mcb 0.9709 다.

E61 이 리더보드에서 이긴 이유도 성능이 아니라 비상관이었다.
joint15 는 멤버 단독 2024 가 851.47 로 binary 848.36 보다 +3.11 뿐인데
binary 와의 상관이 0.9378 로 가장 낮았다.

그래서 두 슬롯을 함께 놓고 조합을 탐색한다.

탐색 공간 (1260 개)
-------------------
    lgbm 슬롯   {bin, j15, cor5, cal6} 의 공집합 아닌 부분집합 15 개의 균등 평균
    cb   슬롯   (1-t) * cb18 + t * Z
                Z in {mcb, e69, mcb+e69, mcb+wml+e69} 의 균등 평균
                t in 0.00, 0.05, ..., 1.00  (21 개)
    나머지 세 멤버와 E42 상위 가중치는 건드리지 않는다.

선택 규칙 (사전 등록. 2024 는 선택에 쓰지 않는다)
-------------------------------------------------
    규칙 A   2022 centered score 최대
    규칙 B   2022 와 2023 이 모두 E61-C 초과인 것 중 2022 최대

2024 는 선택이 끝난 뒤 확인용으로 한 번만 본다.

주의
----
1차 임시 탐색(52 조합)에서는 2024 를 보고 골랐다. 그 숫자는 탐색 결과이며
근거로 쓰지 않는다. 이 스크립트가 그것을 대체한다.

OOF 행 정렬 규약
----------------
모든 .npy 는 ``data/train.csv`` 를 원본 파일 순서 그대로 읽어
``season == S`` 로 필터한 행 순서다. 정렬이나 재색인을 하지 않는다.
E61-C 저장본을 이 규약으로 재조립했을 때 max|diff| = 3.65e-08 로 일치했다.

사용법
------
    python scripts/s26_slot_combination.py
    python scripts/s26_slot_combination.py --no-bootstrap
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# e52 의 구현을 그대로 쓴다. import 가 joblib/lightgbm 까지 끌고 오므로
# 실패하면 동일 로직 폴백을 쓴다. 두 경로의 수식은 완전히 같다.
try:
    from scripts.e52_e42_slope_calibration import (
        centered_loss_difference,
        cluster_bootstrap_mean_ci,
    )
except Exception:  # noqa: BLE001
    try:
        from e52_e42_slope_calibration import (
            centered_loss_difference,
            cluster_bootstrap_mean_ci,
        )
    except Exception:  # noqa: BLE001

        def centered_loss_difference(baseline_prediction, candidate_prediction, target):
            """e52 와 동일. 행별 centered squared-error 차이."""
            baseline = np.asarray(baseline_prediction, dtype="float64")
            candidate = np.asarray(candidate_prediction, dtype="float64")
            y = np.asarray(target, dtype="float64")
            if baseline.shape != candidate.shape or baseline.shape != y.shape:
                raise ValueError("centered loss 배열의 shape가 일치해야 합니다")
            baseline_centered = baseline - baseline.mean() + y.mean()
            candidate_centered = candidate - candidate.mean() + y.mean()
            return (candidate_centered - y) ** 2 - (baseline_centered - y) ** 2

        def cluster_bootstrap_mean_ci(differences, groups, n_bootstrap=2000, seed=20260825):
            """e52 와 동일. 투수 cluster 복원추출 95% 구간."""
            values = np.asarray(differences, dtype="float64")
            group_values = np.asarray(groups)
            if values.shape != group_values.shape or values.ndim != 1:
                raise ValueError("differences와 groups는 같은 길이의 1차원 배열이어야 합니다")
            if n_bootstrap <= 0:
                raise ValueError("bootstrap 횟수는 양수여야 합니다")
            _, inverse = np.unique(group_values, return_inverse=True)
            n_groups = int(inverse.max()) + 1 if len(inverse) else 0
            if n_groups < 2:
                raise ValueError("bootstrap에는 두 개 이상의 cluster가 필요합니다")
            group_sums = np.bincount(inverse, weights=values, minlength=n_groups)
            group_counts = np.bincount(inverse, minlength=n_groups).astype("float64")
            rng = np.random.default_rng(seed)
            estimates = np.empty(n_bootstrap, dtype="float64")
            for index in range(n_bootstrap):
                sampled = rng.integers(0, n_groups, size=n_groups)
                estimates[index] = group_sums[sampled].sum() / group_counts[sampled].sum()
            return {
                "n_clusters": n_groups,
                "n_bootstrap": int(n_bootstrap),
                "seed": int(seed),
                "mean_loss_difference": float(values.mean()),
                "ci_lower_95": float(np.quantile(estimates, 0.025)),
                "ci_upper_95": float(np.quantile(estimates, 0.975)),
            }

REPO = Path(__file__).resolve().parent.parent
PRED_DIR = REPO / "results" / "preds"
E42_PATH = REPO / "results" / "e42_frozen_robust_weights.json"
RESULT_PATH = REPO / "results" / "s26_slot_combination.json"

TARGET = "control_success"
SEASONS = (2022, 2023, 2024)
SELECT_SEASON = 2022
STRESS_SEASON = 2023
EVAL_SEASON = 2024
E61C_ALPHA = 0.771856807188269

MEMBERS = {
    "bin": "e13_{s}_w_wb.npy",
    "j15": "e61c_{s}_joint_seed_average.npy",
    "cor5": "e62_{s}_core5_seed_average.npy",
    "mcb": "e63_{s}_multilabel_catboost_seed_average.npy",
    "wml": "e64_{s}_weighted_multilabel_seed_average.npy",
    "cal6": "e65_{s}_call6_seed_average.npy",
    "cb18": "e18_{s}_catboost.npy",
    "e69": "e69_{s}_joint_catboost_seed42.npy",
}
FIXED = {
    "season_residual": "e17_{s}_residual.npy",
    "xgboost": "e19_{s}_xgboost.npy",
    "trackman_moe": "e28_{s}_moe.npy",
}
LGBM_POOL = ("bin", "j15", "cor5", "cal6")
CB_POOL = (("mcb",), ("e69",), ("mcb", "e69"), ("mcb", "wml", "e69"))
T_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.0001, 0.05))


def centered_score(pred, target):
    pred = np.asarray(pred, dtype="float64")
    target = np.asarray(target, dtype="float64")
    rate = target.mean()
    center_error = pred.mean() - rate
    brier = float(np.mean((pred - target) ** 2))
    return 100000.0 * (1.0 - (brier - center_error**2) / (rate * (1.0 - rate)))


def raw_brier(pred, target):
    return float(np.mean((np.asarray(pred, dtype="float64") - target) ** 2))


def load_frames():
    df = pd.read_csv(
        REPO / "data" / "train.csv",
        encoding="utf-8-sig",
        usecols=["season", "pitcher_id", TARGET],
    )
    target = {}
    pitcher = {}
    for season in SEASONS:
        block = df.loc[df["season"] == season]
        target[season] = block[TARGET].to_numpy(dtype="float64")
        pitcher[season] = block["pitcher_id"].to_numpy()
    return target, pitcher


def load_predictions():
    def read(template):
        out = {}
        for season in SEASONS:
            path = PRED_DIR / template.format(s=season)
            if not path.exists():
                raise FileNotFoundError(f"예측 파일 없음: {path}")
            out[season] = np.load(path).astype("float32")
        return out

    return {k: read(v) for k, v in MEMBERS.items()}, {
        k: read(v) for k, v in FIXED.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    started = time.time()
    print("=" * 84)
    print("  S26  E42 슬롯별 라벨-분해 멤버 조합 탐색")
    print("=" * 84)

    weights = json.loads(E42_PATH.read_text(encoding="utf-8"))["weights"]
    target, pitcher = load_frames()
    member, fixed = load_predictions()
    for name, block in list(member.items()) + list(fixed.items()):
        for season in SEASONS:
            if len(block[season]) != len(target[season]):
                raise RuntimeError(f"{name} {season} 행 수 불일치")

    rest = {
        s: sum(weights[n] * fixed[n][s] for n in FIXED) for s in SEASONS
    }

    def blend(keys, season):
        return np.mean([member[k][season] for k in keys], axis=0)

    def compose(lgbm_keys, cb_keys, t, season):
        lg = blend(lgbm_keys, season)
        cb = (1.0 - t) * member["cb18"][season] + t * blend(cb_keys, season)
        return np.clip(
            weights["wwb_lgbm"] * lg + weights["catboost"] * cb + rest[season], 0.0, 1.0
        )

    baseline = {
        s: np.clip(
            weights["wwb_lgbm"]
            * ((1.0 - E61C_ALPHA) * member["bin"][s] + E61C_ALPHA * member["j15"][s])
            + weights["catboost"] * member["cb18"][s]
            + rest[s],
            0.0,
            1.0,
        )
        for s in SEASONS
    }
    base_score = {s: centered_score(baseline[s], target[s]) for s in SEASONS}
    print("\n  E61-C 재조립 기준선 (LB 1011.98)")
    for s in SEASONS:
        print(f"    {s}  centered {base_score[s]:.2f}")

    print("\n  멤버 단독 centered")
    print(f"    {'member':<8}" + "".join(f"{s:>11}" for s in SEASONS))
    standalone = {}
    for k in MEMBERS:
        standalone[k] = {s: centered_score(member[k][s], target[s]) for s in SEASONS}
        print(f"    {k:<8}" + "".join(f"{standalone[k][s]:>11.2f}" for s in SEASONS))

    correlation = {}
    for season in (SELECT_SEASON, EVAL_SEASON):
        keys = list(MEMBERS)
        matrix = {}
        for a in keys:
            matrix[a] = {
                b: float(np.corrcoef(member[a][season], member[b][season])[0, 1])
                for b in keys
            }
        correlation[str(season)] = matrix
        print(f"\n  멤버 상관 ({season})")
        print("    " + " " * 8 + "".join(f"{k:>8}" for k in keys))
        for a in keys:
            print(f"    {a:<8}" + "".join(f"{matrix[a][b]:>8.4f}" for b in keys))

    subsets = [
        c
        for r in range(1, len(LGBM_POOL) + 1)
        for c in itertools.combinations(LGBM_POOL, r)
    ]
    grid = []
    for lgbm_keys in subsets:
        for cb_keys in CB_POOL:
            for t in T_GRID:
                scores = {
                    s: centered_score(compose(lgbm_keys, cb_keys, t, s), target[s])
                    for s in SEASONS
                }
                grid.append(
                    {
                        "lgbm_slot": list(lgbm_keys),
                        "cb_slot_z": list(cb_keys),
                        "t": float(t),
                        "centered": {str(s): scores[s] for s in SEASONS},
                    }
                )
    print(f"\n  탐색 조합 {len(grid)} 개")

    def pick(rule):
        if rule == "A":
            pool = grid
        else:
            pool = [
                g
                for g in grid
                if g["centered"][str(SELECT_SEASON)] > base_score[SELECT_SEASON]
                and g["centered"][str(STRESS_SEASON)] > base_score[STRESS_SEASON]
            ]
        if not pool:
            return None
        return max(pool, key=lambda g: g["centered"][str(SELECT_SEASON)])

    selections = {}
    print("\n  선택 규칙 적용 (2024 는 선택에 쓰지 않는다)")
    for rule, note in (
        ("A", "2022 최대"),
        ("B", "2022 와 2023 모두 E61-C 초과 + 2022 최대"),
    ):
        chosen = pick(rule)
        if chosen is None:
            print(f"    규칙 {rule}: 후보 없음")
            continue
        lgbm_keys = tuple(chosen["lgbm_slot"])
        cb_keys = tuple(chosen["cb_slot_z"])
        t = chosen["t"]
        print(f"\n    규칙 {rule} ({note})")
        print(f"      lgbm 슬롯 = {'+'.join(lgbm_keys)} 균등")
        print(f"      cb   슬롯 = {1-t:.2f}*cb18 + {t:.2f}*({'+'.join(cb_keys)} 균등)")

        final_weights = {}
        for k in lgbm_keys:
            final_weights[k] = weights["wwb_lgbm"] / len(lgbm_keys)
        final_weights["cb18"] = weights["catboost"] * (1.0 - t)
        for k in cb_keys:
            final_weights[k] = final_weights.get(k, 0.0) + weights["catboost"] * t / len(cb_keys)
        for n in FIXED:
            final_weights[n] = weights[n]
        print("      최종 멤버 가중치")
        for k, v in sorted(final_weights.items(), key=lambda x: -x[1]):
            if v > 1e-9:
                print(f"        {k:<18}{v:.5f}")

        fold_metrics = {}
        for s in SEASONS:
            pred = compose(lgbm_keys, cb_keys, t, s)
            center_error = float(pred.mean() - target[s].mean())
            brier = raw_brier(pred, target[s])
            fold_metrics[str(s)] = {
                "raw_brier": brier,
                "centered_brier": brier - center_error**2,
                "centered_score": centered_score(pred, target[s]),
                "center_error": center_error,
                "delta_centered_vs_e61c": centered_score(pred, target[s]) - base_score[s],
            }
            out = PRED_DIR / f"s26{rule}_{s}_combo_oof.npy"
            np.save(out, pred.astype("float32"))
            print(
                f"      {s}  rawBrier {brier:.8f}  centBrier {brier-center_error**2:.8f}"
                f"  centered {fold_metrics[str(s)]['centered_score']:9.2f}"
                f"  중심오차 {center_error:+.5f}  -> {out.name}"
            )

        bootstrap = None
        if not args.no_bootstrap:
            pred = compose(lgbm_keys, cb_keys, t, EVAL_SEASON)
            differences = centered_loss_difference(
                baseline[EVAL_SEASON], pred, target[EVAL_SEASON]
            )
            bootstrap = cluster_bootstrap_mean_ci(
                differences, pitcher[EVAL_SEASON], n_bootstrap=args.bootstrap_samples
            )
            print(
                f"      2024 투수 cluster bootstrap  평균 {bootstrap['mean_loss_difference']:+.8f}"
                f"  95% CI [{bootstrap['ci_lower_95']:+.8f}, {bootstrap['ci_upper_95']:+.8f}]"
                f"  clusters {bootstrap['n_clusters']}"
            )
            print(
                "      (음수가 개선. 상한 < 0 이면 팀 사전조건 통과)"
            )

        selections[rule] = {
            "note": note,
            "lgbm_slot": list(lgbm_keys),
            "cb_slot_z": list(cb_keys),
            "t": t,
            "final_member_weights": final_weights,
            "folds": fold_metrics,
            "oof_files": {
                str(s): f"results/preds/s26{rule}_{s}_combo_oof.npy" for s in SEASONS
            },
            "bootstrap_2024_vs_e61c": bootstrap,
        }

    payload = {
        "experiment_id": "S26",
        "description": "Slot-aware label-decomposition member combination search",
        "row_order_convention": (
            "data/train.csv 원본 파일 순서에서 season == S 로 필터한 순서. "
            "정렬/재색인 없음. E61-C 재조립 max|diff| = 3.65e-08"
        ),
        "e42_weights": weights,
        "e61c_alpha": E61C_ALPHA,
        "select_season": SELECT_SEASON,
        "stress_season": STRESS_SEASON,
        "eval_season": EVAL_SEASON,
        "eval_season_used_in_selection": False,
        "prior_ad_hoc_scan_used_2024": True,
        "prior_ad_hoc_scan_note": (
            "1차 임시 탐색 52 조합은 2024 를 보고 골랐다. 탐색 결과로만 취급하고 "
            "이 스크립트의 사전등록 선택으로 대체한다."
        ),
        "search_space": {
            "lgbm_pool": list(LGBM_POOL),
            "cb_pool_z": [list(z) for z in CB_POOL],
            "t_grid": list(T_GRID),
            "n_combinations": len(grid),
        },
        "baseline_e61c_centered": {str(s): base_score[s] for s in SEASONS},
        "member_standalone_centered": {
            k: {str(s): v[s] for s in SEASONS} for k, v in standalone.items()
        },
        "member_correlation": correlation,
        "grid": grid,
        "selections": selections,
        "post_submission_rule": (
            "A/B 제출 이후 리더보드 점수를 보고 t 나 가중치를 재조정하지 않는다."
        ),
        "total_sec": time.time() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  결과 저장: {RESULT_PATH}")
    print(f"  총 {time.time()-started:.0f}초")


if __name__ == "__main__":
    main()
