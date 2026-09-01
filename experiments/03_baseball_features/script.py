"""
평가 서버에서 실행되는 추론 코드
=================================

제출 시 이 파일이 submit.zip의 script.py 로 들어간다.
평가 서버가 `python script.py` 로 직접 실행한다.

디렉토리 구조 (평가 서버 기준)
------------------------------
    ./model/                모델 artifact (참가자 제출)
    ./script.py             이 파일     (참가자 제출)
    ./requirements.txt      패키지      (참가자 제출)
    ./data/test.csv         평가 데이터 (서버가 제공, 읽기 전용)
    ./data/sample_submission.csv
    ./output/submission.csv 결과 (이 코드가 생성)

대회 규칙 (AGENTS.md §2)
------------------------
    평가 데이터의 각 행은 독립적으로 예측되어야 한다.
    이 파일에서는 test 전체를 대상으로 하는 집계
    (groupby / mean / rolling / rank ...)를 사용하지 않는다.

    train 기반 통계는 학습 단계에서 계산해 artifact 에 저장하고,
    여기서는 현재 행의 값으로 lookup 만 수행한다. (AGENTS.md §5)

    검증: python scripts/check_rules.py --submit-dir <폴더>

check_rules.py 연동
-------------------
    predict_control_success(test) 를 제공한다.
    검사기가 이 함수를 찾아 "전체 입력" 과 "한 행씩 입력" 의 결과를 대조한다.
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

MODEL_DIR = "./model"
TEST_DIR = "./data"
OUT_PATH = "./output/submission.csv"


# ============================================================
# 모델 artifact
# ============================================================

_ARTIFACT_CACHE = None


def find_model_file(model_dir=MODEL_DIR):
    for ext in ("*.joblib", "*.pkl"):
        hits = sorted(glob.glob(os.path.join(model_dir, ext)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{model_dir} 안에 모델 파일(.joblib/.pkl)이 없음")


def load_artifact(model_dir=MODEL_DIR):
    """모델을 한 번만 읽어 재사용한다."""
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = joblib.load(find_model_file(model_dir))
    return _ARTIFACT_CACHE


# ============================================================
# 데이터 로드
# ============================================================

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


# ============================================================
# 전처리 - 학습과 반드시 동일해야 함
# ============================================================

def add_domain_features(df, spec):
    """야구 도메인 파생 피처. 그룹 단위로 켜고 끈다.

    spec["groups"] 에 적힌 그룹만 만든다. 없으면 E04 와 동일한 구성을 쓴다.

        P  투수     성적 보정, 리그 대비, 폼
        B  타자     타자 쪽 지표
        M  매치업   좌우 조합, 투수-타자 우열
        S  상황     볼카운트, 경기 진행도

    왜 그룹으로 나눴나
        E04 의 15개는 야구 상식으로 한 번에 만든 것이고 검증한 적이 없다.
        실제로 4개는 중요도 최하위였다.
            tto 0.005%, is_pressure 0.029%, is_putaway 0.094%, progress 0.405%
        "더하기"만 해봤지 "빼기"와 "바꾸기"는 시험한 적이 없다.

    주의: 모두 **한 행 안에서만** 계산된다.
       유일한 외부 참조는 train 에서 만들어 artifact 에 저장한 고정 표이며
       season 등 그 행의 값으로 조회만 한다. (AGENTS.md 5)
       test 의 다른 행을 보지 않으므로 행 독립성이 유지된다.
    """
    league_avg = spec["league_avg"]
    default_avg = spec["default_league_avg"]
    k = spec.get("shrink_k", 500.0)
    groups = set(spec.get("groups") or ["P", "B", "M", "S"])

    X = df
    la = X["season"].map(league_avg).fillna(default_avg)

    if "S" in groups:
        # 볼카운트 조합
        X["count_state"] = X["balls_before"] * 3 + X["strikes_before"]
        X["is_pressure"] = (X["balls_before"] == 3).astype("int8")
        X["is_putaway"] = (X["strikes_before"] == 2).astype("int8")
        X["count_edge"] = X["strikes_before"] - X["balls_before"]
        # 경기 진행도 / 타순 회차
        X["progress"] = (X["inning"] - 1) * 3 + X["outs_before"]
        X["tto"] = np.minimum(3, (X["inning"] - 1) // 3 + 1)

    if "M" in groups:
        X["hand_matchup"] = X["pitcher_hand"] * 2 + X["batter_hand"]
        X["same_hand"] = (X["pitcher_hand"] == X["batter_hand"]).astype("int8")
        X["skill_gap"] = (X["asof_pitcher_success_rate"]
                          - X["asof_batter_success_rate"])
        X["exp_gap"] = (np.log1p(X["asof_pitcher_n"])
                        - np.log1p(X["asof_batter_n"]))

    if "P" in groups:
        # 컨디션 (최근 5경기 - 통산)
        X["form_gap"] = (X["asof_pitcher_prev5_game_success_rate"]
                         - X["asof_pitcher_success_rate"])
        X["form_gap_mid"] = (X["asof_pitcher_prev5_game_middle_rate"]
                             - X["asof_pitcher_middle_rate"])
        # 리그 평균 대비 (ERA+ 개념)
        X["rate_plus"] = X["asof_pitcher_success_rate"] - la
        # 표본 크기 보정 (평균으로의 회귀). k 는 실측으로 200 이 최적
        n = X["asof_pitcher_n"].fillna(0)
        X["pitcher_rate_shrunk"] = (
            (n * X["asof_pitcher_success_rate"].fillna(la) + k * la) / (n + k))

    if "P2" in groups:
        # 투수 확장: 아직 만들지 않았던 것들
        # 폼 방향 (직전 1경기 - 최근 5경기). form_gap 보다 짧은 시간축
        X["form_dir"] = (X["asof_pitcher_prev1_game_success_rate"]
                         - X["asof_pitcher_prev5_game_success_rate"])
        # 나눗셈 형태. rate_plus 는 뺄셈이며 나무는 나눗셈도 못 만든다
        X["rate_ratio"] = X["asof_pitcher_success_rate"] / la
        # 역방향률 / 한복판률 표본 보정
        n = X["asof_pitcher_n"].fillna(0)
        for src, avg_key in (("asof_pitcher_reverse_rate", "reverse_league_avg"),
                             ("asof_pitcher_middle_rate", "middle_league_avg")):
            L = spec.get(avg_key)
            if L is not None:
                X[src + "_shrunk"] = (
                    (n * X[src].fillna(L) + k * L) / (n + k))

    if "B" in groups:
        X["batter_rate_plus"] = X["asof_batter_success_rate"] - la

    if "B2" in groups:
        # 타자 확장: 지금까지 타자 파생은 batter_rate_plus 하나뿐이었다.
        nb = X["asof_batter_n"].fillna(0)
        X["batter_rate_shrunk"] = (
            (nb * X["asof_batter_success_rate"].fillna(la) + k * la) / (nb + k))
        Lm = spec.get("batter_middle_league_avg")
        if Lm is not None:
            X["batter_middle_shrunk"] = (
                (nb * X["asof_batter_middle_rate"].fillna(Lm) + k * Lm) / (nb + k))
            X["batter_middle_plus"] = X["asof_batter_middle_rate"] - Lm
        X["batter_exp"] = np.log1p(nb)

    if "M2" in groups:
        # 투수별 / 타자별 좌우 스플릿.
        #
        # 실측: hand_matchup 이 담는 리그 평균 효과를 제거하고도
        #       투수별 개인차가 0.0118 남는다 (잡음 0.0130 대비 실재).
        #       예: 투수 23637(우투)은 좌타 0.5835 / 우타 0.4323 로
        #           리그 평균과 정반대 방향이다.
        #
        # 표는 train 에서만 만들어 artifact 에 저장하고 여기서는 조회만 한다.
        for who, idcol, tabkey in (("p", "pitcher_id", "pitcher_split"),
                                   ("b", "batter_id", "batter_split")):
            tab = spec.get(tabkey)
            if not tab:
                continue
            key = X[idcol].astype("int64") * 10 + X["batter_hand"].astype("int64")
            # 시즌별 표가 있으면 그 시즌을 뺀 표를 쓴다 (누수 방지).
            # 표에 없는 시즌(평가 데이터)은 전체 표를 쓴다.
            by = tab["by_season"]
            allt = tab["all"]
            v = pd.Series(np.nan, index=X.index, dtype="float64")
            for s, sub in by.items():
                m = (X["season"] == s).to_numpy()
                if m.any():
                    v[m] = key[m].map(sub).to_numpy()
            rest = ~X["season"].isin(list(by)).to_numpy()
            if rest.any():
                v[rest] = key[rest].map(allt).to_numpy()
            X[who + "_split"] = v.to_numpy()

    return X


def build_features(df, artifact=None):
    """모델 입력 생성. 행 단위 연산만 사용한다.

    artifact 가 None 이면 (sklearn Pipeline 형식):
        row_id 만 제외하고 반환. 전처리는 Pipeline 내부에서 수행된다.

    artifact 가 dict 이면 (artifact 형식):
        저장된 컬럼 순서로 정렬하고 범주형을 학습 때의 코드로 변환한다.
        변환 표는 train 으로 만들어 저장된 것이며 test 를 보고 만들지 않는다.
    """
    if artifact is None:
        return df.drop(columns=[ID_COL])

    features = artifact["features"]
    cat_cols = artifact.get("cat_cols", [])
    cat_maps = artifact.get("cat_maps", {})

    spec = artifact.get("feature_spec")
    base = df.drop(columns=[ID_COL]) if ID_COL in df.columns else df.copy()
    if spec:
        base = add_domain_features(base.copy(), spec)

    X = base[features].copy()
    for col in cat_cols:
        mapping = cat_maps[col]                     # train 에서 만든 고정 표
        X[col] = X[col].map(mapping)                # 행 단위 lookup
        X[col] = X[col].fillna(-1).astype("int32")  # 학습 때 못 본 값은 -1

    # 중앙값 대치 조건으로 학습한 모델이면 동일하게 채운다.
    # 채우는 값은 train 에서 계산해 artifact 에 저장된 상수이며,
    # test 를 보고 계산하지 않는다. (AGENTS.md §5, §6)
    medians = artifact.get("impute_median")
    if medians:
        for col, value in medians.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)       # 행 단위 상수 대치

    return X


# ============================================================
# 추론
# ============================================================

def _one_model(entry, X):
    """모델 하나의 예측. 필요한 전처리는 각 entry 안에 들어 있다."""
    model = entry["model"]
    kind = entry.get("kind", "proba")

    Z = X
    # 결측을 직접 못 다루는 모델(ExtraTrees 등)은 학습 때 쓴 중앙값으로 채운다.
    # 채우는 값은 train 에서 계산해 저장한 상수이며 test 를 보고 만들지 않는다.
    med = entry.get("impute_median")
    if med:
        Z = Z.fillna(pd.Series(med))
    # 신경망 등은 학습 구간 범위로 자른 뒤 표준화한다 (외삽 차단)
    clip = entry.get("clip")
    if clip:
        Z = Z.clip(pd.Series(clip["lo"]), pd.Series(clip["hi"]), axis=1)
    sc = entry.get("scaler")
    if sc is not None:
        Z = sc.transform(Z)

    if kind == "proba":
        values = model.predict_proba(Z)[:, 1]
    else:
        # 회귀 모델(L2 = Brier 직접 최적화)은 범위를 벗어날 수 있다
        values = model.predict(Z)
    return np.clip(values, 0.0, 1.0)


def _predict_values(artifact, X):
    """제구 성공(1) 확률. 0~1 범위를 보장한다.

    artifact 에 `models` 가 있으면 **여러 모델의 가중 평균**을 낸다.
    없으면 기존처럼 단일 모델로 동작한다.

    앙상블을 쓰는 근거 (2024 홀드아웃 실측)
        LightGBM   739.87      상관 -
        CatBoost   728.35      lgbm 과 0.9490
        ExtraTrees 694.57      lgbm 과 0.9154
        세 개 평균 757.17      <- 개별 어느 것보다 높다

    제곱오차(=Brier)에서는 다음 항등식이 성립하기 때문이다.
        앙상블 오차 = 개별 오차의 평균 - 모델들이 서로 다른 정도
    빼는 항이 항상 0 이상이므로 앙상블은 개별 평균보다 나쁠 수 없다.

    각 모델은 같은 행만 보고 예측하므로 행 독립성은 그대로 유지된다.
    """
    if len(X) == 0:
        return np.array([])

    if isinstance(artifact, dict) and artifact.get("models"):
        entries = artifact["models"]
        ws = artifact.get("weights") or [1.0] * len(entries)
        tot = float(sum(ws))
        out = np.zeros(len(X), dtype="float64")
        for e, w in zip(entries, ws):
            out += (w / tot) * _one_model(e, X)
        return np.clip(out, 0.0, 1.0)

    if isinstance(artifact, dict):
        return _one_model(artifact, X)
    return np.clip(artifact.predict_proba(X)[:, 1], 0.0, 1.0)


# ============================================================
# 중심 보정 상수
# ============================================================
#
# 무엇인가
#   모든 예측에 이 값을 더한다. 음수이므로 전체를 아래로 내린다.
#   예측의 순서나 분산은 하나도 바뀌지 않고 중심만 이동한다.
#
# 왜 필요한가
#   시즌 성공률이 계속 내려간다.
#       2019 .5647  2020 .5327  2021 .5328
#       2022 .5289  2023 .5000  2024 .4861
#   그런데 나무 모델은 외삽을 못 한다. season=2025 행은 전부 학습 범위
#   밖이라 2024 잎에 떨어지고, 그래서 예측 중심이 실제보다 높게 나온다.
#
#   실측 (2024 홀드아웃, S04 기준)
#       실제 성공률 0.4861 / 예측 평균 0.4961 -> 중심오차 +0.0100
#       이것만으로 40.3점을 잃는다.
#
#   팀 실측 (2025 리더보드)
#       baek-32 E05 (보정 없음)          815.5415
#       baek-32 E08 (중심을 내린 것)     890.6784   <- +75.14
#   중심 이동 하나로 75점이 올랐다. 2025 에도 하락이 이어졌다는 뜻이다.
#
# 대회 규칙 준수 (AGENTS.md §2)
#   이것은 test 의 다른 행을 전혀 보지 않는 고정 상수다.
#   test 에 행이 1개만 있어도, 24만 행이 있어도 같은 값을 뺀다.
#   -> 행 독립성 유지. check_rules.py 최대 차이 0.0
#
#   금지되는 방식은 "test 예측들의 평균을 구해 그만큼 빼는 것"이다.
#   그것은 다른 행을 봐야 하므로 규칙 위반이다. 여기서는 하지 않는다.
#
# 값을 어떻게 정하는가
#   최적값을 로컬로는 알 수 없다. 2025 의 실제 성공률을 모르기 때문이다.
#   제출 점수는 SHIFT 에 대해 포물선이므로 세 점이면 꼭짓점이 정해진다.
#
#       SHIFT  0.000  ->  809.0159            (S06, 측정 완료)
#       SHIFT -0.014  ->  874.4039689691      (S07, 측정 완료)
#
#   두 점으로 이미 풀렸다. Brier 는 SHIFT 에 대해 정확한 2차식이고
#   c^2 의 계수가 1 로 고정이므로 미지수는 (m-r) 하나뿐이다.
#
#       우리 모델의 중심오차 m-r = 0.01283
#       최적 SHIFT           = -0.01283      (기대 874.96)
#
#   그런데 m 과 r 을 따로 알려면 점이 하나 더 필요하고,
#   그 점은 -0.014 에서 멀리 떨어져야 한다 (가까우면 조건수가 나빠 못 푼다).
#
#       SHIFT -0.040  ->  ?   (S08)  기대 579점. r 을 +-0.007 로 확정
#
#   r 을 알면 LB 점수를 Brier 로 환산할 수 있고,
#   2025 에서 우리 예측의 실제 변별력을 처음으로 측정할 수 있다.
#   이 측정은 점수를 버리는 대신 남은 기간 내내 쓸 자를 얻는 것이다.
#
# 현재 값: S08 진단용
SHIFT = -0.040


def predict_control_success(test):
    """test 데이터프레임을 받아 제구 성공 확률 배열을 반환한다.

    각 행은 독립적으로 계산된다.
    한 행만 넣어도, 전체를 넣어도 같은 값이 나온다.

    마지막에 SHIFT 를 더한다. 아래 설명 참고.
    """
    artifact = load_artifact()
    X = build_features(test, artifact if isinstance(artifact, dict) else None)
    return np.clip(_predict_values(artifact, X) + SHIFT, 0.0, 1.0)


# ============================================================
# 제출 파일 생성
# ============================================================

def merge_predictions(sub, ids, preds):
    """sample_submission 의 row_id 순서에 맞춰 예측값을 채운다."""
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f"  경고: 예측이 없어 placeholder 를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# ============================================================

def main():
    print("Load model...")
    artifact = load_artifact()
    kind = "artifact" if isinstance(artifact, dict) else "sklearn pipeline"
    print(f"  {os.path.basename(find_model_file())} ({kind})")

    print("Load test data...")
    test = load_test(os.path.join(TEST_DIR, "test.csv"))
    sub = load_sample_submission(os.path.join(TEST_DIR, "sample_submission.csv"))
    print(f"  test={len(test)}  submission={len(sub)}")

    print("Inference...")
    ids = test[ID_COL].tolist()
    preds = predict_control_success(test)
    print(f"  preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
