"""S26-A/B standalone submission inference; every test row is independent."""

from __future__ import annotations

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
_ARTIFACT_CACHE = None
S37_MEMBER_ORDER = (
    "cor5",
    "cb18",
    "mcb",
    "e69",
    "season_residual",
    "trackman_moe",
    "xgboost",
)


def find_model_file(model_dir=MODEL_DIR):
    hits = sorted(glob.glob(os.path.join(model_dir, "*.joblib")))
    if not hits:
        hits = sorted(glob.glob("./**/*.joblib", recursive=True))
    if not hits:
        raise FileNotFoundError(f"{model_dir} 안에 joblib 모델이 없습니다")
    return hits[0]


def load_artifact(model_dir=MODEL_DIR):
    global _ARTIFACT_CACHE
    if _ARTIFACT_CACHE is None:
        _ARTIFACT_CACHE = joblib.load(find_model_file(model_dir))
    return _ARTIFACT_CACHE


def _safe_rate(count, denominator):
    rate = count / denominator.where(denominator.gt(0))
    return rate.where(count.ge(0) & count.le(denominator))


def add_domain_features(df, spec):
    x = df
    league = x["season"].map(spec["league_avg"]).fillna(float(spec["default_league_avg"]))
    shrink_k = float(spec["shrink_k"])
    x["count_state"] = x["balls_before"] * 3 + x["strikes_before"]
    x["is_pressure"] = (x["balls_before"] == 3).astype("int8")
    x["is_putaway"] = (x["strikes_before"] == 2).astype("int8")
    x["count_edge"] = x["strikes_before"] - x["balls_before"]
    x["progress"] = (x["inning"] - 1) * 3 + x["outs_before"]
    x["tto"] = np.minimum(3, (x["inning"] - 1) // 3 + 1)
    x["hand_matchup"] = x["pitcher_hand"] * 2 + x["batter_hand"]
    x["same_hand"] = (x["pitcher_hand"] == x["batter_hand"]).astype("int8")
    x["form_gap"] = x["asof_pitcher_prev5_game_success_rate"] - x["asof_pitcher_success_rate"]
    x["form_gap_mid"] = x["asof_pitcher_prev5_game_middle_rate"] - x["asof_pitcher_middle_rate"]
    x["skill_gap"] = x["asof_pitcher_success_rate"] - x["asof_batter_success_rate"]
    x["exp_gap"] = np.log1p(x["asof_pitcher_n"]) - np.log1p(x["asof_batter_n"])
    x["rate_plus"] = x["asof_pitcher_success_rate"] - league
    x["batter_rate_plus"] = x["asof_batter_success_rate"] - league
    n = x["asof_pitcher_n"].fillna(0)
    x["pitcher_rate_shrunk"] = (
        n * x["asof_pitcher_success_rate"].fillna(league) + shrink_k * league
    ) / (n + shrink_k)
    return x


def add_wwb_features(df, saved):
    x = df
    shrink_k = float(saved["shrink_k"])
    league_rate = float(saved["league_success_rate"])
    pitcher = x["pitcher_id"]
    start = saved["pitcher_season_start"]
    n0 = pitcher.map(start["n0"]).fillna(0.0)
    success0 = pitcher.map(start["success0"]).fillna(0.0)
    middle0 = pitcher.map(start["middle0"]).fillna(0.0)
    reverse0 = pitcher.map(start["reverse0"]).fillna(0.0)
    career_n = x["asof_pitcher_n"].astype("float64")
    season_n = (career_n - n0).where(career_n.ge(n0))
    season_success = np.rint(career_n * x["asof_pitcher_success_rate"].fillna(0.0)) - success0
    season_middle = np.rint(career_n * x["asof_pitcher_middle_rate"].fillna(0.0)) - middle0
    season_reverse = np.rint(career_n * x["asof_pitcher_reverse_rate"].fillna(0.0)) - reverse0
    x["p_season_n"] = season_n
    x["p_season_rate"] = _safe_rate(season_success, season_n)
    x["p_season_rate_shrunk"] = (season_success + shrink_k * league_rate) / (season_n + shrink_k)
    x["p_season_rate_shrunk"] = x["p_season_rate_shrunk"].where(
        season_success.ge(0) & season_success.le(season_n)
    )
    x["p_season_vs_career"] = x["p_season_rate"] - x["asof_pitcher_success_rate"]
    x["p_season_middle_rate"] = _safe_rate(season_middle, season_n)
    x["p_season_reverse_rate"] = _safe_rate(season_reverse, season_n)
    batter = x["batter_id"]
    batter_start = saved["batter_season_start"]
    batter_n0 = batter.map(batter_start["n0"]).fillna(0.0)
    batter_success0 = batter.map(batter_start["success0"]).fillna(0.0)
    batter_career_n = x["asof_batter_n"].astype("float64")
    batter_season_n = (batter_career_n - batter_n0).where(batter_career_n.ge(batter_n0))
    batter_success = np.rint(
        batter_career_n * x["asof_batter_success_rate"].fillna(0.0)
    ) - batter_success0
    x["b_season_n"] = batter_season_n
    x["b_season_rate"] = _safe_rate(batter_success, batter_season_n)
    x["b_season_rate_shrunk"] = (
        batter_success + shrink_k * league_rate
    ) / (batter_season_n + shrink_k)
    x["b_season_rate_shrunk"] = x["b_season_rate_shrunk"].where(
        batter_success.ge(0) & batter_success.le(batter_season_n)
    )
    x["b_season_vs_career"] = x["b_season_rate"] - x["asof_batter_success_rate"]
    return x


def build_member_features(test, member, include_wwb=False):
    base = add_domain_features(test.drop(columns=[ID_COL]).copy(), member["feature_spec"])
    if include_wwb:
        base = add_wwb_features(base, member["wwb_artifact"])
    matrix = base[member["features"]].copy()
    for column in member["cat_cols"]:
        matrix[column] = matrix[column].map(member["cat_maps"][column]).fillna(-1).astype("int32")
    return matrix


def _normalize_probabilities(values):
    array = np.asarray(values, dtype="float64").copy()
    array[~np.isfinite(array)] = 0.0
    array = np.clip(array, 0.0, None)
    totals = array.sum(axis=1, keepdims=True)
    invalid = totals[:, 0] <= 0.0
    array[invalid] = np.asarray([0.50, 0.30, 0.20])
    return array / array.sum(axis=1, keepdims=True)


def build_trackman_features(test, member):
    base = add_domain_features(test.drop(columns=[ID_COL]).copy(), member["feature_spec"])
    base = add_wwb_features(base, member["wwb_artifact"])
    fallback = _normalize_probabilities(
        np.column_stack(
            [
                base["asof_pitcher_fastball_rate"].to_numpy(),
                base["asof_pitcher_breaking_rate"].to_numpy(),
                base["asof_pitcher_offspeed_rate"].to_numpy(),
            ]
        )
    )
    keys = list(zip(base["pitcher_id"], base["balls_before"], base["strikes_before"]))
    key_series = pd.Series(keys, index=base.index)
    lookup = member["probability_lookup"]["columns"]
    probabilities = np.column_stack(
        [key_series.map(lookup[column]).to_numpy(dtype="float64") for column in member["prob_cols"]]
    )
    missing = ~np.isfinite(probabilities).all(axis=1)
    probabilities[missing] = fallback[missing]
    probabilities = _normalize_probabilities(probabilities)
    n_values = key_series.map(lookup["tm_count_n"]).fillna(0.0).to_numpy(dtype="float64")
    for index, column in enumerate(member["prob_cols"]):
        base[column] = probabilities[:, index]
    shrink_k = float(member["probability_lookup"]["shrink_k"])
    base["tm_count_n_log"] = np.log1p(n_values)
    base["tm_count_confidence"] = n_values / (n_values + shrink_k)
    base["tm_fastball_vs_asof"] = probabilities[:, 0] - fallback[:, 0]
    base["tm_breaking_vs_asof"] = probabilities[:, 1] - fallback[:, 1]
    matrix = base[member["features"]].copy()
    for column in member["cat_cols"]:
        matrix[column] = matrix[column].map(member["cat_maps"][column]).fillna(-1).astype("int32")
    return matrix, probabilities


def joint_success_probability(model, matrix, success_labels):
    probabilities = np.asarray(model.predict_proba(matrix), dtype="float64")
    classes = np.asarray(model.classes_, dtype="int64")
    selected = [index for index, label in enumerate(classes) if int(label) in success_labels]
    if set(classes[selected]) != set(success_labels):
        raise ValueError("Joint model success classes are incomplete")
    return probabilities[:, selected].sum(axis=1)


def multilabel_success_probability(model, matrix):
    probabilities = np.asarray(model.predict_proba(matrix), dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError("S26 multilabel CatBoost must return five heads")
    return probabilities[:, 0]


def fixed_s37_prediction(predictions, coefficients, intercept):
    """고정된 S37의 7개 확률 + 7개 logit 메타스택을 계산한다."""
    missing = set(S37_MEMBER_ORDER).difference(predictions)
    if missing:
        raise ValueError(f"S37 member predictions are missing: {sorted(missing)}")
    raw = np.column_stack([predictions[name] for name in S37_MEMBER_ORDER])
    clipped = np.clip(raw, 1e-7, 1.0 - 1e-7)
    design = np.column_stack([raw, np.log(clipped / (1.0 - clipped))])
    coefficients = np.asarray(coefficients, dtype="float64")
    if coefficients.shape != (14,):
        raise ValueError("S37 requires exactly fourteen frozen coefficients")
    linear = design @ coefficients + float(intercept)
    return 1.0 / (1.0 + np.exp(-linear))


def apply_pitch3_residual(s37, pitch3, core5, alpha):
    """S46에서 2022 OOF로 고정한 차이 축 하나만 S37에 더한다."""
    s37 = np.asarray(s37, dtype="float64")
    pitch3 = np.asarray(pitch3, dtype="float64")
    core5 = np.asarray(core5, dtype="float64")
    if s37.shape != pitch3.shape or s37.shape != core5.shape or s37.ndim != 1:
        raise ValueError("S48 residual inputs are not aligned")
    return np.clip(s37 + float(alpha) * (pitch3 - core5), 0.0, 1.0)


def predict_control_success(test):
    artifact = load_artifact()
    artifact_type = artifact.get("artifact_type")
    if artifact_type not in {
        "s26_slot_combo_v1",
        "s37_canonical_v1",
        "s48_s46_residual_v1",
    }:
        raise ValueError("S26-A/B, canonical S37, or S48 submission artifact is required")
    e42 = artifact["e42"]
    if e42.get("artifact_type") != "e42_frozen_robust_v1":
        raise ValueError("The nested frozen E42 artifact is invalid")
    binary = e42["wwb_lgbm"]
    residual = e42["season_residual"]
    catboost = e42["catboost"]
    xgboost = e42["xgboost"]
    trackman = e42["trackman_moe"]
    members = artifact["members"]

    wwb_matrix = build_member_features(test, binary, True)
    predictions = {}
    if "core5" in members:
        core = members["core5"]
        predictions["cor5"] = np.mean(
            np.column_stack([
                joint_success_probability(model, wwb_matrix, core["success_labels"])
                for model in core["models"]
            ]), axis=1,
        )
    if "call6" in members:
        call = members["call6"]
        predictions["cal6"] = np.mean(
            np.column_stack([
                joint_success_probability(model, wwb_matrix, call["success_labels"])
                for model in call["models"]
            ]), axis=1,
        )
    if "joint15" in members:
        joint15 = members["joint15"]
        predictions["j15"] = np.mean(
            np.column_stack([
                joint_success_probability(model, wwb_matrix, joint15["success_labels"])
                for model in joint15["models"]
            ]), axis=1,
        )
    if "pitch3" in members:
        pitch3 = members["pitch3"]
        predictions["pitch3"] = np.mean(
            np.column_stack([
                joint_success_probability(model, wwb_matrix, pitch3["success_labels"])
                for model in pitch3["models"]
            ]), axis=1,
        )
    if "multilabel_catboost" in members:
        predictions["mcb"] = multilabel_success_probability(
            members["multilabel_catboost"]["model"], wwb_matrix
        )
    if "joint_catboost" in members:
        joint = members["joint_catboost"]
        predictions["e69"] = joint_success_probability(
            joint["model"], wwb_matrix, joint["success_labels"]
        )
    residual_prediction = np.clip(
        residual["prediction_offset"]
        + residual["model"].predict(build_member_features(test, residual, False)),
        0.0,
        1.0,
    )
    predictions["season_residual"] = residual_prediction
    predictions["cb18"] = catboost["model"].predict_proba(
        build_member_features(test, catboost, True)
    )[:, 1]
    predictions["xgboost"] = xgboost["model"].predict_proba(
        build_member_features(test, xgboost, True)
    )[:, 1]
    trackman_matrix, trackman_probabilities = build_trackman_features(test, trackman)
    expert_predictions = np.column_stack(
        [
            trackman["experts"][group].predict_proba(trackman_matrix)[:, 1]
            for group in trackman["pitch_groups"]
        ]
    )
    predictions["trackman_moe"] = np.sum(
        expert_predictions * trackman_probabilities, axis=1
    )
    if artifact_type == "s48_s46_residual_v1":
        meta = artifact["s48_meta"]
        s37 = fixed_s37_prediction(
            predictions,
            meta["s37_coefficients"],
            meta["s37_intercept"],
        )
        return apply_pitch3_residual(
            s37,
            predictions["pitch3"],
            predictions["cor5"],
            meta["alpha"],
        )
    if artifact_type == "s37_canonical_v1":
        meta = artifact["s37_meta"]
        if tuple(meta["member_order"]) != S37_MEMBER_ORDER:
            raise ValueError("Canonical S37 member order changed")
        return fixed_s37_prediction(
            predictions,
            meta["coefficients"],
            meta["intercept"],
        )
    weights = artifact["member_weights"]
    if set(weights) != set(predictions):
        raise ValueError(
            f"S26 member mismatch: weights={sorted(weights)}, predictions={sorted(predictions)}"
        )
    prediction = sum(float(weights[name]) * predictions[name] for name in weights)
    # test 분포가 아니라 후보별 2024 validation center error에서 고정한 shift다.
    return np.clip(prediction + float(artifact["prediction_shift"]), 0.0, 1.0)


def main():
    test = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), encoding="utf-8-sig")
    sample = pd.read_csv(os.path.join(TEST_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    if ID_COL not in test or list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("입력 또는 제출 컬럼이 올바르지 않습니다")
    predictions = predict_control_success(test)
    prediction_by_id = dict(zip(test[ID_COL], predictions))
    sample[TARGET_COL] = [prediction_by_id[row_id] for row_id in sample[ID_COL]]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    sample.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH}: rows={len(sample)}")


if __name__ == "__main__":
    main()
