"""V13(Z1+V11) submission inference: 평탄 TabM 게이트 + 3시드 배깅 champion.

V11(= V8 + champion 시드배깅) 대비 유일한 변경:
  TabM 게이트의 "편차크기 구간별 가중치"(0.05/0.05/0.10/0.10, 편차가 클수록 강함)를
  버리고, game_type == R 행 전체에 균일 가중치 w = 0.15 하나만 쓴다.

    corrected = champion + 0.15 * (TabM - S37)      # R 행에만

근거 (results/z1_flat_gate_confirm.json, results/z4_bootstrap_ci.json):
  프로덕션 기준선 재현    2023 +3.262145 / 2024 +4.916895 (W1 기록값과 6자리 일치)
  2023에서만 w 선택       w = 0.15, 프로덕션 대비 +2.234, 그리드 경계 아님
  2024 확인(1회)          프로덕션 대비 +3.432, 3점 기준 통과
  부트스트랩 95% CI       2024 [+1.063, +5.928], P(>0)=0.995
  예측평균 이동           +0.000105 -> CENTER_SHIFT 재산출 불필요

가설: 큰 |TabM-S37| 편차는 TabM이 S37과 가장 크게 어긋난 지점 = 가장 못 믿을
지점이다. 프로덕션은 그 지점을 확대하고 있었다(수축했어야 했다).

보정항 w*(TabM-S37)은 champion과 무관하므로 시드배깅 변경과 층이 겹치지 않는다.
S37 메타 계수, CENTER_SHIFT, 게이트 대상 행(game_type==R)은 모두 그대로다.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd
from rtdl_num_embeddings import PiecewiseLinearEmbeddings
from tabm import TabM


ID_COL = "row_id"
TARGET_COL = "control_success"
S61_SHARE = 0.70
S63_SHARE = 0.30

# V11: champion 시드 배깅 (results/r10_seed_bagging.json에서 검증됨).
SEEDS = (42, 2024, 3407)

# Z1 평탄 게이트 상수. 대상 행(game_type == R)은 S69/S73/V11과 동일하다.
GATE_COLUMN = "game_type"
GATE_VALUE = "R"
# Z1: 구간별 가중치를 버리고 R 행 전체에 균일 가중치 하나.
# w는 2023 OOF에서만 선택(0.01~0.60 그리드, 최적 0.15, 경계 아님) -> 2024에서 1회 확인.
FLAT_GATE_WEIGHT = 0.15

# S73 전용: 전역 중심 편향 보정 (docs/HANDOFF_2026-08-29_CENTER_BIAS_AUDIT.md 섹션 5).
# S69 자신의 2024 예측 평균(0.49476)과 2024 실제 평균(0.48610)의 차이. 변경 없음.
CENTER_SHIFT = -0.00866

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parent if (SCRIPT_PATH.parent / "model").exists() else SCRIPT_PATH.parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
OUT_PATH = ROOT / "output" / "submission.csv"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _artifact_path(name: str) -> Path:
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    return path


def _numeric(rows: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype="float64")


def _s61_raw_features(rows: pd.DataFrame) -> np.ndarray:
    pitcher_success = _numeric(rows, "asof_pitcher_success_rate")
    prev1_success = _numeric(rows, "asof_pitcher_prev1_game_success_rate")
    prev5_success = _numeric(rows, "asof_pitcher_prev5_game_success_rate")
    batter_success = _numeric(rows, "asof_batter_success_rate")
    pitcher_n = np.clip(_numeric(rows, "asof_pitcher_n"), 0.0, None)
    batter_n = np.clip(_numeric(rows, "asof_batter_n"), 0.0, None)
    return np.column_stack(
        [
            pitcher_success,
            _numeric(rows, "asof_pitcher_reverse_rate"),
            _numeric(rows, "asof_pitcher_middle_rate"),
            _numeric(rows, "asof_pitcher_ball_rate"),
            _numeric(rows, "asof_pitcher_strike_rate"),
            prev5_success,
            _numeric(rows, "asof_pitcher_prev5_game_middle_rate"),
            batter_success,
            _numeric(rows, "asof_batter_middle_rate"),
            prev1_success - prev5_success,
            pitcher_success - batter_success,
            np.log1p(pitcher_n),
            np.log1p(batter_n),
            _numeric(rows, "asof_pitcher_fastball_rate"),
            _numeric(rows, "asof_pitcher_breaking_rate"),
            _numeric(rows, "asof_pitcher_offspeed_rate"),
        ]
    )


def _s61_bucket_keys(rows: pd.DataFrame) -> np.ndarray:
    game_type = rows["game_type"].astype("string").str.upper().eq("F").to_numpy(dtype="int64")
    balls = np.nan_to_num(_numeric(rows, "balls_before"), nan=-1.0).astype("int64")
    strikes = np.nan_to_num(_numeric(rows, "strikes_before"), nan=-1.0).astype("int64")
    pitcher_hand = np.nan_to_num(_numeric(rows, "pitcher_hand"), nan=-1.0).astype("int64")
    batter_hand = np.nan_to_num(_numeric(rows, "batter_hand"), nan=-1.0).astype("int64")
    count_state = (balls + 1) * 5 + (strikes + 1)
    hand_matchup = (pitcher_hand + 1) * 4 + (batter_hand + 1)
    return game_type * 10_000 + count_state * 100 + hand_matchup


def _s61_transform(raw_features: np.ndarray, feature_spec: dict) -> np.ndarray:
    values = np.asarray(raw_features, dtype="float64")
    median = np.asarray(feature_spec["median"], dtype="float64")
    filled = np.where(np.isfinite(values), values, median)
    clipped = np.clip(filled, feature_spec["lower"], feature_spec["upper"])
    transformed = (clipped - feature_spec["mean"]) / feature_spec["scale"]
    if not np.isfinite(transformed).all():
        raise RuntimeError("S61 transformed features contain non-finite values")
    return transformed.astype("float32")


def _predict_s61(artifact: dict, base_prediction: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    raw_features = _s61_raw_features(rows)
    features = _s61_transform(raw_features, artifact["feature_spec"])
    keys = _s61_bucket_keys(rows)
    residual = np.full(len(rows), float(artifact["global_residual"]), dtype="float64")
    buckets = artifact["buckets"]
    for bucket_key in np.unique(keys):
        positions = np.flatnonzero(keys == bucket_key)
        bucket = buckets.get(int(bucket_key))
        if bucket is None:
            continue
        query = features[positions].astype("float64")
        centers = np.asarray(bucket["centers"], dtype="float64")
        distance = ((query[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        n_neighbors = min(4, len(centers))
        nearest = np.argpartition(distance, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        nearest_distance = np.take_along_axis(distance, nearest, axis=1)
        nearest_residual = np.asarray(bucket["residuals"], dtype="float64")[nearest]
        weights = 1.0 / (nearest_distance + 1e-3)
        residual[positions] = (weights * nearest_residual).sum(axis=1) / weights.sum(axis=1)
    strength = float(artifact["correction_strength"])
    return np.clip(np.asarray(base_prediction, dtype="float64") + strength * residual, 0.0, 1.0)


def _normalize_ids(rows: pd.DataFrame, column: str) -> np.ndarray:
    return rows[column].astype("string").fillna("__MISSING__").to_numpy(dtype=str)


def _indices(values: np.ndarray, ordered_ids: tuple[str, ...]) -> np.ndarray:
    lookup = {value: index for index, value in enumerate(ordered_ids)}
    return np.fromiter((lookup.get(value, -1) for value in values), dtype="int64")


def _predict_s63(artifact: dict, base_prediction: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    pitcher_ids = tuple(artifact["pitcher_ids"])
    batter_ids = tuple(artifact["batter_ids"])
    pitcher_index = _indices(_normalize_ids(rows, "pitcher_id"), pitcher_ids)
    batter_index = _indices(_normalize_ids(rows, "batter_id"), batter_ids)
    residual = np.full(len(rows), float(artifact["global_residual"]), dtype="float64")
    known_pitcher = pitcher_index >= 0
    known_batter = batter_index >= 0
    residual[known_pitcher] += np.asarray(artifact["pitcher_bias"])[pitcher_index[known_pitcher]]
    residual[known_batter] += np.asarray(artifact["batter_bias"])[batter_index[known_batter]]
    known_pair = known_pitcher & known_batter
    if known_pair.any():
        residual[known_pair] += np.sum(
            np.asarray(artifact["pitcher_factors"])[pitcher_index[known_pair]]
            * np.asarray(artifact["batter_factors"])[batter_index[known_pair]],
            axis=1,
        )
    strength = float(artifact["correction_strength"])
    return np.clip(np.asarray(base_prediction, dtype="float64") + strength * residual, 0.0, 1.0)


def _category_matrix(frame: pd.DataFrame) -> np.ndarray:
    def integer(name: str) -> np.ndarray:
        return pd.to_numeric(frame[name], errors="coerce").fillna(-1).to_numpy(dtype="int64")

    pitcher_hand = integer("pitcher_hand")
    batter_hand = integer("batter_hand")
    balls = integer("balls_before")
    strikes = integer("strikes_before")
    inning = integer("inning")
    return np.column_stack(
        [
            integer("pitcher_id"),
            integer("batter_id"),
            integer("pitcher_team_id"),
            integer("batter_team_id"),
            integer("game_type"),
            integer("top_bottom"),
            integer("base_state"),
            pitcher_hand * 3 + batter_hand,
            balls * 4 + strikes,
            np.clip(inning, 0, 10),
        ]
    )


def _encode_categories(raw: np.ndarray, category_values: list[np.ndarray]) -> np.ndarray:
    values = np.asarray(raw, dtype="int64")
    encoded = np.zeros(values.shape, dtype="int64")
    for column, known_values in enumerate(category_values):
        known = np.asarray(known_values, dtype="int64")
        positions = np.searchsorted(known, values[:, column])
        within = positions < len(known)
        matched = np.zeros(len(values), dtype=bool)
        matched[within] = known[positions[within]] == values[within, column]
        encoded[matched, column] = positions[matched] + 1
    return encoded


def _transform_tabm_rows(frame: pd.DataFrame, raw_categories: np.ndarray, tabm_spec: dict) -> tuple[np.ndarray, np.ndarray]:
    categorical = _encode_categories(raw_categories, tabm_spec["category_values"])
    numeric = frame[tabm_spec["numeric_features"]].replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(tabm_spec["median"]).clip(tabm_spec["lower"], tabm_spec["upper"], axis=1)
    numeric = ((numeric - tabm_spec["mean"]) / tabm_spec["scale"]).to_numpy(dtype="float32")
    if not np.isfinite(numeric).all():
        raise RuntimeError("S67 TabM numeric preprocessing produced a non-finite value")
    return categorical, numeric


def _prepare_tabm_frame(test: pd.DataFrame, preprocessor: dict) -> pd.DataFrame:
    from scripts import ab_feature as legacy
    from scripts.e13_feature_groups import (
        FeatureArtifact,
        make_composition_features,
        transform_current_rows,
    )

    frame = test.drop(columns=[ID_COL], errors="ignore").copy()
    original_features = list(preprocessor["original_features"])
    missing = set(original_features).difference(frame.columns)
    if missing:
        raise ValueError(f"S67 input is missing columns: {sorted(missing)}")

    e04 = legacy.f_e04_all(frame, preprocessor["legacy_spec"]).astype("float32")
    temporal_artifact = FeatureArtifact(**preprocessor["temporal_artifact"])
    temporal = transform_current_rows(frame, temporal_artifact)
    composition = make_composition_features(frame)
    feature_frame = pd.concat([frame, e04, temporal, composition], axis=1)

    for column in legacy.CAT_COLS:
        mapping = preprocessor["legacy_cat_maps"][column]
        feature_frame[column] = feature_frame[column].map(mapping).fillna(-1).astype("int32")

    return feature_frame


def _build_tabm_model(checkpoint: dict, device: torch.device) -> TabM:
    bins = [torch.as_tensor(values, dtype=torch.float32) for values in checkpoint["bins"]]
    num_embeddings = PiecewiseLinearEmbeddings(
        bins,
        d_embedding=int(checkpoint["d_embedding"]),
        activation=True,
        version="B",
    )
    model = TabM.make(
        n_num_features=int(checkpoint["n_numeric"]),
        cat_cardinalities=[int(value) for value in checkpoint["cardinalities"]],
        num_embeddings=num_embeddings,
        d_out=1,
        k=int(checkpoint["k"]),
        n_blocks=int(checkpoint["n_blocks"]),
        d_block=int(checkpoint["d_block"]),
        dropout=0.10,
        arch_type="tabm",
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def _predict_tabm(test: pd.DataFrame) -> tuple[np.ndarray, dict]:
    preprocessor = joblib.load(_artifact_path("s67_final_tabm_preprocessor.joblib"))
    checkpoint = torch.load(
        _artifact_path("s67_final_official_tabm.pt"),
        map_location="cpu",
        weights_only=False,
    )
    feature_frame = _prepare_tabm_frame(test, preprocessor)
    raw_categories = _category_matrix(feature_frame)
    categorical, numeric = _transform_tabm_rows(feature_frame, raw_categories, preprocessor["tabm_spec"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_tabm_model(checkpoint, device)
    outputs = []
    for start in range(0, len(numeric), 4096):
        stop = min(start + 4096, len(numeric))
        x_num = torch.from_numpy(numeric[start:stop]).to(device, non_blocking=device.type == "cuda")
        x_cat = torch.from_numpy(categorical[start:stop]).to(device, non_blocking=device.type == "cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast:
            logits = model(x_num, x_cat).squeeze(-1)
        outputs.append(torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype("float64"), checkpoint


def predict_control_success(test: pd.DataFrame) -> np.ndarray:
    if ID_COL not in test.columns:
        raise ValueError("test.csv must include row_id")

    s37_inference = _load_module("s37_inference", _artifact_path("s37_inference.py"))
    s37_inference._ARTIFACT_CACHE = joblib.load(_artifact_path("s37_canonical.joblib"))
    s37 = np.asarray(s37_inference.predict_control_success(test.copy()), dtype="float64")

    # V11: champion을 시드 3개 각각으로 예측한 뒤 "예측값"을 평균한다
    # (내부 파라미터를 합치는 게 아님 -- results/r10_seed_bagging.json과 동일 방식).
    s61_preds = []
    s63_preds = []
    for seed in SEEDS:
        s61_artifact = joblib.load(_artifact_path(f"s61_seed{seed}.joblib"))
        s63_artifact = joblib.load(_artifact_path(f"s63_seed{seed}.joblib"))
        s61_preds.append(_predict_s61(s61_artifact, s37, test))
        s63_preds.append(_predict_s63(s63_artifact, s37, test))
    s61 = np.mean(s61_preds, axis=0)
    s63 = np.mean(s63_preds, axis=0)
    champion = np.clip(S61_SHARE * s61 + S63_SHARE * s63, 0.0, 1.0)

    tabm, _checkpoint = _predict_tabm(test)

    # Z1 평탄 게이트. w는 2023 OOF에서만 선택했고 test 값이나 리더보드는 쓰지 않았다.
    gate = (
        test[GATE_COLUMN]
        .astype("string")
        .fillna("__MISSING__")
        .eq(GATE_VALUE)
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    dev = tabm - s37
    correction = np.zeros(len(test), dtype="float64")
    correction[gate] = FLAT_GATE_WEIGHT * dev[gate]

    gated = np.clip(champion + correction, 0.0, 1.0)
    return np.clip(gated + CENTER_SHIFT, 0.0, 1.0)


def main() -> None:
    test = pd.read_csv(DATA_DIR / "test.csv", encoding="utf-8-sig")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv", encoding="utf-8-sig")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample_submission.csv columns must be row_id, control_success")

    predictions = predict_control_success(test)
    prediction_by_id = dict(zip(test[ID_COL], predictions))
    sample[TARGET_COL] = [prediction_by_id[row_id] for row_id in sample[ID_COL]]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {OUT_PATH}: rows={len(sample)}")


if __name__ == "__main__":
    main()
