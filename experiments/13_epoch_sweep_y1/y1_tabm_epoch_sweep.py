"""Y1: TabM이 과소학습(undertrained)돼 있는지 직접 재는 진단. Kaggle GPU 전용.

동기 (2026-08-31 저녁 발견):
  프로덕션 TabM(S67)은 EPOCHS=6 하드코딩으로 학습됐고, early stopping이 없다.
  기록된 학습손실이 마지막 에폭까지 계속 떨어진다:
      1: 0.684460  2: 0.682971  3: 0.681955
      4: 0.681005  5: 0.679995  6: 0.678824   <- 평탄화 조짐 없음
  에폭당 58초이므로 총 6분 학습하고 멈춘 것이다. 시드도 1개(42)다.

  그런데 이 TabM은 이 프로젝트에서 실제 리더보드 점수를 크게 올린 **유일한** 축이다
  (S67: 1071.75 -> 1080.19, +8.44, 내부 예측 대비 전이율 96%).
  나머지 축(GBDT 멤버 교체 ±1, 계수 재적합 -27, 스칼라 재튜닝 ~0)은 전부 죽었다.

  외부 근거: TabReD (ICLR 2025)는 "시간 이동(temporal shift)이 있는 실전 표형 데이터"
  벤치마크인데, 그 조건에서 TabM이 최고 성능이고 retrieval 계열(TabR 등)은 무너진다고
  보고한다. 우리 데이터가 정확히 그 조건(2019-2024 학습 -> 2025 평가)이고,
  우리 S64(TabR)가 실제로 기각됐던 것과 일치한다. 즉 아키텍처 선택은 맞았고,
  학습을 덜 시킨 것이 문제일 가능성이 높다.

무엇을 재는가:
  fold별로 EPOCHS를 최대 MAX_EPOCHS까지 늘리면서 **매 에폭 검증 예측을 만들어**
  Brier / centered score를 기록한다. 그러면 다음 세 가지를 한 번에 알 수 있다:
    1) 더 학습하면 좋아지는가 (6에폭이 최적이 아니었는가)
    2) 과적합이 시작되는 지점은 어디인가
    3) 그 최적 에폭에서 S37 대비 얼마나 개선되는가

  체크포인트 에폭에서는 검증 예측을 .npy로 저장한다. 그러면 나중에 게이트
  가중치를 재도출할 때 재학습 없이 아무 에폭이나 골라 쓸 수 있다.

판정 프로토콜 (저장소 관례 그대로):
  2023 fold의 곡선에서 최적 에폭을 고르고, 2024 fold에서 1회 확인한다.
  2024를 보고 에폭을 고르지 않는다.

test 값, test 분포, 리더보드는 전혀 사용하지 않는다.
"""

from __future__ import annotations

import gc
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Kaggle 번들에 wheel이 없을 수도 있으므로 없으면 설치한다 (enable_internet=true 필요).
try:
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from tabm import TabM
except ModuleNotFoundError:  # pragma: no cover - Kaggle 환경 전용 경로
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "tabm", "rtdl_num_embeddings"], check=True)
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from tabm import TabM

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import ab_feature as legacy  # noqa: E402
from e13_feature_groups import TARGET, centered_metrics, make_temporal_features  # noqa: E402
from e13_rolling_validation import _prepare_fold  # noqa: E402
from e61_joint_outcome import feature_contract  # noqa: E402
from s58_tabm_binary import category_matrix, fit_preprocessor, transform_rows  # noqa: E402

try:
    from s37_canonical import load_oof_member_matrix, predict_from_members
    HAVE_S37 = True
except Exception:  # pragma: no cover
    HAVE_S37 = False

RESULT_PATH = REPO / "results" / "y1_tabm_epoch_sweep.json"
PRED_OUT_DIR = REPO / "results" / "preds" / "y1_tabm_sweep"

# --- S67 프로덕션과 동일하게 고정하는 것들 (아키텍처는 하나도 바꾸지 않는다) ---
SEED = 42
K = 32
N_BLOCKS = 2
D_BLOCK = 512
D_EMBEDDING = 8
N_BINS = 32
BIN_SAMPLE_SIZE = 200_000
BATCH_SIZE = 1024
INFERENCE_BATCH_SIZE = 4096
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 3e-4

# --- 이 실험에서만 바꾸는 것: 에폭 수 ---
MAX_EPOCHS = 40
CHECKPOINT_EPOCHS = (6, 12, 18, 24, 30, 36, 40)   # 이 에폭들의 검증 예측을 저장
FOLDS = (2022, 2023, 2024)
PRODUCTION_EPOCHS = 6                              # 비교 기준 (현재 배포된 값)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(numeric_train: np.ndarray, cardinalities: list[int], device: torch.device):
    rng = np.random.default_rng(SEED)
    sample_size = min(BIN_SAMPLE_SIZE, len(numeric_train))
    sample_indices = rng.choice(len(numeric_train), size=sample_size, replace=False)
    bin_tensors = compute_bins(torch.from_numpy(numeric_train[sample_indices]), n_bins=N_BINS)
    num_embeddings = PiecewiseLinearEmbeddings(
        bin_tensors, d_embedding=D_EMBEDDING, activation=True, version="B")
    model = TabM.make(
        n_num_features=numeric_train.shape[1],
        cat_cardinalities=cardinalities,
        num_embeddings=num_embeddings,
        d_out=1, k=K, n_blocks=N_BLOCKS, d_block=D_BLOCK,
        dropout=0.10, arch_type="tabm",
    ).to(device)
    return model


@torch.inference_mode()
def predict(model, categorical: np.ndarray, numeric: np.ndarray, device) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(numeric), INFERENCE_BATCH_SIZE):
        stop = min(start + INFERENCE_BATCH_SIZE, len(numeric))
        x_num = torch.from_numpy(numeric[start:stop]).to(device, non_blocking=True)
        x_cat = torch.from_numpy(categorical[start:stop]).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x_num, x_cat).squeeze(-1)
        # 공식 권장: head logit이 아니라 head별 확률을 평균한다.
        outputs.append(torch.sigmoid(logits.float()).mean(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype("float64")


def prepare_fold(frame_all, original_features, temporal, valid_season: int):
    frame, train_mask, valid_mask, base_features = _prepare_fold(
        frame_all, original_features, temporal, valid_season)
    features = feature_contract(base_features)
    train_positions = np.flatnonzero(train_mask.to_numpy())
    valid_positions = np.flatnonzero(valid_mask.to_numpy())
    raw_categories = category_matrix(frame)
    spec = fit_preprocessor(frame, raw_categories, features, train_positions)
    train_arrays = transform_rows(frame, raw_categories, train_positions, spec)
    valid_arrays = transform_rows(frame, raw_categories, valid_positions, spec)
    cardinalities = [len(values) + 1 for values in spec["category_values"]]
    target_train = frame.loc[train_mask, TARGET].to_numpy(dtype="float32")
    target_valid = frame.loc[valid_mask, TARGET].to_numpy(dtype="float64")
    del raw_categories
    gc.collect()
    return train_arrays, target_train, valid_arrays, target_valid, cardinalities


def run_fold(valid_season: int, frame_all, original_features, temporal, device) -> dict:
    print(f"\n{'='*70}\nfold {valid_season}: 데이터 준비 중...", flush=True)
    train_arrays, y_train, valid_arrays, y_valid, cardinalities = prepare_fold(
        frame_all, original_features, temporal, valid_season)
    categorical_train, numeric_train = train_arrays
    categorical_valid, numeric_valid = valid_arrays
    print(f"  train={len(numeric_train):,}  valid={len(numeric_valid):,}  "
          f"numeric={numeric_train.shape[1]}", flush=True)

    s37_reference = None
    if HAVE_S37:
        try:
            s37 = predict_from_members(load_oof_member_matrix(valid_season))
            if len(s37) == len(y_valid):
                s37_reference = centered_metrics(s37, y_valid)
                print(f"  S37 기준선: brier={s37_reference['brier']:.8f} "
                      f"centered={s37_reference['centered_score']:.2f}", flush=True)
        except Exception as exc:
            print(f"  (S37 기준선 생략: {exc})", flush=True)

    set_seed()
    model = build_model(numeric_train, cardinalities, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(SEED)

    history = []
    PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        started = time.time()
        order = rng.permutation(len(numeric_train))
        losses = []
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            positions = order[start:start + BATCH_SIZE]
            x_num = torch.from_numpy(numeric_train[positions]).to(device, non_blocking=True)
            x_cat = torch.from_numpy(categorical_train[positions]).to(device, non_blocking=True)
            y = torch.from_numpy(y_train[positions]).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x_num, x_cat).squeeze(-1)
                # 공식 지침: head 평균의 loss가 아니라 head별 loss의 평균을 최적화한다.
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y[:, None].expand_as(logits))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))

        prediction = predict(model, categorical_valid, numeric_valid, device)
        metrics = centered_metrics(prediction, y_valid)
        record = {
            "epoch": epoch,
            "train_bce": float(np.mean(losses)),
            "valid_brier": metrics["brier"],
            "valid_centered_score": metrics["centered_score"],
            "valid_pred_mean": float(prediction.mean()),
            "sec": time.time() - started,
        }
        if s37_reference is not None:
            record["delta_brier_vs_s37"] = metrics["brier"] - s37_reference["brier"]
            record["delta_centered_vs_s37"] = (
                metrics["centered_score"] - s37_reference["centered_score"])
        history.append(record)

        flag = ""
        if epoch in CHECKPOINT_EPOCHS:
            np.save(PRED_OUT_DIR / f"y1_{valid_season}_tabm_epoch{epoch}.npy",
                    prediction.astype("float32"))
            flag = " [저장]"
        print(f"  epoch={epoch:2d}/{MAX_EPOCHS} train_bce={record['train_bce']:.6f} "
              f"valid_brier={record['valid_brier']:.8f} "
              f"centered={record['valid_centered_score']:.2f} "
              f"({record['sec']:.0f}s){flag}", flush=True)

    best = min(history, key=lambda r: r["valid_brier"])
    production = next(r for r in history if r["epoch"] == PRODUCTION_EPOCHS)
    K_POINTS = 0.2485
    gain_points = -100000.0 * (best["valid_brier"] - production["valid_brier"]) / K_POINTS
    print(f"\n  fold {valid_season} 요약: 최적 에폭={best['epoch']} "
          f"(brier {production['valid_brier']:.8f} -> {best['valid_brier']:.8f}, "
          f"현재 6에폭 대비 {gain_points:+.2f}점)", flush=True)

    del model, train_arrays, valid_arrays, numeric_train, categorical_train
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "valid_season": valid_season,
        "history": history,
        "s37_reference": s37_reference,
        "best_epoch_by_valid_brier": best["epoch"],
        "production_epoch_record": production,
        "best_record": best,
        "gain_points_vs_production_epochs": gain_points,
    }


def main() -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("Y1은 GPU 실험이다. Kaggle accelerator를 GPU로 설정할 것.")
    device = torch.device("cuda")
    print(f"device={torch.cuda.get_device_name(0)}", flush=True)

    print("데이터 로딩...", flush=True)
    frame_all, original_features = legacy.load_data()
    temporal = make_temporal_features(frame_all)

    folds = {}
    for season in FOLDS:
        folds[str(season)] = run_fold(season, frame_all, original_features, temporal, device)

    # 판정: 2023에서 최적 에폭을 고르고 2024에서 1회 확인한다 (2024로 고르지 않는다).
    selection = folds["2023"]
    chosen_epoch = selection["best_epoch_by_valid_brier"]
    holdout = folds["2024"]
    holdout_at_chosen = next(r for r in holdout["history"] if r["epoch"] == chosen_epoch)
    holdout_at_production = holdout["production_epoch_record"]
    K_POINTS = 0.2485
    confirm_points = -100000.0 * (
        holdout_at_chosen["valid_brier"] - holdout_at_production["valid_brier"]) / K_POINTS
    passes = confirm_points > 0.0

    payload = {
        "experiment_id": "Y1",
        "description": "TabM epoch sweep: is the production 6-epoch TabM undertrained?",
        "motivation": "S67 TabM trained 6 epochs with train loss still falling; it is the only axis that ever moved the real leaderboard (+8.44, 96% transfer)",
        "external_evidence": "TabReD (ICLR 2025): under temporal shift TabM is the best deep tabular model while retrieval methods degrade -- matches our S64/TabR rejection",
        "architecture_changed": False,
        "fixed_hyperparameters": {
            "k": K, "n_blocks": N_BLOCKS, "d_block": D_BLOCK, "d_embedding": D_EMBEDDING,
            "n_bins": N_BINS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY, "seed": SEED,
        },
        "max_epochs": MAX_EPOCHS,
        "production_epochs": PRODUCTION_EPOCHS,
        "folds": folds,
        "protocol": "choose epoch on 2023 curve, confirm once on 2024",
        "chosen_epoch_from_2023": chosen_epoch,
        "confirm_2024_points_vs_production": confirm_points,
        "passes": bool(passes),
        "checkpoint_epochs_saved": list(CHECKPOINT_EPOCHS),
        "test_values_read": False,
        "leaderboard_used": False,
        "elapsed_sec": time.time() - started,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("Y1 결론")
    print("=" * 70)
    for season in FOLDS:
        f = folds[str(season)]
        print(f"  {season}: 최적 에폭 {f['best_epoch_by_valid_brier']:2d}  "
              f"6에폭 대비 {f['gain_points_vs_production_epochs']:+.2f}점")
    print(f"\n  2023에서 고른 에폭: {chosen_epoch}")
    print(f"  2024에서 확인: {confirm_points:+.2f}점 (현재 6에폭 대비)")
    print(f"  통과: {passes}")
    print(f"\n결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
