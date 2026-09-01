"""E08 제출 artifact의 평가 규모 추론 성능을 점검한다.

이 스크립트는 오직 실행 시간, 메모리 사용량, 출력 유효성 및 행 독립성을
검증하기 위한 개발용 도구다. ``data/test.csv``에서는 입력 컬럼 이름만 읽고,
실제 프록시 입력은 ``data/train.csv``의 앞 245,789행에서 target을 제외해 만든다.
test 행 집계/통계나 예측값 분포를 이용한 보정은 전혀 수행하지 않는다.

``psutil``은 이 개발용 벤치마크에서만 사용하며 제출 ``requirements.txt``에
추가하지 않는다.

실행 예시::

    python scripts/benchmark_e08_submission.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil


REPO = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from script import predict_from_artifact  # noqa: E402


DEFAULT_ROWS = 245_789
TARGET_COL = "control_success"
MIB = 1024**2


class PeakRssSampler:
    """현재 프로세스 RSS를 짧은 간격으로 표본 추출한다."""

    def __init__(self, interval_seconds: float = 0.005):
        if interval_seconds <= 0:
            raise ValueError("RSS 표본 추출 간격은 0보다 커야 합니다.")
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
        self.baseline_bytes = self.process.memory_info().rss
        self.peak_bytes = self.baseline_bytes
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            self.peak_bytes = max(
                self.peak_bytes,
                self.process.memory_info().rss,
            )
        except psutil.Error:
            # 프로세스 자체를 측정하므로 일반적으로 발생하지 않지만, 측정 실패가
            # 추론 작업을 중단시키지는 않도록 한다.
            pass

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread.start()

    def stop(self) -> int:
        self._sample()
        self._stop_event.set()
        self._thread.join()
        self._sample()
        return self.peak_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E08 제출 모델의 평가 규모 추론 시간/RSS/행 독립성 검증",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO / "models" / "e08_season_residual_ensemble.joblib",
        help="검증할 E08 joblib artifact 경로",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=REPO / "data" / "train.csv",
        help="입력 모양 프록시의 원본 train.csv 경로",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=REPO / "data" / "test.csv",
        help="공식 입력 컬럼 이름만 읽을 test.csv 경로",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help=f"프록시 입력 행 수 (기본값: {DEFAULT_ROWS:,})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "e08_inference_benchmark.json",
        help="벤치마크 결과 JSON 경로",
    )
    parser.add_argument(
        "--rss-sample-interval",
        type=float,
        default=0.005,
        help="RSS 표본 추출 간격(초)",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """상대 경로는 저장소 루트 기준으로 해석한다."""

    return path if path.is_absolute() else REPO / path


def read_input_proxy(train_path: Path, test_path: Path, rows: int) -> tuple[pd.DataFrame, list[str]]:
    """test 컬럼과 같은 모양의 train 기반 런타임 프록시를 읽는다.

    test에서는 헤더만 읽는다. groupby, 평균, 빈도 등 어떤 test 통계도
    계산하지 않으며 target도 프록시 입력에 포함하지 않는다.
    """

    if rows <= 0:
        raise ValueError("프록시 행 수는 0보다 커야 합니다.")

    test_columns = pd.read_csv(
        test_path,
        nrows=0,
        encoding="utf-8-sig",
    ).columns.tolist()
    if TARGET_COL in test_columns:
        raise ValueError(f"test 입력 컬럼에 target({TARGET_COL})이 포함되어 있습니다.")

    proxy = pd.read_csv(
        train_path,
        usecols=test_columns,
        nrows=rows,
        encoding="utf-8-sig",
    )
    if len(proxy) != rows:
        raise ValueError(
            f"요청한 {rows:,}행을 만들 수 없습니다. 읽은 행 수: {len(proxy):,}",
        )

    # pandas 버전에 따른 usecols 반환 순서 차이를 막아 공식 test 순서로 고정한다.
    proxy = proxy.loc[:, test_columns]
    return proxy, test_columns


def mib(value_bytes: int) -> float:
    return float(value_bytes / MIB)


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    train_path = resolve_path(args.train)
    test_path = resolve_path(args.test)
    output_path = resolve_path(args.output)

    for required_path in (model_path, train_path, test_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    process = psutil.Process(os.getpid())
    sampler = PeakRssSampler(args.rss_sample_interval)
    sampler.start()
    total_started = time.perf_counter()

    load_started = time.perf_counter()
    artifact = joblib.load(model_path)
    model_load_seconds = time.perf_counter() - load_started

    read_started = time.perf_counter()
    proxy, input_columns = read_input_proxy(train_path, test_path, args.rows)
    proxy_read_seconds = time.perf_counter() - read_started

    predict_started = time.perf_counter()
    predictions = np.asarray(
        predict_from_artifact(proxy, artifact),
        dtype=float,
    ).reshape(-1)
    prediction_seconds = time.perf_counter() - predict_started

    validation_started = time.perf_counter()
    expected_length = len(proxy)
    length_pass = len(predictions) == expected_length
    finite_pass = bool(np.isfinite(predictions).all())
    if len(predictions):
        prediction_min = float(np.min(predictions))
        prediction_max = float(np.max(predictions))
        unit_interval_pass = prediction_min >= 0.0 and prediction_max <= 1.0
    else:
        prediction_min = None
        prediction_max = None
        unit_interval_pass = False

    independence_rows = min(20, expected_length)
    single_predictions = np.array(
        [
            float(predict_from_artifact(proxy.iloc[[idx]].copy(), artifact)[0])
            for idx in range(independence_rows)
        ],
        dtype=float,
    )
    batch_predictions = predictions[:independence_rows]
    batch_vs_single_max_abs_diff = float(
        np.max(np.abs(batch_predictions - single_predictions)),
    )
    row_independence_pass = batch_vs_single_max_abs_diff <= 1e-12
    validation_seconds = time.perf_counter() - validation_started

    total_seconds = time.perf_counter() - total_started
    sampled_peak_bytes = sampler.stop()
    final_rss_bytes = process.memory_info().rss
    memory_info = process.memory_info()
    os_peak_working_set_bytes = getattr(memory_info, "peak_wset", None)

    all_checks_pass = bool(
        length_pass
        and finite_pass
        and unit_interval_pass
        and row_independence_pass
    )

    result = {
        "experiment_id": "E08",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": "inference_runtime_validation_only",
        "safety_note": (
            "test.csv에서는 입력 컬럼 헤더만 읽었다. 프록시는 train.csv의 target 제외 "
            "입력 행이며, test 집계/통계/예측 분포 보정은 수행하지 않았다."
        ),
        "psutil_scope": (
            "개발용 RSS 측정에만 사용하며 제출 requirements.txt에는 포함하지 않는다."
        ),
        "artifact": {
            "path": str(model_path.relative_to(REPO)),
            "size_bytes": model_path.stat().st_size,
        },
        "proxy": {
            "source": str(train_path.relative_to(REPO)),
            "test_header_source": str(test_path.relative_to(REPO)),
            "construction": (
                "train.csv 선두 행에서 공식 test 입력 컬럼만 선택; target 제외; "
                "집계/groupby/통계 없음"
            ),
            "rows": expected_length,
            "columns": len(input_columns),
            "target_excluded": TARGET_COL not in proxy.columns,
        },
        "timing_seconds": {
            "model_load": model_load_seconds,
            "proxy_read": proxy_read_seconds,
            "prediction": prediction_seconds,
            "validation_and_first20_single_prediction": validation_seconds,
            "total_after_imports": total_seconds,
        },
        "memory": {
            "measurement": "psutil RSS sampled in-process",
            "sample_interval_ms": args.rss_sample_interval * 1000.0,
            "baseline_rss_mib": mib(sampler.baseline_bytes),
            "sampled_peak_rss_mib": mib(sampled_peak_bytes),
            "sampled_peak_increase_mib": mib(
                max(0, sampled_peak_bytes - sampler.baseline_bytes),
            ),
            "final_rss_mib": mib(final_rss_bytes),
            "os_peak_working_set_mib": (
                mib(os_peak_working_set_bytes)
                if os_peak_working_set_bytes is not None
                else None
            ),
        },
        "checks": {
            "expected_prediction_length": expected_length,
            "actual_prediction_length": len(predictions),
            "length_pass": bool(length_pass),
            "finite_pass": finite_pass,
            "unit_interval_pass": bool(unit_interval_pass),
            "prediction_min": prediction_min,
            "prediction_max": prediction_max,
            "batch_vs_single_rows": independence_rows,
            "batch_vs_single_max_abs_diff": batch_vs_single_max_abs_diff,
            "row_independence_tolerance": 1e-12,
            "row_independence_pass": bool(row_independence_pass),
            "all_pass": all_checks_pass,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "psutil_development_only": psutil.__version__,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[E08 benchmark] rows={expected_length:,}, columns={len(input_columns)}")
    print(
        "  time(s): "
        f"load={model_load_seconds:.3f}, read={proxy_read_seconds:.3f}, "
        f"predict={prediction_seconds:.3f}, total={total_seconds:.3f}",
    )
    print(
        "  RSS(MiB): "
        f"baseline={mib(sampler.baseline_bytes):.1f}, "
        f"peak={mib(sampled_peak_bytes):.1f}, "
        f"increase={mib(max(0, sampled_peak_bytes - sampler.baseline_bytes)):.1f}",
    )
    print(
        "  checks: "
        f"length={length_pass}, finite={finite_pass}, [0,1]={unit_interval_pass}, "
        f"batch-vs-single max diff={batch_vs_single_max_abs_diff:.3e}",
    )
    print(f"  result: {output_path.relative_to(REPO)}")

    if not all_checks_pass:
        raise SystemExit("E08 추론 벤치마크 검증 실패")


if __name__ == "__main__":
    main()
