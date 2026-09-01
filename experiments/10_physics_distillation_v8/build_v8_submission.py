"""V8 제출본 빌드: 트리 수 교정 + 메타 계수 동결.

V7과의 차이 (V7은 LB 1089.81 -> 1066.74로 -23.07점 실패):
  1. 트리 수를 공식 빌더와 같은 절차로 교정 (1000 임의고정 -> early stopping 값)
  2. 메타러너 계수를 **건드리지 않는다** (V7은 비음수 재적합을 했다)

즉 공식 제출본(S73) 대비 바뀌는 것은 cb18 멤버의 학습 타깃 하나뿐이다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

REPO = Path(__file__).resolve().parent.parent
ARTIFACT_IN = REPO / "models" / "s37_canonical.joblib"
ARTIFACT_OUT = REPO / "models" / "v8_distilled_s37.joblib"
DISTILLED_MODEL = REPO / "results" / "deploy_distilled_cb18.cbm"
BASE_ZIP = REPO / "submissions" / "s73_center_shift.zip"
ZIP_PATH = REPO / "submissions" / "v8_physics_frozen_coef.zip"
RESULT_PATH = REPO / "results" / "v8_final_submission_build.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def assemble_artifact() -> dict:
    artifact = joblib.load(ARTIFACT_IN)
    old_coef = list(artifact["s37_meta"]["coefficients"])
    old_intercept = float(artifact["s37_meta"]["intercept"])

    new_model = CatBoostClassifier()
    new_model.load_model(str(DISTILLED_MODEL))
    artifact["e42"]["catboost"]["model"] = new_model

    # 계수는 절대 건드리지 않는다 (V7 실패의 두 번째 원인)
    assert list(artifact["s37_meta"]["coefficients"]) == old_coef
    assert float(artifact["s37_meta"]["intercept"]) == old_intercept
    artifact["s37_meta"]["coefficient_source"] += " | V8: cb18 member replaced by physics-distilled variant, coefficients UNCHANGED"
    return artifact, new_model.tree_count_


def package_zip() -> list[dict]:
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_bytes = ARTIFACT_OUT.read_bytes()
    with zipfile.ZipFile(BASE_ZIP) as src:
        if "model/s37_canonical.joblib" not in src.namelist():
            raise RuntimeError("베이스 zip 구조 이상")
        with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as out:
            for item in src.infolist():
                if item.filename == "model/s37_canonical.joblib":
                    out.writestr("model/s37_canonical.joblib", new_bytes)
                else:
                    out.writestr(item, src.read(item.filename))
    with zipfile.ZipFile(ZIP_PATH) as z:
        if z.testzip() is not None:
            raise RuntimeError("zip 손상")
        if any("\\" in n for n in z.namelist()):
            raise RuntimeError("경로 구분자 이상")
        return [{"name": i.filename, "size": i.file_size, "crc": f"{i.CRC:08X}"} for i in z.infolist()]


def smoke_test() -> dict:
    with tempfile.TemporaryDirectory(prefix="v8_sub_") as tmp_name:
        tmp = Path(tmp_name)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(tmp)
        d = tmp / "data"
        d.mkdir()
        shutil.copy2(REPO / "data" / "test.csv", d / "test.csv")
        shutil.copy2(REPO / "data" / "sample_submission.csv", d / "sample_submission.csv")
        completed = subprocess.run([sys.executable, "script.py"], cwd=tmp,
                                   text=True, capture_output=True, check=True)
        out = tmp / "output" / "submission.csv"
        sub = pd.read_csv(out)
        if list(sub.columns) != ["row_id", "control_success"]:
            raise RuntimeError("컬럼 이상")
        vals = sub["control_success"].to_numpy(dtype="float64")
        if not ((vals >= 0) & (vals <= 1)).all() or not np.isfinite(vals).all():
            raise RuntimeError("확률 범위 이상")

        # script.py가 numpy/pandas보다 torch를 먼저 import하므로 순서를 지킨다.
        audit_code = (
            "import importlib.util, json; "
            "spec=importlib.util.spec_from_file_location('m','script.py'); "
            "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "import numpy as np, pandas as pd; "
            "t=pd.read_csv('data/test.csv', encoding='utf-8-sig'); "
            "b=m.predict_control_success(t.copy()); "
            "s=np.array([m.predict_control_success(t.iloc[[i]].copy().reset_index(drop=True))[0] for i in range(len(t))]); "
            "r=m.predict_control_success(t.iloc[::-1].copy().reset_index(drop=True))[::-1]; "
            "print(json.dumps({'batch_single': float(np.max(np.abs(b-s))), 'reverse': float(np.max(np.abs(b-r)))}))"
        )
        audit = subprocess.run([sys.executable, "-c", audit_code], cwd=tmp,
                               text=True, capture_output=True, check=True)
        ap = json.loads(audit.stdout.strip().splitlines()[-1])
        if ap["batch_single"] > 1e-8 or ap["reverse"] > 1e-8:
            raise RuntimeError(f"행 독립성 위반: {ap}")
        return {"rows": len(sub), "min": float(vals.min()), "max": float(vals.max()),
                "row_independence": ap,
                "stdout_tail": completed.stdout.strip().splitlines()[-2:]}


def main() -> None:
    started = time.time()
    if not DISTILLED_MODEL.exists():
        raise FileNotFoundError(f"{DISTILLED_MODEL} 없음 -- 캐글 V8 학습을 먼저 끝낼 것")

    print("[1/3] 아티팩트 조립 (계수 동결)...")
    artifact, tree_count = assemble_artifact()
    joblib.dump(artifact, ARTIFACT_OUT, compress=3)
    print(f"  distilled cb18 트리수: {tree_count}")

    print("[2/3] zip 패키징...")
    entries = package_zip()

    print("[3/3] 스모크 테스트...")
    smoke = smoke_test()

    v8_meta = json.loads((REPO / "results" / "deploy_distilled_cb18_meta.json").read_text(encoding="utf-8"))
    result = {
        "experiment_id": "V8-submission",
        "description": "S73 pipeline with cb18 replaced by physics-distilled variant; tree count calibrated like the official builder; meta coefficients FROZEN",
        "fixes_vs_v7": ["tree count calibrated (V7 used 1000 arbitrarily)",
                         "no meta-learner refit (V7 refit non-negative)"],
        "v7_lb_result": {"score": 1066.7396556619, "vs_s73": -23.07},
        "s73_reference_lb": 1089.8103798611,
        "training_meta": v8_meta,
        "distilled_tree_count": int(tree_count),
        "zip_path": str(ZIP_PATH.relative_to(REPO)),
        "zip_sha256": sha256(ZIP_PATH),
        "zip_entries": entries,
        "smoke_test": smoke,
        "test_values_read": False,
        "leaderboard_used": False,
        "status": "built_not_submitted",
        "elapsed_sec": time.time() - started,
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "zip_entries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
