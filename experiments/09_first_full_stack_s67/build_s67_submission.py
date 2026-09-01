"""Build the S67/S66B final submission ZIP."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
ZIP_PATH = REPO / "submissions" / "s67_final_s66b_gated_tabm.zip"
RESULT_PATH = REPO / "results" / "s67_final_submission_build.json"
ROW_AUDIT_TOLERANCE = 1e-8

SCRIPT_PATH = REPO / "scripts" / "s67_script.py"
S37_INFERENCE_PATH = REPO / "scripts" / "s26_script.py"
HELPER_SCRIPTS = [
    REPO / "scripts" / "ab_feature.py",
    REPO / "scripts" / "e13_feature_groups.py",
]
ARTIFACTS = [
    (REPO / "models" / "s37_canonical.joblib", "model/s37_canonical.joblib"),
    (
        REPO / "models" / "s61_contextual_prototype_retrieval.joblib",
        "model/s61_contextual_prototype_retrieval.joblib",
    ),
    (
        REPO / "models" / "s63_low_rank_matchup_residual.joblib",
        "model/s63_low_rank_matchup_residual.joblib",
    ),
    (
        REPO / "artifacts" / "s67_final_tabm" / "model" / "s67_final_official_tabm.pt",
        "model/s67_final_official_tabm.pt",
    ),
    (
        REPO / "artifacts" / "s67_final_tabm" / "model" / "s67_final_tabm_preprocessor.joblib",
        "model/s67_final_tabm_preprocessor.joblib",
    ),
    (
        REPO / "artifacts" / "s67_final_tabm" / "s67_final_tabm_training.json",
        "model/s67_final_tabm_training.json",
    ),
    (
        REPO / "transfer_s65_tabm_kaggle" / "wheels" / "tabm-0.0.3-py3-none-any.whl",
        "model/wheels/tabm-0.0.3-py3-none-any.whl",
    ),
    (
        REPO / "transfer_s65_tabm_kaggle" / "wheels" / "rtdl_num_embeddings-0.0.12-py3-none-any.whl",
        "model/wheels/rtdl_num_embeddings-0.0.12-py3-none-any.whl",
    ),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_requirements(archive: zipfile.ZipFile) -> None:
    requirements = "\n".join(
        [
            "catboost==1.2.7",
            "lightgbm==4.5.0",
            "scikit-learn==1.9.0",
            "xgboost==2.1.1",
            "./model/wheels/tabm-0.0.3-py3-none-any.whl",
            "./model/wheels/rtdl_num_embeddings-0.0.12-py3-none-any.whl",
            "",
        ]
    )
    archive.writestr("requirements.txt", requirements)


def assert_required_files_exist() -> None:
    required = [SCRIPT_PATH, S37_INFERENCE_PATH, *HELPER_SCRIPTS, *[path for path, _ in ARTIFACTS]]
    missing = [str(path.relative_to(REPO)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files for S67 packaging: {missing}")


def package_zip() -> list[dict]:
    assert_required_files_exist()
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(SCRIPT_PATH, "script.py")
        archive.write(S37_INFERENCE_PATH, "model/s37_inference.py")
        archive.writestr("model/__init__.py", "")
        archive.writestr("model/scripts/__init__.py", "")
        for helper in HELPER_SCRIPTS:
            archive.write(helper, f"model/scripts/{helper.name}")
        for source, arcname in ARTIFACTS:
            archive.write(source, arcname)
        write_requirements(archive)

    with zipfile.ZipFile(ZIP_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP entry: {bad}")
        names = archive.namelist()
        if any("\\" in name for name in names):
            raise RuntimeError("ZIP contains Windows backslash paths")
        required_names = {
            "script.py",
            "requirements.txt",
            "model/__init__.py",
            "model/s37_inference.py",
            "model/scripts/__init__.py",
            "model/scripts/ab_feature.py",
            "model/scripts/e13_feature_groups.py",
            "model/s37_canonical.joblib",
            "model/s61_contextual_prototype_retrieval.joblib",
            "model/s63_low_rank_matchup_residual.joblib",
            "model/s67_final_official_tabm.pt",
            "model/s67_final_tabm_preprocessor.joblib",
        }
        missing = sorted(required_names.difference(names))
        if missing:
            raise RuntimeError(f"ZIP is missing required entries: {missing}")
        return [
            {"name": item.filename, "size": item.file_size, "crc": f"{item.CRC:08X}"}
            for item in archive.infolist()
        ]


def verify_packaged_script() -> dict:
    with tempfile.TemporaryDirectory(prefix="s67_submission_") as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(temp)
        data_dir = temp / "data"
        data_dir.mkdir()
        shutil.copy2(REPO / "data" / "test.csv", data_dir / "test.csv")
        shutil.copy2(REPO / "data" / "sample_submission.csv", data_dir / "sample_submission.csv")
        command = [sys.executable, "script.py"]
        completed = subprocess.run(
            command,
            cwd=temp,
            text=True,
            capture_output=True,
            check=True,
        )
        output = temp / "output" / "submission.csv"
        if not output.exists():
            raise RuntimeError("Packaged script did not create output/submission.csv")
        import pandas as pd

        submission = pd.read_csv(output)
        if list(submission.columns) != ["row_id", "control_success"]:
            raise RuntimeError("Submission columns are invalid")
        if len(submission) != len(pd.read_csv(data_dir / "sample_submission.csv")):
            raise RuntimeError("Submission row count does not match sample_submission")
        values = submission["control_success"].to_numpy(dtype="float64")
        if not ((values >= 0.0) & (values <= 1.0)).all():
            raise RuntimeError("Submission has probabilities outside [0, 1]")
        row_audit_code = (
            "import importlib.util, json; "
            "spec=importlib.util.spec_from_file_location('s67_submission','script.py'); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "import numpy as np, pandas as pd; "
            "test=pd.read_csv('data/test.csv', encoding='utf-8-sig'); "
            "batch=module.predict_control_success(test.copy()); "
            "singles=np.array([module.predict_control_success(test.iloc[[i]].copy().reset_index(drop=True))[0] for i in range(len(test))]); "
            "reverse=module.predict_control_success(test.iloc[::-1].copy().reset_index(drop=True))[::-1]; "
            "print(json.dumps({'batch_single_max_abs_diff': float(np.max(np.abs(batch-singles))), "
            "'reverse_order_max_abs_diff': float(np.max(np.abs(batch-reverse))), "
            "'finite': bool(np.isfinite(batch).all())}))"
        )
        row_audit = subprocess.run(
            [sys.executable, "-c", row_audit_code],
            cwd=temp,
            text=True,
            capture_output=True,
            check=True,
        )
        row_audit_payload = json.loads(row_audit.stdout.strip().splitlines()[-1])
        if row_audit_payload["batch_single_max_abs_diff"] > ROW_AUDIT_TOLERANCE:
            raise RuntimeError(
                "Batch and single-row predictions differ: "
                f"{row_audit_payload['batch_single_max_abs_diff']}"
            )
        if row_audit_payload["reverse_order_max_abs_diff"] > ROW_AUDIT_TOLERANCE:
            raise RuntimeError(
                "Predictions changed under row reordering: "
                f"{row_audit_payload['reverse_order_max_abs_diff']}"
            )
        return {
            "stdout_tail": completed.stdout.strip().splitlines()[-3:],
            "stderr_tail": completed.stderr.strip().splitlines()[-3:],
            "rows": int(len(submission)),
            "min_prediction": float(values.min()),
            "max_prediction": float(values.max()),
            "row_independence_audit": row_audit_payload,
        }


def build() -> dict:
    entries = package_zip()
    smoke = verify_packaged_script()
    result = {
        "experiment_id": "S67-final-submission",
        "source_candidate": "S66B gated official TabM correction over S61/S63 champion",
        "formula": "0.5*S61 + 0.5*S63 + 1.0*(0.05*(TabM-S37) when game_type == R else 0)",
        "selection_source": "2023 OOF only",
        "holdout_result_2024_delta_centered_score_vs_champion": 3.6813607252606744,
        "holdout_result_2024_delta_brier_vs_champion": -7.977175562001726e-06,
        "test_values_read": False,
        "leaderboard_used_for_weights": False,
        "zip_path": str(ZIP_PATH.relative_to(REPO)),
        "zip_sha256": sha256(ZIP_PATH),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_entries": entries,
        "packaged_smoke_test": smoke,
        "status": "built_not_submitted",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    build()
