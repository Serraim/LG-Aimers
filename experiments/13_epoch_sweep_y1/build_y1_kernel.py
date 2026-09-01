"""Y1 캐글 커널 빌더: 학습 스크립트를 커널 script.py 안에 통째로 넣는다.

이렇게 하면 데이터셋을 새로 올릴 필요 없이 커널 파일 하나만 푸시하면 된다
(기존 번들 데이터셋은 그대로 attach해서 train.csv와 scripts/를 쓴다).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "scripts" / "y1_tabm_epoch_sweep.py"
KERNEL_DIR = REPO / "kaggle_y1"
BUNDLE_SLUG = "serraim0109/lg-aimers-physics-distillation-bundle"
KERNEL_ID = "serraim0109/lg-aimers-y1-tabm-epoch-sweep"

# 캐글 데이터셋(lg-aimers-physics-distillation-bundle)에는 data/, models/,
# results/preds/만 있고 scripts/가 없다 (2026-08-31 Y1 ERROR로 확인됨:
# find_bundle()이 scripts/+data/ 둘 다 있는 디렉터리를 찾다가 실패).
# 350MB+ 데이터셋을 재업로드하는 대신, y1 스크립트가 import하는 헬퍼 모듈을
# 여기서 통째로 커널에 같이 임베드한다 (전부 삼중따옴표 없음, 서로 순환 의존 없음).
HELPER_MODULES = [
    "ab_feature.py",
    "e13_feature_groups.py",
    "e13_rolling_validation.py",
    "e61_joint_outcome.py",
    "s58_tabm_binary.py",
]

BOOTSTRAP = '''"""Y1 커널: TabM 에폭 스윕 진단 (자동 생성됨 -- scripts/build_y1_kernel.py)."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = "/kaggle/working/repo"


def find_bundle() -> str:
    """attach된 입력 중 data/를 가진 디렉터리를 찾는다 (scripts/는 요구하지 않음 --
    이 데이터셋에는 data/, models/, results/preds/만 있고 scripts/가 없다)."""
    roots = [Path("/kaggle/input")]
    for root in roots:
        if not root.exists():
            continue
        for candidate in [root, *root.rglob("*")]:
            if not candidate.is_dir():
                continue
            if (candidate / "data" / "train.csv").exists():
                return str(candidate)
    listing = []
    for root in roots:
        if root.exists():
            listing = [str(p) for p in root.iterdir()]
    raise FileNotFoundError(f"번들을 못 찾음. /kaggle/input 내용: {listing}")


def fix_gpu_torch_compat() -> None:
    """캐글이 Tesla P100(Pascal, sm_60)을 배정하면 기본 설치된 PyTorch(sm_70+ 전용)가
    'no kernel image is available for execution on the device'로 죽는다.
    이건 Kaggle/docker-python 이슈 #1546으로 보고된, 아직 안 고쳐진 캐글 플랫폼 버그다
    (https://github.com/Kaggle/docker-python/issues/1546). Pascal을 지원하는 마지막
    안정 버전은 PyTorch 2.7.x다 (2.8부터 sm_60 드롭). P100이 감지되면 그 버전으로
    바꿔 설치한다. bootstrap 프로세스 자체는 torch를 아직 import하지 않았으므로
    (y1 학습 스크립트는 이후 별도 subprocess로 fork된다) 여기서 재설치해도 안전하다.
    """
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        gpu_line = result.stdout
    except Exception as exc:
        print(f"nvidia-smi 확인 실패 (무시하고 계속): {exc}", flush=True)
        return
    print("GPU:", gpu_line.strip(), flush=True)
    if "P100" not in gpu_line:
        return
    print("P100 감지 -> Pascal 호환 PyTorch(2.7.1)로 교체 설치 중 (몇 분 소요)...", flush=True)
    # torchvision은 이 스크립트에서 쓰지 않으므로 재설치하지 않는다 (실패 표면 최소화).
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         "torch==2.7.1"],
        check=True,
    )
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch; print(torch.__version__, torch.cuda.is_available(), "
         "torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"],
        capture_output=True, text=True,
    )
    print("교체 후 torch 상태:", check.stdout.strip(), check.stderr.strip()[-500:], flush=True)


def main() -> None:
    bundle = find_bundle()
    print("번들:", bundle, flush=True)
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    shutil.copytree(bundle, REPO_DIR)
    os.chdir(REPO_DIR)
    os.makedirs("results", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)

    fix_gpu_torch_compat()

    for name, content in HELPER_SOURCES.items():
        (Path("scripts") / name).write_text(content, encoding="utf-8")
    print(f"헬퍼 모듈 {len(HELPER_SOURCES)}개 기록: {list(HELPER_SOURCES)}", flush=True)

    script_path = Path("scripts/y1_tabm_epoch_sweep.py")
    script_path.write_text(Y1_SOURCE, encoding="utf-8")
    print("학습 스크립트 기록:", script_path, flush=True)

    result = subprocess.run([sys.executable, str(script_path)])
    if result.returncode != 0:
        sys.exit(result.returncode)

    # 결과를 캐글 output으로 복사
    shutil.copytree("results", "/kaggle/working/results", dirs_exist_ok=True)
    print("완료", flush=True)


'''


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if "'''" in source:
        raise RuntimeError("학습 스크립트에 삼중 홑따옴표가 있어 그대로 임베드할 수 없다")

    helper_blocks = ["HELPER_SOURCES = {}\n"]
    for name in HELPER_MODULES:
        content = (REPO / "scripts" / name).read_text(encoding="utf-8")
        if "'''" in content:
            raise RuntimeError(f"{name}에 삼중 홑따옴표가 있어 그대로 임베드할 수 없다")
        helper_blocks.append(f"HELPER_SOURCES[{name!r}] = r'''{content}'''\n")
    helper_source = "\n".join(helper_blocks)

    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    kernel_script = KERNEL_DIR / "script.py"
    kernel_script.write_text(
        BOOTSTRAP + helper_source + f"\nY1_SOURCE = r'''{source}'''\n\n\n"
        "if __name__ == \"__main__\":\n    main()\n",
        encoding="utf-8",
    )

    metadata = {
        "id": KERNEL_ID,
        "title": "lg-aimers-y1-tabm-epoch-sweep",
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [BUNDLE_SLUG],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (KERNEL_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"생성됨: {kernel_script}  ({kernel_script.stat().st_size:,} bytes)")
    print(f"생성됨: {KERNEL_DIR / 'kernel-metadata.json'}")
    print()
    print("푸시 명령 (PowerShell):")
    print(f'  .\\.venv\\Scripts\\python.exe -m kaggle kernels push -p "{KERNEL_DIR}"')
    print()
    print("상태 확인:")
    print(f'  .\\.venv\\Scripts\\python.exe -m kaggle kernels status {KERNEL_ID}')
    print()
    print("결과 회수 (완료 후):")
    print(f'  .\\.venv\\Scripts\\python.exe -m kaggle kernels output {KERNEL_ID} -p kaggle_y1_output')


if __name__ == "__main__":
    main()
