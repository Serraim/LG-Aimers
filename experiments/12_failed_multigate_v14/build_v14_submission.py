"""V13 제출물 패키징: V11 시드배깅 zip의 script.py만 Z1 평탄게이트 판으로 교체.

베이스로 v11_distilled_cb18_seed_bagged_champion.zip 을 쓴다 -- 이 zip만
s61_seed{42,2024,3407}.joblib / s63_seed*.joblib 아티팩트를 담고 있다
(v8_physics_frozen_coef.zip 에는 단일시드 아티팩트만 있다).
"""
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE_ZIP = REPO / "submissions" / "v11_distilled_cb18_seed_bagged_champion.zip"
ZIP_PATH = REPO / "submissions" / "v14_two_tier_gate_r3_share.zip"
SCRIPT = REPO / "scripts" / "v14_two_tier_gate_script.py"


def package_zip() -> None:
    script_bytes = SCRIPT.read_bytes()
    seen_script = False
    with zipfile.ZipFile(BASE_ZIP) as src, \
            zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            if item.filename == "script.py":
                out.writestr("script.py", script_bytes)
                seen_script = True
            else:
                out.writestr(item, src.read(item.filename))
    if not seen_script:
        raise RuntimeError("base zip에 script.py가 없다")
    print(f"Packaged {ZIP_PATH.name}")


if __name__ == "__main__":
    package_zip()
