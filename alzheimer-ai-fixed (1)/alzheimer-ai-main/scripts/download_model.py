"""
Local helper: fetch the real trained model into app/models/ so the app can run
without Git LFS. Set MODEL_URL (env var) to a direct-download link first.

    MODEL_URL="https://.../multimodal_final.h5" python scripts/download_model.py
"""
import os
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "app" / "models" / "multimodal_final.h5"


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    with open(path, "rb") as f:
        return f.read(64).startswith(b"version https://git-lfs.github.com")


def main() -> int:
    if DEST.exists() and DEST.stat().st_size > 1024 and not is_lfs_pointer(DEST):
        print(f"Model already present: {DEST} ({DEST.stat().st_size/1e6:.0f} MB)")
        return 0

    url = os.environ.get("MODEL_URL")
    if not url:
        print("ERROR: set MODEL_URL to a direct-download link to multimodal_final.h5")
        return 1

    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_name(DEST.name + ".part")
    print(f"Downloading model from {url} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(DEST)
    print(f"Saved {DEST} ({DEST.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
