"""Snapshot the legacy Spider/Agent Lightning course code without copying Spider data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"D:\01_project\找工作的项目\agentic RL")
SPIDER = SOURCE_ROOT / "项目源码" / "spider"
GUIDE = SOURCE_ROOT / "Agent Lightning快速入门.md"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    files = [path for path in SPIDER.rglob("*") if path.is_file() and path.suffix != ".parquet"]
    manifest = {
        "source_root": str(SOURCE_ROOT),
        "copied": [{"source": str(path), "target": str(ROOT / "legacy_reproduction" / "spider" / path.relative_to(SPIDER)), "sha256": digest(path), "bytes": path.stat().st_size} for path in files] + [{"source": str(GUIDE), "target": str(ROOT / "course_materials" / GUIDE.name), "sha256": digest(GUIDE), "bytes": GUIDE.stat().st_size}],
        "external_only": [{"source": str(SOURCE_ROOT / "微调SQL数据集"), "reason": "canonical Spider data remains in its original location"}, {"source": str(SPIDER / "train_spider.parquet"), "reason": "course dataset is referenced rather than duplicated"}],
    }
    if not args.dry_run:
        for path in files:
            target = ROOT / "legacy_reproduction" / "spider" / path.relative_to(SPIDER)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        target = ROOT / "course_materials" / GUIDE.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(GUIDE, target)
        (ROOT / "course_materials" / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
