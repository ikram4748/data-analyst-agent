"""Download BIRD mini-dev and extract only what this project needs.

    python scripts/setup_data.py            # download (~800 MB), extract SQLite DBs (~1.5 GB), delete the zip
    python scripts/setup_data.py --keep-zip

Questions + gold SQL come from the canonical Hugging Face release (birdsql/bird_mini_dev, SQLite split);
the databases come from the official BIRD archive.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analyst_agent.config import get_settings  # noqa: E402

ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
MEMBER = re.compile(r"^minidev/MINIDEV/dev_databases/([A-Za-z0-9_]+)/(\1\.sqlite|database_description/[^/]+\.csv)$")


def download(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%}", end="", flush=True)
    print()


def extract(zip_path: Path, out: Path) -> int:
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            m = MEMBER.match(info.filename)
            if not m:                       # skip MySQL/Postgres dumps, __MACOSX, everything else
                continue
            # build the target path ourselves rather than trusting names inside the archive
            db, rel = m.group(1), m.group(2)
            target = out / "dev_databases" / db / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def questions(out: Path) -> None:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("birdsql/bird_mini_dev", "data/mini_dev_sqlite-00000-of-00001.json", repo_type="dataset")
    text = Path(p).read_text(encoding="utf-8")
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    (out / "mini_dev_sqlite.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"questions: {len(rows)} -> {out / 'mini_dev_sqlite.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    out = s.bird_dir
    raw = s.data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "minidev.zip"
    if not zip_path.exists():
        download(ZIP_URL, zip_path)
    print(f"extracted {extract(zip_path, out)} files -> {out / 'dev_databases'}")
    questions(out)
    if not args.keep_zip:
        zip_path.unlink()
        print("deleted the zip (use --keep-zip to keep it)")


if __name__ == "__main__":
    main()
