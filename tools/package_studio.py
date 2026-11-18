"""Build a portable zip of the Training Studio (stdlib only). Run with any Python 3.10+:

    python tools\\package_studio.py [--out DIR] [--with-db] [--full-dataset NAME ...] [--no-design]

Robot files go to training-management-server/design/ (the default STUDIO_DESIGN_DIR) and .env.example is included, so the app
needs no path changes on the other machine; any other location is set in training-management-server/.env. By default it contains the app, the pre-built frontend and a LIGHT copy of the data: results/*.json,
dataset metadata (no videos, no data parquet), model config.json files (no weights), train logs (for loss curves) and the small
robot description files. Training jobs need the training environment (env\\python.exe + GPU) which is NOT packaged; without it the
app is a viewer. Use --full-dataset NAME to also ship one dataset completely (videos + parquet) so episodes can be played.
"""
import argparse
import fnmatch
import json
import os
import shutil
import sqlite3
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # training-management-server/
sys.path.insert(0, str(ROOT / "backend"))
from app import config as SC                        # noqa: E402  (studio settings: .env / STUDIO_DESIGN_DIR)
DESIGN = SC.DESIGN
VERSION = "0.2.0"
MAX_ASSET_BYTES = 8 * 1024 * 1024                   # robot URDF/USD files larger than this are skipped
MAX_LOG_BYTES = 30 * 1024 * 1024


def copy(src: Path, dst: Path, stats):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    stats["files"] += 1
    stats["bytes"] += src.stat().st_size


def copy_tree(src: Path, dst: Path, stats, skip=()):
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", "node_modules") and d not in skip]
        for f in files:
            if f.endswith((".pyc",)):
                continue
            p = Path(root) / f
            copy(p, dst / p.relative_to(src), stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "packages"), help="output folder (default training-management-server/packages)")
    ap.add_argument("--with-db", action="store_true", help="include backend/data/studio.db (robot/model edits, assignments, job history)")
    ap.add_argument("--full-dataset", action="append", default=[], metavar="NAME", help="ship this dataset completely (repeatable)")
    ap.add_argument("--no-design", action="store_true", help="do not ship URDF/USD files (robots then show as missing assets)")
    a = ap.parse_args()

    dist = ROOT / "frontend" / "dist"
    if not (dist / "index.html").exists():
        sys.exit("frontend/dist is missing: run `npm run build` in training-management-server/frontend first")

    missing = [n for n in a.full_dataset if not (ROOT / "datasets" / "lerobot" / n).is_dir()]
    if missing:
        sys.exit(f"--full-dataset: not found under datasets/lerobot: {missing}")

    name = f"training-studio-{VERSION}"
    stage = Path(a.out) / name
    if stage.exists():
        shutil.rmtree(stage)
    pkg_root = stage / "training-management-server"
    stats = {"files": 0, "bytes": 0}

    # 1) the app
    copy_tree(ROOT / "backend" / "app", pkg_root / "backend" / "app", stats)
    for f in ("requirements.txt",):
        copy(ROOT / "backend" / f, pkg_root / "backend" / f, stats)
    copy_tree(dist, pkg_root / "frontend" / "dist", stats)
    copy(ROOT / "studio.bat", pkg_root / "studio.bat", stats)
    copy(ROOT / "STUDIO.md", pkg_root / "STUDIO.md", stats)
    copy(ROOT / ".env.example", pkg_root / ".env.example", stats)
    copy(ROOT / "README_STUDIO.md", pkg_root / "README_STUDIO.md", stats)
    for f in ("setup_train_env.bat", "setup_train_env.sh"):
        copy(ROOT / f, pkg_root / f, stats)   # never the machine's own .env
    (pkg_root / "backend" / "data").mkdir(parents=True, exist_ok=True)
    if a.with_db and (ROOT / "backend" / "data" / "studio.db").exists():
        src = sqlite3.connect(ROOT / "backend" / "data" / "studio.db")           # consistent copy even while the app is running
        dst = sqlite3.connect(pkg_root / "backend" / "data" / "studio.db")
        src.backup(dst)
        src.close(); dst.close()
        stats["files"] += 1

    # 2) results (eval + train summaries, tables)
    if (ROOT / "results").exists():
        copy_tree(ROOT / "results", pkg_root / "results", stats)

    # 3) datasets: metadata only, plus full copies on request
    ds_root = ROOT / "datasets" / "lerobot"
    shipped_full = []
    if ds_root.exists():
        for d in sorted(p for p in ds_root.iterdir() if p.is_dir()):
            if d.name in a.full_dataset:
                copy_tree(d, pkg_root / "datasets" / "lerobot" / d.name, stats)
                shipped_full.append(d.name)
            elif (d / "meta").exists():
                copy_tree(d / "meta", pkg_root / "datasets" / "lerobot" / d.name / "meta", stats)

    # 4) models: only the small json files of every numeric checkpoint (weights stay behind)
    out = ROOT / "outputs"
    if out.exists():
        for run in sorted(p for p in out.iterdir() if p.is_dir()):
            ck = run / "checkpoints"
            if not ck.exists():
                continue
            for step in sorted(c for c in ck.iterdir() if c.is_dir() and c.name.isdigit()):
                pm = step / "pretrained_model"
                for fn in ("config.json", "train_config.json"):
                    if (pm / fn).exists():
                        copy(pm / fn, pkg_root / "outputs" / run.name / "checkpoints" / step.name / "pretrained_model" / fn, stats)

    # 5) train logs (loss curves)
    logs = ROOT / "logs"
    if logs.exists():
        for p in sorted(logs.glob("train_*.log")):
            if p.stat().st_size <= MAX_LOG_BYTES:
                copy(p, pkg_root / "logs" / p.name, stats)
        studio_logs = logs / "studio"
        if studio_logs.exists():
            copy_tree(studio_logs, pkg_root / "logs" / "studio", stats)

    # 6) design: only files the studio's robots point at (+ embodiment descriptors) and the humanoid body config
    skipped_assets = []
    hj = ROOT / "scripts" / "data" / "humanoid.json"
    if hj.exists():
        copy(hj, pkg_root / "scripts" / "data" / "humanoid.json", stats)
    db_path = ROOT / "backend" / "data" / "studio.db"
    if not a.no_design and db_path.exists():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        paths = set()
        for r in con.execute("SELECT design_dir, urdf, usd, usd_ros FROM robot WHERE hidden=0"):
            if not r["design_dir"]:
                continue
            for k in ("urdf", "usd", "usd_ros"):
                if r[k]:
                    paths.add(f"{r['design_dir']}/{r[k]}")       # asset paths are relative to <design>/<design_dir>/
            for emb in ("embodiment.yaml", "robot.yaml"):
                if (DESIGN / r["design_dir"] / emb).exists():
                    paths.add(f"{r['design_dir']}/{emb}")
        con.close()
        for rel in sorted(paths):
            p = DESIGN / rel
            if not p.is_file():
                continue
            if p.stat().st_size > MAX_ASSET_BYTES:
                skipped_assets.append(f"design/{rel}")
                continue
            copy(p, pkg_root / "design" / rel, stats)   # the packaged app's default STUDIO_DESIGN_DIR

    # 7) README + manifest
    (stage / "README.txt").write_text(f"""Training Studio {VERSION}

Run (Windows, Python 3.10+ and internet needed on the first run):
    training-management-server\\studio.bat
then open http://127.0.0.1:8700 .  The first run creates backend\\.venv and installs backend\\requirements.txt.
Unzip to a SHORT folder (e.g. C:\\studio): Windows limits paths to 260 characters and the install fails in deep folders.
(Alternative: set STUDIO_VENV=C:\\studio-venv before running studio.bat.)

What is inside: the web app, the built frontend, and a light copy of the project data (results, dataset metadata, model configs,
train logs, small robot files).{' Full datasets: ' + ', '.join(shipped_full) + '.' if shipped_full else ''}
Not inside: dataset videos/parquet, model weights, the training environment (env\\python.exe), Isaac Sim. So on this machine the
app browses data and manages robots/models; it cannot train or play episodes unless you add them.

To use your own data: copy datasets\\lerobot\\<name>, outputs\\<run> and results\\ into training-management-server\\ (same layout as the
project), then press "Quét lại dữ liệu" on the Overview page. To train, also install the training environment so that
training-management-server\\env\\python.exe exists, or set STUDIO_TRAIN_PYTHON in training-management-server\\.env.

The app has no login: keep it on 127.0.0.1 (default). Settings: copy training-management-server\\.env.example to .env
(host/port, data folders, design folder, Isaac Sim, train python).
See training-management-server\\STUDIO.md for the full documentation.
""", encoding="utf-8")
    manifest = {"name": name, "version": VERSION, "files": stats["files"], "bytes": stats["bytes"], "full_datasets": shipped_full,
                "with_db": a.with_db, "skipped_large_assets": skipped_assets}
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # 8) zip
    zpath = Path(a.out) / f"{name}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, Path(name) / p.relative_to(stage))
    shutil.rmtree(stage)
    print(f"wrote {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB zipped, {stats['files']} files, {stats['bytes'] / 1e6:.1f} MB raw)")
    if skipped_assets:
        print("skipped (larger than 8 MB):", *skipped_assets, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
