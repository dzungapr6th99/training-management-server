"""Per-episode view of LeRobot v3 datasets: episode list (meta/episodes), the video segment of each camera
(videos/<key>/chunk-X/file-Y.mp4 + from/to timestamp), and the state/action series (data/chunk-X/file-Y.parquet).
Read-only."""
import re
import threading
from functools import lru_cache
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config as C
from . import db

CAM_RE = re.compile(r"^[A-Za-z0-9_]+$")
_lock = threading.Lock()
_cache: dict = {}


def _dataset_row(ds_id):
    with db.session() as con:
        return db.one(con, "SELECT * FROM dataset WHERE id=?", (ds_id,))


def _root(ds):
    return C.ROOT / ds["path"]


def load_meta(ds_id):
    """Episode table of a dataset as a list of dicts (cached until meta/episodes changes)."""
    ds = _dataset_row(ds_id)
    if not ds:
        return None
    root = _root(ds)
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    sig = tuple((str(f), f.stat().st_mtime_ns) for f in files)
    with _lock:
        hit = _cache.get(ds_id)
        if hit and hit[0] == sig:
            return hit[1]
    cams = db.jload(ds["cameras"], [])
    cols = ["episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"]
    for c in cams:
        k = f"videos/observation.images.{c}/"
        cols += [k + "chunk_index", k + "file_index", k + "from_timestamp", k + "to_timestamp"]
    rows = []
    for f in files:
        have = set(pq.ParquetFile(f).schema_arrow.names)
        rows += pq.read_table(f, columns=[c for c in cols if c in have]).to_pylist()
    rows.sort(key=lambda r: r["episode_index"])
    out = []
    for r in rows:
        t = r.get("tasks")
        out.append({
            "episode": r["episode_index"], "task": (t[0] if isinstance(t, list) and t else t), "length": r["length"],
            "from_index": r["dataset_from_index"], "to_index": r["dataset_to_index"],
            "data_chunk": r["data/chunk_index"], "data_file": r["data/file_index"],
            "videos": {c: {"chunk": r[f"videos/observation.images.{c}/chunk_index"],
                           "file": r[f"videos/observation.images.{c}/file_index"],
                           "from": r[f"videos/observation.images.{c}/from_timestamp"],
                           "to": r[f"videos/observation.images.{c}/to_timestamp"]}
                       for c in cams if f"videos/observation.images.{c}/chunk_index" in r},
        })
    with _lock:
        _cache[ds_id] = (sig, out)
    return out


def list_episodes(ds_id, offset=0, limit=50, q=None):
    meta = load_meta(ds_id)
    if meta is None:
        return None
    rows = meta
    if q:
        ql = q.lower()
        rows = [r for r in meta if ql in (r["task"] or "").lower() or ql == str(r["episode"])]
    fps = (_dataset_row(ds_id) or {}).get("fps") or 25
    page = [{"episode": r["episode"], "task": r["task"], "length": r["length"], "seconds": round(r["length"] / fps, 1)}
            for r in rows[offset: offset + limit]]
    return {"total": len(rows), "offset": offset, "episodes": page}


def episode_detail(ds_id, ep):
    meta = load_meta(ds_id)
    if meta is None:
        return None
    ds = _dataset_row(ds_id)
    if not 0 <= ep < len(meta) or meta[ep]["episode"] != ep:
        return {"error": "episode not found"}
    r = meta[ep]
    return {
        "dataset": ds_id, "episode": ep, "task": r["task"], "length": r["length"], "fps": ds["fps"],
        "seconds": round(r["length"] / ds["fps"], 2),
        "videos": {c: {**v, "url": f"/api/media/{ds_id}/{c}/{v['chunk']}/{v['file']}",
                       "available": video_path(ds_id, c, v["chunk"], v["file"]) is not None} for c, v in r["videos"].items()},
        "has_data": (_root(ds) / "data" / f"chunk-{r['data_chunk']:03d}" / f"file-{r['data_file']:03d}.parquet").is_file(),
        "feature_names": db.jload(ds["feature_names"], {}),
    }


def series(ds_id, ep):
    """state/action of one episode. LeRobot writes one parquet row group per episode, so read that group directly and
    fall back to a filtered read when the layout differs."""
    meta = load_meta(ds_id)
    ds = _dataset_row(ds_id)
    if meta is None or not 0 <= ep < len(meta):
        return None
    r = meta[ep]
    f = _root(ds) / "data" / f"chunk-{r['data_chunk']:03d}" / f"file-{r['data_file']:03d}.parquet"
    if not f.is_file():
        return None  # metadata-only copy of the dataset (no data/ folder on this machine)
    pf = pq.ParquetFile(f)
    cols = ["observation.state", "action", "frame_index", "episode_index"]
    t = None
    first_ep = min(m["episode"] for m in meta if (m["data_chunk"], m["data_file"]) == (r["data_chunk"], r["data_file"]))
    g = ep - first_ep
    if 0 <= g < pf.num_row_groups:
        cand = pf.read_row_group(g, columns=cols)
        if len(cand) and set(cand["episode_index"].to_pylist()) == {ep}:
            t = cand
    if t is None:
        full = pq.read_table(f, columns=cols)
        t = full.filter(pc.equal(full["episode_index"], ep))
    return {
        "frames": t["frame_index"].to_pylist(),
        "state": t["observation.state"].to_pylist(),
        "action": t["action"].to_pylist(),
    }


@lru_cache(maxsize=256)
def video_path(ds_id, camera, chunk, file):
    ds = _dataset_row(ds_id)
    if not ds or camera not in db.jload(ds["cameras"], []) or not CAM_RE.match(camera):
        return None
    p = _root(ds) / "videos" / f"observation.images.{camera}" / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
    return p if p.is_file() else None
