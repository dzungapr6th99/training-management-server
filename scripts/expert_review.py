"""Expert-data quality review for the humanoid pallet set -> logs/expert_review/.

  env\\python.exe scripts\\expert_review.py metrics          # per-episode metrics.json (all episodes, 4 processes)
  env\\python.exe scripts\\expert_review.py videos           # H.264 mp4 of every episode (skips existing)
  env\\python.exe scripts\\expert_review.py videos --ep 12 40
  env\\python.exe scripts\\expert_review.py page            # index.html: pick an episode, watch its video

Per episode: length, arm, level, DART noise, active-arm commanded speed vs hardware velocity limit, action jerk,
state-vs-previous-action tracking error, idle-arm drift, gripper close/open count, joint-limit margin, hesitation
frames (active arm still, gripper unchanged), end-of-episode still frames, camera brightness and frozen frames, plus
the expert's own meta (grasp offset, min grip opening, replanned place, end pose checks).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw" / "humanoid_pallet_all"
OUT = ROOT / "logs" / "expert_review"
LIM = json.loads((ROOT / "scripts" / "data" / "humanoid.json").read_text())["arm_limits"]
LO, HI = np.array(LIM["lower_rad"]), np.array(LIM["upper_rad"])
VMAX = np.array(LIM["velocity_used_rad_s"])
CAMS = ("camera1", "camera2", "camera3")


def arm_cols(arm):
    o = 0 if arm == "L" else 8
    return slice(o, o + 7), o + 7


def episode_metrics(path):
    d = np.load(path, allow_pickle=True)
    s, a = d["state"].astype(np.float64), d["action"].astype(np.float64)
    meta = json.loads(str(d["meta"]))
    fps = float(d["fps"])
    T = len(s)
    arm = meta["arm"]
    j, g = arm_cols(arm)
    ij, ig = arm_cols("R" if arm == "L" else "L")

    va = np.diff(a[:, j], axis=0) * fps                      # commanded joint speed, rad/s
    jerk = np.diff(a[:, j], n=3, axis=0) * 1000              # mrad/step^3
    track = np.abs(s[1:, j] - a[:-1, j]) * 1000              # mrad
    margin = np.minimum(s[:, j] - LO, HI - s[:, j]).min(0)   # rad, per joint
    gc = a[:, g]
    closes = int(np.sum((gc[:-1] >= 0.5) & (gc[1:] < 0.5)))
    opens = int(np.sum((gc[:-1] < 0.5) & (gc[1:] >= 0.5)))
    first_close = int(np.argmax(gc < 0.5)) if closes else -1
    still = (np.abs(va).max(1) < 0.01) & (np.diff(gc) == 0)  # active arm not moving, gripper unchanged
    tail = 0
    for x in still[::-1]:
        if not x:
            break
        tail += 1
    # hesitation = still frames not at the very end, excluding the gripper's own settle windows
    hes = int(still.sum() - tail)
    longest = 0
    run = 0
    for x in still[: len(still) - tail]:
        run = run + 1 if x else 0
        longest = max(longest, run)

    cam = {}
    for c in CAMS:
        v = d[c]
        m = v.reshape(T, -1).astype(np.float32)
        bright = m.mean(1)
        frozen = int(np.sum(np.abs(np.diff(m[:, ::97], axis=0)).max(1) == 0))
        cam[c] = {"bright_mean": round(float(bright.mean()), 1), "bright_min": round(float(bright.min()), 1),
                  "frozen_frames": frozen}

    return {
        "ep": int(Path(path).stem.split("_")[1]),
        "frames": T,
        "seconds": round(T / fps, 2),
        "level": int(d["level"]),
        "task": str(d["task"]),
        "arm": arm,
        "dart": meta.get("dart_noise", 0.0),
        "target_shape": meta["parts"][meta["target"]]["shape"],
        "target_color": meta["parts"][meta["target"]]["color"],
        "target_scale": meta["parts"][meta["target"]]["scale"],
        "n_parts": len(meta["parts"]),
        "n_prefill": len(meta.get("prefill", [])),
        "target_cell": meta.get("target_cell"),
        "speed_ratio_max": round(float((np.abs(va) / VMAX).max()), 3),
        "jerk_p99_mrad": round(float(np.percentile(np.abs(jerk), 99)), 2),
        "jerk_max_mrad": round(float(np.abs(jerk).max()), 2),
        "track_p95_mrad": round(float(np.percentile(track, 95)), 2),
        "track_max_mrad": round(float(track.max()), 2),
        "idle_arm_drift_mrad": round(float(np.abs(s[:, ij] - s[0, ij]).max() * 1000), 2),
        "idle_grip_min": round(float(s[:, ig].min()), 3),
        "grip_closes": closes,
        "grip_opens": opens,
        "first_close_s": round(first_close / fps, 2) if first_close >= 0 else None,
        "joint_margin_min_rad": round(float(margin.min()), 3),
        "joint_margin_argmin": int(margin.argmin()) + 1,
        "hesitation_frames": hes,
        "longest_pause_frames": longest,
        "end_still_frames": tail,
        "cams": cam,
        "grip_min_open": meta.get("grip_min_open"),
        "grasp_offset_mm": meta.get("grasp_offset_mm"),
        "grasp_offset_norm_mm": round(float(np.linalg.norm(meta.get("grasp_offset_mm", [0, 0, 0]))), 1),
        "replanned": meta.get("replanned"),
        "rejected_layouts": meta.get("rejected_layouts"),
        "end_checks": {k: meta.get(k) for k in ("held_after_lift", "cell_was_free", "low", "still", "upright")},
        "others_on_pallet": len(meta.get("others_on_pallet", [])),
        "seed": meta.get("seed"),
    }


def cmd_metrics(args):
    files = sorted(RAW.glob("episode_*.npz"))
    if args.limit:
        files = files[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, r in enumerate(ex.map(episode_metrics, files, chunksize=4)):
            rows.append(r)
            if i % 50 == 0:
                print(f"{i + 1}/{len(files)}", flush=True)
    (OUT / "metrics.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print("wrote", OUT / "metrics.json", len(rows))


def label(img, text, xy, scale=0.45, color=(255, 255, 255)):
    import cv2
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def write_video(ep, out):
    import av
    d = np.load(RAW / f"episode_{ep:04d}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    a = d["action"]
    T = len(a)
    fps = int(d["fps"])
    c = av.open(str(out), "w")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = 768, 256, "yuv420p"
    st.options = {"crf": "26", "preset": "veryfast", "movflags": "+faststart"}
    names = ("head", "wrist L", "wrist R")
    frames = np.concatenate([d[k] for k in CAMS], axis=2)    # load each camera once (npz re-reads on every d[k])
    for t in range(T):
        row = frames[t].copy()
        for i, n in enumerate(names):
            label(row, n, (6 + 256 * i, 16))
        label(row, f"{t / fps:5.2f}s", (6, 248))
        for side, o, x in (("L", 0, 262), ("R", 8, 518)):
            gcmd = a[t, o + 7]
            col = (120, 255, 120) if gcmd >= 0.5 else (255, 140, 60)
            label(row, f"grip {'open' if gcmd >= 0.5 else 'CLOSE'}", (x, 248), color=col)
        label(row, f"arm {meta['arm']}", (518 + 170, 16), color=(255, 230, 80))
        fr = av.VideoFrame.from_ndarray(row, format="rgb24")
        for p in st.encode(fr):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()


def cmd_videos(args):
    vd = OUT / "videos"
    vd.mkdir(parents=True, exist_ok=True)
    eps = args.ep or sorted(int(f.stem.split("_")[1]) for f in RAW.glob("episode_*.npz"))
    todo = [(e, vd / f"ep{e:04d}.mp4") for e in eps]
    todo = [(e, o) for e, o in todo if args.overwrite or not o.exists()]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(write_video, *zip(*todo)) if todo else []):
            if i % 25 == 0:
                print(f"{i + 1}/{len(todo)}", flush=True)
    print("done", len(todo), flush=True)


def cmd_page(args):
    rows = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    tpl = (ROOT / "scripts" / "expert_review_page.html").read_text(encoding="utf-8")
    out = OUT / "index.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(rows, ensure_ascii=False)), encoding="utf-8")
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("--workers", type=int, default=4)
    m.add_argument("--limit", type=int, default=0)
    v = sub.add_parser("videos")
    v.add_argument("--ep", type=int, nargs="*")
    v.add_argument("--overwrite", action="store_true")
    v.add_argument("--workers", type=int, default=6)
    sub.add_parser("page")
    args = p.parse_args()
    {"metrics": cmd_metrics, "videos": cmd_videos, "page": cmd_page}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
