"""Convert expert episodes recorded by openarm_bimanual_sim.py / yellowarm_sim.py --expert --record DIR (*.npz) into a LeRobot v3
dataset (videos + parquet) that lerobot-train can use directly.

Run with the SmolVLA env:
  env\\python.exe scripts\\npz_to_lerobot.py --raw datasets\\raw\\openarm_cube_in_box_isaac
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def state_names(n):
    if n == 9:  # OpenArm left arm: 7 joints + 2 finger joints
        return [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]
    if n == 16:  # humanoid: left YellowArm (7 joints + gripper opening), then the right one
        return [f"{s}_{j}" for s in ("left", "right") for j in [f"joint{i}" for i in range(1, 8)] + ["gripper"]]
    if n == 8:  # 7 joints + gripper opening (YellowArm)
        return [f"joint{i}" for i in range(1, 8)] + ["gripper"]
    return [f"s{i}" for i in range(n)]


def action_names(n):
    if n == 16:
        return [f"{s}_{j}_target" for s in ("left", "right") for j in [f"joint{i}" for i in range(1, 8)] + ["gripper"]]
    return [f"joint{i}_target" for i in range(1, 8)] + ["gripper_target"] if n == 8 else [f"a{i}" for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with episode_*.npz")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "datasets" / "lerobot" / "openarm_cube_in_box_isaac"))
    ap.add_argument("--repo-id", default="local/openarm_cube_in_box_isaac")
    ap.add_argument("--robot-type", default="openarm_bimanual_isaac")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=0, help="convert only the first N episodes (0 = all)")
    ap.add_argument("--vcodec", default="h264_nvenc",
                    help="video codec: h264_nvenc (NVIDIA GPU, fast), libsvtav1 (lerobot default, slow on CPU), auto")
    args = ap.parse_args()

    files = []
    for f in sorted(Path(args.raw).glob("episode_*.npz")):
        try:  # skip a file the generator is still writing
            with np.load(f) as z:
                z["action"]
            files.append(f)
        except Exception as e:
            print(f"[convert] skipping unreadable {f.name}: {e}")
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        sys.exit(f"no readable episode_*.npz in {args.raw}")
    with np.load(files[0]) as first:
        cams = [k for k in first.files if first[k].ndim == 4]  # (T, H, W, 3) image arrays
        h, w = first[cams[0]].shape[1:3]
        ns, na = first["state"].shape[1], first["action"].shape[1]
        fps = int(round(float(first["fps"])))
        # episodes that carry their own joint names (dual_orca_pick_sim.py: embodiment order) use them
        names = [str(n) for n in first["names"]] if "names" in first.files else None
    print(f"[convert] cameras {cams}, state {ns}, action {na}, {h}x{w} @ {fps} fps")
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)

    features = {
        "observation.state": {"dtype": "float32", "shape": (ns,),
                              "names": names if names and len(names) == ns else state_names(ns)},
        "action": {"dtype": "float32", "shape": (na,),
                   "names": [f"{n}_target" for n in names] if names and len(names) == na else action_names(na)},
        **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
           for c in cams},
    }
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=fps, features=features, root=out,
                               robot_type=args.robot_type, use_videos=True,
                               streaming_encoding=True,  # encode frames directly instead of writing PNGs first
                               # The streaming encoder DROPS a frame when its queue is full (default 30), which shifts
                               # the rest of that episode's video; buffer a whole episode so add_frame never drops.
                               encoder_queue_maxsize=4000,
                               rgb_encoder=RGBEncoderConfig(
                                   vcodec=args.vcodec,
                                   # NVENC rejects lerobot's GOP of 2 unless B-frames are off.
                                   extra_options={"bf": 0} if "nvenc" in args.vcodec else {}))
    total = 0
    for i, f in enumerate(files):
        with np.load(f) as z:  # decompress each array once (indexing an NpzFile re-reads the whole array)
            d = {k: z[k] for k in z.files}
        task = str(d["task"])
        T = len(d["action"])
        for t in range(T):
            frame = {"observation.state": d["state"][t], "action": d["action"][t], "task": task}
            for c in cams:
                frame[f"observation.images.{c}"] = d[c][t]
            ds.add_frame(frame)
        ds.save_episode()
        total += T
        print(f"[convert] {i + 1}/{len(files)} {f.name}: {T} frames", flush=True)
    ds.finalize()
    print(f"[convert] done: {len(files)} episodes, {total} frames, {fps} fps -> {out}")


if __name__ == "__main__":
    main()
