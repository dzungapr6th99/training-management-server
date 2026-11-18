"""Checks for the humanoid (2 x YellowArm) episodes of the parts-to-pallet task, used by verify_npz.py and
verify_dataset.py when state/action are 16-D (left joint1..7 + gripper opening 0..1, then the right arm).

- audit_humanoid: joint values against the YellowArm URDF limits, action = absolute joint target (tracking error of
  the working arm), gripper events (exactly one arm closes per episode, the fingers stop on the part, the waiting arm's
  gripper stays open), start-pose diversity, duplicate trajectories.
- check_lerobot: the converted LeRobot dataset holds the same episodes / frames / tasks as the raw npz, with
  bit-identical state and action and video frames close to the raw images.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SIDES = ("left", "right")
NAMES = [f"{s[0].upper()}.{n}" for s in SIDES for n in [f"joint{i}" for i in range(1, 8)] + ["gripper"]]


def _hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def yellowarm_limits():
    import humanoid_kin as hk

    ch = hk.Chain()
    lo = np.concatenate([ch.lo, [0.0], ch.lo, [0.0]])
    hi = np.concatenate([ch.hi, [1.0], ch.hi, [1.0]])
    return lo, hi


def audit_humanoid(S, A, files, problems, warnings):
    allS, allA = np.concatenate(S), np.concatenate(A)
    lo, hi = yellowarm_limits()

    _hr("3. STATE vs YELLOWARM URDF LIMITS (both arms; gripper = opening 0..1)")
    print(f"{'dof':10s} {'min':>9s} {'max':>9s} {'limit lo':>9s} {'limit hi':>9s} {'used %':>7s} {'pinned %':>9s}  note")
    for j in range(16):
        col = allS[:, j]
        mn, mx = col.min(), col.max()
        used = (mx - mn) / (hi[j] - lo[j]) * 100
        tol = 1e-3 * max(1.0, hi[j] - lo[j])
        pin = float(((col <= lo[j] + tol) | (col >= hi[j] - tol)).mean() * 100)
        note = ""
        if mn < lo[j] - 1e-3 or mx > hi[j] + 1e-3:
            note = "<-- OUT OF LIMIT"
            problems.append(f"{NAMES[j]} outside its URDF limits ({mn:.3f}..{mx:.3f})")
        elif pin > 5 and j % 8 != 7:  # an open gripper sits at 1.0 by design
            note = "<-- RIDES ITS LIMIT"
            warnings.append(f"{NAMES[j]} at its joint limit in {pin:.0f}% of the frames")
        elif used < 1:
            note = "<-- barely moves"
        print(f"{NAMES[j]:10s} {mn:9.4f} {mx:9.4f} {lo[j]:9.4f} {hi[j]:9.4f} {used:7.1f} {pin:9.1f}  {note}")
    out_a = int(((allA < lo - 1e-3) | (allA > hi + 1e-3)).sum())
    print(f"\naction values outside the limits: {out_a}")
    if out_a:
        problems.append(f"{out_a} action values outside the joint / gripper limits")

    _hr("4. ACTION SEMANTICS (absolute joint targets expected)")
    working, err_w, err_d, idle_move = [], [], [], []
    for s, a in zip(S, A):
        closes = [k for k in range(2) if a[:, 8 * k + 7].min() < 0.5]
        w = closes[0] if len(closes) == 1 else int(np.argmax([np.ptp(a[:, 8 * k:8 * k + 7], 0).max() for k in range(2)]))
        working.append((w, closes))
        j = slice(8 * w, 8 * w + 7)
        err_w.append(np.abs(a[:-1, j] - s[1:, j]).mean())
        err_d.append(np.abs(a[:-1, j] - (s[1:, j] - s[:-1, j])).mean())
        o = 1 - w
        idle_move.append(np.ptp(s[:, 8 * o:8 * o + 7], 0).max())
    ea, ed = float(np.mean(err_w)), float(np.mean(err_d))
    print(f"working arm: mean|action - next_state| = {ea:.4f} rad, mean|action - delta| = {ed:.4f} rad "
          f"-> {'ABSOLUTE targets' if ea < ed else 'DELTA commands'}")
    if ea >= ed:
        problems.append("actions look like deltas, expected absolute joint targets")
    elif ea > 0.05:
        warnings.append(f"working-arm tracking error {ea:.3f} rad (controller lags its targets)")
    im = np.array(idle_move)
    print(f"waiting arm: largest joint travel per episode median {np.median(im):.4f} rad, max {im.max():.4f} rad")
    if im.max() > 0.1:
        warnings.append(f"the waiting arm moves up to {im.max():.2f} rad in some episode")

    _hr("5. GRIPPER")
    n_bad, held, sides = 0, [], {0: 0, 1: 0}
    for f, s, a, (w, closes) in zip(files, S, A, working):
        if len(closes) != 1:
            n_bad += 1
            if n_bad <= 5:
                print(f"  {os.path.basename(f)}: {len(closes)} arms close their gripper")
            continue
        sides[w] += 1
        g_cmd, g_state = a[:, 8 * w + 7], s[:, 8 * w + 7]
        closed = np.where(g_cmd < 0.5)[0]
        held.append(float(np.median(g_state[closed[len(closed) // 2:]])))  # fingers settled on the part
        if a[:, 8 * (1 - w) + 7].min() < 0.99:
            n_bad += 1
            print(f"  {os.path.basename(f)}: the waiting arm's gripper does not stay open")
    print(f"working arm: left {sides[0]}, right {sides[1]}; episodes with a wrong gripper pattern: {n_bad}")
    if n_bad:
        problems.append(f"{n_bad} episodes without exactly one closing gripper / with a closing waiting arm")
    if held:
        h = np.array(held)
        empty = int((h < 0.05).sum())
        print(f"gripper opening while closed on the part (75 mm = 1.0): median {np.median(h):.3f}, "
              f"min {h.min():.3f}, max {h.max():.3f}; closed on nothing (< 0.05): {empty}")
        if empty:
            problems.append(f"{empty} episodes where the gripper closes on nothing")

    _hr("6. DIVERSITY")
    init = np.array([s[0] for s in S])
    print("start pose std per dof (rad): " + " ".join(f"{v:.3f}" for v in init.std(0)))
    if init.std(0).max() < 1e-4:
        warnings.append("all episodes start from an identical pose")
    sig = [f"{s.shape[0]}_{s.sum():.6f}" for s in S]
    dup = len(sig) - len(set(sig))
    seeds = []
    for f in files:
        with np.load(f) as z:
            if "meta" in z.files:
                seeds.append(json.loads(str(z["meta"])).get("seed"))
    dup_seed = len(seeds) - len(set(seeds))
    print(f"identical-trajectory duplicates: {dup}; duplicate seeds: {dup_seed} ({len(seeds)} episodes with meta)")
    if dup or dup_seed:
        problems.append(f"{dup} duplicate trajectories, {dup_seed} duplicate seeds")


def check_lerobot(root, files, problems, warnings, n_check=6):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _hr(f"8. LEROBOT DATASET {root}")
    info = json.load(open(os.path.join(root, "meta", "info.json"), encoding="utf-8"))
    ds = LeRobotDataset(repo_id="local/" + os.path.basename(str(root)), root=root, video_backend="pyav")
    lens = []
    for f in files:
        with np.load(f) as z:
            lens.append(len(z["action"]))
    print(f"episodes {ds.num_episodes} (npz {len(files)}), frames {ds.num_frames} (npz {sum(lens)}), fps {ds.fps}, "
          f"robot_type {info.get('robot_type')}")
    if ds.num_episodes != len(files) or ds.num_frames != sum(lens):
        problems.append("LeRobot episode / frame count differs from the npz")
    for k in ("observation.state", "action"):
        sh = tuple(ds.features[k]["shape"])
        print(f"{k}: {sh}")
        if sh != (16,):
            problems.append(f"{k} shape {sh} != (16,)")
    cams = [k for k in ds.features if k.startswith("observation.images.")]
    print(f"cameras: {cams}")
    short = []  # a video segment shorter than its episode = frames dropped by the encoder
    for e in range(ds.meta.total_episodes):
        m = ds.meta.episodes[e]
        for c in cams:
            n = round((m[f"videos/{c}/to_timestamp"] - m[f"videos/{c}/from_timestamp"]) * ds.fps)
            if n != m["length"]:
                short.append((e, c.split(".")[-1], m["length"], n))
    print(f"video segments whose frame count differs from the episode length: {len(short)} {short[:10]}")
    if short:
        problems.append(f"{len(short)} video segments with a wrong frame count (encoder dropped frames)")
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    worst_img, worst_sa, bad_task = 0.0, 0.0, 0
    for e in np.linspace(0, len(files) - 1, min(n_check, len(files))).astype(int):
        with np.load(files[e]) as z:
            d = {k: z[k] for k in z.files}
        for t in (0, lens[e] // 2, lens[e] - 1):
            item = ds[int(starts[e] + t)]
            worst_sa = max(worst_sa, float(np.abs(item["observation.state"].numpy() - d["state"][t]).max()),
                           float(np.abs(item["action"].numpy() - d["action"][t]).max()))
            bad_task += item["task"] != str(d["task"])
            for c in cams:
                img = item[c].numpy().transpose(1, 2, 0) * 255.0  # (C, H, W) float 0..1 -> (H, W, C)
                worst_img = max(worst_img, float(np.abs(img - d[c.split(".")[-1]][t]).mean()))
    print(f"sampled frames: max |state/action - npz| {worst_sa:.2e}, worst mean |image - npz| {worst_img:.2f}/255 "
          f"(video compression), task mismatches {bad_task}")
    if worst_sa > 1e-5 or bad_task:
        problems.append("LeRobot frames do not match the npz (state/action/task)")
    if worst_img > 8:
        warnings.append(f"video frames differ from the raw images by {worst_img:.1f}/255 on average")
