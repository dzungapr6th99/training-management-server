#!/usr/bin/env python3
"""Audit raw .npz demonstration episodes before converting to LeRobotDataset.

Checks structure, value sanity, action/state relationship, gripper behaviour,
per-episode diversity, and image health. Prints a report; exits non-zero if a
blocking problem is found.

Usage:  python verify_npz.py <dir_with_npz> [--images N]
"""
import sys, os, glob, zipfile, argparse
import numpy as np
import numpy.lib.format as fmt

# OpenArm v2.0 revolute limits (from openarm_description v2.urdf)
LIMITS_LEFT = [(-3.4907, 1.3963), (-3.3161, 0.17453), (-1.5708, 1.5708),
               (0.0, 2.4435), (-1.5708, 1.5708), (-0.7854, 0.7854),
               (-1.5708, 1.5708), (0.0, 0.7854), (0.0, 0.7854)]
LIMITS_RIGHT = [(-1.3963, 3.4907), (-0.17453, 3.3161), (-1.5708, 1.5708),
                (0.0, 2.4435), (-1.5708, 1.5708), (-0.7854, 0.7854),
                (-1.5708, 1.5708), (-0.7854, 0.0), (-0.7854, 0.0)]

def headers(path):
    """Read shapes/dtypes without decompressing arrays."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            with z.open(n) as f:
                ver = fmt.read_magic(f)
                rd = fmt.read_array_header_1_0 if ver == (1, 0) else fmt.read_array_header_2_0
                shape, _, dt = rd(f)
            out[n[:-4]] = (shape, dt)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--images", type=int, default=3, help="how many episodes to image-check")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, "*.npz")))
    if not files:
        print("No .npz found in", a.dir); return 1
    print(f"Found {len(files)} episodes in {a.dir}\n")

    problems, warnings = [], []

    # ---------- 1. structure consistency ----------
    print("=" * 78); print("1. STRUCTURE"); print("=" * 78)
    ref = headers(files[0])
    ref_keys = set(ref)
    print(f"{'key':14s} {'shape (ep0)':24s} {'dtype'}")
    for k, (s, d) in ref.items():
        print(f"{k:14s} {str(s):24s} {d}")
    bad_struct = []
    for f in files:
        h = headers(f)
        if set(h) != ref_keys:
            bad_struct.append((os.path.basename(f), "key mismatch: " + str(set(h) ^ ref_keys)))
            continue
        T = h["state"][0][0]
        for k, (s, d) in h.items():
            if s and s[0] != T:
                bad_struct.append((os.path.basename(f), f"{k} length {s[0]} != state length {T}"))
            if d != ref[k][1] and not (d.kind == "U" and ref[k][1].kind == "U"):  # string lengths differ
                bad_struct.append((os.path.basename(f), f"{k} dtype {d} != {ref[k][1]}"))
            if len(s) > 1 and s[1:] != ref[k][0][1:]:
                bad_struct.append((os.path.basename(f), f"{k} inner shape {s[1:]} != {ref[k][0][1:]}"))
    if bad_struct:
        problems.append(f"{len(bad_struct)} structure mismatches")
        print("\nMISMATCHES:")
        for n, m in bad_struct[:15]:
            print("  ", n, "-", m)
    else:
        print("\nOK: all episodes share keys, dtypes and inner shapes; lengths align across arrays.")

    # ---------- 2. load light arrays for all episodes ----------
    print("\n" + "=" * 78); print("2. VALUES"); print("=" * 78)
    S, A, lens, tasks, fpss = [], [], [], [], []
    for f in files:
        with np.load(f) as z:
            S.append(z["state"].astype(np.float64))
            A.append(z["action"].astype(np.float64))
            lens.append(len(S[-1]))
            tasks.append(str(z["task"])) if "task" in z else None
            fpss.append(float(z["fps"])) if "fps" in z else None
    lens = np.array(lens)
    allS, allA = np.concatenate(S), np.concatenate(A)

    nan_s = int(np.isnan(allS).sum() + np.isinf(allS).sum())
    nan_a = int(np.isnan(allA).sum() + np.isinf(allA).sum())
    print(f"frames total        : {len(allS)}")
    print(f"episode length      : min {lens.min()}  max {lens.max()}  mean {lens.mean():.1f}  std {lens.std():.1f}")
    print(f"NaN/Inf in state    : {nan_s}")
    print(f"NaN/Inf in action   : {nan_a}")
    if nan_s or nan_a:
        problems.append("NaN/Inf present")
    if len(set(tasks)) == 1:
        print(f"task string         : {tasks[0]!r} (identical in all)")
    else:
        print(f"task string         : {len(set(tasks))} distinct -> {sorted(set(tasks))[:4]}")
    if len(set(fpss)) == 1:
        print(f"fps                 : {fpss[0]}")
    else:
        warnings.append("fps differs between episodes")
        print(f"fps                 : INCONSISTENT {sorted(set(fpss))}")
    print(f"duration @fps       : min {lens.min()/fpss[0]:.1f}s  max {lens.max()/fpss[0]:.1f}s")

    if allS.shape[1] == 16:  # humanoid: 2 x YellowArm (scripts/verify_humanoid.py)
        from verify_humanoid import audit_humanoid
        audit_humanoid(S, A, files, problems, warnings)
    else:
        # ---------- 3. joint limits ----------
        print("\n" + "=" * 78); print("3. STATE vs OPENARM JOINT LIMITS"); print("=" * 78)
        def violations(lim):
            v = 0
            for j, (lo, hi) in enumerate(lim[:allS.shape[1]]):
                v += int(((allS[:, j] < lo - 1e-3) | (allS[:, j] > hi + 1e-3)).sum())
            return v
        vL, vR = violations(LIMITS_LEFT), violations(LIMITS_RIGHT)
        which = "LEFT" if vL <= vR else "RIGHT"
        lim = LIMITS_LEFT if vL <= vR else LIMITS_RIGHT
        print(f"out-of-limit samples: left-arm limits {vL}, right-arm limits {vR}  -> data matches {which} arm\n")
        names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "finger1", "finger2"]
        print(f"{'dof':9s} {'min':>9s} {'max':>9s} {'limit lo':>9s} {'limit hi':>9s} {'used %':>7s}  {'range'}")
        for j in range(allS.shape[1]):
            lo, hi = lim[j]
            mn, mx = allS[:, j].min(), allS[:, j].max()
            used = (mx - mn) / (hi - lo) * 100 if hi > lo else 0
            flag = "  <-- OUT OF LIMIT" if (mn < lo - 1e-3 or mx > hi + 1e-3) else ("  <-- barely moves" if used < 1 else "")
            print(f"{names[j]:9s} {mn:9.4f} {mx:9.4f} {lo:9.4f} {hi:9.4f} {used:7.1f}{flag}")

        # ---------- 4. action semantics ----------
        print("\n" + "=" * 78); print("4. ACTION SEMANTICS"); print("=" * 78)
        print("If action = ABSOLUTE joint target, action[t] should track state[t+1] closely.")
        print("If action = DELTA, action[t] should track state[t+1]-state[t].\n")
        err_abs, err_del, corr = [], [], []
        for s, ac in zip(S, A):
            n = min(ac.shape[1], s.shape[1])
            a7, s_next, s_now = ac[:-1, :7], s[1:, :7], s[:-1, :7]
            err_abs.append(np.abs(a7 - s_next).mean())
            err_del.append(np.abs(a7 - (s_next - s_now)).mean())
        err_abs, err_del = np.mean(err_abs), np.mean(err_del)
        print(f"mean|action - next_state|        = {err_abs:.5f} rad")
        print(f"mean|action - (next-now) state|  = {err_del:.5f} rad")
        verdict = "ABSOLUTE joint targets" if err_abs < err_del else "DELTA joint commands"
        print(f"-> action is most consistent with: {verdict}")
        if err_abs < err_del:
            print(f"   tracking error {err_abs*1000:.2f} mrad "
                  f"({'tight, controller follows well' if err_abs < 0.02 else 'LOOSE — controller lags its target'})")
            if err_abs > 0.05:
                warnings.append(f"action/state tracking error high ({err_abs:.3f} rad) — gains may be too soft")

        # ---------- 5. gripper ----------
        print("\n" + "=" * 78); print("5. GRIPPER"); print("=" * 78)
        g = allA[:, 7]
        u = np.unique(np.round(g, 4))
        print(f"action gripper: min {g.min():.4f} max {g.max():.4f}, {len(u)} distinct values")
        print("  -> binary open/close" if len(u) <= 3 else "  -> continuous")
        f1, f2 = allS[:, 7], allS[:, 8]
        print(f"state finger1 : min {f1.min():.4f} max {f1.max():.4f}")
        print(f"state finger2 : min {f2.min():.4f} max {f2.max():.4f}")
        mirror_err = np.abs(np.abs(f1) - np.abs(f2)).mean()
        print(f"|f1|-|f2| mean abs diff = {mirror_err:.5f}  "
              f"({'fingers mirror correctly' if mirror_err < 0.01 else 'FINGERS DISAGREE'})")
        if mirror_err >= 0.01:
            warnings.append("finger joints not mirrored")
        # did the gripper actually actuate in every episode?
        idle = [os.path.basename(f) for f, ac in zip(files, A) if np.ptp(ac[:, 7]) < 1e-4]
        print(f"episodes where gripper never moves: {len(idle)}" + (f"  {idle[:5]}" if idle else ""))
        if idle:
            problems.append(f"{len(idle)} episodes have a motionless gripper — no grasp happened")

        # ---------- 6. diversity across episodes ----------
        print("\n" + "=" * 78); print("6. DIVERSITY (is randomisation working?)"); print("=" * 78)
        init = np.array([s[0] for s in S])
        fin = np.array([s[-1] for s in S])
        print(f"{'dof':9s} {'init std':>10s} {'final std':>10s}")
        for j in range(init.shape[1]):
            print(f"{names[j]:9s} {init[:, j].std():10.5f} {fin[:, j].std():10.5f}")
        if init.std(axis=0).max() < 1e-4:
            warnings.append("all episodes start from an identical pose — no start-state randomisation")
            print("\nNOTE: every episode starts at the SAME joint configuration.")
            print("      Object pose may still be randomised (not visible in joint state).")
        # duplicate episode detection
        sig = [f"{s.shape[0]}_{s.sum():.6f}" for s in S]
        dup = len(sig) - len(set(sig))
        print(f"\nidentical-trajectory duplicates: {dup}")
        if dup:
            problems.append(f"{dup} duplicate trajectories")

    # ---------- 7. images ----------
    print("\n" + "=" * 78); print("7. IMAGES"); print("=" * 78)
    cams = [k for k, (s, d) in ref.items() if len(s) == 4]
    pick = np.linspace(0, len(files) - 1, min(a.images, len(files))).astype(int)
    print(f"{'episode':16s} {'camera':13s} {'mean':>7s} {'std':>7s} {'black':>6s} {'dup-frames':>11s}")
    for i in pick:
        with np.load(files[i]) as z:
            for c in cams:
                im = z[c]
                fm = im.reshape(len(im), -1).mean(1)
                d = np.abs(np.diff(im.reshape(len(im), -1).astype(np.int16), axis=0)).mean(1)
                nblack = int((fm < 5).sum())
                ndup = int((d < 0.1).sum())
                print(f"{os.path.basename(files[i])[:-4]:16s} {c:13s} {fm.mean():7.1f} {fm.std():7.2f} "
                      f"{nblack:6d} {ndup:11d}")
                if nblack:
                    problems.append(f"{os.path.basename(files[i])}/{c}: {nblack} black frames")
                if ndup > len(im) * 0.2:
                    warnings.append(f"{os.path.basename(files[i])}/{c}: {ndup}/{len(im)} near-duplicate frames "
                                    "(render may lag physics)")
                del im

    # ---------- summary ----------
    print("\n" + "=" * 78); print("SUMMARY"); print("=" * 78)
    if problems:
        print(f"BLOCKING ({len(problems)}):")
        for p in problems[:20]:
            print("  x", p)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print("  !", w)
    if not problems and not warnings:
        print("Clean. Dataset looks ready to convert.")
    return 1 if problems else 0

if __name__ == "__main__":
    sys.exit(main())
