"""Reach probe for the YellowArm humanoid: which torso/shoulder configuration gives the widest workspace shared by
both arms, with the fixed head camera still seeing all of it.

  Phase A (numpy, any python):   env\\python.exe scripts\\humanoid_reach_probe.py --phase reach
  Phase B (Isaac Sim renders):   D:\\Isaac\\isaacsim\\python.bat scripts\\humanoid_reach_probe.py --phase render [--headless]

Phase A sweeps every candidate mount in scripts/data/humanoid.json (axis x roll x shoulder height x separation x
torso-table gap). A table point counts as reachable by an arm when top-down IK succeeds at the grasp height AND the
approach height for every grasp yaw in the config, inside the URDF joint limits, with no arm point inside the table or
the torso / shoulder bar (per-link collision boxes from the USD), and with static gravity torque (USD link masses + payload) under torque_margin x URDF effort.
Score = shared area (both arms). For the best configuration of each mount style it also picks the head-camera pitch
that centres the shared area and the smallest square FOV that sees all of it.
Outputs go to logs/humanoid_probe/: reach_results.json, reach_cache.npz, map_<mount>.png.

Phase B (run once per mount with --mounts: RTX cameras only render the first stage of a process) builds the
best configuration of each mount in Isaac (torso, table, both arms with URDF efforts and the real
moteus velocity limits, coloured reach tiles), renders the head camera and an overview camera, and checks in physics
that both arms hold a few shared grasp poses (TCP error after 2 s with gravity on).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import humanoid_kin as hk  # noqa: E402

OUT = os.path.join(ROOT, "logs", "humanoid_probe")
TORSO_INFLATE = 0.01  # m of clearance kept between the arm's collision boxes and torso / shoulder bar
TABLE_CLEAR = 0.005  # m above the table top for every link box; fingertips may come down to the table top
BAR = {"half_x": 0.04, "z0": 0.005, "z1": 0.065, "past_plate": 0.04}  # shoulder bar (non-lateral mounts), mount frame


def mirror(axis):
    return [axis[0], -axis[1], axis[2]]


# ================================================== phase A: reachability ==================================================
def solve_arm(ch, cfg, axis, log, sign):
    """Reachability of one arm (mount at the origin) on a mount-relative xy lattice, for every z level implied by the
    shoulder heights and every mount roll. IK runs once with joint1 free and roll 0; a roll r then only shifts joint1
    (q1 -> q1 - r), so each roll is a joint1-limit check on the same solutions.
    Returns dict: xs, ys, and per (h, roll): ok (nx, ny) bool and inward (nx, ny) = smallest mount-relative y (signed
    so that the torso is on the -y side) of any arm point in the torso's x band, maximised over the valid IK
    solutions; -inf where unreachable."""
    s = cfg["search"]
    lim = cfg["arm_limits"]
    step = s["grid_step"]
    xs = np.round(np.arange(-0.30, 0.95 + 1e-9, step), 4)
    ys = np.round(np.arange(-0.95, 0.95 + 1e-9, step), 4)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    R = hk.mount_rotation(axis, 0.0)
    j2 = R @ ch.origins[0][:3, 3] + R @ (ch.origins[0][:3, :3] @ ch.origins[1][:3, 3])  # joint2 origin at q1 = 0
    reach = 0.32 + 0.28 + 0.133 + 0.02
    rng = np.random.default_rng(0)
    seeds = np.vstack([np.zeros(7), rng.uniform(ch.lo * 0.85, ch.hi * 0.85, (15, 7))])
    seeds[:, 0] = np.linspace(-np.pi, np.pi, len(seeds), endpoint=False)
    eff = np.array(lim["effort_nm"]) * lim["torque_margin"]
    half_depth = cfg["torso"]["depth"] / 2 + TORSO_INFLATE
    # sign: +1 for the left arm (torso towards mount-relative -y), -1 for the right arm. It must come from the side,
    # not from axis[1]: forward45 / down have axis y = 0 for both arms (bug fixed 2026-10-05, see PLAN Nhật ký).
    has_bar = abs(axis[1]) < 0.99  # every mount except "lateral" hangs from a shoulder bar
    res = {"xs": xs, "ys": ys}
    rolls = s["rolls_deg"]
    for h in s["heights"]:
        ok_all = {r: np.ones(X.shape, bool) for r in rolls}
        inward = {r: np.full(X.shape, np.inf) for r in rolls}
        for zt in (s["grasp_z"], s["approach_z"]):
            z = zt - h
            cand = np.linalg.norm(np.stack([X - j2[0], Y - j2[1], np.full(X.shape, z - j2[2])], -1), axis=-1) < reach
            idx = np.argwhere(cand & np.any([ok_all[r] for r in rolls], 0))
            ok_lvl = {r: np.zeros(X.shape, bool) for r in rolls}
            inw_lvl = {r: np.full(X.shape, -np.inf) for r in rolls}
            if len(idx):
                T = np.stack([X[idx[:, 0], idx[:, 1]], Y[idx[:, 0], idx[:, 1]], np.full(len(idx), z)], 1)
                ok_yaws = {r: np.ones(len(idx), bool) for r in rolls}
                inw_yaws = {r: np.full(len(idx), np.inf) for r in rolls}
                for yaw in s["grasp_yaws_deg"]:
                    q, ok, _, _ = ch.ik_down(T, np.deg2rad(yaw), R, np.zeros(3), seeds, iters=70, free_j1=True)
                    M, S = ok.shape
                    qf = q.reshape(M * S, 7)
                    table_ok, ys_band = arm_clearance(ch, qf, R, h, half_depth, sign, has_bar)
                    tau = ch.gravity_torque(qf, R, np.zeros(3), payload=lim["payload_kg"])
                    torque_ok = np.all(np.abs(tau) <= eff, 1)
                    valid0 = ok.reshape(-1) & table_ok & torque_ok
                    for r in rolls:
                        valid = (valid0 & ch.j1_ok_for_roll(qf[:, 0], r)).reshape(M, S)
                        best = np.where(valid, ys_band.reshape(M, S), -np.inf).max(1)
                        ok_yaws[r] &= valid.any(1)
                        inw_yaws[r] = np.minimum(inw_yaws[r], best)
                for r in rolls:
                    ok_lvl[r][idx[:, 0], idx[:, 1]] = ok_yaws[r]
                    inw_lvl[r][idx[:, 0], idx[:, 1]] = np.where(ok_yaws[r], inw_yaws[r], -np.inf)
            for r in rolls:
                ok_all[r] &= ok_lvl[r]
                inward[r] = np.minimum(inward[r], inw_lvl[r])
        for r in rolls:
            res[(h, r)] = {"ok": ok_all[r], "inward": np.where(ok_all[r], inward[r], -np.inf)}
        log(f"    h={h:.2f}: reachable lattice points per roll " + str({r: int(ok_all[r].sum()) for r in rolls}))
    return res


def arm_clearance(ch, q, R, h, half_depth, sign, has_bar, chunk=4096):
    """Collision-box checks for arm solutions q (M, 7), mount at the origin with rotation R, table at z = -h.
    Returns table_ok (M,) (also includes the shoulder bar) and ys_band (M,): the smallest signed mount-relative y of
    any box point in the torso's x band below the torso top (the torso test itself depends on the separation)."""
    table_ok = np.empty(len(q), bool)
    ys_band = np.empty(len(q))
    fing = hk.FINGER_MASK
    for a in range(0, len(q), chunk):
        p = ch.collision_points(q[a:a + chunk], R, np.zeros(3))
        z = p[..., 2]
        ok = (z[:, ~fing].min(1) > -h + TABLE_CLEAR) & (z[:, fing].min(1) > -h - 0.001)
        if has_bar:
            inb = (np.abs(p[..., 0]) < BAR["half_x"] + TORSO_INFLATE) & (z > BAR["z0"] - TORSO_INFLATE) & \
                  (z < BAR["z1"] + TORSO_INFLATE) & (sign * p[..., 1] < BAR["past_plate"] + TORSO_INFLATE)
            ok &= ~inb.any(1)
        band = (np.abs(p[..., 0]) < half_depth) & (z < 0.10 + TORSO_INFLATE)
        table_ok[a:a + chunk] = ok
        ys_band[a:a + chunk] = np.where(band, sign * p[..., 1], np.inf).min(1)
    return table_ok, ys_band


def torso_width(cfg, mount, sep):
    t = cfg["torso"]
    if mount == "lateral":
        return sep
    return float(np.clip(sep - 2 * t["shoulder_clearance"], t["min_width"], t["width"]))


def table_grid(cfg, sep, gap, step):
    """World table points (x, y) on a lattice aligned with the arm lattices (y = sep/2 + k * step)."""
    t = cfg["table"]
    edge = cfg["torso"]["depth"] / 2 + gap
    m = t["usable_margin"]
    xs = np.round(np.arange(edge + m, edge + t["depth"] - m + 1e-9, step), 4)
    k0 = int(np.ceil((-t["width"] / 2 + m - sep / 2) / step))
    k1 = int(np.floor((t["width"] / 2 - m - sep / 2) / step))
    ys = np.round(sep / 2 + np.arange(k0, k1 + 1) * step, 4)
    return xs, ys, edge


def lookup(arm, h, roll, X, Y, need_inward):
    xs, ys = arm["xs"], arm["ys"]
    step = xs[1] - xs[0]
    ix = np.round((X - xs[0]) / step).astype(int)
    iy = np.round((Y - ys[0]) / step).astype(int)
    inside = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    out = np.zeros(X.shape, bool)
    d = arm[(h, roll)]
    ixc, iyc = np.clip(ix, 0, len(xs) - 1), np.clip(iy, 0, len(ys) - 1)
    out[inside] = d["ok"][ixc[inside], iyc[inside]] & (d["inward"][ixc[inside], iyc[inside]] >= need_inward)
    return out


def camera_fit(cfg, h, pts):
    """Pitch (deg, down from horizontal) that minimises the square FOV needed to see all pts, and that FOV."""
    hc = cfg["head_camera"]
    eye = np.array([hc["x"], 0.0, h + hc["height_above_mount"]])
    d = pts - eye
    best = (None, 1e9)
    for pitch in np.arange(10.0, 89.9, 0.5):
        p = np.deg2rad(pitch)
        fwd = np.array([np.cos(p), 0.0, -np.sin(p)])
        up = np.array([np.sin(p), 0.0, np.cos(p)])
        left = np.array([0.0, 1.0, 0.0])
        zf = d @ fwd
        if np.any(zf <= 0.01):
            continue
        need = 2 * np.degrees(max(np.arctan(np.abs(d @ left) / zf).max(), np.arctan(np.abs(d @ up) / zf).max()))
        if need < best[1]:
            best = (float(pitch), float(need))
    return best


def phase_reach(cfg, args):
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "reach.log"), "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    ch = hk.Chain()
    s = cfg["search"]
    step = s["grid_step"]
    mounts = args.mounts or list(cfg["candidates"])
    t0 = time.time()
    arms = {}
    arms_cache = os.path.join(OUT, "arms_cache.pkl")
    if args.reuse and os.path.exists(arms_cache):
        import pickle

        arms = pickle.load(open(arms_cache, "rb"))
        log(f"[reach] reusing IK solutions from {arms_cache}")
    for m in mounts:
        axL = cfg["candidates"][m]["axis_left"]
        for side, ax in (("L", axL), ("R", mirror(axL))):
            if (m, side) in arms:
                continue
            log(f"[reach] {m} {side}  ({time.time() - t0:.0f}s)")
            arms[(m, side)] = solve_arm(ch, cfg, ax, log, 1.0 if side == "L" else -1.0)
    import pickle

    pickle.dump(arms, open(arms_cache, "wb"))

    rows = []
    for m in mounts:
        for h in s["heights"]:
            for sep in s["separations"]:
                torso_w = torso_width(cfg, m, sep)
                need = torso_w / 2 + TORSO_INFLATE - sep / 2  # smallest allowed mount-relative inward y
                for gap in s["gaps"]:
                    xs, ys, edge = table_grid(cfg, sep, gap, step)
                    X, Y = np.meshgrid(xs, ys, indexing="ij")
                    best = {}
                    for side, sy in (("L", 1), ("R", -1)):
                        cands = []
                        for roll in s["rolls_deg"]:
                            r = lookup(arms[(m, side)], h, roll, X, Y - sy * sep / 2, need)
                            cands.append((int(r.sum()), roll, r))
                        best[side] = max(cands, key=lambda c: c[0])
                    L, R = best["L"][2], best["R"][2]
                    shared = L & R
                    rows.append({"mount": m, "height": h, "separation": sep, "gap": gap, "torso_width": torso_w,
                                 "roll_left": best["L"][1], "roll_right": best["R"][1],
                                 "area_shared_m2": round(shared.sum() * step * step, 4),
                                 "area_left_m2": round(L.sum() * step * step, 4),
                                 "area_right_m2": round(R.sum() * step * step, 4),
                                 "area_union_m2": round((L | R).sum() * step * step, 4),
                                 "table_edge_x": round(edge, 4)})
    rows.sort(key=lambda r: (-r["area_shared_m2"], -r["area_union_m2"]))

    best_per_mount = {}
    cache = {}
    for m in mounts:
        r = next(x for x in rows if x["mount"] == m)
        xs, ys, edge = table_grid(cfg, r["separation"], r["gap"], step)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        need = r["torso_width"] / 2 + TORSO_INFLATE - r["separation"] / 2
        L = lookup(arms[(m, "L")], r["height"], r["roll_left"], X, Y - r["separation"] / 2, need)
        R = lookup(arms[(m, "R")], r["height"], r["roll_right"], X, Y + r["separation"] / 2, need)
        sh = L & R
        pts = np.stack([X[sh], Y[sh], np.zeros(sh.sum())], 1)
        pitch, fov_need = camera_fit(cfg, r["height"], pts) if len(pts) else (None, None)
        sx, sy_ = (X[sh], Y[sh]) if sh.any() else (np.zeros(1), np.zeros(1))
        r2 = dict(r, cam_pitch_deg=pitch, cam_fov_needed_deg=None if fov_need is None else round(fov_need, 1),
                  cam_fov_ok=bool(fov_need is not None and fov_need <= cfg["head_camera"]["fov_deg"]),
                  shared_x_range=[float(sx.min()), float(sx.max())], shared_y_range=[float(sy_.min()), float(sy_.max())])
        best_per_mount[m] = r2
        cache[f"{m}_xs"], cache[f"{m}_ys"], cache[f"{m}_L"], cache[f"{m}_R"] = xs, ys, L, R
        save_map(os.path.join(OUT, f"map_{m}.png"), xs, ys, L, R, r2)
        log(f"[reach] best {m}: " + json.dumps(r2))
    np.savez_compressed(os.path.join(OUT, "reach_cache.npz"), **cache)
    with open(os.path.join(OUT, "reach_results.json"), "w", encoding="utf-8") as f:
        json.dump({"best_per_mount": best_per_mount, "top20": rows[:20], "n_configs": len(rows),
                   "seconds": round(time.time() - t0, 1), "all": rows}, f, indent=1)
    log(f"[reach] done: {len(rows)} configurations in {time.time() - t0:.0f}s")


def save_map(path, xs, ys, L, R, r):
    """Top-down map, robot at the bottom: x (forward) up, y (left) to the left. green both, blue left, orange right."""
    from PIL import Image, ImageDraw

    c = 10
    W, H = len(ys) * c, len(xs) * c
    img = Image.new("RGB", (W, H + 40), (235, 235, 235))
    d = ImageDraw.Draw(img)
    for i in range(len(xs)):
        for j in range(len(ys)):
            col = (60, 170, 70) if L[i, j] and R[i, j] else (70, 110, 220) if L[i, j] else (240, 150, 40) if R[i, j] \
                else (205, 200, 190)
            x0, y0 = (len(ys) - 1 - j) * c, (len(xs) - 1 - i) * c
            d.rectangle([x0, y0, x0 + c - 2, y0 + c - 2], fill=col)
    d.text((4, H + 4), f"{r['mount']} h={r['height']} sep={r['separation']} gap={r['gap']} "
                       f"shared={r['area_shared_m2']} m2 union={r['area_union_m2']} m2", fill=(0, 0, 0))
    d.text((4, H + 20), "green both / blue left / orange right; robot at bottom edge", fill=(0, 0, 0))
    img.save(path)


# ================================================== select: approved configuration ==================================================
def camera_need(eye, pitch_deg, pts, d_pos=None, d_rot=None):
    """Square FOV (deg) needed to see every point in pts from a head camera at eye looking forward-down by pitch_deg.
    d_pos (K, 3) / d_rot (K, 3) = mounting errors (m; roll, pitch, yaw in deg) -> needed FOV per error (K,)."""
    d_pos = np.zeros((1, 3)) if d_pos is None else d_pos
    d_rot = np.zeros((1, 3)) if d_rot is None else d_rot
    p = np.deg2rad(pitch_deg)
    fwd0 = np.array([np.cos(p), 0.0, -np.sin(p)])
    up0 = np.array([np.sin(p), 0.0, np.cos(p)])
    left0 = np.array([0.0, 1.0, 0.0])
    C0 = np.stack([fwd0, left0, up0], 1)  # columns: camera forward, left, up in world
    out = np.empty(len(d_pos))
    for k in range(len(d_pos)):
        r, pt, y = np.deg2rad(d_rot[k])
        # errors applied in the camera frame: roll about forward, pitch about left, yaw about up
        E = hk.rot_axis((1, 0, 0), r) @ hk.rot_axis((0, 1, 0), pt) @ hk.rot_axis((0, 0, 1), y)
        C = C0 @ E
        d = pts - (eye + d_pos[k])
        zf = d @ C[:, 0]
        if np.any(zf <= 0.01):
            out[k] = 180.0
            continue
        out[k] = 2 * np.degrees(max(np.arctan(np.abs(d @ C[:, 1]) / zf).max(), np.arctan(np.abs(d @ C[:, 2]) / zf).max()))
    return out


def phase_select(cfg, args):
    """Approved configuration -> scripts/data/humanoid_workspace.npz (left/right/shared reach masks on the table) and
    humanoid.json "selected" (mount, torso, head camera pose and the smallest FOV in --fovs that still sees the whole
    shared area under +/- pos_err / rot_err mounting errors)."""
    import pickle

    m, h, sep, gap = args.mount, args.height, args.separation, args.gap
    s = cfg["search"]
    step = s["grid_step"]
    arms = pickle.load(open(os.path.join(OUT, "arms_cache.pkl"), "rb"))
    xs, ys, edge = table_grid(cfg, sep, gap, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    tw = torso_width(cfg, m, sep)
    need = tw / 2 + TORSO_INFLATE - sep / 2
    L = lookup(arms[(m, "L")], h, args.roll, X, Y - sep / 2, need)
    R = lookup(arms[(m, "R")], h, args.roll, X, Y + sep / 2, need)
    sh = L & R
    print(f"[select] {m} h={h} sep={sep} gap={gap} roll={args.roll}: shared {sh.sum() * step * step:.4f} m2, "
          f"union {(L | R).sum() * step * step:.4f} m2")
    np.savez_compressed(os.path.join(HERE, "data", "humanoid_workspace.npz"), xs=xs, ys=ys, left=L, right=R, shared=sh,
                        step=step)
    save_map(os.path.join(OUT, f"map_{m}_selected.png"), xs, ys, L, R,
             {"mount": m, "height": h, "separation": sep, "gap": gap, "area_shared_m2": round(sh.sum() * step * step, 4),
              "area_union_m2": round((L | R).sum() * step * step, 4)})

    # head camera: on the front face of the head box; the shared area counted with cell corners, table top up to 6 cm
    hc = cfg["head_camera"]
    head_x = hc["x"]
    eye = np.array([head_x + 0.065, 0.0, h + hc["height_above_mount"]])
    c = np.stack([X[sh], Y[sh]], 1)
    corners = np.concatenate([c + np.array([dx, dy]) * step / 2 for dx in (-1, 1) for dy in (-1, 1)])
    pts = np.concatenate([np.c_[corners, np.full(len(corners), z)] for z in (0.0, 0.06)])
    e_p, e_r = args.pos_err, args.rot_err
    grid = np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    d_pos, d_rot = np.repeat(grid * e_p, len(grid), 0), np.tile(grid * e_r, (len(grid), 1))  # 27 x 27 corners
    best = None
    for pitch in np.arange(40.0, 80.01, 0.5):
        w = camera_need(eye, pitch, pts, d_pos, d_rot).max()
        if best is None or w < best[1]:
            best = (float(pitch), float(w))
    pitch, worst = best
    nominal = float(camera_need(eye, pitch, pts)[0])
    choice = next((f for f in sorted(args.fovs) if f >= worst), None)
    print(f"[select] head camera eye={eye.round(3).tolist()} pitch {pitch} deg: needed FOV nominal {nominal:.1f}, "
          f"worst over +/-{100 * e_p:.0f} cm / +/-{e_r:.0f} deg = {worst:.1f} -> choose {choice} from {args.fovs}")
    if choice is None:
        raise SystemExit("no candidate FOV covers the worst case")
    worst_k = np.argsort(-camera_need(eye, pitch, pts, d_pos, d_rot))[:2]
    sel = {
        "_approved": "2026-10-05 by the user (PLAN.md 2.0)",
        "mount": m, "axis_left": cfg["candidates"][m]["axis_left"], "roll_left_deg": args.roll, "roll_right_deg": args.roll,
        "mount_height": h, "separation": sep, "torso_table_gap": gap, "torso_width": tw, "torso_depth": cfg["torso"]["depth"],
        "table_edge_x": round(edge, 4),
        "head_camera": {"pos": eye.round(4).tolist(), "head_box_center_x": head_x, "pitch_deg": pitch, "fov_deg": choice,
                        "resolution": hc["resolution"], "fov_needed_nominal_deg": round(nominal, 1),
                        "fov_needed_worst_deg": round(worst, 1), "mount_error_checked": {"pos_m": e_p, "rot_deg": e_r},
                        "worst_errors": [{"d_pos": d_pos[k].round(3).tolist(), "d_rot_deg": d_rot[k].tolist()}
                                         for k in worst_k],
                        "fov_candidates": sorted(args.fovs)},
        "workspace": {"file": "scripts/data/humanoid_workspace.npz",
                      "area_shared_m2": round(sh.sum() * step * step, 4), "area_union_m2": round((L | R).sum() * step * step, 4),
                      "shared_x_range": [float(X[sh].min()), float(X[sh].max())],
                      "shared_y_range": [float(Y[sh].min()), float(Y[sh].max())]},
    }
    path = args.config
    txt = open(path, encoding="utf-8").read()
    data = json.loads(txt)
    data["selected"] = sel
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    print("[select] wrote selected -> " + path)


# ================================================== phase B: Isaac renders ==================================================
def phase_render(cfg, args):
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    import omni.timeline
    import omni.usd
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.experimental.materials import OmniPbrMaterial
    from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
    from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
    from PIL import Image
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    res = json.load(open(os.path.join(OUT, "reach_results.json"), encoding="utf-8"))
    cache = np.load(os.path.join(OUT, "reach_cache.npz"))
    lim = cfg["arm_limits"]
    hc = cfg["head_camera"]
    ch = hk.Chain()
    USD = hk.URDF.replace("yellow_arm.urdf", "yellow_arm/yellow_arm.usd")
    report = {}
    mounts = args.mounts or list(res["best_per_mount"])

    def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
        q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
        return np.array([q.GetReal(), *q.GetImaginary()])

    def camera(path, eye, target, fov_deg, res_px):
        cam = RtxCamera(path, translations=np.array(eye, float), orientations=look_at_quat(eye, target))
        c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
        ap = c.GetHorizontalApertureAttr().Get()
        c.GetVerticalApertureAttr().Set(ap)
        c.GetFocalLengthAttr().Set(float(ap / (2 * np.tan(np.deg2rad(fov_deg) / 2))))
        c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        return CameraSensor(cam, resolution=(res_px, res_px), annotators=["rgb"])

    for m in mounts:
        r = res["best_per_mount"][m]
        h, sep, gap = r["height"], r["separation"], r["gap"]
        stage_utils.create_new_stage()
        SimulationManager.set_physics_dt(1.0 / 100.0)
        DomeLight("/World/Light").set_intensities(1200)
        t = cfg["table"]
        GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, -t["height"]])
        stage = omni.usd.get_context().get_stage()
        mats = {}

        def box(path, size, pos, color, collide=True):
            c = Cube(path, sizes=1.0, positions=np.array(pos, float), scales=np.array(size, float))
            if collide:
                GeomPrim(path, apply_collision_apis=True)
            key = tuple(color)
            if key not in mats:
                mats[key] = OmniPbrMaterial(f"/World/Materials/m{len(mats)}")
                mats[key].set_input_values("diffuse_color_constant", list(color))
            c.apply_visual_materials(mats[key])

        edge = r["table_edge_x"]
        box("/World/table", (t["depth"], t["width"], t["height"]), (edge + t["depth"] / 2, 0, -t["height"] / 2),
            (0.55, 0.45, 0.33), collide=not args.no_obstacles)
        td, tw = cfg["torso"]["depth"], r["torso_width"]
        top = h + 0.10
        box("/World/torso", (td, tw - 0.01, top + t["height"]), (0, 0, (top - t["height"]) / 2), cfg["torso"]["color"],
            collide=not args.no_obstacles)
        if m != "lateral":  # shoulder bar out to the mounting plates
            box("/World/shoulder_bar", (2 * BAR["half_x"], sep + 2 * BAR["past_plate"], BAR["z1"] - BAR["z0"]),
                (0, 0, h + (BAR["z0"] + BAR["z1"]) / 2), cfg["torso"]["color"],
                collide=not args.no_obstacles)
        box("/World/head", (0.12, 0.14, 0.14), (0.0, 0, h + hc["height_above_mount"]), (0.25, 0.25, 0.28), collide=False)
        xs, ys, L, R = cache[f"{m}_xs"], cache[f"{m}_ys"], cache[f"{m}_L"], cache[f"{m}_R"]
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if L[i, j] or R[i, j]:
                    col = (0.2, 0.75, 0.25) if L[i, j] and R[i, j] else (0.25, 0.4, 0.9) if L[i, j] else (0.95, 0.55, 0.1)
                    box(f"/World/tiles/t_{i}_{j}", (0.022, 0.022, 0.002), (x, y, 0.001), col, collide=False)

        arms = {}
        axL = cfg["candidates"][m]["axis_left"]
        for side, ax, roll, sy in (("L", axL, r["roll_left"], 1), ("R", mirror(axL), r["roll_right"], -1)):
            path = f"/World/arm_{side}"
            stage_utils.add_reference_to_stage(usd_path=USD, path=path)
            Rm = hk.mount_rotation(ax, roll)
            pm = np.array([0.0, sy * sep / 2, h])
            p = stage.GetPrimAtPath(path)
            p.GetAttribute("xformOp:orient").Set(Gf.Quatd(*map(float, hk.mat_to_quat(Rm))))
            p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, pm)))
            # real limits: URDF effort, moteus velocity (PhysX wants deg/s for revolute joints)
            for k, jn in enumerate(hk.ARM_JOINTS):
                jp = stage.GetPrimAtPath(f"{path}/joints/{jn}") if stage.GetPrimAtPath(f"{path}/joints/{jn}") else None
                if jp is None:
                    jp = next(x for x in stage.Traverse() if x.GetPath().pathString.startswith(path) and x.GetName() == jn)
                jp.GetAttribute("drive:angular:physics:maxForce").Set(float(lim["effort_nm"][k]))
                jp.GetAttribute("physxJoint:maxJointVelocity").Set(float(np.degrees(lim["velocity_used_rad_s"][k])))
            arms[side] = (path, Rm, pm)

        eye = np.array([hc["x"], 0.0, h + hc["height_above_mount"]])
        p = np.deg2rad(r["cam_pitch_deg"] or 45.0)
        target = eye + np.array([np.cos(p), 0.0, -np.sin(p)])
        cams = {"head": camera("/World/head_cam", eye + np.array([0.065, 0, 0]), target + np.array([0.065, 0, 0]),
                               hc["fov_deg"], hc["resolution"]),
                "head_hi": camera("/World/head_cam_hi", eye + np.array([0.065, 0, 0]), target + np.array([0.065, 0, 0]),
                                  hc["fov_deg"], 640),
                "overview": camera("/World/overview", (edge + 1.3, 0.9, h + 0.9), (edge * 0.5 + 0.1, 0, h * 0.4), 60, 720),
                "top": camera("/World/top", (edge + 0.3, 0, 2.4), (edge + 0.3, 0, 0), 50, 720)}

        robots = {s: Articulation(arms[s][0]) for s in arms}
        tl = omni.timeline.get_timeline_interface()
        tl.play()
        app.update()

        # pose: both arms above the shared area (approach height), then physics hold checks at grasp height
        sh = L & R
        Xg, Yg = np.meshgrid(xs, ys, indexing="ij")
        spts = np.stack([Xg[sh], Yg[sh]], 1)
        cy = spts[:, 1].mean() if len(spts) else 0.0
        cx = spts[:, 0].mean() if len(spts) else edge + 0.25
        rng = np.random.default_rng(1)
        seeds = np.vstack([np.zeros(7), rng.uniform(ch.lo * 0.85, ch.hi * 0.85, (15, 7))])

        sgn = {"L": 1.0, "R": -1.0}
        need = tw / 2 + TORSO_INFLATE - sep / 2

        def ik(side, pt):
            path, Rm, pm = arms[side]
            q, ok, _, _ = ch.ik_down(np.array([pt]) - pm, 0.0, Rm, np.zeros(3), seeds, iters=150)
            t_ok, ys_b = arm_clearance(ch, q[0], Rm, h, td / 2 + TORSO_INFLATE, sgn[side], m != "lateral")
            valid = ok[0] & t_ok & (ys_b >= need)
            if not valid.any():
                return None
            return q[0][np.argmax(valid)]

        def pose_and_hold(qs, seconds):
            for s, q in qs.items():
                rb = robots[s]
                names = rb.dof_names
                full = np.zeros(len(names))
                full[[names.index(j) for j in hk.ARM_JOINTS]] = q
                for fj in ("yellowarm_joint8", "yellowarm_joint9"):
                    full[names.index(fj)] = lim["gripper"]["open"]
                rb.set_dof_positions(full)
                rb.set_dof_velocities(np.zeros_like(full))
                rb.set_dof_position_targets(full)
            for _ in range(int(seconds * 100)):
                app.update()

        def tcp_world(side):
            l7 = RigidPrim(f"{arms[side][0]}/yellowarm_link7")
            p7, q7 = (x.numpy()[0].astype(float) for x in l7.get_world_poses())
            w, x, y, z = q7
            zax = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
            return p7 + zax * hk.TCP_OFFSET

        park = {}  # waiting pose: tcp out on the arm's own side, high above the table
        for side, sy in (("L", 1), ("R", -1)):
            for x, y, z in ((edge + 0.10, 0.45, 0.30), (edge + 0.05, 0.40, 0.35), (edge + 0.20, 0.50, 0.25),
                            (edge + 0.15, 0.35, 0.40)):
                park[side] = ik(side, np.array([x, sy * (sep / 2 + y - 0.18), z]))
                if park[side] is not None:
                    break
        holds = []
        if len(spts):
            pick = spts[rng.choice(len(spts), size=min(args.hold_points, len(spts)), replace=False)]
            for pt in pick:
                for side, other in (("L", "R"), ("R", "L")):
                    tgt = np.array([pt[0], pt[1], cfg["search"]["grasp_z"]])
                    q = ik(side, tgt)
                    if q is None:
                        holds.append({"side": side, "target": tgt.round(3).tolist(), "ik": False})
                        continue
                    pose_and_hold({side: q, **({other: park[other]} if park[other] is not None else {})}, 1.0)
                    err = float(np.linalg.norm(tcp_world(side) - tgt))
                    holds.append({"side": side, "target": tgt.round(3).tolist(), "ik": True, "hold_err_mm": round(1000 * err, 1)})
        qL = ik("L", np.array([cx, cy + 0.12, 0.15]))
        qR = ik("R", np.array([cx, cy - 0.12, 0.15]))
        pose_and_hold({k: v for k, v in (("L", qL), ("R", qR)) if v is not None}, 1.5)
        for _ in range(60):  # let the RTX cameras settle before grabbing
            app.update()
        for n, cam in cams.items():
            d, _ = cam.get_data("rgb")
            if d is not None:
                Image.fromarray(d.numpy()[..., :3].astype(np.uint8)).save(os.path.join(OUT, f"{m}_{n}.png"))
        report[m] = {"config": r, "hold_checks": holds, "park_found": {s: park[s] is not None for s in park},
                     "hold_err_mm_max": max((x["hold_err_mm"] for x in holds if x.get("ik")), default=None),
                     "hold_ok_fraction": round(sum(1 for x in holds if x.get("ik") and x["hold_err_mm"] < 10.0)
                                               / max(1, len(holds)), 3)}
        print(f"[render] {m}: " + json.dumps(report[m]["hold_checks"]), flush=True)
        tl.stop()
    name = os.path.join(OUT, "render_report_no_obstacles.json" if args.no_obstacles else "render_report.json")
    if os.path.exists(name):  # one process per mount (RTX cameras only render the first stage of a process)
        report = {**json.load(open(name, encoding="utf-8")), **report}
    with open(name, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    app.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["reach", "render", "select"], required=True)
    ap.add_argument("--mount", default="forward45", help="select")
    ap.add_argument("--height", type=float, default=0.45, help="select")
    ap.add_argument("--separation", type=float, default=0.55, help="select")
    ap.add_argument("--gap", type=float, default=0.05, help="select")
    ap.add_argument("--roll", type=float, default=90, help="select (both arms)")
    ap.add_argument("--fovs", type=float, nargs="*", default=[45, 55, 65], help="select: candidate head-camera FOVs")
    ap.add_argument("--pos-err", type=float, default=0.02, help="select: camera mounting error, m per axis")
    ap.add_argument("--rot-err", type=float, default=3.0, help="select: camera mounting error, deg per axis")
    ap.add_argument("--config", default=os.path.join(HERE, "data", "humanoid.json"))
    ap.add_argument("--mounts", nargs="*", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-obstacles", action="store_true", help="render: table/torso without colliders (diagnostics)")
    ap.add_argument("--reuse", action="store_true", help="reach: reuse logs/humanoid_probe/arms_cache.pkl")
    ap.add_argument("--hold-points", type=int, default=30, help="render: shared points checked in physics per arm")
    args, _ = ap.parse_known_args()
    cfg = hk.load_config(args.config)
    {"reach": phase_reach, "render": phase_render, "select": phase_select}[args.phase](cfg, args)


if __name__ == "__main__":
    main()
