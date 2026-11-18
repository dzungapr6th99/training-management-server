"""Retarget the NVIDIA Isaac Lab Mimic GR1T2 pick-place demos onto the dual YellowArm + ORCA robot (numpy only).

Source: datasets/external/gr1/dataset_annotated_gr1.hdf5 (5 human teleop demos, Isaac-PickPlace-GR1T2-Abs-Mimic-v0,
20 Hz): left hand picks a steering wheel (scale 0.75) from the table and hands it to the right hand, which drops it in
a bin. Link order: datasets/external/gr1/gr1_names.json (scripts/gr1_inspect.py, verified against the GR1 USD).
Both robots face +Y with the left arm at -X (YellowArm needs embodiment.yaml sim_overrides for that), and the hands
have the same handedness, so GR1 left -> YellowArm left, GR1 right -> YellowArm right, no mirroring.

Per frame:
  layout   every position p goes through one similarity about the shoulder midpoint: C_o + s * (p - C_g),
           s = YellowArm arm length / GR1 arm length (table, object, bin, hand paths keep GR1's proportions).
  arm      weighted damped least squares on arm joint1..7 + wrist: knuckle centre (mean finger base) at the target,
           hand orientation = GR1 hand orientation (finger direction / palm normal / across-palm axis, built the same
           way from landmarks on both hands), elbow toward GR1's shoulder->elbow direction, smooth w.r.t. the
           previous frame. Near the object the hand->object offset is kept at true size (hands are about the same
           size), far from it the scaled layout is used: target = map(B) + w * (1 - s) * (B - obj), w = 1 within
           0.10 m of the object, 0 beyond 0.30 m.
  fingers  16 ORCA finger joints: fingertip positions (5) and thumb->finger vectors (4) in the hand frame match
           GR1's (scaled by finger length ratio), DLS from the previous frame.
Output (per demo): datasets/external/gr1/retarget/<demo>.npz
  q (T, 48) joint_full order of embodiment.yaml, t (T,) s at 25 Hz, obj_pos/obj_quat (T,) mapped object pose,
  plus kinematic errors; and datasets/external/gr1/retarget/report.json.

  env\\python.exe scripts\\gr1_retarget.py [--demos demo_0 ...] [--hz 25]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import embodiment as EMB
from dual_orca_kin import FINGERS, Kin, rot_err
from mhh_paths import ISAAC_SIM_DIR, ROOT

GR1 = ROOT / "datasets" / "external" / "gr1"
OUT = GR1 / "retarget"
SIDES = {"left": ("left", "L"), "right": ("right", "R")}  # ours -> (GR1 arm prefix, GR1 hand prefix)
GR1_FINGERS = ("index", "middle", "ring", "pinky")


def h5():
    # h5py ships inside an Isaac Sim extension (cp312 wheel, same as env python): no pip install needed
    sys.path.insert(0, str(ISAAC_SIM_DIR / "exts" / "isaacsim.replicator.episode_recorder" / "pip_prebundle"))
    import h5py

    return h5py


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hand_frame(W, B, I, P):
    """Same recipe on both hands: x = wrist -> knuckles (finger direction), z = pinky -> index base (across palm),
    y = z cross x (palm normal side); re-orthogonalised."""
    x = B - W
    x /= np.linalg.norm(x)
    z = I - P
    z -= x * (x @ z)
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


# ------------------------------------------------------------------ GR1 side
class GR1Demo:
    def __init__(self, grp, names):
        self.L = names["link_names"]
        self.J = names["dof_names"]
        self.ls = grp["obs/robot_links_state"][()].astype(np.float64)  # (T, 55, 13): pos, quat wxyz, lin, ang
        self.q = grp["states/articulation/robot/joint_position"][()].astype(np.float64)
        self.obj_p = grp["obs/object_pos"][()].astype(np.float64)
        self.obj_q = grp["obs/object_rot"][()].astype(np.float64)
        self.T = len(self.q)
        self.dt = 6 / 120.0  # sim dt 1/120, decimation 6 (pickplace_gr1t2_env_cfg.py)
        self._tip_loc = {}

    def p(self, link, t):
        return self.ls[t, self.L.index(link), :3]

    def R(self, link, t):
        return quat_wxyz_to_R(self.ls[t, self.L.index(link), 3:7])

    def arm(self, side, t):
        """(shoulder centre, elbow, wrist). Link origins are not joint centres on GR1: the shoulder centre is taken
        at the yaw link's x/y and the pitch link's height (where the three shoulder axes meet)."""
        a = SIDES[side][0]
        sp, sy = self.p(f"{a}_upper_arm_pitch_link", t), self.p(f"{a}_upper_arm_yaw_link", t)
        return (np.array([sy[0], sy[1], sp[2]]), self.p(f"{a}_lower_arm_pitch_link", t),
                self.p(f"{a}_hand_roll_link", t))

    def landmarks(self, side, t):
        h = SIDES[side][1]
        W = self.p(f"{SIDES[side][0]}_hand_pitch_link", t)
        base = {f: self.p(f"{h}_{f}_proximal_link", t) for f in GR1_FINGERS}
        B = np.mean(list(base.values()), 0)
        return W, B, base["index"], base["pinky"]

    def frame(self, side, t):
        W, B, I, P = self.landmarks(side, t)
        return B, hand_frame(W, B, I, P)

    def tips(self, side, t):
        """Approximate fingertips: the last phalanx continues along the proximal->last-link direction measured in the
        most open frame (GR1 links stop at the last joint), length = 0.8 x proximal segment."""
        h = SIDES[side][1]
        chains = {f: (f"{h}_{f}_proximal_link", f"{h}_{f}_intermediate_link") for f in GR1_FINGERS}
        chains["thumb"] = (f"{h}_thumb_proximal_link", f"{h}_thumb_distal_link")
        if side not in self._tip_loc:
            t0 = int(np.argmin([self.q[k, self.J.index(f"{h}_index_proximal_joint")] ** 2 for k in range(self.T)]))
            loc = {}
            for f, (a, b) in chains.items():
                d = self.p(b, t0) - self.p(a, t0)
                loc[f] = self.R(b, t0).T @ d * 0.8
            self._tip_loc[side] = loc
        return {f: self.p(b, t) + self.R(b, t) @ self._tip_loc[side][f] for f, (a, b) in chains.items()}


# ------------------------------------------------------------------ YellowArm + ORCA side
class Ours:
    def __init__(self):
        self.E = EMB.Embodiment()
        self.names = self.E.action_space("joint_full")
        self.k = Kin()
        self.lo = {n: self.k.limits[n][0] for n in self.k.limits}
        self.hi = {n: self.k.limits[n][1] for n in self.k.limits}
        W0 = self.k.fk({})
        self.loc = {}
        for s in ("left", "right"):
            Pi = np.linalg.inv(W0[f"{s}_palm"])
            loc = lambda l: (Pi @ W0[l])[:3, 3]  # noqa: E731
            self.loc[s] = {"W": loc(f"{s}_wrist_jointbody"), **{f: loc(f"{s}_{f}_pp") for f in FINGERS}}
        self.home = dict(zip(self.names, self.E.home(self.names)))

    def arm_names(self, side):
        return self.k.chain(side)

    def finger_names(self, side):
        return [n for n in self.names if n.startswith(side) and n not in self.k.chain(side)]

    def landmarks(self, side, W):
        M = W[f"{side}_palm"]
        g = lambda v: (M @ np.r_[v, 1.0])[:3]  # noqa: E731
        L = self.loc[side]
        base = {f: g(L[f]) for f in FINGERS}
        return g(L["W"]), np.mean(list(base.values()), 0), base["index"], base["pinky"]

    def frame(self, side, W):
        Wr, B, I, P = self.landmarks(side, W)
        return B, hand_frame(Wr, B, I, P)

    def shoulder(self, side, W):
        return W[f"{side}_yellowarm_link2"][:3, 3]

    def elbow(self, side, W):
        return W[f"{side}_yellowarm_link4"][:3, 3]

    def tips(self, side, W):
        return {f: W[f"{side}_{f}_fingertip"][:3, 3] for f in (*FINGERS, "thumb")}


def swivel_dir(S, E, Wr):
    """Unit direction of the elbow away from the shoulder->wrist axis (the arm's swivel angle)."""
    a = (Wr - S) / np.linalg.norm(Wr - S)
    u = (E - S) - a * ((E - S) @ a)
    return u / max(np.linalg.norm(u), 1e-9)


def swivel_err(o, side, W, u_t):
    S, E, Wr = o.shoulder(side, W), o.elbow(side, W), W[f"{side}_yellowarm_link6"][:3, 3]
    return float(np.arccos(np.clip(swivel_dir(S, E, Wr) @ u_t, -1, 1)))


def solve_arm(o, side, q, B_t, R_t, u_t, iters=40, w_rot=0.3, w_reg=0.05, damping=0.03, elbow_iters=25,
              hand_tol=0.005):
    """Two-phase IK. Phase 1 (must hold): knuckle centre at B_t and hand orientation R_t, damped least squares
    started from the given pose (+ a small pull toward it for smoothness). Phase 2 (style): move the elbow toward
    the point of its swivel circle that points along u_t (GR1's swivel direction), inside the null space of phase 1;
    a step is kept only if the hand stays within hand_tol of phase 1."""
    names = o.arm_names(side)
    lo = np.array([o.lo[n] for n in names]) + 0.01
    hi = np.array([o.hi[n] for n in names]) - 0.01
    x0 = np.clip(np.array([q[n] for n in names]), lo, hi)

    def fk(xv):
        qq = dict(q)
        qq.update(zip(names, xv))
        return o.k.fk(qq)

    def e_hand(xv):
        B, R = o.frame(side, fk(xv))
        return np.concatenate([B_t - B, w_rot * rot_err(R, R_t)])

    def e_elb(xv):
        W = fk(xv)
        S, E, Wr = o.shoulder(side, W), o.elbow(side, W), W[f"{side}_yellowarm_link6"][:3, 3]
        a = (Wr - S) / np.linalg.norm(Wr - S)
        c = S + a * ((E - S) @ a)
        u = u_t - a * (u_t @ a)
        return c + np.linalg.norm(E - c) * u / max(np.linalg.norm(u), 1e-9) - E

    def jac(f, xv, e):
        J = np.zeros((len(e), 8))
        for i in range(8):
            d = xv.copy()
            d[i] += 1e-5
            J[:, i] = (e - f(d)) / 1e-5
        return J

    x = x0.copy()
    e = e_hand(x)
    for _ in range(iters):
        J = jac(e_hand, x, e)
        step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        if w_reg > 0:
            step += (np.eye(8) - np.linalg.pinv(J) @ J) @ (w_reg * (x0 - x))
        n = np.linalg.norm(step)
        if n > 0.2:
            step *= 0.2 / n
        x = np.clip(x + step, lo, hi)
        e = e_hand(x)
        if np.linalg.norm(e) < 1e-4:
            break
    base_err = np.linalg.norm(e[:3])
    for _ in range(elbow_iters):
        J1 = jac(e_hand, x, e)
        ee = e_elb(x)
        if np.linalg.norm(ee) < 0.005:
            break
        J2 = jac(e_elb, x, ee)
        N = np.eye(8) - np.linalg.pinv(J1, rcond=1e-3) @ J1
        g = N @ (J2.T @ ee)
        if np.linalg.norm(g) < 1e-8:
            break
        for a in (0.5, 0.25, 0.1):  # line search on the elbow error, guarded by the hand error
            xt = np.clip(x + a * g / max(1.0, np.linalg.norm(g) / 0.1), lo, hi)
            et = e_hand(xt)
            for _ in range(3):  # re-project onto the hand task
                Jt = jac(e_hand, xt, et)
                xt = np.clip(xt + Jt.T @ np.linalg.solve(Jt @ Jt.T + damping ** 2 * np.eye(6), et), lo, hi)
                et = e_hand(xt)
            if np.linalg.norm(et[:3]) <= base_err + hand_tol and np.linalg.norm(e_elb(xt)) < np.linalg.norm(ee):
                x, e = xt, et
                break
        else:
            break
    q = dict(q)
    q.update(zip(names, x))
    return q


ARM_SEEDS = [(j1, j3, j4) for j1 in (-0.4, 0.4, 1.2) for j3 in (-2.0, -0.7, 0.7, 2.0) for j4 in (-1.5, 0.8)]


def score_arm(o, side, q, q_prev, B_t, R_t, u_t, w_jump):
    W = o.k.fk(q)
    B, R = o.frame(side, W)
    he, re = np.linalg.norm(B - B_t), np.linalg.norm(rot_err(R, R_t))
    names = o.arm_names(side)
    jump = np.abs(np.array([q[n] - q_prev[n] for n in names])).max()
    ok = he < 0.01 and re < 0.06
    return (0 if ok else 1, np.degrees(swivel_err(o, side, W, u_t)) + w_jump * jump + 100 * he + 10 * re), ok


def pick_arm(o, side, q_prev, B_t, R_t, u_t, search=False, swivel_bad=35.0, w_jump=40.0):
    """Warm start from the previous frame (continuity); search the seed grid at the first frame, or when the warm
    start loses the hand target or drifts to a different elbow posture than GR1's. Best = hand exact, then smallest
    swivel error + penalty on the joint jump from the previous frame (no branch flips mid-motion)."""
    q = solve_arm(o, side, q_prev, B_t, R_t, u_t, iters=80 if search else 30)
    sc, ok = score_arm(o, side, q, q_prev, B_t, R_t, u_t, 0.0 if search else w_jump)
    sw = np.degrees(swivel_err(o, side, o.k.fk(q), u_t))
    if not search and ok and sw < swivel_bad:
        return q
    best, bsc = q, sc
    names = o.arm_names(side)
    for j1, j3, j4 in ARM_SEEDS:
        q2 = dict(q_prev)
        q2.update(dict(zip(names, [j1, 0.3, j3, j4, 0.0, 0.0, 0.0, 0.0])))
        q2 = solve_arm(o, side, q2, B_t, R_t, u_t, iters=100, w_reg=0.0, elbow_iters=10)
        sc2, _ = score_arm(o, side, q2, q_prev, B_t, R_t, u_t, 0.0 if search else w_jump)
        if sc2 < bsc:
            best, bsc = q2, sc2
    return best


def solve_fingers(o, side, q, tips_t, iters=25, w_vec=0.7, w_reg=0.02, damping=0.02):
    """tips_t: {finger: target tip in the palm-link frame}."""
    names = o.finger_names(side)
    names = [n for n in names if not n.endswith("_wrist")]
    lo = np.array([o.lo[n] for n in names]) + 1e-3
    hi = np.array([o.hi[n] for n in names]) - 1e-3
    x0 = np.clip(np.array([q[n] for n in names]), lo, hi)
    x = x0.copy()
    keys = (*FINGERS, "thumb")

    def res(xv):
        qq = dict(q)
        qq.update(zip(names, xv))
        W = o.k.fk(qq)
        Pi = np.linalg.inv(W[f"{side}_palm"])
        tp = {f: (Pi @ np.r_[p, 1.0])[:3] for f, p in o.tips(side, W).items()}
        r = [tips_t[f] - tp[f] for f in keys]
        r += [w_vec * ((tips_t[f] - tips_t["thumb"]) - (tp[f] - tp["thumb"])) for f in FINGERS]
        r.append(w_reg * (x0 - xv))
        return np.concatenate(r)

    e = res(x)
    for _ in range(iters):
        J = np.zeros((len(e), len(x)))
        for i in range(len(x)):
            d = x.copy()
            d[i] += 1e-5
            J[:, i] = (e - res(d)) / 1e-5
        step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
        n = np.linalg.norm(step)
        if n > 0.3:
            step *= 0.3 / n
        x = np.clip(x + step, lo, hi)
        e2 = res(x)
        if np.linalg.norm(e2) > np.linalg.norm(e) - 1e-8:
            e = e2
            break
        e = e2
    q = dict(q)
    q.update(zip(names, x))
    return q, float(np.sqrt(np.mean(e[:15] ** 2)) * np.sqrt(3))


THUMB = ("thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip")


def solve_thumb(o, side, q, dir_t, aperture_t, iters=40, w_ap=1.0, w_reg=0.05, damping=0.03):
    """ORCA thumb (4 joints): unit direction knuckle-centre -> thumb tip in the hand frame = dir_t, and
    |thumb tip - index tip| / (wrist->knuckle length) = aperture_t. Returns (q, residual)."""
    names = [f"{side}_{j}" for j in THUMB]
    lo = np.array([o.lo[n] for n in names]) + 1e-3
    hi = np.array([o.hi[n] for n in names]) - 1e-3
    x0 = np.clip(np.array([q[n] for n in names]), lo, hi)
    x = x0.copy()

    def res(xv):
        qq = dict(q)
        qq.update(zip(names, xv))
        W = o.k.fk(qq)
        Wr, B, _, _ = o.landmarks(side, W)
        _, R = o.frame(side, W)
        tp = o.tips(side, W)
        v = R.T @ (tp["thumb"] - B)
        ap = np.linalg.norm(tp["thumb"] - tp["index"]) / np.linalg.norm(B - Wr)
        return np.concatenate([dir_t - v / np.linalg.norm(v), [w_ap * (aperture_t - ap)], w_reg * (x0 - xv)])

    e = res(x)
    for _ in range(iters):
        J = np.zeros((len(e), 4))
        for i in range(4):
            dd = x.copy()
            dd[i] += 1e-5
            J[:, i] = (e - res(dd)) / 1e-5
        step = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
        n = np.linalg.norm(step)
        if n > 0.3:
            step *= 0.3 / n
        x = np.clip(x + step, lo, hi)
        e2 = res(x)
        if abs(np.linalg.norm(e) - np.linalg.norm(e2)) < 1e-7:
            e = e2
            break
        e = e2
    q = dict(q)
    q.update(zip(names, x))
    return q, float(np.linalg.norm(e[:4]))


def resample(T, dt, hz):
    t_src = np.arange(T) * dt
    t_out = np.arange(0, t_src[-1] + 1e-9, 1.0 / hz)
    return t_src, t_out


def retarget_demo(d, o, hz=25, verbose=True):
    # layout similarity
    Sg = {s: d.arm(s, 0)[0] for s in SIDES}
    W0 = o.k.fk(o.home)
    So = {s: o.shoulder(s, W0) for s in SIDES}
    def reach_g(s):
        S, E, Wr = d.arm(s, 0)
        return np.linalg.norm(E - S) + np.linalg.norm(Wr - E) + np.linalg.norm(d.frame(s, 0)[0] - Wr)

    def reach_o(s):
        e, w6 = o.elbow(s, W0), W0[f"{s}_yellowarm_link6"][:3, 3]
        return np.linalg.norm(e - So[s]) + np.linalg.norm(w6 - e) + np.linalg.norm(o.frame(s, W0)[0] - w6)

    s_scale = float(np.mean([reach_o(s) / reach_g(s) for s in SIDES]))  # y, z: total reach shoulder -> knuckles
    s_x = float(np.linalg.norm(So["left"] - So["right"]) / np.linalg.norm(Sg["left"] - Sg["right"]))  # shoulders
    Cg, Co = (Sg["left"] + Sg["right"]) / 2, (So["left"] + So["right"]) / 2
    Sv = np.array([s_x, s_scale, s_scale])
    mp = lambda p: Co + Sv * (p - Cg)  # noqa: E731
    upper_o = {s: np.linalg.norm(o.elbow(s, W0) - So[s]) for s in SIDES}

    # finger-length ratio (index base -> tip, open hand)
    tg = d.tips("left", 0)
    _, Bg, Ig, _ = d.landmarks("left", 0)
    Wq = o.k.fk({})
    _, Bo, Io, _ = o.landmarks("left", Wq)
    f_scale = np.linalg.norm(Wq["left_middle_fingertip"][:3, 3] - Bo) / np.linalg.norm(tg["middle"] - Bg)

    t_src, t_out = resample(d.T, d.dt, hz)
    q = dict(o.home)
    rows, errs = [], []
    t0 = time.time()
    for ti, tt in enumerate(t_out):
        a = min(int(tt / d.dt), d.T - 2)
        u = tt / d.dt - a
        lerp = lambda f: (1 - u) * f(a) + u * f(a + 1)  # noqa: E731
        obj = lerp(lambda k: d.obj_p[k])
        er = {}
        for s in SIDES:
            Bg_t = lerp(lambda k: d.frame(s, k)[0])
            Rg_t = d.frame(s, a if u < 0.5 else a + 1)[1]
            Sg_t, Eg_t, Wg_t = (lerp(lambda k: d.arm(s, k)[i]) for i in range(3))
            dist = np.linalg.norm(Bg_t - obj)
            w = float(np.clip((0.30 - dist) / 0.20, 0.0, 1.0))
            B_t = mp(Bg_t) + w * (1 - Sv) * (Bg_t - obj)
            u_t = swivel_dir(Sg_t, Eg_t, Wg_t)
            q_prev = dict(q)
            q = pick_arm(o, s, q_prev, B_t, Rg_t, u_t, search=(ti == 0))
            # fingers: index..pinky copy GR1's flexion angles (both hands are anthropomorphic, 90 deg MCP = 90 deg);
            # thumb: direction of the thumb tip from the knuckles + thumb-index aperture (GR1 thumb has 3 joints)
            gq = lambda n: float(lerp(lambda k: d.q[k, d.J.index(n)]))  # noqa: E731
            h = SIDES[s][1]
            for fg in FINGERS:
                q[f"{s}_{fg}_mcp"] = float(np.clip(-gq(f"{h}_{fg}_proximal_joint"), o.lo[f"{s}_{fg}_mcp"],
                                                   o.hi[f"{s}_{fg}_mcp"]))
                q[f"{s}_{fg}_pip"] = float(np.clip(-gq(f"{h}_{fg}_intermediate_joint"), max(0.0, o.lo[f"{s}_{fg}_pip"]),
                                                   o.hi[f"{s}_{fg}_pip"]))
                q[f"{s}_{fg}_abd"] = 0.0
            tips_g = {fg: lerp(lambda k: d.tips(s, k)[fg]) for fg in ("index", "thumb")}
            hand_g = np.linalg.norm(Bg_t - d.landmarks(s, a)[0])
            th_dir = Rg_t.T @ (tips_g["thumb"] - Bg_t)
            q, tip_rms = solve_thumb(o, s, q, th_dir / np.linalg.norm(th_dir),
                                     np.linalg.norm(tips_g["thumb"] - tips_g["index"]) / hand_g)
            W = o.k.fk(q)
            B, R = o.frame(s, W)
            er[s] = dict(hand_pos=float(np.linalg.norm(B - B_t)), hand_rot=float(np.linalg.norm(rot_err(R, Rg_t))),
                         elbow=swivel_err(o, s, W, u_t), tip_rms=tip_rms,
                         load=float(o.k.gravity_load(s, q).max()))
        rows.append([q[n] for n in o.names])
        errs.append(er)
        if verbose and ti % 25 == 0:
            print(f"  t={tt:5.2f}s " + " | ".join(
                f"{s}: hand {100 * e['hand_pos']:.1f}cm {np.degrees(e['hand_rot']):.0f}deg swivel {np.degrees(e['elbow']):.0f}deg "
                f"thumb {e['tip_rms']:.2f}" for s, e in er.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    obj_out = np.array([mp((1 - (tt / d.dt - min(int(tt / d.dt), d.T - 2))) * d.obj_p[min(int(tt / d.dt), d.T - 2)]
                           + (tt / d.dt - min(int(tt / d.dt), d.T - 2)) * d.obj_p[min(int(tt / d.dt), d.T - 2) + 1])
                        for tt in t_out])
    objq = np.array([d.obj_q[min(int(round(tt / d.dt)), d.T - 1)] for tt in t_out])
    return dict(q=np.array(rows, np.float32), t=t_out.astype(np.float32), obj_pos=obj_out.astype(np.float32),
                obj_quat_wxyz=objq.astype(np.float32), scale=Sv.tolist(), finger_scale=f_scale,
                layout=dict(C_gr1=Cg.tolist(), C_ours=Co.tolist(), scale_xyz=Sv.tolist(),
                            obj0_gr1=d.obj_p[0].tolist(), obj0_ours=mp(d.obj_p[0]).tolist()), errs=errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="*", default=None)
    ap.add_argument("--hz", type=float, default=25.0)
    a = ap.parse_args()
    h5py = h5()
    names = json.loads((GR1 / "gr1_names.json").read_text(encoding="utf-8"))
    f = h5py.File(GR1 / "dataset_annotated_gr1.hdf5", "r")
    o = Ours()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"source": "dataset_annotated_gr1.hdf5", "hz": a.hz, "demos": {}}
    for dn in a.demos or list(f["data"].keys()):
        print(f"[retarget] {dn}", flush=True)
        r = retarget_demo(GR1Demo(f["data"][dn], names), o, hz=a.hz)
        summ = {}
        for s in SIDES:
            for k in ("hand_pos", "hand_rot", "elbow", "tip_rms", "load"):
                v = np.array([e[s][k] for e in r["errs"]])
                summ[f"{s}_{k}_mean"], summ[f"{s}_{k}_max"] = round(float(v.mean()), 4), round(float(v.max()), 4)
        np.savez_compressed(OUT / f"{dn}.npz", q=r["q"], t=r["t"], obj_pos=r["obj_pos"],
                            obj_quat_wxyz=r["obj_quat_wxyz"], names=np.array(o.names),
                            layout=json.dumps(r["layout"]), errors=json.dumps(summ))
        report["demos"][dn] = {"frames": len(r["t"]), "layout": r["layout"], "finger_scale": r["finger_scale"], **summ}
        print(f"[retarget] {dn}: {json.dumps(summ)}", flush=True)
    rp = OUT / (f"report_{'_'.join(a.demos)}.json" if a.demos else "report.json")
    rp.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("[retarget] ->", OUT)


if __name__ == "__main__":
    main()
