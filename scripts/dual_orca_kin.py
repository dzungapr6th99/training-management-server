"""Kinematics of the dual YellowArm+ORCA robot from its URDF (numpy only; runs in env python and Isaac python).

  Kin.fk(q)                    -> {link: 4x4 world transform} (q: {joint: rad}, missing joints = 0)
  Kin.palm(side, q)            -> (grasp-centre position, R) ; R columns = (finger dir, palm normal, side axis)
  Kin.ik(side, pos, R, q0)     -> 8 joint values (arm joint1..7 + wrist) with damped least squares, limits respected
  grasp_pose(g, side)          -> {hand joint: rad}, g = 0 open ... 1 closed power grasp (synergy primitive)

Frames (robot base 'dual_base' = world when the robot stands at the origin): +Z up, left arm at -X, right arm at +X.
Limits/efforts = URDF + embodiment.yaml sim_overrides (Kin(sim=True), default): J1 range moved so the robot faces +Y
(thumbs forward, J1 increasing swings the arm forward), industrial-class arm torques. Kin(sim=False) = raw URDF (the
v1 dataset was recorded with it, robot facing -Y). See dual_robot_probe.py for the checks.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from mhh_paths import DUAL_URDF

SIDES = ("left", "right")
ARM = [f"yellowarm_joint{i}" for i in range(1, 8)]
FINGERS = ("index", "middle", "ring", "pinky")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(o):
    M = np.eye(4)
    if o is not None:
        M[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
        M[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
    return M


def _rot(axis, t):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def rot_err(R, Rt):
    """Rotation vector taking R to Rt (world frame)."""
    E = Rt @ R.T
    c = np.clip((np.trace(E) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if np.pi - th < 1e-4:  # ~180 deg: axis from the symmetric part
        w, V = np.linalg.eigh((E + np.eye(3)) / 2)
        return V[:, -1] * th
    return v * th / (2 * np.sin(th))


class Kin:
    def __init__(self, urdf=DUAL_URDF, sim=True):
        root = ET.parse(str(urdf)).getroot()
        self.joints = {}  # name -> (parent, child, T_origin, axis or None)
        self.children = {}
        self.limits = {}
        child_links = set()
        for j in root.findall("joint"):
            a = j.find("axis")
            mov = j.get("type") in ("revolute", "continuous", "prismatic")
            ax = np.array([float(v) for v in a.get("xyz").split()]) if (mov and a is not None) else None
            p, c = j.find("parent").get("link"), j.find("child").get("link")
            self.joints[j.get("name")] = (p, c, _T(j.find("origin")), ax)
            self.children.setdefault(p, []).append(j.get("name"))
            child_links.add(c)
            if mov:
                lim = j.find("limit")
                self.limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        self.root = next(l.get("name") for l in root.findall("link") if l.get("name") not in child_links)
        self.effort = {j.get("name"): float(j.find("limit").get("effort")) for j in root.findall("joint")
                       if j.get("type") == "revolute"}
        self.inertial = {}  # link -> (mass, com in link frame)
        for l in root.findall("link"):
            i = l.find("inertial")
            if i is not None and float(i.find("mass").get("value")) > 0:
                self.inertial[l.get("name")] = (float(i.find("mass").get("value")), _T(i.find("origin"))[:3, 3])
        if sim:
            ov = _overrides()
            for n, (lo, hi) in ov.get("joint_limits_rad", {}).items():
                self.limits[n] = (float(lo), float(hi))
            for n, e in ov.get("effort_nm", {}).items():
                self.effort[n] = float(e)
        self.parent_joint = {c: jn for jn, (p, c, _, _) in self.joints.items()}
        self._palm_local = {}
        for s in SIDES:
            self._palm_local[s] = self._calibrate_palm(s)

    # ---------------- forward kinematics ----------------
    def fk(self, q, base=None):
        W = {self.root: np.eye(4) if base is None else base}
        stack = [self.root]
        while stack:
            pl = stack.pop()
            for jn in self.children.get(pl, []):
                p, c, To, ax = self.joints[jn]
                M = W[p] @ To
                if ax is not None:
                    R = np.eye(4)
                    R[:3, :3] = _rot(ax, q.get(jn, 0.0))
                    M = M @ R
                W[c] = M
                stack.append(c)
        return W

    def chain(self, side):
        return [f"{side}_{j}" for j in ARM] + [f"{side}_wrist"]

    def _calibrate_palm(self, side):
        """Grasp frame in palm-link coordinates: origin between palm and curled fingers, x = finger direction,
        y = palm normal (side the fingers curl to), z = x cross y."""
        W0 = self.fk({})
        Wc = self.fk(grasp_pose(1.0, side))
        Pi = np.linalg.inv(W0[f"{side}_palm"])
        loc = lambda W, l: (Pi @ W[l])[:3, 3]  # noqa: E731 (palm does not move with the fingers)
        base = np.mean([loc(W0, f"{side}_{f}_pp") for f in FINGERS], 0)
        tip_open = np.mean([loc(W0, f"{side}_{f}_fingertip") for f in FINGERS], 0)
        tip_closed = np.mean([loc(Wc, f"{side}_{f}_fingertip") for f in FINGERS], 0)
        x = tip_open - base
        x /= np.linalg.norm(x)
        y = tip_closed - tip_open
        y -= x * (x @ y)
        y /= np.linalg.norm(y)
        z = np.cross(x, y)
        R = np.stack([x, y, z], 1)
        palm_c = np.zeros(3)  # palm link origin
        # grasp centre: ~55% along the open finger length, half-way between palm surface and curled tips
        off = base + 0.45 * (tip_open - base) + 0.5 * ((tip_closed - base) @ y) * y
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = off if np.isfinite(off).all() else palm_c
        return M

    def palm(self, side, q, W=None):
        W = W if W is not None else self.fk(q)
        M = W[f"{side}_palm"] @ self._palm_local[side]
        return M[:3, 3].copy(), M[:3, :3].copy()

    def _subtree(self, link):
        out, st = [], [link]
        while st:
            l = st.pop()
            out.append(l)
            st += [self.joints[j][1] for j in self.children.get(l, [])]
        return out

    def gravity_load(self, side, q):
        """Static gravity torque / URDF effort for each arm joint 1..7 (|value| > 1: the motor cannot hold)."""
        W = self.fk(q)
        out = []
        for jn in self.chain(side)[:7]:
            p, c, To, ax = self.joints[jn]
            Mj = W[p] @ To
            axis, org = Mj[:3, :3] @ (ax / np.linalg.norm(ax)), Mj[:3, 3]
            tau = 0.0
            for l in self._subtree(c):
                if l in self.inertial:
                    m, com = self.inertial[l]
                    pc = (W[l] @ np.r_[com, 1.0])[:3]
                    tau += np.cross(pc - org, [0, 0, -9.81 * m]) @ axis
            out.append(abs(tau) / self.effort[jn])
        return np.array(out)

    def ik_best(self, side, pos, R, seeds, **kw):
        """ik() from several seeds; returns the best (q8, pos_err, rot_err)."""
        best = None
        for s in seeds:
            r = self.ik(side, pos, R, q0=s, **kw)
            if best is None or r[1] + 0.05 * r[2] < best[1] + 0.05 * best[2]:
                best = r
            if r[1] < 2e-3 and r[2] < 0.03:
                break
        return best

    # ---------------- inverse kinematics ----------------
    def ik(self, side, pos, R=None, q0=None, iters=200, w_rot=0.3, damping=0.02, tol=1e-3, rest=None, w_rest=0.0):
        """Damped least squares on the 8-joint chain (arm 1..7 + wrist). R=None: position only.
        Returns (q8, pos_err_m, rot_err_rad)."""
        names = self.chain(side)
        lo = np.array([self.limits[n][0] for n in names]) + 0.02
        hi = np.array([self.limits[n][1] for n in names]) - 0.02
        q = np.clip(np.array(q0 if q0 is not None else np.zeros(8), float), lo, hi)
        rest = None if rest is None else np.asarray(rest, float)

        def err(qv):
            qq = dict(zip(names, qv))
            p, Rc = self.palm(side, qq)
            e = [pos - p]
            if R is not None:
                e.append(w_rot * rot_err(Rc, R))
            return np.concatenate(e)

        e = err(q)
        for _ in range(iters):
            if np.linalg.norm(e[:3]) < tol and (R is None or np.linalg.norm(e[3:]) / w_rot < 10 * tol):
                break
            J = np.zeros((len(e), 8))
            h = 1e-5
            for k in range(8):
                dq = q.copy()
                dq[k] += h
                J[:, k] = (e - err(dq)) / h  # d(-e)/dq = J of the pose
            A = J @ J.T + damping ** 2 * np.eye(len(e))
            step = J.T @ np.linalg.solve(A, e)
            if rest is not None and w_rest > 0:  # null-space pull toward the rest pose
                N = np.eye(8) - np.linalg.pinv(J) @ J
                step += N @ (w_rest * (rest - q))
            n = np.linalg.norm(step)
            if n > 0.3:
                step *= 0.3 / n
            q = np.clip(q + step, lo, hi)
            e = err(q)
        pe = float(np.linalg.norm(e[:3]))
        re = float(np.linalg.norm(e[3:]) / w_rot) if R is not None else 0.0
        return q, pe, re


# ---------------- grasp synergy 'power_v1' (top-down cage grasp of a ~50 mm object) ----------------
# Joint values (rad, same for both hands: the left URDF is the mirror with mirrored axes). Chosen from URDF FK, NOT
# from hardware (P3.1 will tune them):
#   g = 0    open: fingers straight, thumb folded back toward the wrist in the palm plane (at URDF zero the thumb
#            points 8.6 cm out of the palm and hits the table before the palm reaches the object)
#   g = 0.5  pre-shape: thumb tip at grasp-frame (-0.066, 0, 0) beside the object's near face, nothing more than
#            1.6 cm below the palm; fingers still straight. The thumb's swing from open to here dips ~6 cm below
#            the palm (its base is under the palm), so the expert pre-shapes while the hand is still high.
#   g = 1    closed: fingers curl onto the far face; thumb abducts to (-0.059, 0, 0) and squeezes the near face.
# Object centre in the grasp frame for this synergy: GRASP_OBJ_OFFSET (used by dual_orca_pick_plan).
HAND_OPEN = {"thumb_mcp": -0.85, "thumb_abd": -1.0, "thumb_pip": -0.75, "thumb_dip": 0.0,
             "index_mcp": 0.0, "index_pip": 0.0, "middle_mcp": 0.0, "middle_pip": 0.0,
             "ring_mcp": 0.0, "ring_pip": 0.0, "pinky_mcp": 0.0, "pinky_pip": 0.0,
             "index_abd": 0.0, "middle_abd": 0.0, "ring_abd": 0.0, "pinky_abd": 0.0}
THUMB_PRE = {"thumb_mcp": 0.68, "thumb_abd": -1.07, "thumb_pip": 1.22, "thumb_dip": 1.44}
HAND_CLOSED = {"thumb_mcp": 0.68, "thumb_abd": -0.72, "thumb_pip": 1.22, "thumb_dip": 1.44,
               "index_mcp": 1.17, "index_pip": 1.25, "middle_mcp": 1.17, "middle_pip": 1.25,
               "ring_mcp": 1.17, "ring_pip": 1.25, "pinky_mcp": 1.17, "pinky_pip": 1.25,
               "index_abd": 0.0, "middle_abd": 0.0, "ring_abd": 0.0, "pinky_abd": 0.0}
GRASP_OBJ_OFFSET = (-0.035, 0.0, 0.0)  # object centre in the grasp frame when grasped with power_v1
G_PRESHAPE = 0.5
HAND = ["wrist", "thumb_mcp", "thumb_abd", "thumb_pip", "thumb_dip", "index_abd", "index_mcp", "index_pip",
        "middle_abd", "middle_mcp", "middle_pip", "ring_abd", "ring_mcp", "ring_pip", "pinky_abd", "pinky_mcp",
        "pinky_pip"]
_LIM = None


def _overrides():
    import json

    from mhh_paths import DUAL_EMBODIMENT

    return json.load(open(DUAL_EMBODIMENT, encoding="utf-8")).get("sim_overrides", {})


def _limits():
    global _LIM
    if _LIM is None:
        _LIM = {}
        for j in ET.parse(str(DUAL_URDF)).getroot().findall("joint"):
            l = j.find("limit")
            if l is not None and j.get("type") == "revolute":
                _LIM[j.get("name")] = (float(l.get("lower")), float(l.get("upper")))
        for n, (lo, hi) in _overrides().get("joint_limits_rad", {}).items():
            _LIM[n] = (float(lo), float(hi))
    return _LIM


def grasp_pose(g, side):
    """Hand joint targets (the 16 hand joints except the wrist) for synergy value g in [0, 1] (see power_v1 above):
    0..G_PRESHAPE moves only the thumb (open -> pre-shape), G_PRESHAPE..1 closes the fingers and squeezes the thumb."""
    L = _limits()
    g = float(np.clip(g, 0.0, 1.0))
    a = min(g / G_PRESHAPE, 1.0)  # thumb pre-shape phase
    b = max((g - G_PRESHAPE) / (1.0 - G_PRESHAPE), 0.0)  # closing phase
    out = {}
    for j in HAND_CLOSED:
        lo, hi = L[f"{side}_{j}"]
        if j in THUMB_PRE:
            v = HAND_OPEN[j] + a * (THUMB_PRE[j] - HAND_OPEN[j]) + b * (HAND_CLOSED[j] - THUMB_PRE[j])
        else:
            v = HAND_OPEN[j] + b * (HAND_CLOSED[j] - HAND_OPEN[j])
        out[f"{side}_{j}"] = float(np.clip(v, lo + 1e-3, hi - 1e-3))
    return out


def grasp_closure(qmap, side):
    """Measured synergy value (inverse of grasp_pose, 0..1): thumb pre-shape progress below G_PRESHAPE, mean finger
    flexion (mcp/pip of the 4 fingers) above it."""
    fing = np.mean([(qmap[f"{side}_{f}_{j}"] - HAND_OPEN[f"{f}_{j}"]) / (HAND_CLOSED[f"{f}_{j}"] - HAND_OPEN[f"{f}_{j}"])
                    for f in FINGERS for j in ("mcp", "pip")])
    if fing > 0.02:
        return float(np.clip(G_PRESHAPE + (1 - G_PRESHAPE) * fing, 0.0, 1.0))
    th = (qmap[f"{side}_thumb_mcp"] - HAND_OPEN["thumb_mcp"]) / (THUMB_PRE["thumb_mcp"] - HAND_OPEN["thumb_mcp"])
    return float(np.clip(G_PRESHAPE * th, 0.0, G_PRESHAPE))


if __name__ == "__main__":
    k = Kin()
    for s in SIDES:
        p, R = k.palm(s, {})
        print(s, "grasp centre at zero pose", np.round(p, 3), "finger dir", np.round(R[:, 0], 2), "palm normal",
              np.round(R[:, 1], 2))
    # IK self-test: right hand, palm down, fingers forward, 20 cm in front of the right shoulder at 0.55 m
    Rt = np.stack([[0, -1, 0], [0, 0, -1], np.cross([0, -1, 0], [0, 0, -1])], 1).astype(float)
    for s, x in (("left", -0.25), ("right", 0.25)):
        q, pe, re = k.ik(s, np.array([x, -0.35, 0.55]), Rt, q0=np.r_[-0.6, 0, 0, -1.0, 0, 0, 0, 0])
        print(s, "IK", np.round(q, 2), f"pos err {pe * 1000:.1f} mm rot err {np.degrees(re):.1f} deg")
