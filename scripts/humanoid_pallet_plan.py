"""Pure-numpy planning for the humanoid parts-to-pallet task (PLAN.md 2.2-2.4). Numpy only, so it runs in the env
python (fast tests) and inside Isaac's python (the sim imports it):

  env\\python.exe scripts\\humanoid_pallet_plan.py --selftest [--level 1|2 --n 50]

- Scene layout: pallet (2x3 tray) and parts sampled ONLY inside the shared workspace of the approved configuration
  (`shared` mask of scripts/data/humanoid_workspace.npz), with clearance for the open fingers around the part to grasp.
  Split `wide` (eval only, PLAN 2.6): the target part lies OUTSIDE the training parts region, in a ring of WIDE_RING
  around it that one arm reaches and the head camera sees under the worst mounting error (`wide` mask).
- Arm choice from the reach map: a part reachable by one arm only -> that arm; in the shared area -> the arm whose IK
  solution is further from its joint limits.
- Top-down grasp / place IK with the same validity checks as humanoid_reach_probe (URDF limits, table, torso and
  shoulder-bar clearance), Cartesian straight lines for every vertical move, joint-space transits; every trajectory is
  timed so no joint exceeds `vel_frac` x the real (moteus) velocity limit.
- Arm-vs-arm clearance: the whole planned trajectory of the working arm against the other arm parked at its waiting pose
  (points on every link box of one arm vs the exact boxes of the other, both ways).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import humanoid_kin as hk  # noqa: E402
import humanoid_reach_probe as P  # noqa: E402

SIDES = ("L", "R")
TIP_BELOW_TCP = 0.117 - hk.TCP_OFFSET  # fingertips are 32 mm below the TCP (finger boxes span link7 z 0.064..0.117)
FINGER_REACH = 0.0375 + 0.011  # half the max opening + finger thickness: how far the open fingers reach from the TCP
ARM_CLEAR = 0.03  # m, required clearance between the two arms
SPLITS = ("train", "heldout_color", "heldout_shape", "wide")
# seed ranges (PLAN 2.6): episode i of a run uses seed + i. Training data only below SEED_DATA_END; evaluation on the
# fixed test seeds; exec-steps / checkpoint choices only on the validation seeds, never on the test seeds.
SEED_DATA_END = 100000
SEED_TEST = (100000, 100049)
SEED_VAL_START = 200000
WIDE_RING = 0.10  # m: the `wide` split places the target part up to this far outside the training parts region


def wrap(a, period):
    """Wrap angle(s) into [-period/2, period/2)."""
    return (np.asarray(a) + period / 2) % period - period / 2


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def scurve(tau_total, ramp, dt):
    """Progress samples (0..tau_total] of a trapezoidal velocity profile with raised-cosine ramps of `ramp` seconds:
    peak speed = 1 (one unit of tau per second), total time = tau_total + ramp (each ramp loses ramp/2)."""
    ramp = min(ramp, tau_total)
    T = tau_total + ramp
    n = max(1, int(np.ceil(T / dt)))
    t = np.linspace(0.0, T, 20 * n + 1)
    v = np.ones_like(t)
    if ramp > 0:
        v = np.minimum(v, 0.5 - 0.5 * np.cos(np.pi * np.clip(t / ramp, 0, 1)))
        v = np.minimum(v, 0.5 - 0.5 * np.cos(np.pi * np.clip((T - t) / ramp, 0, 1)))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    s *= tau_total / s[-1]
    return np.interp(np.arange(1, n + 1) * (T / n), t, s)


# ------------------------------------------------------------------ parts ------------------------------------------------
class Part:
    """One part instance: shape key, size scale, colour key (+ the shape spec from parts.json)."""

    def __init__(self, parts, shape, scale, color):
        self.shape, self.scale, self.color = shape, float(scale), color
        sp = parts["shapes"][shape]
        self.spec = sp
        self.dims = np.array(sp["dims"]) * self.scale  # full extent x, y, z (local frame)
        self.height = float(self.dims[2])
        self.grip_width = float(self.dims[1])
        self.radius = float(np.hypot(self.dims[0], self.dims[1]) / 2)  # footprint bounding circle
        self.period = np.deg2rad(sp["yaw_period_deg"]) if sp["yaw_period_deg"] else 0.0
        self.name = sp["name"]
        self.rgb = parts["colors"][color]["rgb"]
        if sp["kind"] == "box":
            vol = float(np.prod(self.dims))
        elif sp["kind"] == "cylinder":
            d = self.dims[1]
            ln = self.dims[0] if sp.get("axis") == "x" else self.dims[2]
            vol = float(np.pi * d * d / 4 * ln)
        else:  # hex prism, across flats = dims[1]
            vol = float(np.sqrt(3) / 2 * self.dims[1] ** 2 * self.dims[2])
        self.volume = vol
        self.mass = vol * parts["density_kg_m3"]

    def describe(self):
        return f"{self.color} {self.name}"

    def meta(self):
        return {"shape": self.shape, "scale": self.scale, "color": self.color, "name": self.name,
                "dims": np.round(self.dims, 4).tolist(), "mass_kg": round(self.mass, 4)}

    def grasp_yaws(self, part_yaw):
        """Candidate gripper yaws in [-180, 180) deg: fingers close along the part's local y; every yaw equivalent
        under the part's symmetry (both finger orders, so the wrist can take the short way). Round parts: every 45 deg."""
        if self.period == 0.0:
            return np.deg2rad(np.arange(-180.0, 180.0, 45.0))
        per = min(self.period, np.pi)
        c = wrap(part_yaw + np.arange(-12, 13) * per, 2 * np.pi)
        return np.unique(np.round(c, 6))

    def grasp_z(self):
        """TCP height for the grasp: fingertips ~3 mm above the table for low parts, mid-height for tall ones."""
        return float(np.clip(self.height * 0.5 + 0.012, 0.035, 0.050))

    def tilt_ok(self, quat, max_deg=30.0):
        R = quat_to_mat(quat)
        if self.spec["kind"] == "cylinder" and self.spec.get("axis") == "x":  # lying cylinder: its axis stays level
            return bool(np.degrees(np.arcsin(min(1.0, abs(R[2, 0])))) <= max_deg)
        return bool(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) <= max_deg)


# ------------------------------------------------------------------ world ------------------------------------------------
class World:
    def __init__(self, ctrl_hz=25.0, vel_frac=0.85, ramp=0.5, max_transit=8.0):
        self.cfg = hk.load_config()
        sel = self.cfg["selected"]
        assert sel, "humanoid.json has no 'selected' configuration (run humanoid_reach_probe.py --phase select)"
        self.sel = sel
        self.parts = json.load(open(os.path.join(HERE, "data", "parts.json"), encoding="utf-8"))
        ws = np.load(os.path.join(HERE, "data", "humanoid_workspace.npz"))
        self.ws = {k: ws[k] for k in ("xs", "ys", "left", "right", "shared")}
        self.ws_step = float(ws["step"])
        self.ctrl_hz, self.vel_frac, self.ramp, self.max_transit = ctrl_hz, vel_frac, ramp, max_transit
        self.h, self.sep = sel["mount_height"], sel["separation"]
        self.edge = sel["table_edge_x"]
        self.ch = hk.Chain()
        lim = self.cfg["arm_limits"]
        self.vmax = np.array(lim["velocity_used_rad_s"])
        self.lo, self.hi = self.ch.lo, self.ch.hi
        self.mount = {}
        for side, sy in (("L", 1.0), ("R", -1.0)):
            ax = sel["axis_left"] if side == "L" else P.mirror(sel["axis_left"])
            roll = sel["roll_left_deg"] if side == "L" else sel["roll_right_deg"]
            self.mount[side] = (hk.mount_rotation(ax, roll), np.array([0.0, sy * self.sep / 2, self.h]), sy)
        self.half_depth = sel["torso_depth"] / 2 + P.TORSO_INFLATE
        self.need = sel["torso_width"] / 2 + P.TORSO_INFLATE - self.sep / 2
        rng = np.random.default_rng(0)
        self.seeds = np.vstack([np.zeros(7), rng.uniform(self.lo * 0.85, self.hi * 0.85, (23, 7))])
        self.seeds[:, 0] = np.linspace(self.lo[0] + 0.1, self.hi[0] - 0.1, len(self.seeds))
        self.box_c = np.array([(np.array(lo) + np.array(hi)) / 2 for lo, hi in hk.LINK_BOXES])
        self.box_h = np.array([(np.array(hi) - np.array(lo)) / 2 for lo, hi in hk.LINK_BOXES])
        pl = self.parts["pallet"]
        self.pitch = pl["cell_inner"] + pl["wall"]
        self.pallet_size = np.array([pl["rows"] * self.pitch + pl["wall"], pl["cols"] * self.pitch + pl["wall"]])
        S = self.ws["shared"]
        X, Y = np.meshgrid(self.ws["xs"], self.ws["ys"], indexing="ij")
        self.shared_center = np.array([X[S].mean(), Y[S].mean()])
        # working branch: max-margin solution above the middle of the shared area; everything else stays near it
        self.q_work = {s: self.ik(s, (*self.shared_center, 0.12), 0.0) for s in SIDES}
        self.park = self._find_parks()
        self.pallet_nominal = self._find_pallet_nominal()
        self.ws["parts"] = self._parts_region()
        self.ws["wide"] = self._wide_region()
        self.ws["table"] = self.ws["left"] | self.ws["right"]

    # ---------------- masks ----------------
    def in_mask(self, key, xy):
        xy = np.atleast_2d(xy)
        xs, ys = self.ws["xs"], self.ws["ys"]
        ix = np.round((xy[:, 0] - xs[0]) / self.ws_step).astype(int)
        iy = np.round((xy[:, 1] - ys[0]) / self.ws_step).astype(int)
        inside = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        out = np.zeros(len(xy), bool)
        out[inside] = self.ws[key][ix[inside], iy[inside]]
        return out

    # ---------------- kinematics ----------------
    def fk_tcp(self, side, q):
        R, p, _ = self.mount[side]
        _, _, Rl, tcp = self.ch.fk(np.atleast_2d(q), R, p)
        return tcp, Rl

    def valid(self, side, q):
        """Joint limits + table / torso / shoulder-bar clearance (same test as the reach probe). q (N, 7)."""
        q = np.atleast_2d(q)
        R, _, sy = self.mount[side]
        lim_ok = np.all((q >= self.lo - 1e-6) & (q <= self.hi + 1e-6), 1)
        t_ok, ys_b = P.arm_clearance(self.ch, q, R, self.h, self.half_depth, sy, True)
        return lim_ok & t_ok & (ys_b >= self.need)

    def margin(self, q):
        """Distance (rad) of the closest joint to its limit."""
        q = np.atleast_2d(q)
        return np.minimum(q - self.lo, self.hi - q).min(1)

    def ik(self, side, pt, yaw, q_ref=None, iters=120, min_margin=0.03):
        """Best valid top-down IK solution for one TCP point: nearest (in time) to q_ref if given (among solutions at
        least min_margin from the joint limits, if any), else the one furthest from the joint limits. q (7,) or None."""
        R, p, _ = self.mount[side]
        seeds = self.seeds if q_ref is None else np.vstack([q_ref, self.seeds])
        q, ok, _, _ = self.ch.ik_down(np.array([pt], float), yaw, R, p, seeds, iters=iters)
        q, ok = q[0], ok[0]
        if not ok.any():
            return None
        q = q[ok]
        q = q[self.valid(side, q)]
        if not len(q):
            return None
        mg = self.margin(q)
        if q_ref is None:
            return q[np.argmax(mg)]
        if (mg >= min_margin).any():
            q = q[mg >= min_margin]
        return q[np.argmin((np.abs(q - q_ref) / self.vmax).max(1))]

    def line(self, side, p0, p1, yaw, q0, step=0.004, yaw1=None):
        """Straight TCP line p0 -> p1, top-down, yaw fixed (or turning linearly to yaw1 the short way), IK seeded point
        by point from q0 so the joints move continuously. Returns (K, 7) or None."""
        R, p, _ = self.mount[side]
        dyaw = 0.0 if yaw1 is None else float(wrap(yaw1 - yaw, 2 * np.pi))
        n = max(1, int(np.ceil(max(np.linalg.norm(np.subtract(p1, p0)) / step, abs(dyaw) / 0.05))))
        out, q = [], np.asarray(q0, float)
        for k in range(1, n + 1):
            pt = np.asarray(p0) + (np.asarray(p1) - np.asarray(p0)) * k / n
            qs, ok, _, _ = self.ch.ik_down(pt[None], yaw + dyaw * k / n, R, p, q[None], iters=40)
            if not ok[0, 0] or not self.valid(side, qs[0])[0]:
                return None
            q = qs[0, 0]
            out.append(q)
        return np.array(out)

    def retime(self, path, min_seconds=0.0):
        """Joint path (K, 7) (first row = start) -> control-step targets (T, 7): trapezoidal speed with raised-cosine
        ramps, no joint faster than vel_frac x its velocity limit; at least min_seconds long."""
        path = np.asarray(path, float)
        if len(path) < 2:
            return path[1:]
        d = (np.abs(np.diff(path, axis=0)) / (self.vel_frac * self.vmax)).max(1)  # seconds per segment at full speed
        tau = np.concatenate([[0.0], np.cumsum(d)])
        if tau[-1] < 1e-9:
            return path[-1:].repeat(max(1, int(np.ceil(min_seconds * self.ctrl_hz))), 0)
        slow = max(1.0, (min_seconds - self.ramp) / tau[-1]) if min_seconds > 0 else 1.0  # stretch to min_seconds
        s = scurve(tau[-1] * slow, self.ramp, 1.0 / self.ctrl_hz) / slow
        n = len(s)
        out = np.empty((n, 7))
        for j in range(7):
            out[:, j] = np.interp(s, tau, path[:, j])
        return out

    # ---------------- arm vs arm ----------------
    def link_frames(self, side, q):
        R, p, _ = self.mount[side]
        _, _, _, _, Rs, ps = self.ch.fk(np.atleast_2d(q), R, p, frames=True)
        Rb = np.concatenate([Rs, Rs[:, 6:7]], 1)
        pb = np.concatenate([ps, ps[:, 6:7]], 1)
        return Rb, pb  # (N, 8, 3, 3), (N, 8, 3)

    def arm_distance(self, side_a, qa, side_b, qb, links_b=slice(0, 8)):
        """Smallest distance between arm a (configs qa (N, 7)) and arm b (one config qb; only its links `links_b`, of
        link1..7 + fingers): sample points of each arm against the exact link boxes of the other, both ways. (N,)"""
        qa = np.atleast_2d(qa)
        out = np.full(len(qa), np.inf)
        pa_all = self.ch.collision_points(qa, *self.mount[side_a][:2])  # (N, 216, 3)
        pb = self.ch.collision_points(np.atleast_2d(qb), *self.mount[side_b][:2])[0].reshape(8, 27, 3)[links_b]
        pb = pb.reshape(-1, 3)
        Rb, tb = self.link_frames(side_b, qb)
        Ra, ta = self.link_frames(side_a, qa)
        for i0 in range(0, len(qa), 64):
            sl = slice(i0, i0 + 64)
            d1 = self._pts_to_boxes(pa_all[sl], Rb[0][links_b], tb[0][links_b], links_b)
            d2 = np.stack([self._pts_to_boxes(pb[None], Ra[n], ta[n])[0] for n in range(sl.start, min(sl.stop, len(qa)))])
            out[sl] = np.minimum(d1, d2)
        return out

    def _pts_to_boxes(self, pts, Rb, tb, links=slice(0, 8)):
        """pts (N, K, 3) world, boxes `links` with rotations Rb (B, 3, 3) and origins tb (B, 3) -> min distance (N,)."""
        rel = pts[:, None, :, :] - tb[None, :, None, :]  # (N, B, K, 3)
        loc = np.einsum("bji,nbkj->nbki", Rb, rel) - self.box_c[links][None, :, None, :]
        d = np.linalg.norm(np.maximum(np.abs(loc) - self.box_h[links][None, :, None, :], 0.0), axis=-1)
        return d.reshape(len(pts), -1).min(1)

    # ---------------- waiting pose ----------------
    def gripper_yaw(self, side, q):
        """Yaw of a top-down gripper (inverse of hk.down_rot)."""
        Rl = self.fk_tcp(side, q)[1][0]
        return float(np.arctan2(-Rl[0, 1], Rl[1, 1]))

    def transit(self, side, q0, p1, yaw1, step=0.01):
        """Free-space move from configuration q0 to the top-down TCP pose (p1, yaw1): a Cartesian line (continuous IK,
        no sweeping elbow) if possible, else a joint-space line to the nearest IK solution. Returns (path (K, 7) incl.
        q0, kind) or (None, None)."""
        tcp0, R0 = self.fk_tcp(side, q0)
        if R0[0][2, 2] < -0.99:  # starts top-down: try a Cartesian line first
            path = self.line(side, tcp0[0], p1, self.gripper_yaw(side, q0), q0, step=step, yaw1=yaw1)
            if path is not None:
                return np.vstack([q0, path]), "cartesian"
        q1 = self.ik(side, p1, yaw1, q_ref=q0)
        if q1 is None:
            return None, None
        # a joint-space line must stay clear of table / torso / bar all the way, not only at its ends
        n = max(2, int(np.ceil(np.abs(q1 - q0).max() / 0.05)))
        mid = q0 + (q1 - q0) * np.linspace(0, 1, n + 1)[:, None]
        if not self.valid(side, mid).all():
            return None, None
        return np.vstack([q0, q1]), "joint"

    def time_to(self, qa, qb):
        """Seconds for the slowest joint between two configurations at vel_frac x the velocity limit."""
        return float((np.abs(np.asarray(qb) - np.asarray(qa)) / (self.vel_frac * self.vmax)).max())

    def _work_configs(self, side, step=2):
        """Configurations of one arm working over the shared area, where all parts and the pallet are (grasp,
        approach and carry heights, two yaws)."""
        X, Y = np.meshgrid(self.ws["xs"], self.ws["ys"], indexing="ij")
        m = self.ws["shared"][::step, ::step]
        qs = []
        for x, y in np.stack([X[::step, ::step][m], Y[::step, ::step][m]], 1):
            for z in (0.035, 0.16):
                for yaw in (0.0, np.pi / 2):
                    q = self.ik(side, (x, y, z), yaw, q_ref=self.q_work[side], iters=80)
                    if q is not None:
                        qs.append(q)
        return np.array(qs)

    def _find_parks(self, min_margin=0.05):
        """Waiting pose of each arm: top-down, out on the arm's own side beyond the shared area (|y| >= 0.30 m), in the
        working IK branch (q_work), >= min_margin rad from every joint limit, reachable from q_work by a Cartesian line.
        Score: the fraction of the OTHER arm's working configurations over the shared area that keep >= ARM_CLEAR from
        it (links 1-2 of the waiting arm are left out: they sit at the shoulder whatever the pose), then the smallest
        distance. Cached in scripts/data/humanoid_park.json (keyed by the body config)."""
        cache = os.path.join(HERE, "data", "humanoid_park.json")
        key = json.dumps({k: self.sel[k] for k in ("mount", "mount_height", "separation", "torso_table_gap",
                                                    "roll_left_deg", "roll_right_deg")}, sort_keys=True)
        if os.path.exists(cache):
            c = json.load(open(cache, encoding="utf-8"))
            if c.get("key") == key:
                self.park_clearance = c["clearance"]
                return {s: np.array(c["park"][s]) for s in SIDES}
        work = {s: self._work_configs(s) for s in SIDES}
        parks, clear = {}, {}
        for s in SIDES:
            o = "R" if s == "L" else "L"
            _, _, sy = self.mount[s]
            pw = self.tcp_of_work = self.fk_tcp(s, self.q_work[s])[0][0]
            cands = []
            for x in (0.15, 0.25, 0.35, 0.45, 0.55, 0.65):
                for yo in (0.30, 0.38, 0.46, 0.54):
                    for z in (0.18, 0.25, 0.32):
                        for yaw in (0.0, np.pi / 2):
                            q = self.ik(s, (x, sy * yo, z), yaw, q_ref=self.q_work[s], min_margin=min_margin)
                            if q is None or self.margin(q)[0] < min_margin:
                                continue
                            d = self.arm_distance(o, work[o], s, q, links_b=slice(2, 8))
                            cands.append([float((d >= ARM_CLEAR).mean()), float(d.min()), q, (x, yo, z, yaw)])
            cands.sort(key=lambda c: (-c[0], -c[1]))
            best = next((c for c in cands if self.line(s, pw, self.fk_tcp(s, c[2])[0][0], 0.0, self.q_work[s], step=0.01,
                                                         yaw1=c[3][3]) is not None), None)
            assert best is not None, f"no waiting pose for arm {s}"
            parks[s], clear[s] = best[2], {"free_fraction": round(best[0], 3), "min_m": round(best[1], 3),
                                           "tcp_yaw": best[3]}
        self.park_clearance = clear
        json.dump({"key": key, "park": {s: parks[s].tolist() for s in SIDES}, "clearance": clear,
                   "_doc": "waiting poses (rad) from humanoid_pallet_plan.World._find_parks; clearance = min distance (m) "
                           "from links 3.. of the waiting arm to the other arm working over the shared area"},
                  open(cache, "w", encoding="utf-8"), indent=1)
        return parks

    def _parts_region(self, max_transit=4.0):
        """Where parts may be placed: shared-area cells from which, for EVERY grasp-yaw class (0/45/90/135 deg; both
        finger orders tried), at least one arm gets there from its waiting pose by a Cartesian transit (no IK-branch
        flip) within max_transit seconds and can go straight down to the grasp height. Near the torso the forward45 arms
        must flip the elbow to grasp top-down (6-11 s joint moves that sweep towards the other arm), so those cells are
        left out. Cached in scripts/data/humanoid_parts_region.npz."""
        cache = os.path.join(HERE, "data", "humanoid_parts_region.npz")
        park_key = json.dumps({s: np.round(self.park[s], 4).tolist() for s in SIDES})
        if os.path.exists(cache):
            c = np.load(cache)
            if str(c["park_key"]) == park_key and c["mask"].shape == self.ws["shared"].shape:
                return c["mask"]
        S = self.ws["shared"]
        mask = np.zeros_like(S)
        for i, j in np.argwhere(S):
            x, y = self.ws["xs"][i], self.ws["ys"][j]
            ok_all = True
            for yc in np.deg2rad([0.0, 45.0, 90.0, 135.0]):
                ok = False
                for s in SIDES:
                    for yaw in (yc, yc - np.pi):
                        path, kind = self.transit(s, self.park[s], (x, y, 0.12), yaw)
                        if kind != "cartesian" or len(self.retime(path)) / self.ctrl_hz > max_transit:
                            continue
                        if self.line(s, (x, y, 0.12), (x, y, 0.035), yaw, path[-1]) is not None:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    ok_all = False
                    break
            mask[i, j] = ok_all
        np.savez_compressed(cache, mask=mask, park_key=park_key)
        return mask

    def _wide_region(self, ring=WIDE_RING, fov_margin_deg=1.0):
        """Target area of the `wide` eval split: grid cells outside the training parts region, at most `ring` from it,
        reachable by at least one arm, and fully in the head-camera image (a 80 x 80 x 50 mm box around the cell centre,
        FOV - fov_margin_deg) under every corner of the DR mounting-error box (+/- d_pos, +/- d_rot). Whether the
        expert really gets there with a given part and yaw is decided per layout by the planner (else resampled)."""
        import itertools

        hc = self.sel["head_camera"]
        dr = json.load(open(os.path.join(HERE, "data", "domain_rand.json"), encoding="utf-8"))["head_camera"]
        dp = np.array(list(itertools.product((-dr["d_pos_m"], dr["d_pos_m"]), repeat=3)))
        da = np.array(list(itertools.product((-dr["d_rot_deg"], dr["d_rot_deg"]), repeat=3)))
        d_pos, d_rot = np.repeat(dp, len(da), 0), np.tile(da, (len(dp), 1))
        X, Y = np.meshgrid(self.ws["xs"], self.ws["ys"], indexing="ij")
        Pm = self.ws["parts"]
        pp = np.stack([X[Pm], Y[Pm]], 1)
        dist = np.sqrt(((np.stack([X, Y], -1)[:, :, None] - pp[None, None]) ** 2).sum(-1)).min(-1)
        cand = (self.ws["left"] | self.ws["right"]) & ~Pm & (dist <= ring + 1e-9)
        box = np.array([[dx, dy, z] for dx in (-0.04, 0.04) for dy in (-0.04, 0.04) for z in (0.0, 0.05)])
        mask = np.zeros_like(cand)
        for i, j in np.argwhere(cand):
            need = P.camera_need(np.array(hc["pos"]), hc["pitch_deg"], box + [X[i, j], Y[i, j], 0.0], d_pos, d_rot)
            mask[i, j] = need.max() <= hc["fov_deg"] - fov_margin_deg
        return mask

    # ---------------- pallet ----------------
    def pallet_cells(self, center, yaw):
        """World xy of the 6 cell centres, (rows * cols, 2), row-major (row 0 nearest the robot)."""
        pl = self.parts["pallet"]
        u = (np.arange(pl["rows"]) - (pl["rows"] - 1) / 2) * self.pitch
        v = (np.arange(pl["cols"]) - (pl["cols"] - 1) / 2) * self.pitch
        U, V = np.meshgrid(u, v, indexing="ij")
        c, s = np.cos(yaw), np.sin(yaw)
        return np.stack([center[0] + c * U - s * V, center[1] + s * U + c * V], -1).reshape(-1, 2)

    def pallet_points(self, center, yaw, margin=0.0, step=0.01):
        hx, hy = self.pallet_size / 2 + margin
        u, v = np.meshgrid(np.arange(-hx, hx + 1e-9, step), np.arange(-hy, hy + 1e-9, step), indexing="ij")
        c, s = np.cos(yaw), np.sin(yaw)
        return np.stack([center[0] + c * u - s * v, center[1] + s * u + c * v], -1).reshape(-1, 2)

    def pallet_fits(self, center, yaw):
        """The inner area of all 6 cells (the tray minus its 6 mm outer rim) lies in the shared workspace."""
        return bool(self.in_mask("shared", self.pallet_points(center, yaw, -self.parts["pallet"]["wall"], 0.005)).all())

    def _find_pallet_nominal(self):
        """Nominal pallet centre: every jittered pose (+/- jitter_xy, +/- jitter_yaw) fits in the shared area; the
        feasible centre closest to the middle of all feasible centres."""
        pl = self.parts["pallet"]
        j, jy = pl["jitter_xy"], np.deg2rad(pl["jitter_yaw_deg"])
        good = []
        for x in np.arange(0.40, 0.70, 0.005):
            for y in np.arange(-0.06, 0.0601, 0.005):
                if all(self.pallet_fits((x + dx, y + dy), dyaw) for dx in (-j, 0, j) for dy in (-j, 0, j)
                       for dyaw in (-jy, 0, jy)):
                    good.append((float(x), float(y)))
        assert good, "the pallet (with its jitter) does not fit in the shared area"
        good = np.array(good)
        return good[np.argmin(np.linalg.norm(good - good.mean(0), axis=1))]

    def dist_to_pallet(self, xy, center, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        d = np.atleast_2d(xy) - center
        u, v = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
        hx, hy = self.pallet_size / 2
        return np.hypot(np.maximum(np.abs(u) - hx, 0), np.maximum(np.abs(v) - hy, 0))

    def in_cell(self, xy, center, yaw, cell):
        """Is xy inside the inner area of cell `cell` (row-major index)?"""
        pl = self.parts["pallet"]
        cc = self.pallet_cells(center, yaw)[cell]
        c, s = np.cos(yaw), np.sin(yaw)
        d = np.asarray(xy[:2]) - cc
        u, v = c * d[0] + s * d[1], -s * d[0] + c * d[1]
        return bool(max(abs(u), abs(v)) < pl["cell_inner"] / 2)

    def cell_of(self, xy, center, yaw):
        for k in range(len(self.pallet_cells(center, yaw))):
            if self.in_cell(xy, center, yaw, k):
                return k
        return None

    # ---------------- layout ----------------
    def part_footprint(self, part, xy, yaw):
        hx, hy = part.dims[0] / 2, part.dims[1] / 2
        c, s = np.cos(yaw), np.sin(yaw)
        loc = np.array([[0, 0], [hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy], [hx, 0], [-hx, 0], [0, hy], [0, -hy]])
        return np.asarray(xy) + loc @ np.array([[c, s], [-s, c]])

    def sample_parts(self, rng, split="train", n=1, exclude=()):
        """n distinct (shape, colour) parts, also distinct from the parts in `exclude`. train: train shapes and colours;
        heldout_shape / heldout_color: only the heldout shapes / colours (the other attribute from the train set);
        wide: train shapes and colours (only the positions differ)."""
        assert split in SPLITS, split
        sh = [k for k, v in self.parts["shapes"].items() if v["split"] == ("heldout" if split == "heldout_shape" else "train")]
        cols = [k for k, v in self.parts["colors"].items() if v["split"] == ("heldout" if split == "heldout_color" else "train")]
        out, used = [], {(p.shape, p.color) for p in exclude}
        while len(out) < n:
            s_, c_ = sh[rng.integers(len(sh))], cols[rng.integers(len(cols))]
            if (s_, c_) in used:
                continue
            used.add((s_, c_))
            out.append(Part(self.parts, s_, self.parts["sizes"][rng.integers(len(self.parts["sizes"]))], c_))
        return out

    def sample_layout(self, rng, parts, target, max_tries=400, split="train"):
        """Pallet pose + part poses (xy, yaw), all in the shared area. Returns (pallet_center, pallet_yaw, poses) or None.
        Part `target` gets room for the open fingers; the others only must not touch anything. Split `wide`: the target
        centre in the `wide` mask, the other parts in the training region or the wide ring, footprints on the reachable
        table instead of the shared area."""
        pl = self.parts["pallet"]
        pc = self.pallet_nominal + rng.uniform(-pl["jitter_xy"], pl["jitter_xy"], 2)
        pyaw = float(np.deg2rad(rng.uniform(-pl["jitter_yaw_deg"], pl["jitter_yaw_deg"])))
        X, Y = np.meshgrid(self.ws["xs"], self.ws["ys"], indexing="ij")
        foot = "table" if split == "wide" else "shared"
        order = [target] + [i for i in range(len(parts)) if i != target]
        poses = [None] * len(parts)
        for i in order:
            p = parts[i]
            keys = (("wide",) if i == target else ("parts", "wide")) if split == "wide" else ("parts",)
            S = np.logical_or.reduce([self.ws[k] for k in keys])
            cells = np.stack([X[S], Y[S]], 1)
            for _ in range(max_tries):
                xy = cells[rng.integers(len(cells))] + rng.uniform(-self.ws_step / 2, self.ws_step / 2, 2)
                yaw = float(rng.uniform(-np.pi, np.pi))
                if not self.in_mask(foot, self.part_footprint(p, xy, yaw)).all() or \
                        not any(self.in_mask(k, xy)[0] for k in keys):
                    continue
                need_pal = FINGER_REACH + 0.006 if i == target else p.radius + 0.01
                if self.dist_to_pallet(xy, pc, pyaw)[0] < need_pal:
                    continue
                ok = True
                for j, pj in enumerate(parts):
                    if poses[j] is None:
                        continue
                    d = np.linalg.norm(xy - poses[j][0])
                    gap = (FINGER_REACH + 0.006 + pj.radius) if i == target else \
                        (FINGER_REACH + 0.006 + p.radius) if j == target else (p.radius + pj.radius + 0.01)
                    if d < gap:
                        ok = False
                        break
                if ok:
                    poses[i] = (xy, yaw)
                    break
            if poses[i] is None:
                return None
        return pc, pyaw, poses

    def sample_task_parts(self, rng, split, n):
        """The n table parts of an episode and the index of the target. train / wide: n distinct training parts, random
        target. heldout_color / heldout_shape: only the TARGET has the heldout colour / shape, the other parts are
        training parts (there are too few heldout combinations for distinct parts, and in L2 the policy must pick the
        unseen part out of familiar ones)."""
        if split in ("train", "wide"):
            parts = self.sample_parts(rng, n=n)
            return parts, int(rng.integers(len(parts)))
        tgt = int(rng.integers(n))
        target = self.sample_parts(rng, split=split, n=1)[0]
        others = self.sample_parts(rng, n=n - 1, exclude=[target])
        return others[:tgt] + [target] + others[tgt:], tgt

    def sample_prefill(self, rng, table_parts, n, pallet_c, pallet_yaw):
        """n parts already lying in n random pallet cells at the start: distinct (shape, colour) from the table parts
        (the L2 command stays unambiguous) and from each other; turned to the pallet axes (+/- 8 deg, any multiple of
        90 deg) and jittered, with the whole footprint >= 3 mm inside the cell walls. Training parts in every split.
        Returns [(Part, cell, xy, yaw)]."""
        if n <= 0:
            return []
        pl = self.parts["pallet"]
        parts = self.sample_parts(rng, n=n, exclude=table_parts)
        cells = rng.permutation(pl["rows"] * pl["cols"])[:n]
        cc = self.pallet_cells(pallet_c, pallet_yaw)
        c, s = np.cos(pallet_yaw), np.sin(pallet_yaw)
        lim = pl["cell_inner"] / 2 - 0.003
        out = []
        for p, k in zip(parts, cells):
            yaw0 = pallet_yaw + int(rng.integers(4)) * np.pi / 2
            xy, yaw = cc[k], yaw0  # centred and aligned always fits (largest footprint 72 mm < 74 mm)
            for _ in range(60):
                xy_t = cc[k] + rng.uniform(-0.008, 0.008, 2)
                yaw_t = yaw0 + np.deg2rad(rng.uniform(-8.0, 8.0))
                d = self.part_footprint(p, xy_t, yaw_t) - cc[k]
                if np.all(np.abs(np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)) < lim):
                    xy, yaw = xy_t, yaw_t
                    break
            out.append((p, int(k), np.asarray(xy, float), float(wrap(yaw, 2 * np.pi))))
        return out

    def release_clear(self, pp, fill, margin=0.005):
        """Do the open fingers and the wrist at the release pose (the lowest point of the place) keep `margin` from the
        parts already in the pallet? Parts are taken as upright cylinders of their footprint's bounding radius."""
        if not fill:
            return True
        R = hk.down_rot(pp["place_yaw"])
        half = pp["release_open"] * 0.0375 + 0.011
        g = [np.linspace(-0.012, 0.012, 3), np.linspace(-half, half, 9), np.linspace(0.064, 0.117, 3)]
        fing = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        lo7, hi7 = hk.LINK_BOXES[6]
        g = [np.linspace(lo7[0], hi7[0], 4), np.linspace(lo7[1], hi7[1], 9), [hi7[2]]]
        wrist = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
        pts = pp["release_tcp"] + (np.vstack([fing, wrist]) - np.array([0.0, 0.0, hk.TCP_OFFSET])) @ R.T
        top0 = self.parts["pallet"]["base"]
        for p, _, xy, _ in fill:
            near = np.linalg.norm(pts[:, :2] - xy, axis=1) < p.radius + margin
            if np.any(near & (pts[:, 2] < top0 + p.height + margin)):
                return False
        return True

    # ---------------- task plan ----------------
    def plan_pick(self, side, part, xy, part_yaw, q_start):
        """Best grasp yaw for this arm and the pick trajectory. Returns dict or None."""
        zg = part.grasp_z()
        z_above = max(0.12, part.height + 0.07)
        y0 = self.gripper_yaw(side, q_start)
        best = None
        for yaw in sorted(part.grasp_yaws(part_yaw), key=lambda a: abs(wrap(a - y0, 2 * np.pi))):
            path, kind = self.transit(side, q_start, (xy[0], xy[1], z_above), yaw)
            if path is None:
                continue
            qa = path[-1]
            down = self.line(side, (xy[0], xy[1], z_above), (xy[0], xy[1], zg), yaw, qa)
            if down is None:
                continue
            qg = down[-1]
            m = float(min(self.margin(qg)[0], self.margin(qa)[0]))
            tr = self.retime(path)
            t = len(tr) / self.ctrl_hz
            if t > self.max_transit or m < 0.02:  # a long sweeping transit, or a pose right at a joint limit
                continue
            if best is not None and best["transit_kind"] == "cartesian" and kind != "cartesian":
                continue  # a continuous Cartesian transit beats any joint-space one
            if best is None or (kind == "cartesian" and best["transit_kind"] != "cartesian") or                     t < best["transit_s"] - 0.2:  # the grasp yaw with the quickest transit
                best = {"yaw": float(yaw), "q_above": qa, "q_grasp": qg, "margin": m, "z_grasp": zg, "z_above": z_above,
                        "transit_s": t, "transit_kind": kind, "transit": tr,
                        "down": self.retime(np.vstack([qa, down]), min_seconds=(z_above - zg) / 0.08)}
            if best is not None and best["transit_kind"] == "cartesian" and len(best["transit"]) > 0 and t < 3.0:
                break  # good enough: keep planning time down
        return best

    def plan_place(self, side, part, q_grasp, off_g, part_yaw_now, grasp_yaw, pallet_c, pallet_yaw, cell,
                   z_carry=0.16, q_from=None):
        """Lift -> carry over `cell` -> lower -> release -> up -> back to the waiting pose. off_g = part centre in the
        gripper (link7) frame. The part is turned so that its local axes line up with the pallet's.
        q_from: re-plan after the lift (the arm is already at carry height there; 'lift' is then empty)."""
        pl = self.parts["pallet"]
        if q_from is None:
            xy_g = self.fk_tcp(side, q_grasp)[0][0]
            lift = self.line(side, xy_g, (xy_g[0], xy_g[1], z_carry), grasp_yaw, q_grasp)
            if lift is None:
                return None
            lift_t = self.retime(np.vstack([q_grasp, lift]), min_seconds=(z_carry - xy_g[2]) / 0.10)
        else:
            lift = np.asarray(q_from, float)[None]
            lift_t = lift[:0]
        dyaw = wrap(pallet_yaw - part_yaw_now, np.pi / 2)  # cells are square: line the part up to 90 deg
        place_yaw = float(wrap(grasp_yaw + dyaw, np.pi))  # gripper is symmetric under 180 deg
        Rp = hk.down_rot(place_yaw)
        cc = self.pallet_cells(pallet_c, pallet_yaw)[cell]
        wall_top = pl["base"] + pl["wall_height"]
        # release height: fingertips clear the cell walls; the part drops at most a little more than a wall height
        tip_above_bottom = part.grasp_z() - TIP_BELOW_TCP
        bottom = max(pl["base"] + 0.004, wall_top + 0.004 - tip_above_bottom)
        obj_target = np.array([cc[0], cc[1], bottom + part.height / 2])
        tcp_rel = obj_target - Rp @ off_g
        tcp_over = np.array([tcp_rel[0], tcp_rel[1], z_carry])
        carry, carry_kind = self.transit(side, lift[-1], tcp_over, place_yaw)
        if carry is None:
            return None
        q_over = carry[-1]
        lower = self.line(side, tcp_over, tcp_rel, place_yaw, q_over)
        if lower is None:
            return None
        up = self.line(side, tcp_rel, tcp_rel + np.array([0.0, 0.0, 0.08]), place_yaw, lower[-1])
        if up is None:
            return None
        pk = self.park[side]
        home, _ = self.transit(side, up[-1], self.fk_tcp(side, pk)[0][0], self.gripper_yaw(side, pk))
        if home is None:
            return None
        home = np.vstack([home, pk])  # small final joint correction onto the exact waiting pose
        return {
            "lift": lift_t,
            "carry": self.retime(carry), "carry_kind": carry_kind,
            "lower": self.retime(np.vstack([q_over, lower]), min_seconds=(z_carry - tcp_rel[2]) / 0.08),
            "up": self.retime(np.vstack([lower[-1], up]), min_seconds=0.08 / 0.10),
            "home": self.retime(home),
            "place_yaw": place_yaw, "release_tcp": tcp_rel, "obj_target": obj_target,
            "release_open": float(np.clip((part.grip_width + 0.02) / 0.075, 0.0, 1.0)),
        }

    def nominal_offset(self, part, yaw_grasp):
        """Part centre in the gripper frame for a centred grasp (used to check the place plan before grasping)."""
        R = hk.down_rot(yaw_grasp)
        return R.T @ np.array([0.0, 0.0, part.height / 2 - part.grasp_z()])

    def choose_and_plan(self, part, xy, part_yaw, pallet_c, pallet_yaw, cells_free, q_start, rng=None, fill=()):
        """Arm choice (reach map, then joint-limit margin), cell choice and the full nominal plan, checked for arm-vs-arm
        clearance against the other arm's waiting pose and for finger clearance from the parts already in the pallet
        (`fill`, from sample_prefill). Cell: a random free cell if `rng` is given (the next one in a random order when
        the plan fails), else the nearest free cell. Returns dict or None."""
        reach = {s: bool(self.in_mask("left" if s == "L" else "right", xy)[0]) for s in SIDES}
        sides = [s for s in SIDES if reach[s]]
        cands = []
        for s in sides:
            pk = self.plan_pick(s, part, xy, part_yaw, q_start[s])
            if pk is not None:
                cands.append((s, pk))
        if len(cands) == 2:
            cands.sort(key=lambda c: -c[1]["margin"])
        cc = self.pallet_cells(pallet_c, pallet_yaw)
        if rng is None:
            order_cells = sorted(cells_free, key=lambda k: np.linalg.norm(cc[k] - xy))
        else:
            order_cells = [cells_free[i] for i in rng.permutation(len(cells_free))]
        for s, pk in cands:
            other = "R" if s == "L" else "L"
            for cell in order_cells:
                pp = self.plan_place(s, part, pk["q_grasp"], self.nominal_offset(part, pk["yaw"]), part_yaw, pk["yaw"],
                                     pallet_c, pallet_yaw, cell)
                if pp is None or not self.release_clear(pp, fill):
                    continue
                traj = np.vstack([pk["transit"], pk["down"], pp["lift"], pp["carry"], pp["lower"], pp["up"], pp["home"]])
                dmin = float(self.arm_distance(s, traj[::2], other, self.park[other]).min())
                if dmin < ARM_CLEAR:
                    continue
                return {"side": s, "reach": reach, "pick": pk, "place": pp, "cell": int(cell), "arm_clearance": dmin,
                        "n_candidates": len(cands)}
        return None


# ------------------------------------------------------------------ self-test ------------------------------------------------
def selftest(level, n, seed, prefill_max=3, split="train"):
    import time

    w = World()
    print(f"park L tcp {np.round(w.fk_tcp('L', w.park['L'])[0][0], 3)}  R tcp {np.round(w.fk_tcp('R', w.park['R'])[0][0], 3)}"
          f"  pallet nominal {np.round(w.pallet_nominal, 3)} size {np.round(w.pallet_size, 3)}")
    d = w.arm_distance("L", w.park["L"], "R", w.park["R"])[0]
    print(f"park-park arm distance {d:.3f} m")
    for p in (Part(w.parts, s, max(w.parts["sizes"]), "red") for s in w.parts["shapes"]):
        assert p.grip_width <= 0.060 + 1e-9, (p.shape, p.grip_width)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    stats = {"ok": 0, "no_layout": 0, "no_plan": 0, "sides": {"L": 0, "R": 0}, "both_reach": 0, "steps": [],
             "cells": np.zeros(6, int), "n_fill": np.zeros(prefill_max + 1, int)}
    for i in range(n):
        parts, tgt = w.sample_task_parts(rng, split, 1 if level == 1 else int(rng.integers(3, 5)))
        lay = w.sample_layout(rng, parts, tgt, split=split)
        if lay is None:
            stats["no_layout"] += 1
            continue
        pc, pyaw, poses = lay
        xy, yaw = poses[tgt]
        fill = w.sample_prefill(rng, parts, int(rng.integers(prefill_max + 1)), pc, pyaw)
        free = [k for k in range(6) if k not in {f[1] for f in fill}]
        plan = w.choose_and_plan(parts[tgt], xy, yaw, pc, pyaw, free, w.park, rng=rng, fill=fill)
        if plan is None:
            stats["no_plan"] += 1
            continue
        stats["ok"] += 1
        stats["cells"][plan["cell"]] += 1
        stats["n_fill"][len(fill)] += 1
        stats["sides"][plan["side"]] += 1
        stats["both_reach"] += all(plan["reach"].values())
        T = sum(len(plan["pick"][k]) for k in ("transit", "down")) + sum(len(plan["place"][k]) for k in
                                                                        ("lift", "carry", "lower", "up", "home"))
        stats["steps"].append(T + 15 + 12)
    st = np.array(stats["steps"]) if stats["steps"] else np.zeros(1)
    print(f"wide region {w.ws['wide'].sum() * w.ws_step ** 2:.3f} m2, training parts region "
          f"{w.ws['parts'].sum() * w.ws_step ** 2:.3f} m2")
    print(f"level {level} split {split}: {n} layouts in {time.time() - t0:.1f}s -> planned {stats['ok']}, no layout "
          f"{stats['no_layout']}, no plan {stats['no_plan']}, arms {stats['sides']}, in both-arm area "
          f"{stats['both_reach']}, steps median {np.median(st):.0f} max {st.max():.0f} ({np.median(st) / w.ctrl_hz:.1f} s), "
          f"target cells {stats['cells'].tolist()}, prefilled 0..{prefill_max}: {stats['n_fill'].tolist()}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefill-max", type=int, default=3)
    ap.add_argument("--split", default="train", choices=SPLITS)
    a = ap.parse_args()
    selftest(a.level, a.n, a.seed, a.prefill_max, a.split)
