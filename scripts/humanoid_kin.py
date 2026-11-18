"""Numpy kinematics for the YellowArm humanoid (two arms on one torso). Stdlib + numpy only, so it runs in both the
env python and Isaac Sim's python.

- The chain (joint origins, axes, limits, efforts) is parsed from the YellowArm URDF; FK/IK are batched over many
  targets at once (damped least squares, several seeds per target, joint limits enforced).
- An arm is placed on the torso by a mount frame: base +Z = `axis` (unit vector, world), rotated by `roll` about it.
  Left/right mounts mirror the axis across the XZ plane; each arm keeps its own roll.
- Checks used by the reach probe: arm volume (per-link boxes measured from the USD collision meshes) vs table top,
  torso and shoulder bar, and static gravity torque vs joint effort.
  Link masses come from the USD (the URDF inertials are placeholders: 1.883 kg everywhere); link COMs are taken at
  the middle of each link segment. Both are assumptions until real CAD masses exist.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from mhh_paths import YELLOWARM_URDF as _URDF  # noqa: E402
URDF = _URDF.as_posix()
ARM_JOINTS = [f"yellowarm_joint{i}" for i in range(1, 8)]
TCP_OFFSET = 0.085  # grasp point along link7 +Z (same as yellowarm_sim.py)
# USD masses (kg) of link1..link7 (+ both fingers on link7); link0 is fixed to the torso.
LINK_MASS = np.array([1.5, 0.5, 1.5, 0.5, 1.0, 0.3, 0.4 + 0.1])
# Axis-aligned bounds (link frame, metres) of each link's USD collision meshes, link1..link7, then both fingers
# (link8/9, attached to link7; bounds at q = 0 cover open..closed). Measured from yellow_arm.usd.
LINK_BOXES = [
    ((-0.093, -0.081, 0.000), (0.093, 0.061, 0.238)),
    ((-0.078, -0.066, -0.034), (0.078, 0.074, 0.070)),
    ((-0.093, -0.073, 0.000), (0.093, 0.064, 0.303)),
    ((-0.078, -0.066, -0.034), (0.078, 0.074, 0.070)),
    ((-0.069, -0.073, 0.000), (0.069, 0.059, 0.257)),
    ((-0.050, -0.037, -0.022), (0.050, 0.037, 0.051)),
    ((-0.034, -0.079, -0.006), (0.027, 0.079, 0.063)),
    ((-0.011, -0.039, 0.064), (0.012, 0.039, 0.117)),
]


def _box_grid(lo, hi, n=3):
    g = [np.linspace(lo[k], hi[k], n) for k in range(3)]
    return np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)


BOX_POINTS = np.stack([_box_grid(lo, hi) for lo, hi in LINK_BOXES])  # (8, 27, 3)
FINGER_MASK = np.zeros(BOX_POINTS.shape[:2], bool)
FINGER_MASK[7] = True
FINGER_MASK = FINGER_MASK.reshape(-1)


def load_config(path=os.path.join(HERE, "data", "humanoid.json")):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ rotations ------------------------------------------
def rot_axis(axis, ang):
    """Rotation matrices about a fixed unit axis for an array of angles -> (..., 3, 3)."""
    ang = np.asarray(ang, float)
    x, y, z = axis
    c, s = np.cos(ang), np.sin(ang)
    C = 1 - c
    R = np.empty(ang.shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def rpy_matrix(r, p, y):
    return rot_axis((0, 0, 1), y) @ rot_axis((0, 1, 0), p) @ rot_axis((1, 0, 0), r)


def mat_to_quat(R):
    """3x3 -> (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def down_rot(yaw):
    """Top-down grasp: link7 +Z points down, fingers close along the yaw-rotated world Y (as yellowarm_sim.down_quat)."""
    return rot_axis((0, 0, 1), yaw) @ rot_axis((0, 1, 0), np.pi * np.ones_like(np.asarray(yaw, float)))


def mount_rotation(axis, roll_deg):
    """Base frame in world: +Z along `axis`, x chosen from world -Z / +X and turned by roll about the axis."""
    z = np.asarray(axis, float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, -1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = ref - z * (ref @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R0 = np.stack([x, y, z], axis=1)
    return rot_axis(z, np.deg2rad(roll_deg)) @ R0


# ------------------------------------------------------------------ chain -----------------------------------------------
class Chain:
    def __init__(self, urdf=URDF):
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}
        self.origins, self.axes, self.lo, self.hi, self.effort, self.vel = [], [], [], [], [], []
        for name in ARM_JOINTS:
            j = joints[name]
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz").split()])
            rpy = [float(v) for v in o.get("rpy").split()]
            T = np.eye(4)
            T[:3, :3] = rpy_matrix(*rpy)
            T[:3, 3] = xyz
            self.origins.append(T)
            self.axes.append(np.array([float(v) for v in j.find("axis").get("xyz").split()]))
            lim = j.find("limit")
            self.lo.append(float(lim.get("lower")))
            self.hi.append(float(lim.get("upper")))
            self.effort.append(float(lim.get("effort")))
            self.vel.append(float(lim.get("velocity")))
        self.lo, self.hi = np.array(self.lo), np.array(self.hi)
        self.effort, self.vel = np.array(self.effort), np.array(self.vel)

    def fk(self, q, base_R=np.eye(3), base_p=np.zeros(3), frames=False):
        """q (N, 7) -> joint origins/axes in world (N, 7, 3), link7 rotation (N, 3, 3), tcp (N, 3).
        frames=True also returns every link frame: rotations (N, 7, 3, 3) and origins (N, 7, 3) of link1..link7."""
        q = np.atleast_2d(q)
        N = q.shape[0]
        R = np.broadcast_to(base_R, (N, 3, 3)).copy()
        p = np.broadcast_to(base_p, (N, 3)).copy()
        org, ax = np.empty((N, 7, 3)), np.empty((N, 7, 3))
        Rs = np.empty((N, 7, 3, 3)) if frames else None
        for i in range(7):
            T = self.origins[i]
            p = p + R @ T[:3, 3]
            R = R @ T[:3, :3]
            org[:, i], ax[:, i] = p, R @ self.axes[i]
            R = R @ rot_axis(self.axes[i], q[:, i])
            if frames:
                Rs[:, i] = R
        tcp = p + R[:, :, 2] * TCP_OFFSET
        if frames:
            return org, ax, R, tcp, Rs, org.copy()
        return org, ax, R, tcp

    def collision_points(self, q, base_R, base_p):
        """World points sampled on every link's collision box (N, 8 * 27, 3); FINGER_MASK marks the finger points."""
        _, _, _, _, Rs, ps = self.fk(q, base_R, base_p, frames=True)
        Rb = np.concatenate([Rs, Rs[:, 6:7]], 1)  # fingers ride on link7
        pb = np.concatenate([ps, ps[:, 6:7]], 1)
        pts = pb[:, :, None, :] + np.einsum("nbij,bkj->nbki", Rb, BOX_POINTS)
        return pts.reshape(len(q), -1, 3)

    def ik_down(self, targets, yaw, base_R, base_p, seeds, iters=120, pos_tol=0.004, rot_tol=0.035, margin=0.03,
                free_j1=False):
        """Top-down IK for many targets. targets (N, 3), yaw scalar or (N,), seeds (S, 7).
        Returns q (N, S, 7) per seed, ok (N, S) and the final errors. Joint limits shrunk by `margin` rad.
        free_j1: no limit on joint1 (the caller checks it later, e.g. for several mount rolls at once)."""
        N, S = len(targets), len(seeds)
        tgt = np.repeat(targets, S, axis=0)
        Rg = np.repeat(np.broadcast_to(down_rot(np.broadcast_to(yaw, (N,))), (N, 3, 3)), S, axis=0)
        q = np.tile(seeds, (N, 1)).astype(float)
        lo, hi = self.lo + margin, self.hi - margin
        if free_j1:
            lo, hi = lo.copy(), hi.copy()
            lo[0], hi[0] = -1e3, 1e3
        lam2 = 0.03 ** 2
        for it in range(iters):
            org, ax, R, tcp = self.fk(q, base_R, base_p)
            ep = tgt - tcp
            Re = Rg @ np.transpose(R, (0, 2, 1))
            er = 0.5 * np.stack([Re[:, 2, 1] - Re[:, 1, 2], Re[:, 0, 2] - Re[:, 2, 0], Re[:, 1, 0] - Re[:, 0, 1]], 1)
            e = np.concatenate([np.clip(ep, -0.1, 0.1), er * 0.5], 1)  # orientation weighted (rad ~ 0.5 m)
            Jp = np.cross(ax, tcp[:, None, :] - org)  # (M, 7, 3)
            J = np.concatenate([Jp, ax * 0.5], 2).transpose(0, 2, 1)  # (M, 6, 7)
            A = J @ J.transpose(0, 2, 1) + lam2 * np.eye(6)
            dq = (J.transpose(0, 2, 1) @ np.linalg.solve(A, e[..., None]))[..., 0]
            q = np.clip(q + np.clip(dq, -0.3, 0.3), lo, hi)
        org, ax, R, tcp = self.fk(q, base_R, base_p)
        perr = np.linalg.norm(tgt - tcp, axis=1)
        Re = Rg @ np.transpose(R, (0, 2, 1))
        rerr = np.arccos(np.clip((np.trace(Re, axis1=1, axis2=2) - 1) / 2, -1, 1))
        ok = (perr < pos_tol) & (rerr < rot_tol)
        return q.reshape(N, S, 7), ok.reshape(N, S), perr.reshape(N, S), rerr.reshape(N, S)

    def j1_ok_for_roll(self, q1, roll_deg, margin=0.03):
        """joint1 value q1 solved with roll 0 -> is it inside the limits once the mount is rolled by roll_deg?
        (Rolling the mount about its axis is the same as adding the roll to joint1.)"""
        lo, hi = self.lo[0] + margin, self.hi[0] - margin
        v = np.mod(q1 - np.deg2rad(roll_deg) - lo, 2 * np.pi)
        return v <= hi - lo

    def gravity_torque(self, q, base_R, base_p, payload=0.15):
        """Static joint torques (N, 7) holding the arm against gravity (link COMs at segment midpoints)."""
        org, ax, R, tcp = self.fk(q, base_R, base_p)
        ends = np.concatenate([org[:, 1:], tcp[:, None]], 1)  # link i spans joint i origin -> joint i+1 origin
        com = 0.5 * (org + ends)
        g = np.array([0.0, 0.0, -9.81])
        F = LINK_MASS[None, :, None] * g  # (1, 7, 3)
        tau = np.zeros(q.shape)
        for i in range(7):
            r = com[:, i:] - org[:, i:i + 1]
            m = np.cross(r, F[:, i:]).sum(1) + np.cross(tcp - org[:, i], payload * g)
            tau[:, i] = (m * ax[:, i]).sum(1)
        return tau


def segment_points(org, tcp, n=4):
    """Points sampled along the arm from joint2 to the tcp (N, P, 3), for collision checks."""
    pts = np.concatenate([org[:, 1:], tcp[:, None]], 1)
    out = [pts[:, :1]]
    for a, b in zip(range(pts.shape[1] - 1), range(1, pts.shape[1])):
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(pts[:, a:a + 1] * (1 - t) + pts[:, b:b + 1] * t)
    return np.concatenate(out, 1)


def in_box(pts, center, half, inflate=0.0):
    return np.all(np.abs(pts - center) < np.asarray(half) + inflate, axis=-1)
