"""Scene constants + expert planning for dual_orca_pick_sim.py (numpy only; `env\\python.exe dual_orca_pick_plan.py
--selftest` checks the planner without Isaac).

Robot base = world origin, front = -Y (see dual_orca_kin). Right hand picks a cube top-down (palm down, fingers
forward); the left arm stays at home. All arm motions are in the 8-joint chain (yellowarm_joint1..7 + ORCA wrist).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

import dual_orca_kin as K
import json
import os

KIN = K.Kin()
TABLE_TOP = 0.38  # m; palm-down grasps feasible at grasp-centre z 0.40-0.50 (IK + gravity-load scan)
TABLE_Y = (-0.30, -0.80)  # front edge .. back edge (the left hand at home reaches y = -0.27)
FLOOR_Z = -0.20  # floor below the stand: at home the fingertips hang down to z = -0.015 (J1 cannot reach 0)
PEDESTAL = (0.34, 0.34)  # x, y size of the block the stand base sits on (floor .. z = 0)
TABLE_X = (-0.60, 0.60)
CUBE = 0.05  # edge (m)
CUBE_MASS = 0.08
CUBE_X = (0.12, 0.30)
CUBE_Y = (-0.34, -0.39)  # on the table; hand centre sits 3.5 cm further forward (power_v1 object offset)
PRE_DZ = 0.10  # pre-grasp height above the grasp point
LIFT_DZ = 0.16
_F, _D = np.array([0, -1.0, 0]), np.array([0, 0, -1.0])
R_DOWN = np.stack([_F, _D, np.cross(_F, _D)], 1)  # grasp frame: fingers forward, palm normal down
READY_POS = np.array([0.40, -0.33, 0.55])  # right grasp centre at the start: beside (not above) the cube region,
# so the head camera sees the cube at the start of every episode
SEEDS = [np.r_[-1.35, -0.59, 1.68, -2.25, -0.48, 1.55, -0.57, 0.88], np.r_[-0.8, 0, 0, -1.2, 0, 0, 0, 0],
         np.r_[-1.0, -0.3, 0.8, -1.8, 0, 1.0, 0, 0], np.r_[-0.5, 0.3, 0, -1.5, 0, 0.8, 0, 0]]
_HJ = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "humanoid.json"), encoding="utf-8"))
V_ARM = np.array([float(v) for v in _HJ["arm_limits"]["velocity_used_rad_s"]] + [1.5])  # rad/s: arm joints 1..7 (real moteus) + ORCA wrist (conservative)
LOAD_MAX = 0.85  # static gravity torque / URDF effort allowed on any arm joint along the plan


def ik_right(pos, q0=None):
    seeds = ([q0] if q0 is not None else []) + SEEDS
    q, pe, re = KIN.ik_best("right", pos, R_DOWN, seeds, iters=150)
    load = KIN.gravity_load("right", dict(zip(KIN.chain("right"), q))).max()
    return q, pe, re, load


_READY = None


def ready_pose():
    global _READY
    if _READY is None:
        q, pe, re, load = ik_right(READY_POS)
        if pe > 3e-3 or re > 0.08 or load > LOAD_MAX:
            raise RuntimeError(f"ready pose infeasible: {pe * 1000:.1f} mm {np.degrees(re):.1f} deg load {load:.2f}")
        _READY = q
    return _READY


def joint_traj(q0, q1, hz, min_s=0.6):
    """Min-jerk joint interpolation; duration from the real joint velocity limits with a 1.4x margin."""
    T = max(min_s, 1.4 * float(np.max(np.abs(q1 - q0) / V_ARM)))
    n = max(2, int(round(T * hz)))
    s = np.linspace(0, 1, n + 1)[1:]
    s = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    return [q0 + (q1 - q0) * si for si in s]


def cart_traj(q0, p0, p1, seconds, hz):
    """Straight-line grasp-centre motion (palm kept down) by IK at every control step."""
    n = max(3, int(round(seconds * hz)))
    out, q = [], q0
    for k in range(1, n + 1):
        p = p0 + (p1 - p0) * (k / n)
        qn, pe, re, load = ik_right(p, q0=q)
        if pe > 5e-3 or re > 0.1 or np.max(np.abs(qn - q)) > 0.25:
            return None
        out.append(qn)
        q = qn
    return out


GRASP_DZ = 0.008  # grasp centre above the synergy's nominal object centre: palm just clears the cube top


def plan_episode(cube_pos, hz=25.0, grasp_dz=GRASP_DZ, grasp_dy=0.0):
    # grasp centre such that the cube centre sits at the power_v1 object offset in the grasp frame
    g = cube_pos - R_DOWN @ np.array(K.GRASP_OBJ_OFFSET) + np.array([0.0, -grasp_dy, grasp_dz])
    pre = g + np.array([0.0, 0.0, PRE_DZ])
    q_pre, pe, re, load = ik_right(pre, q0=ready_pose())
    if pe > 3e-3 or re > 0.08 or load > LOAD_MAX:
        return None, f"pre-grasp IK {pe * 1000:.1f}mm {np.degrees(re):.1f}deg load {load:.2f}"
    down = cart_traj(q_pre, pre, g, 1.0, hz)
    if down is None:
        return None, "descent IK"
    up = cart_traj(down[-1], g, g + np.array([0, 0, LIFT_DZ]), 1.2, hz)
    if up is None:
        return None, "lift IK"
    loads = [KIN.gravity_load("right", dict(zip(KIN.chain("right"), q))).max() for q in down + up]
    if max(loads) > LOAD_MAX:
        return None, f"gravity load {max(loads):.2f}"
    return {"transit": joint_traj(ready_pose(), q_pre, hz), "down": down, "up": up, "grasp_point": g,
            "max_load": float(max(loads + [load]))}, ""


def sample_layout(rng):
    pos = np.array([rng.uniform(*CUBE_X), rng.uniform(CUBE_Y[1], CUBE_Y[0]), TABLE_TOP + CUBE / 2])
    yaw = rng.uniform(-np.pi / 8, np.pi / 8)
    return pos, yaw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    t = time.time()
    print("ready", np.round(ready_pose(), 3), f"({time.time() - t:.1f}s)")
    ok = 0
    for s in range(a.n):
        rng = np.random.default_rng(s)
        pos, yaw = sample_layout(rng)
        t = time.time()
        plan, why = plan_episode(pos)
        ok += plan is not None
        print(f"seed {s} cube {np.round(pos, 3)}: {'OK' if plan else 'REJECT ' + why}"
              + (f" transit {len(plan['transit'])} down {len(plan['down'])} up {len(plan['up'])} load {plan['max_load']:.2f}"
                 if plan else "") + f" ({time.time() - t:.1f}s)")
    print(f"{ok}/{a.n} feasible")
