"""Bimanual OpenArm "place the cube in the box" scene in Isaac Sim 6, driven by a SmolVLA checkpoint
fine-tuned for that task (default: qualia-robotics/smolvla-openarm-cube-in-box-v2).

Launch with Isaac Sim's python:  D:\\Isaac\\isaacsim\\python.bat openarm_bimanual_sim.py [--headless] [--probe]
Requires policy_server.py running with that checkpoint (see run_bimanual_all.bat).

The checkpoint was trained in a MuJoCo sim of the same robot (100 RL-teacher demos, 25 fps, ~1 s each):
 - every episode starts with the LEFT gripper open around the cube (pre-grasp); the policy closes, carries
   the cube over and drops it in a blue box;
 - state (9) = left joint1..7 (rad) + finger1, finger2 (MuJoCo units, 0.7 = open);
   action (8) = left joint1..7 targets + gripper ctrl (0 = close, 0.785 = open);
 - 3 fisheye 480x480 cameras: base (body, looking at the table), wrist_left, wrist_right.
This scene mirrors that setup: episodes start from the dataset's recorded start poses (data/qualia_cube_in_box.json),
the table height and cube are placed under the left fingertips, and the box sits where the teacher released.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--task", default="place the cube in the box")
parser.add_argument("--episodes", type=int, default=0, help="0 = run forever until the window is closed")
parser.add_argument("--keep-open", action="store_true")
parser.add_argument("--episode-seconds", type=float, default=2.0,
                    help="training episodes last ~1 s; past that the policy is out of distribution and swings the arm")
parser.add_argument("--control-hz", type=float, default=25.0, help="dataset fps")
parser.add_argument("--physics-hz", type=float, default=100.0, help="4 physics steps per control step, as in training")
parser.add_argument("--exec-steps", type=int, default=16, help="actions executed per chunk before re-planning")
parser.add_argument("--max-step-rad", type=float, default=0.3, help="per control step joint target rate limit")
parser.add_argument("--arm-stiffness", type=float, default=800.0)
parser.add_argument("--arm-damping", type=float, default=40.0)
parser.add_argument("--img-size", type=int, default=480)
parser.add_argument("--no-fisheye", action="store_true", help="use pinhole cameras instead of the training fisheye")
parser.add_argument("--fisheye-fov", type=float, default=180.0, help="field of view of the image circle (deg)")
parser.add_argument("--start-index", type=int, default=-1, help="dataset start pose to use (-1 = random each episode)")
parser.add_argument("--box-pos", type=float, nargs=2, default=None, metavar=("X", "Y"),
                    help="box centre (default: where the training teacher released the cube)")
parser.add_argument("--gripper-state-open", type=float, default=0.7, help="state value of a fully open finger")
parser.add_argument("--gripper-action-open", type=float, default=0.785, help="action value that fully opens")
parser.add_argument("--cube-friction", type=float, default=1.5)
parser.add_argument("--grasp-depth", type=float, default=0.03,
                    help="how much of the 4 cm cube is between the fingers at the start (m)")
# Scripted-expert data generation (no policy server needed).
parser.add_argument("--expert", action="store_true", help="drive the left arm with a scripted IK pick-and-place")
parser.add_argument("--record", default="", help="directory to save successful expert episodes (.npz)")
parser.add_argument("--start-mode", choices=["dataset", "expert"], default="dataset",
                    help="policy mode: 'dataset' = qualia training start poses (cube already between the fingers); "
                         "'expert' = same start as our scripted demos (ready pose above the table, random cube)")
parser.add_argument("--ready-jitter", type=float, default=0.03, help="random +/- offset (m) of the start (ready) pose per axis")
parser.add_argument("--table-z", type=float, default=0.60, help="table height in expert mode")
parser.add_argument("--cube-x", type=float, nargs=2, default=[0.28, 0.42], help="cube x range in expert mode")
parser.add_argument("--cube-y", type=float, nargs=2, default=[0.02, 0.20], help="cube y range in expert mode")
parser.add_argument("--debug", action="store_true", help="print finger command/target/position every control step")
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--save-frames", default="")
parser.add_argument("--probe", action="store_true", help="build scene, reset one episode, save frames, exit")
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

sys.stdout.reconfigure(line_buffering=True)  # progress lines show up immediately when redirected to a log

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.materials import OmniPbrMaterial, RigidBodyMaterial  # noqa: E402
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/
sys.path.insert(0, HERE)
from wire import HOST, recv_msg, send_msg  # noqa: E402

ROOT = os.path.dirname(HERE)
OPENARM_USD = os.path.join(
    ROOT, "vendor", "openarm_isaac_lab", "source", "openarm", "openarm", "tasks", "manager_based",
    "openarm_manipulation", "usds", "openarm_bimanual", "openarm_bimanual.usd",
).replace("\\", "/")
TASK_DATA = json.load(open(os.path.join(HERE, "data", "qualia_cube_in_box.json")))

ROBOT = "/World/openarm"
LEFT_ARM = [f"openarm_left_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"openarm_right_joint{i}" for i in range(1, 8)]
LEFT_FINGERS = ["openarm_left_finger_joint1", "openarm_left_finger_joint2"]
RIGHT_FINGERS = ["openarm_right_finger_joint1", "openarm_right_finger_joint2"]
LEFT_LO = np.deg2rad([-200, -190, -90, 0, -90, -45, -90])
LEFT_HI = np.deg2rad([80, 10, 90, 140, 90, 45, 90])
FINGER_OPEN = 0.044  # m (Isaac prismatic finger)
CAM_NAMES = ("base", "wrist_left", "wrist_right")
CUBE = 0.04
BOX_OUTER, BOX_WALL, BOX_H = 0.16, 0.01, 0.05
BOX_XY = np.array(args.box_pos if args.box_pos else TASK_DATA["release_tcp_mean"][:2])
rng = np.random.default_rng(args.seed)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    q = view.GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)


def make_camera(path, eye, target, up=(0.0, 0.0, 1.0), focal=12.0) -> CameraSensor:
    s = args.img_size
    kw = dict(translations=np.array(eye, dtype=np.float64), orientations=look_at_quat(eye, target, up))
    if not args.no_fisheye:
        f = (s / 2) / np.deg2rad(args.fisheye_fov / 2)  # equidistant: r = f * theta
        kw["schemas"] = ["OmniLensDistortionOpenCvFisheyeAPI"]
        kw["attributes"] = {
            "omni:lensdistortion:opencvFisheye:cx": s / 2, "omni:lensdistortion:opencvFisheye:cy": s / 2,
            "omni:lensdistortion:opencvFisheye:fx": f, "omni:lensdistortion:opencvFisheye:fy": f,
            "omni:lensdistortion:opencvFisheye:k1": 0.0, "omni:lensdistortion:opencvFisheye:k2": 0.0,
            "omni:lensdistortion:opencvFisheye:k3": 0.0, "omni:lensdistortion:opencvFisheye:k4": 0.0,
            "omni:lensdistortion:opencvFisheye:imageSize": Gf.Vec2i(s, s),
        }
    cam = RtxCamera(path, **kw)
    if not args.no_fisheye:
        cam.prims[0].GetAttribute("omni:lensdistortion:model").Set("opencvFisheye")
    usd_cam = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    usd_cam.GetFocalLengthAttr().Set(focal)
    usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(s, s), annotators=["rgb"])


def set_translate(path, xyz):
    UsdGeom.XformCommonAPI(omni.usd.get_context().get_stage().GetPrimAtPath(path)).SetTranslate(Gf.Vec3d(*map(float, xyz)))


def build_scene():
    stage_utils.create_new_stage()
    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    DomeLight("/World/DomeLight").set_intensities(1200)
    GroundPlane("/World/ground", sizes=20.0)
    stage_utils.add_reference_to_stage(usd_path=OPENARM_USD, path=ROBOT)
    stage = omni.usd.get_context().get_stage()

    def box(path, size, pos, color, dynamic=False):
        c = Cube(path, sizes=1.0, positions=np.array(pos), scales=np.array(size))
        geom = GeomPrim(path, apply_collision_apis=True)
        if dynamic:
            RigidPrim(path)
        mat = OmniPbrMaterial("/World/Materials/" + path.strip("/").replace("/", "_"))
        mat.set_input_values("diffuse_color_constant", list(color))
        c.apply_visual_materials(mat)
        return geom

    # Static props live under Xforms whose translate is re-set each episode (table height follows the start pose).
    UsdGeom.Xform.Define(stage, "/World/table_rig")
    box("/World/table_rig/top", (0.70, 1.0, 0.03), (0.45, 0.0, -0.015), (0.45, 0.30, 0.18))
    UsdGeom.Xform.Define(stage, "/World/box_rig")
    blue, o, w, h = (0.1, 0.25, 0.85), BOX_OUTER, BOX_WALL, BOX_H
    box("/World/box_rig/floor", (o, o, w), (0, 0, w / 2), blue)
    box("/World/box_rig/wall_px", (w, o, h), ((o - w) / 2, 0, h / 2), blue)
    box("/World/box_rig/wall_nx", (w, o, h), (-(o - w) / 2, 0, h / 2), blue)
    box("/World/box_rig/wall_py", (o, w, h), (0, (o - w) / 2, h / 2), blue)
    box("/World/box_rig/wall_ny", (o, w, h), (0, -(o - w) / 2, h / 2), blue)
    cube_geom = box("/World/cube", (CUBE, CUBE, CUBE), (0.4, 0.1, 1.0), (0.95, 0.35, 0.05), dynamic=True)
    # Grippy cube: combine mode "max" wins over the fingers' default material, so every contact uses mu=1.5.
    grip = RigidBodyMaterial("/World/PhysicsMaterials/grippy")
    grip.set_friction_coefficients(static_frictions=args.cube_friction, dynamic_frictions=args.cube_friction)
    grip.set_combine_modes(frictions="max")
    cube_geom.apply_physics_materials(grip)

    cams = {
        # Training "base" view: just in front of the chest, pitched ~41 deg down, table filling the lower half.
        "base": make_camera("/World/cam_base", eye=(0.25, 0.0, 1.02), target=(1.00, 0.0, 0.37)),
        # openarm_*_hand frame: +Z = approach. In the dataset's grasp poses the left hand's +X faces up,
        # so the left wrist camera sits on that side looking along the fingers.
        "wrist_left": make_camera(f"{ROBOT}/openarm_left_hand/cam_wrist", eye=(0.06, 0.0, 0.0),
                                  target=(0.0, 0.0, 0.2), up=(1.0, 0.0, 0.0)),
        "wrist_right": make_camera(f"{ROBOT}/openarm_right_hand/cam_wrist", eye=(-0.06, 0.0, 0.0),
                                   target=(0.0, 0.0, 0.2), up=(-1.0, 0.0, 0.0)),
    }
    return cams


def grab(cams):
    out = []
    for name in CAM_NAMES:
        data, _ = cams[name].get_data("rgb")
        img = np.zeros((args.img_size, args.img_size, 3), np.uint8) if data is None else data.numpy()[..., :3]
        out.append(np.ascontiguousarray(img.astype(np.uint8)))
    return out


def save_frames(imgs, tag):
    if not args.save_frames:
        return
    from PIL import Image

    os.makedirs(args.save_frames, exist_ok=True)
    for name, img in zip(CAM_NAMES, imgs):
        Image.fromarray(img).save(os.path.join(args.save_frames, f"{tag}_{name}.png"))


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def run_expert(robot, cube, tcp, cams, la, lf, rf, ctrl_idx, decimation, read_state, cube_in_box,
               policy_episode=None):
    """Scripted pick-and-place demonstrator for the left arm (differential IK, top-down grasp).

    Every control step records exactly what the policy sees/does: state (9) + 3 camera images at the start of
    the step, and the absolute joint/gripper targets commanded over it (action, 8) -- same units and conventions
    as qualia-robotics/openarm-cube-in-box-v2, so the data can fine-tune that checkpoint directly.
    """
    names = robot.dof_names
    a_open = args.gripper_action_open
    tcp_jac = int(robot.get_link_indices("openarm_left_ee_tcp").numpy()[0]) - 1
    down_q = np.array([0.0, 0.0, 1.0, 0.0])  # TCP approach (+Z) pointing down, fingers close along world Y
    table_z = args.table_z
    set_translate("/World/table_rig", (0.0, 0.0, table_z))
    set_translate("/World/box_rig", (BOX_XY[0], BOX_XY[1], table_z))
    # Home: the recorded start pose nearest the average one (bent arm in front of the robot).
    st = np.array(TASK_DATA["start_tcp"])
    home_q = np.array(TASK_DATA["start_states"][int(np.argmin(np.linalg.norm(st - st.mean(0), axis=1)))][:7])
    out_dir = args.record
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    saved = len([f for f in os.listdir(out_dir) if f.endswith(".npz")]) if out_dir else 0

    def tcp_pose():
        p, q = tcp.get_world_poses()
        return p.numpy()[0].astype(np.float64), q.numpy()[0].astype(np.float64)

    def ik_target(goal_p):
        q = robot.get_dof_positions().numpy()[0].astype(np.float64)
        jac = robot.get_jacobian_matrices().numpy()[0, tcp_jac][:, la].astype(np.float64)
        p, rq = tcp_pose()
        dq_err = quat_mul(down_q, np.array([rq[0], -rq[1], -rq[2], -rq[3]]))
        err = np.concatenate([goal_p - p, dq_err[1:] * np.sign(dq_err[0])])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.05 ** 2 * np.eye(6), err)
        return np.clip(q[la] + np.clip(dq, -0.15, 0.15), LEFT_LO, LEFT_HI)

    rec = None

    def step(arm_t, grip):
        if rec is not None:
            rec["state"].append(read_state())
            for n, img in zip(CAM_NAMES, grab(cams)):
                rec[n].append(img)
            rec["action"].append(np.concatenate([arm_t, [grip]]).astype(np.float32))
        f = grip / a_open * FINGER_OPEN
        robot.set_dof_position_targets(np.concatenate([arm_t, [f, f]]), dof_indices=ctrl_idx)
        for _ in range(decimation):
            simulation_app.update()

    def move(goal_p, grip, speed, settle=0):
        p0 = tcp_pose()[0]
        n = max(1, int(np.ceil(np.linalg.norm(goal_p - p0) / (speed / args.control_hz))))
        for k in range(1, n + 1):
            step(ik_target(p0 + (goal_p - p0) * k / n), grip)
        for _ in range(settle):
            step(ik_target(goal_p), grip)

    def hold(goal_p, grip, n):
        for _ in range(n):
            step(ik_target(goal_p), grip)

    ep, tries, wins = saved, 0, 0
    while simulation_app.is_running() and (not args.episodes or ep < args.episodes):
        if policy_episode is not None and args.episodes and tries >= args.episodes:
            break
        tries += 1
        # ---- reset (not recorded): home pose, IK to a jittered ready pose above the table, random cube ----
        full = np.zeros(len(names), np.float32)
        full[la] = home_q
        full[lf] = full[rf] = FINGER_OPEN
        cube.set_world_poses(positions=np.array([[0.0, 0.8, 0.05]]))
        robot.set_dof_positions(full)
        robot.set_dof_velocities(np.zeros_like(full))
        robot.set_dof_position_targets(full)
        for _ in range(5):
            simulation_app.update()
        ready = np.array([0.32, 0.10, table_z + 0.18]) + rng.uniform(-args.ready_jitter, args.ready_jitter, 3)
        rec = None
        move(ready, a_open, 0.3, settle=40)
        ready_err = np.linalg.norm(tcp_pose()[0] - ready)
        c_xy = np.array([rng.uniform(*args.cube_x), rng.uniform(*args.cube_y)])
        yaw = rng.uniform(-0.3, 0.3)
        cube.set_world_poses(positions=np.array([[c_xy[0], c_xy[1], table_z + CUBE / 2 + 0.001]]),
                             orientations=np.array([[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]]))
        cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        hold(ready, a_open, 5)

        if policy_episode is not None:  # evaluation: the policy drives from the expert's start condition
            policy_episode(tries - 1, f"cube {np.round(c_xy, 3)}")
            for _ in range(20):
                simulation_app.update()
            ok, c = cube_in_box(table_z)
            wins += ok
            print(f"[sim] episode {tries - 1} done: cube start {np.round(c_xy, 3)} end {np.round(c, 3)} -> "
                  f"{'SUCCESS (cube in box)' if ok else 'fail'}   [{wins}/{tries} successes]")
            continue

        # ---- recorded demonstration ----
        rec = {"state": [], "action": [], **{n: [] for n in CAM_NAMES}}
        above = np.array([c_xy[0], c_xy[1], table_z + 0.12])
        grasp = np.array([c_xy[0], c_xy[1], table_z + 0.012])
        lift = np.array([c_xy[0], c_xy[1], table_z + 0.16])
        over_box = np.array([BOX_XY[0], BOX_XY[1], table_z + 0.16])
        drop = np.array([BOX_XY[0], BOX_XY[1], table_z + 0.11])
        move(above, a_open, 0.30)
        move(grasp, a_open, 0.15, settle=4)
        hold(grasp, 0.0, 12)  # close
        move(lift, 0.0, 0.20)
        move(over_box, 0.0, 0.30)
        move(drop, 0.0, 0.15, settle=2)
        hold(drop, a_open, 8)  # release
        move(drop + np.array([0.0, 0.0, 0.08]), a_open, 0.20)
        episode, rec = rec, None
        for _ in range(20):
            simulation_app.update()
        ok, c = cube_in_box(table_z)
        T = len(episode["action"])
        print(f"[expert] try {tries}: ready err {1000 * ready_err:.0f} mm, cube start {np.round(c_xy, 3)}, "
              f"{T} frames ({T / args.control_hz:.1f} s) -> {'SUCCESS' if ok else 'fail'}")
        if ok and out_dir:
            path = os.path.join(out_dir, f"episode_{ep:04d}.npz")
            np.savez_compressed(path, state=np.stack(episode["state"]), action=np.stack(episode["action"]),
                                **{n: np.stack(episode[n]) for n in CAM_NAMES}, task=args.task, fps=args.control_hz)
            print(f"[expert] saved {path}")
        if ok:
            ep += 1
    if policy_episode is None:
        print(f"[expert] done: {ep - saved} new successful episodes out of {tries} tries")
    else:
        print(f"[sim] evaluation done: {wins}/{tries} cubes in the box")


def main():
    cams = build_scene()
    robot = Articulation(ROBOT)
    cube = RigidPrim("/World/cube")
    tcp = RigidPrim(f"{ROBOT}/openarm_left_ee_tcp")
    hand = RigidPrim(f"{ROBOT}/openarm_left_hand")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    names = robot.dof_names
    idx = lambda js: [names.index(j) for j in js]  # noqa: E731
    la, ra, lf, rf = idx(LEFT_ARM), idx(RIGHT_ARM), idx(LEFT_FINGERS), idx(RIGHT_FINGERS)
    ctrl_idx = la + lf
    robot.set_dof_gains(stiffnesses=[args.arm_stiffness] * 14, dampings=[args.arm_damping] * 14, dof_indices=la + ra)
    robot.set_dof_gains(stiffnesses=[2e3] * 4, dampings=[1e2] * 4, dof_indices=lf + rf)
    robot.set_dof_max_efforts([333.33] * 4, dof_indices=lf + rf)  # openarm_isaac_lab gripper effort limit
    # Position-controlled arms like the MuJoCo teacher; gravity off so the PD targets are tracked closely.
    robot.set_link_enabled_gravities(np.zeros((1, robot.num_links), dtype=bool))
    decimation = max(1, round(args.physics_hz / args.control_hz))
    s_open, a_open = args.gripper_state_open, args.gripper_action_open

    def read_state():
        q = robot.get_dof_positions().numpy()[0]
        return np.concatenate([q[la], q[lf] / FINGER_OPEN * s_open]).astype(np.float32)

    def reset(start_i):
        start = np.array(TASK_DATA["start_states"][start_i], dtype=np.float32)
        full = np.zeros(len(names), np.float32)
        full[la] = start[:7]
        full[lf] = full[rf] = FINGER_OPEN
        cube.set_world_poses(positions=np.array([[0.0, 0.8, 0.05]]))  # park the cube out of the way
        robot.set_dof_positions(full)
        robot.set_dof_velocities(np.zeros_like(full))
        robot.set_dof_position_targets(full)
        for _ in range(5):
            simulation_app.update()
        p = tcp.get_world_poses()[0].numpy()[0]
        hq = hand.get_world_poses()[1].numpy()[0]
        # Table top just below the fingertips so the cube sits between the open fingers (training start state),
        # with --grasp-depth of cube above the fingertips (the policy lifts ~3 cm before the fingers close).
        table_z = float(p[2] - (CUBE - args.grasp_depth))
        set_translate("/World/table_rig", (0.0, 0.0, table_z))
        set_translate("/World/box_rig", (BOX_XY[0], BOX_XY[1], table_z))
        # Align the cube with the finger closing axis (hand +Y) projected on the table.
        m = Gf.Matrix3d(Gf.Quatd(float(hq[0]), Gf.Vec3d(*map(float, hq[1:]))))
        yaw = np.arctan2(m[1][1], m[1][0])
        cq = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        cube.set_world_poses(positions=np.array([[p[0], p[1], table_z + CUBE / 2 + 0.002]]),
                             orientations=cq[None])
        cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        for _ in range(10):
            simulation_app.update()
        return table_z, p

    def cube_in_box(table_z):
        c = cube.get_world_poses()[0].numpy()[0]
        inner = BOX_OUTER / 2 - BOX_WALL
        inside = np.all(np.abs(c[:2] - BOX_XY) < inner) and c[2] < table_z + BOX_H + 0.01
        return inside, c

    pick = (lambda: args.start_index) if args.start_index >= 0 else (lambda: int(rng.integers(len(TASK_DATA["start_states"]))))

    if args.probe:
        i = pick()
        table_z, p = reset(i)
        for _ in range(60):
            simulation_app.update()
        imgs = grab(cams)
        c = cube.get_world_poses()[0].numpy()[0]
        print(f"[sim] probe start#{i}: tcp={np.round(p, 3)} table_z={table_z:.3f} cube={np.round(c, 3)} "
              f"box={np.round(BOX_XY, 3)} state={np.round(read_state(), 2)} mean px {[round(float(im.mean()), 1) for im in imgs]}")
        cache = UsdGeom.XformCache()
        stage = omni.usd.get_context().get_stage()
        for n, pth in (("base", "/World/cam_base"), ("wrist_left", f"{ROBOT}/openarm_left_hand/cam_wrist"),
                       ("wrist_right", f"{ROBOT}/openarm_right_hand/cam_wrist")):
            prim = stage.GetPrimAtPath(pth)
            if not prim.IsValid():
                print(f"[sim] cam {n}: prim missing at {pth}")
                continue
            m = cache.GetLocalToWorldTransform(prim)
            fwd = m.TransformDir(Gf.Vec3d(0, 0, -1))
            print(f"[sim] cam {n}: pos={np.round(list(m.ExtractTranslation()), 3)} forward={np.round(list(fwd), 2)}")
        save_frames(imgs, "probe")
        timeline.stop()
        simulation_app.close()
        return

    if args.expert:
        run_expert(robot, cube, tcp, cams, la, lf, rf, ctrl_idx, decimation, read_state, cube_in_box)
        timeline.stop()
        simulation_app.close()
        return

    sock = socket.create_connection((HOST, args.port), timeout=300)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"cmd": "info"})
    print(f"[sim] connected to policy server: {recv_msg(sock)}")

    steps_per_episode = int(args.episode_seconds * args.control_hz)

    def policy_episode(ep, label):
        send_msg(sock, {"cmd": "reset"})
        recv_msg(sock)
        target = read_state()[:8].copy()
        target[7] = a_open
        step = 0
        while step < steps_per_episode and simulation_app.is_running():
            imgs = grab(cams)
            state = read_state()
            send_msg(sock, {"cmd": "infer", "state": state, "images": imgs, "cam_names": list(CAM_NAMES),
                            "task": args.task})
            resp = recv_msg(sock)
            chunk = resp["actions"][: args.exec_steps]
            print(f"[sim] ep {ep} ({label}) step {step}: infer {resp['ms']:.0f} ms, "
                  f"grip state={state[7]:.2f} a0={np.round(chunk[0], 2)} cube={np.round(cube.get_world_poses()[0].numpy()[0], 3)}")
            save_frames(imgs, f"ep{ep}_step{step:04d}")
            for a in chunk:
                desired = a.astype(np.float32).copy()
                desired[:7] = np.clip(desired[:7], LEFT_LO, LEFT_HI)
                desired[7] = np.clip(desired[7], 0.0, a_open)
                target[:7] += np.clip(desired[:7] - target[:7], -args.max_step_rad, args.max_step_rad)
                target[7] = desired[7]
                finger = target[7] / a_open * FINGER_OPEN
                robot.set_dof_position_targets(np.concatenate([target[:7], [finger, finger]]), dof_indices=ctrl_idx)
                for _ in range(decimation):
                    simulation_app.update()
                if args.debug:
                    qd = robot.get_dof_positions().numpy()[0]
                    tg = robot.get_dof_position_targets().numpy()[0]
                    print(f"[dbg] step {step}: finger cmd={finger:.4f} target={np.round(tg[lf], 4)} "
                          f"pos={np.round(qd[lf], 4)} tcp={np.round(tcp.get_world_poses()[0].numpy()[0], 3)}")
                step += 1
                if step >= steps_per_episode or not simulation_app.is_running():
                    break

    if args.start_mode == "expert":
        # Same reset as the scripted expert (ready pose above a fixed-height table, random cube), policy drives.
        run_expert(robot, cube, tcp, cams, la, lf, rf, ctrl_idx, decimation, read_state, cube_in_box,
                   policy_episode=policy_episode)
    else:
        successes, ep = 0, -1
        while simulation_app.is_running():
            ep += 1
            if args.episodes and ep >= args.episodes:
                break
            start_i = pick()
            table_z, _ = reset(start_i)
            policy_episode(ep, f"start#{start_i}")
            ok, c = cube_in_box(table_z)
            successes += ok
            print(f"[sim] episode {ep} done: cube={np.round(c, 3)} box={np.round(BOX_XY, 3)} -> "
                  f"{'SUCCESS (cube in box)' if ok else 'fail'}   [{successes}/{ep + 1} successes]")

    sock.close()
    if args.keep_open and not args.headless:
        print("[sim] episodes finished; window stays open (close it to exit)")
        while simulation_app.is_running():
            simulation_app.update()
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[sim] total {time.time() - t0:.1f}s")
