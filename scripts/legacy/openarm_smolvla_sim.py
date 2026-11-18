"""Run a SmolVLA policy on the OpenArm (unimanual) in Isaac Sim 6.

Launch with Isaac Sim's python:   D:\\Isaac\\isaacsim\\python.bat openarm_smolvla_sim.py [--headless] [--task "..."]
Requires policy_server.py to be running (SmolVLA conda env). See run_all.bat.

Loop: render 3 cameras (front / wrist / side) + read 8-D state (7 arm joints + gripper opening)
-> policy server returns an action chunk -> execute the first --exec-steps actions as joint position
targets at --control-hz, then re-plan.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--task", default="pick up the red cube and place it on the green target")
parser.add_argument("--episodes", type=int, default=0, help="number of episodes; 0 = run forever until the window is closed")
parser.add_argument("--keep-open", action="store_true", help="keep the Isaac Sim window open after the last episode")
parser.add_argument("--episode-seconds", type=float, default=30.0)
parser.add_argument("--control-hz", type=float, default=30.0)
parser.add_argument("--exec-steps", type=int, default=25, help="actions executed from each chunk before re-planning")
parser.add_argument("--action-mode", choices=["abs", "delta"], default="abs",
                    help="abs: actions are joint targets; delta: actions are added to the current joint positions")
parser.add_argument("--max-step-rad", type=float, default=0.05, help="per control step joint target rate limit (rad)")
parser.add_argument("--img-size", type=int, default=256)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--save-frames", default="", help="directory to dump camera frames (every re-plan)")
parser.add_argument("--probe", action="store_true", help="build scene, print robot info, save one frame set, exit")
# End-effector-delta policies (OpenVLA): the server reports action_type="eef_delta".
parser.add_argument("--policy-hz", type=float, default=5.0, help="rate the eef-delta policy was trained at (Bridge: 5 Hz)")
parser.add_argument("--eef-pos-scale", type=float, default=1.0)
parser.add_argument("--eef-rot-scale", type=float, default=1.0)
parser.add_argument("--eef-ready", choices=["topdown", "home"], default="topdown",
                    help="pose to move the gripper to (via IK) before an eef-delta episode starts")
parser.add_argument("--ik-damping", type=float, default=0.05)
# Scene coordinates (world frame, metres; robot base at origin, +X forward over the table, +Y to the robot's left,
# table top at Z=0).
parser.add_argument("--cube-pos", type=float, nargs=2, default=[0.40, 0.10], metavar=("X", "Y"))
parser.add_argument("--cube-jitter", type=float, default=0.0, help="random +/- offset (m) added to the cube each episode")
parser.add_argument("--target-pos", type=float, nargs=2, default=[0.40, -0.15], metavar=("X", "Y"))
parser.add_argument("--ready-pos", type=float, nargs=3, default=[0.40, 0.0, 0.20], metavar=("X", "Y", "Z"),
                    help="TCP (between the fingertips) position the hand moves to before an eef-delta episode")
parser.add_argument("--home-deg", type=float, nargs=7, default=None, metavar="J",
                    help="start pose of the arm: 7 joint angles in degrees (default 90 0 -90 90 0 0 0)")
parser.add_argument("--eef-axis-signs", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"),
                    help="multiply the policy's dx/dy/dz by these, e.g. 1 -1 1 if its Y axis is mirrored")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.materials import OmniPbrMaterial  # noqa: E402
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
    "openarm_manipulation", "usds", "openarm_unimanual", "openarm_unimanual.usd",
).replace("\\", "/")

ROBOT = "/World/openarm"
ARM_JOINTS = [f"openarm_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["openarm_finger_joint1", "openarm_finger_joint2"]
# Same home pose / gains as openarm_isaac_lab's OPEN_ARM_CFG.
HOME = np.array([1.57, 0.0, -1.57, 1.57, 0.0, 0.0, 0.0], dtype=np.float32)
GRIPPER_OPEN, GRIPPER_CLOSED = 0.044, 0.0
ARM_LO = np.deg2rad([-80, -100, -90, 0, -90, -45, -90]).astype(np.float32)
ARM_HI = np.deg2rad([200, 100, 90, 140, 90, 45, 90]).astype(np.float32)

PHYSICS_HZ = 120.0
CAM_NAMES = ("front", "wrist", "side")
# eef-delta mode: TCP workspace box (world, m) and the top-down ready pose (TCP +Z = approach pointing down).
EEF_LO = np.array([0.15, -0.45, 0.01])
EEF_HI = np.array([0.75, 0.45, 0.50])
if args.home_deg is not None:
    HOME = np.clip(np.deg2rad(args.home_deg), ARM_LO, ARM_HI).astype(np.float32)
READY_POS = np.array(args.ready_pos)
CUBE_XY = np.array(args.cube_pos)
TARGET_XY = np.array(args.target_pos)
TOPDOWN_QUAT = np.array([0.0, 0.0, 1.0, 0.0])  # wxyz, 180 deg about world Y


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_rpy(r, p, y):
    """Extrinsic X-Y-Z (roll, pitch, yaw) -> wxyz."""
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """wxyz orientation of a USD camera (looks down -Z, +Y up) at `eye` looking at `target` (parent frame)."""
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up))
    q = view.GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()], dtype=np.float64)


def make_camera(path: str, eye, target, up=(0.0, 0.0, 1.0), focal=18.0) -> CameraSensor:
    cam = RtxCamera(path, translations=np.array(eye, dtype=np.float64), orientations=look_at_quat(eye, target, up))
    usd_cam = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    usd_cam.GetFocalLengthAttr().Set(focal)
    usd_cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(args.img_size, args.img_size), annotators=["rgb"])


def build_scene():
    stage_utils.create_new_stage()
    SimulationManager.set_physics_dt(1.0 / PHYSICS_HZ)
    DomeLight("/World/DomeLight").set_intensities(1200)
    GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, -0.75])

    stage_utils.add_reference_to_stage(usd_path=OPENARM_USD, path=ROBOT)

    def box(path, size, pos, color, dynamic=False):
        c = Cube(path, sizes=1.0, positions=np.array(pos), scales=np.array(size))
        GeomPrim(path, apply_collision_apis=True)
        if dynamic:
            RigidPrim(path)
        mat = OmniPbrMaterial(path.replace("/World/", "/World/Materials/"))
        mat.set_input_values("diffuse_color_constant", list(color))
        c.apply_visual_materials(mat)
        return c

    # Table top surface at z = 0 (robot base sits on it, like the openarm_isaac_lab lift task).
    box("/World/table", (1.2, 1.2, 0.75), (0.45, 0.0, -0.375), (0.55, 0.42, 0.3))
    box("/World/cube", (0.04, 0.04, 0.04), (*CUBE_XY, 0.02), (0.9, 0.1, 0.1), dynamic=True)
    box("/World/target", (0.10, 0.10, 0.002), (*TARGET_XY, 0.001), (0.1, 0.8, 0.2))

    cams = {
        "front": make_camera("/World/cam_front", eye=(1.05, 0.0, 0.5), target=(0.3, 0.0, 0.05)),
        # openarm_hand frame: +Z = approach (TCP at z=0.093), -X = back of the gripper ("up" at home pose).
        "wrist": make_camera(f"{ROBOT}/openarm_hand/cam_wrist", eye=(-0.06, 0.0, 0.0), target=(0.0, 0.0, 0.2),
                             up=(-1.0, 0.0, 0.0), focal=12.0),
        "side": make_camera("/World/cam_side", eye=(0.45, 1.0, 0.6), target=(0.4, 0.0, 0.05)),
    }
    return cams


def grab(cams) -> list[np.ndarray]:
    out = []
    for name in CAM_NAMES:
        data, _ = cams[name].get_data("rgb")
        if data is None:
            img = np.zeros((args.img_size, args.img_size, 3), np.uint8)
        else:
            img = data.numpy()
            if not hasattr(grab, "logged"):
                grab.logged = True
                print(f"[sim] rgb annotator: shape={img.shape} dtype={img.dtype} max={img.max()}")
            img = img[..., :3].astype(np.uint8)
        out.append(np.ascontiguousarray(img))
    return out


def save_frames(imgs, tag):
    if not args.save_frames:
        return
    from PIL import Image

    os.makedirs(args.save_frames, exist_ok=True)
    for name, img in zip(CAM_NAMES, imgs):
        Image.fromarray(img).save(os.path.join(args.save_frames, f"{tag}_{name}.png"))


def main():
    cams = build_scene()
    robot = Articulation(ROBOT)
    cube = RigidPrim("/World/cube")
    tcp = RigidPrim(f"{ROBOT}/openarm_ee_tcp")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    names = robot.dof_names
    arm_idx = [names.index(j) for j in ARM_JOINTS]
    fing_idx = [names.index(j) for j in FINGER_JOINTS]
    act_idx = arm_idx + fing_idx
    print(f"[sim] dofs={names}")

    home_full = np.zeros(len(names), np.float32)
    home_full[arm_idx] = HOME
    home_full[fing_idx] = GRIPPER_OPEN
    robot.set_dof_gains(stiffnesses=[80.0] * 7, dampings=[4.0] * 7, dof_indices=arm_idx)
    robot.set_dof_gains(stiffnesses=[2e3] * 2, dampings=[1e2] * 2, dof_indices=fing_idx)
    robot.set_dof_max_efforts([40, 40, 27, 27, 7, 7, 7], dof_indices=arm_idx)

    def reset():
        robot.set_dof_positions(home_full)
        robot.set_dof_velocities(np.zeros_like(home_full))
        robot.set_dof_position_targets(home_full)
        cube_xy = CUBE_XY + np.random.uniform(-args.cube_jitter, args.cube_jitter, 2)
        cube.set_world_poses(positions=np.array([[cube_xy[0], cube_xy[1], 0.02]]),
                             orientations=np.array([[1.0, 0.0, 0.0, 0.0]]))
        cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        for _ in range(30):
            simulation_app.update()

    def read_state():
        q = robot.get_dof_positions().numpy()[0]
        return np.concatenate([q[arm_idx], q[fing_idx[:1]]]).astype(np.float32)

    reset()
    tcp_pos = tcp.get_world_poses()[0].numpy()[0]
    print(f"[sim] home state={np.round(read_state(), 3)} tcp_world={np.round(tcp_pos, 3)}")

    if args.probe:
        hp, hq = (x.numpy()[0] for x in RigidPrim(f"{ROBOT}/openarm_hand").get_world_poses())
        rot = np.array(Gf.Matrix3d(Gf.Quatd(float(hq[0]), Gf.Vec3d(*map(float, hq[1:])))))  # rows = local axes in world
        print(f"[sim] hand pos={np.round(hp, 3)} quat={np.round(hq, 3)}")
        print(f"[sim] hand local x/y/z in world: {np.round(rot, 2).tolist()}; tcp in hand frame="
              f"{np.round(rot @ (tcp_pos - hp), 3)}")
        for i in range(300):
            simulation_app.update()
            if i % 50 == 49:
                imgs = grab(cams)
                print(f"[sim] warmup {i + 1}: mean px " + ", ".join(f"{im.mean():.1f}" for im in imgs))
        save_frames(imgs, "probe")
        print(f"[sim] probe frames -> {args.save_frames}")
        timeline.stop()
        simulation_app.close()
        return

    sock = socket.create_connection((HOST, args.port), timeout=300)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"cmd": "info"})
    info = recv_msg(sock)
    action_type = info.get("action_type", "joint")
    print(f"[sim] connected to policy server: {info}")

    decimation = max(1, round(PHYSICS_HZ / args.control_hz))
    steps_per_episode = int(args.episode_seconds * args.control_hz)

    # ---- end-effector-delta control (OpenVLA) via damped-least-squares IK on the TCP link ----
    tcp_jac_idx = int(robot.get_link_indices("openarm_ee_tcp").numpy()[0]) - 1  # fixed base: no root row
    if action_type == "eef_delta":
        # Same as openarm_isaac_lab's OPEN_ARM_HIGH_PD_CFG (used there for task-space IK control).
        robot.set_dof_gains(stiffnesses=[400.0] * 7, dampings=[80.0] * 7, dof_indices=arm_idx)
        robot.set_link_enabled_gravities(np.zeros((1, robot.num_links), dtype=bool))

    def tcp_pose():
        p, q = tcp.get_world_poses()
        return p.numpy()[0].astype(np.float64), q.numpy()[0].astype(np.float64)

    def ik_arm_target(goal_pos, goal_quat):
        q = robot.get_dof_positions().numpy()[0]
        jac = robot.get_jacobian_matrices().numpy()[0, tcp_jac_idx][:, arm_idx].astype(np.float64)  # (6, 7)
        pos, quat = tcp_pose()
        dq_err = quat_mul(goal_quat, quat_conj(quat))
        err = np.concatenate([goal_pos - pos, dq_err[1:] * np.sign(dq_err[0])])
        lam = args.ik_damping ** 2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam, err)
        dq = np.clip(dq, -args.max_step_rad * 2, args.max_step_rad * 2)
        return np.clip(q[arm_idx] + dq, ARM_LO, ARM_HI).astype(np.float32), np.linalg.norm(err[:3])

    def drive_eef(goal_pos, goal_quat, gripper, n_steps):
        for _ in range(n_steps):
            arm, pos_err = ik_arm_target(goal_pos, goal_quat)
            robot.set_dof_position_targets(np.concatenate([arm, [gripper, gripper]]), dof_indices=act_idx)
            for _ in range(decimation):
                simulation_app.update()
        return pos_err

    def go_ready():
        _, q_now = tcp_pose()
        goal_q = TOPDOWN_QUAT if args.eef_ready == "topdown" else q_now
        err = drive_eef(READY_POS, goal_q, GRIPPER_OPEN, int(3 * args.control_hz))
        print(f"[sim] ready pose ({args.eef_ready}) reached, pos err {1000 * err:.1f} mm, tcp={np.round(tcp_pose()[0], 3)}")

    steps_per_eef_action = max(1, round(args.control_hz / args.policy_hz))

    ep = -1
    while simulation_app.is_running():
        ep += 1
        if args.episodes and ep >= args.episodes:
            break
        reset()
        send_msg(sock, {"cmd": "reset"})
        recv_msg(sock)
        target = read_state()
        step, replans = 0, 0
        if action_type == "eef_delta":
            go_ready()
            gripper = GRIPPER_OPEN
        while step < steps_per_episode and simulation_app.is_running():
            imgs = grab(cams)
            state = read_state()
            send_msg(sock, {"cmd": "infer", "state": state, "images": imgs, "cam_names": list(CAM_NAMES),
                            "task": args.task})
            resp = recv_msg(sock)
            if action_type == "eef_delta":
                a = resp["actions"][0].astype(np.float64)
                pos, quat = tcp_pose()
                goal_pos = np.clip(pos + a[:3] * np.array(args.eef_axis_signs) * args.eef_pos_scale, EEF_LO, EEF_HI)
                goal_quat = quat_mul(quat_from_rpy(*(a[3:6] * args.eef_rot_scale)), quat)
                gripper = GRIPPER_OPEN if a[6] > 0.5 else GRIPPER_CLOSED  # Bridge convention: 1 = open
                if replans % 10 == 0:
                    print(f"[sim] ep {ep} step {step}: infer {resp['ms']:.0f} ms, tcp={np.round(pos, 3)}, "
                          f"a={np.round(a, 3)}, cube={np.round(cube.get_world_poses()[0].numpy()[0], 3)}")
                    save_frames(imgs, f"ep{ep}_step{step:05d}")
                replans += 1
                drive_eef(goal_pos, goal_quat, gripper, steps_per_eef_action)
                step += steps_per_eef_action
                continue
            chunk = resp["actions"][: args.exec_steps]
            if replans % 10 == 0:
                print(f"[sim] ep {ep} step {step}: infer {resp['ms']:.0f} ms, state={np.round(state, 2)}, "
                      f"a0={np.round(chunk[0], 2)}, cube={np.round(cube.get_world_poses()[0].numpy()[0], 3)}")
                save_frames(imgs, f"ep{ep}_step{step:05d}")
            replans += 1
            for a in chunk:
                a = a.astype(np.float32)
                desired = read_state() + a if args.action_mode == "delta" else a
                desired[:7] = np.clip(desired[:7], ARM_LO, ARM_HI)
                desired[7] = np.clip(desired[7], GRIPPER_CLOSED, GRIPPER_OPEN)
                # Rate-limit so a zero-shot policy can't slam the arm.
                target = target + np.clip(desired - target, -args.max_step_rad, args.max_step_rad)
                robot.set_dof_position_targets(np.concatenate([target, target[7:8]]), dof_indices=act_idx)
                for _ in range(decimation):
                    simulation_app.update()
                step += 1
                if step >= steps_per_episode or not simulation_app.is_running():
                    break
        cube_final = cube.get_world_poses()[0].numpy()[0]
        dist = np.linalg.norm(cube_final[:2] - TARGET_XY)
        print(f"[sim] episode {ep} done: {step} steps, cube final={np.round(cube_final, 3)}, "
              f"distance to green target {100 * dist:.1f} cm -> {'SUCCESS' if dist < 0.05 else 'fail'}")

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
