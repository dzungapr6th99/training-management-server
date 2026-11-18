"""YellowArm (SkyentificGit/YellowArmSim) pick-and-place in Isaac Sim 6: scripted-expert data generation and
SmolVLA policy evaluation in one scene.

  Expert demos : D:\\Isaac\\isaacsim\\python.bat yellowarm_sim.py --expert --record DIR --episodes 300 --headless
  Policy eval  : D:\\Isaac\\isaacsim\\python.bat yellowarm_sim.py --port 5556 [--headless --episodes 20]
  Probe        : ... --probe --save-frames DIR

Robot: 7 revolute joints + 2 mirrored prismatic fingers (joint8/9: -0.022 = open, 75 mm gap; +0.015 = closed).
The USD mounts the arm sideways at 1.2 m; here it stands upright on the table top (z = 0), reaching along +X.
State (8) = joint1..7 (rad) + gripper opening (0 closed .. 1 open). Action (8) = joint1..7 targets + gripper command.
Cameras (named like lerobot/smolvla_base's keys): camera1 = front, camera2 = wrist, camera3 = side.
Every episode randomizes the cube pose, the box position and the arm's start pose.
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
parser.add_argument("--task", default="put the cube in the box")
parser.add_argument("--episodes", type=int, default=0, help="0 = run until the window is closed")
parser.add_argument("--episode-seconds", type=float, default=10.0)
parser.add_argument("--control-hz", type=float, default=25.0)
parser.add_argument("--physics-hz", type=float, default=100.0)
parser.add_argument("--exec-steps", type=int, default=8, help="policy actions executed per chunk before re-planning (8 beat 16: 18/20 vs 12/20)")
parser.add_argument("--max-step-rad", type=float, default=0.3)
parser.add_argument("--img-size", type=int, default=256)
parser.add_argument("--port", type=int, default=5556)
parser.add_argument("--save-frames", default="")
parser.add_argument("--probe", action="store_true")
parser.add_argument("--seed", type=int, default=0, help="policy eval: episode i is seeded with seed + i (resumable)")
parser.add_argument("--results-json", default="", help="policy eval: append one JSON line per episode to this file")
parser.add_argument("--keep-open", action="store_true")
# randomization (world frame, metres; robot base at the origin on the table top, +X forward, +Y left)
parser.add_argument("--cube-x", type=float, nargs=2, default=[0.22, 0.46])
parser.add_argument("--cube-y", type=float, nargs=2, default=[-0.05, 0.25])
parser.add_argument("--cube-yaw", type=float, default=45.0, help="+/- degrees")
parser.add_argument("--box-x", type=float, nargs=2, default=[0.24, 0.44])
parser.add_argument("--box-y", type=float, nargs=2, default=[-0.32, -0.16])
parser.add_argument("--ready-jitter", type=float, default=0.08, help="+/- m per axis around the ready pose")
parser.add_argument("--cube-friction", type=float, default=1.5)
# expert
parser.add_argument("--expert", action="store_true", help="scripted IK pick-and-place instead of a policy")
parser.add_argument("--record", default="", help="save successful expert episodes (.npz) here")
args, _ = parser.parse_known_args()
sys.stdout.reconfigure(line_buffering=True)

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wire import HOST, recv_msg, send_msg  # noqa: E402

from mhh_paths import YELLOWARM_USD as _USD  # noqa: E402
USD = _USD.as_posix()
ROBOT = "/World/yellowarm"
ARM = [f"yellowarm_joint{i}" for i in range(1, 8)]
FINGERS = ["yellowarm_joint8", "yellowarm_joint9"]
F_OPEN, F_CLOSED = -0.022, 0.015
TCP_OFFSET = 0.085  # grasp point along link7 +Z (fingertips span 0.064-0.117)
CAM_NAMES = ("camera1", "camera2", "camera3")
CUBE, BOX_OUT, BOX_WALL, BOX_H = 0.04, 0.14, 0.01, 0.04
rng = np.random.default_rng(args.seed)


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, np.array([0.0, *v])), np.array([q[0], -q[1], -q[2], -q[3]]))[1:]


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def down_quat(yaw):  # link7 +Z pointing down, fingers closing along the yaw-rotated world Y
    return quat_mul(yaw_quat(yaw), np.array([0.0, 0.0, 1.0, 0.0]))


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def make_camera(path, eye, target, up=(0.0, 0.0, 1.0), focal=18.0):
    cam = RtxCamera(path, translations=np.array(eye, float), orientations=look_at_quat(eye, target, up))
    c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    c.GetFocalLengthAttr().Set(focal)
    c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(args.img_size, args.img_size), annotators=["rgb"])


def set_translate(path, xyz):
    UsdGeom.XformCommonAPI(omni.usd.get_context().get_stage().GetPrimAtPath(path)).SetTranslate(Gf.Vec3d(*map(float, xyz)))


def build_scene():
    stage_utils.create_new_stage()
    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    DomeLight("/World/Light").set_intensities(1200)
    GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, -0.75])
    stage_utils.add_reference_to_stage(usd_path=USD, path=ROBOT)
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(ROBOT)
    root.GetAttribute("xformOp:orient").Set(Gf.Quatd(1, 0, 0, 0))  # upright, base on the table top
    root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))

    def box(path, size, pos, color, dynamic=False):
        c = Cube(path, sizes=1.0, positions=np.array(pos), scales=np.array(size))
        g = GeomPrim(path, apply_collision_apis=True)
        if dynamic:
            RigidPrim(path)
        m = OmniPbrMaterial("/World/Materials/" + path.strip("/").replace("/", "_"))
        m.set_input_values("diffuse_color_constant", list(color))
        c.apply_visual_materials(m)
        return g

    box("/World/table", (1.1, 1.3, 0.75), (0.30, 0.0, -0.375), (0.55, 0.45, 0.33))
    UsdGeom.Xform.Define(stage, "/World/box_rig")
    blue, o, w, h = (0.12, 0.3, 0.85), BOX_OUT, BOX_WALL, BOX_H
    box("/World/box_rig/floor", (o, o, w), (0, 0, w / 2), blue)
    box("/World/box_rig/wall_px", (w, o, h), ((o - w) / 2, 0, h / 2), blue)
    box("/World/box_rig/wall_nx", (w, o, h), (-(o - w) / 2, 0, h / 2), blue)
    box("/World/box_rig/wall_py", (o, w, h), (0, (o - w) / 2, h / 2), blue)
    box("/World/box_rig/wall_ny", (o, w, h), (0, -(o - w) / 2, h / 2), blue)
    cube = box("/World/cube", (CUBE, CUBE, CUBE), (0.3, 0.1, 0.02), (0.92, 0.2, 0.1), dynamic=True)
    grip = RigidBodyMaterial("/World/PhysicsMaterials/grippy")
    grip.set_friction_coefficients(static_frictions=args.cube_friction, dynamic_frictions=args.cube_friction)
    grip.set_combine_modes(frictions="max")
    cube.apply_physics_materials(grip)
    return {
        "camera1": make_camera("/World/cam_front", eye=(1.05, 0.0, 0.62), target=(0.32, 0.0, 0.0)),
        # link7 frame: +Z = approach, fingers along Y; with the top-down grasp link7 +X points back at the robot.
        "camera2": make_camera(f"{ROBOT}/yellowarm_link7/cam_wrist", eye=(0.10, 0.0, 0.03), target=(0.0, 0.0, 0.17),
                               up=(1.0, 0.0, 0.0), focal=11.0),
        "camera3": make_camera("/World/cam_side", eye=(0.34, 0.95, 0.55), target=(0.34, -0.05, 0.0)),
    }


def grab(cams):
    out = []
    for n in CAM_NAMES:
        d, _ = cams[n].get_data("rgb")
        img = np.zeros((args.img_size, args.img_size, 3), np.uint8) if d is None else d.numpy()[..., :3]
        out.append(np.ascontiguousarray(img.astype(np.uint8)))
    return out


def save_frames(imgs, tag):
    if not args.save_frames:
        return
    from PIL import Image

    os.makedirs(args.save_frames, exist_ok=True)
    for n, img in zip(CAM_NAMES, imgs):
        Image.fromarray(img).save(os.path.join(args.save_frames, f"{tag}_{n}.png"))


def main():
    global rng
    cams = build_scene()
    robot = Articulation(ROBOT)
    cube = RigidPrim("/World/cube")
    link7 = RigidPrim(f"{ROBOT}/yellowarm_link7")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    names = robot.dof_names
    ai = [names.index(j) for j in ARM]
    fi = [names.index(j) for j in FINGERS]
    ctrl = ai + fi
    lo, hi = (x.numpy()[0] for x in robot.get_dof_limits())
    LO, HI = lo[ai], hi[ai]
    jac_row = int(robot.get_link_indices("yellowarm_link7").numpy()[0]) - 1
    decimation = max(1, round(args.physics_hz / args.control_hz))

    def tcp_pose():
        p, q = (x.numpy()[0].astype(np.float64) for x in link7.get_world_poses())
        return p + quat_rotate(q, np.array([0.0, 0.0, TCP_OFFSET])), q

    def read_state():
        q = robot.get_dof_positions().numpy()[0]
        g = np.clip((F_CLOSED - q[fi].mean()) / (F_CLOSED - F_OPEN), 0.0, 1.0)
        return np.concatenate([q[ai], [g]]).astype(np.float32)

    def apply(arm, grip):
        f = F_CLOSED + (F_OPEN - F_CLOSED) * float(np.clip(grip, 0.0, 1.0))
        robot.set_dof_position_targets(np.concatenate([arm, [f, f]]), dof_indices=ctrl)
        for _ in range(decimation):
            simulation_app.update()

    def ik_target(goal_p, goal_q):
        q = robot.get_dof_positions().numpy()[0].astype(np.float64)
        J = robot.get_jacobian_matrices().numpy()[0, jac_row][:, ai].astype(np.float64)
        p7, q7 = (x.numpy()[0].astype(np.float64) for x in link7.get_world_poses())
        r = quat_rotate(q7, np.array([0.0, 0.0, TCP_OFFSET]))
        Jt = J.copy()  # velocity of the TCP point = v_link7 + w x r
        Jt[:3] = J[:3] - np.cross(r[None, :], J[3:].T).T
        p, rq = p7 + r, q7
        dq_err = quat_mul(goal_q, np.array([rq[0], -rq[1], -rq[2], -rq[3]]))
        err = np.concatenate([goal_p - p, dq_err[1:] * np.sign(dq_err[0])])
        dq = Jt.T @ np.linalg.solve(Jt @ Jt.T + 0.05 ** 2 * np.eye(6), err)
        return np.clip(q[ai] + np.clip(dq, -0.15, 0.15), LO, HI)

    def cube_in_box(box_xy):
        c = cube.get_world_poses()[0].numpy()[0]
        inside = np.all(np.abs(c[:2] - box_xy) < BOX_OUT / 2 - BOX_WALL) and c[2] < BOX_H + 0.02
        return bool(inside), c

    # -------- episode reset: random box, arm to a jittered ready pose (IK, unrecorded), random cube --------
    HOME = np.zeros(len(names), np.float32)
    HOME[fi] = F_OPEN

    def reset():
        cube.set_world_poses(positions=np.array([[0.0, 0.9, 0.05]]))
        robot.set_dof_positions(HOME)
        robot.set_dof_velocities(np.zeros_like(HOME))
        robot.set_dof_position_targets(HOME)
        box_xy = np.array([rng.uniform(*args.box_x), rng.uniform(*args.box_y)])
        set_translate("/World/box_rig", (box_xy[0], box_xy[1], 0.0))
        for _ in range(3):
            simulation_app.update()
        ready = np.array([0.32, 0.05, 0.22]) + rng.uniform(-args.ready_jitter, args.ready_jitter, 3)
        ready_q = down_quat(rng.uniform(-0.4, 0.4))
        for _ in range(int(3.0 * args.control_hz)):
            apply(ik_target(ready, ready_q), 1.0)
        err = np.linalg.norm(tcp_pose()[0] - ready)
        while True:  # cube away from the box
            c_xy = np.array([rng.uniform(*args.cube_x), rng.uniform(*args.cube_y)])
            if np.linalg.norm(c_xy - box_xy) > 0.16:
                break
        yaw = np.deg2rad(rng.uniform(-args.cube_yaw, args.cube_yaw))
        cube.set_world_poses(positions=np.array([[c_xy[0], c_xy[1], CUBE / 2 + 0.001]]), orientations=yaw_quat(yaw)[None])
        cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        for _ in range(10):
            simulation_app.update()
        return box_xy, c_xy, yaw, ready, err

    if args.probe:
        box_xy, c_xy, yaw, ready, err = reset()
        for _ in range(60):
            simulation_app.update()
        imgs = grab(cams)
        print(f"[yellow] probe: ready err {1000 * err:.1f} mm tcp={np.round(tcp_pose()[0], 3)} cube={np.round(c_xy, 3)} "
              f"box={np.round(box_xy, 3)} state={np.round(read_state(), 2)} px={[round(float(i.mean()), 1) for i in imgs]}")
        save_frames(imgs, "probe")
        timeline.stop()
        simulation_app.close()
        return

    wins = tries = saved = 0
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        saved = len([f for f in os.listdir(args.record) if f.endswith(".npz")])

    if args.expert:
        rec = None

        def step(arm, grip):
            if rec is not None:
                rec["state"].append(read_state())
                for n, img in zip(CAM_NAMES, grab(cams)):
                    rec[n].append(img)
                rec["action"].append(np.concatenate([arm, [grip]]).astype(np.float32))
            apply(arm, grip)

        def move(goal_p, goal_q, grip, speed, settle=0, noise=0.0):
            p0 = tcp_pose()[0]
            n = max(1, int(np.ceil(np.linalg.norm(goal_p - p0) / (speed / args.control_hz))))
            # a smooth random detour (zero at both ends) so the demos also show recovering from being off-path
            bend = rng.normal(0, noise, 3) if noise else np.zeros(3)
            for k in range(1, n + 1):
                s = k / n
                step(ik_target(p0 + (goal_p - p0) * s + bend * np.sin(np.pi * s), goal_q), grip)
            for _ in range(settle):
                step(ik_target(goal_p, goal_q), grip)

        while simulation_app.is_running() and (not args.episodes or saved < args.episodes):
            tries += 1
            box_xy, c_xy, yaw, ready, err = reset()
            gq = down_quat(((yaw + np.pi / 4) % (np.pi / 2)) - np.pi / 4)  # grasp across the cube faces
            rec = {"state": [], "action": [], **{n: [] for n in CAM_NAMES}}
            above = np.array([c_xy[0], c_xy[1], 0.12])
            move(above, gq, 1.0, 0.30, noise=0.03)
            move(np.array([c_xy[0], c_xy[1], 0.018]), gq, 1.0, 0.15, settle=4)
            for _ in range(10):
                step(ik_target(np.array([c_xy[0], c_xy[1], 0.018]), gq), 0.0)
            move(np.array([c_xy[0], c_xy[1], 0.15]), gq, 0.0, 0.20)
            over = np.array([box_xy[0], box_xy[1], 0.15])
            move(over, down_quat(0.0), 0.0, 0.30, noise=0.03)
            move(np.array([box_xy[0], box_xy[1], 0.10]), down_quat(0.0), 0.0, 0.15, settle=2)
            for _ in range(8):
                step(ik_target(np.array([box_xy[0], box_xy[1], 0.10]), down_quat(0.0)), 1.0)
            move(np.array([box_xy[0], box_xy[1], 0.18]), down_quat(0.0), 1.0, 0.20)
            ep, rec = rec, None
            for _ in range(20):
                simulation_app.update()
            ok, c = cube_in_box(box_xy)
            T = len(ep["action"])
            print(f"[expert] try {tries}: ready err {1000 * err:.0f} mm cube {np.round(c_xy, 3)} yaw {np.degrees(yaw):.0f} "
                  f"box {np.round(box_xy, 3)} {T} frames -> {'SUCCESS' if ok else 'fail'}")
            if ok and args.record:
                path = os.path.join(args.record, f"episode_{saved:04d}.npz")
                np.savez_compressed(path, state=np.stack(ep["state"]), action=np.stack(ep["action"]),
                                    **{n: np.stack(ep[n]) for n in CAM_NAMES}, task=args.task, fps=args.control_hz)
            if ok:
                saved += 1
        print(f"[expert] done: {saved} episodes saved, {tries} tries")
        timeline.stop()
        simulation_app.close()
        return

    # ---------------------------------------- policy evaluation ----------------------------------------
    sock = socket.create_connection((HOST, args.port), timeout=300)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"cmd": "info"})
    print(f"[yellow] connected to policy server: {recv_msg(sock)}")
    steps = int(args.episode_seconds * args.control_hz)
    results = None
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        results = open(args.results_json, "a", encoding="utf-8")
    while simulation_app.is_running() and (not args.episodes or tries < args.episodes):
        ep_seed = args.seed + tries
        rng = np.random.default_rng(ep_seed)  # per-episode seed: a crashed run can resume at seed + done
        t_ep = time.time()
        box_xy, c_xy, yaw, ready, err = reset()
        send_msg(sock, {"cmd": "reset"})
        recv_msg(sock)
        target = read_state()
        step = 0
        infer_ms, server_ms = [], []
        max_z, first_in = 0.0, None
        while step < steps and simulation_app.is_running():
            imgs = grab(cams)
            t_inf = time.perf_counter()
            send_msg(sock, {"cmd": "infer", "state": read_state(), "images": imgs, "cam_names": list(CAM_NAMES),
                            "task": args.task})
            reply = recv_msg(sock)
            chunk = reply["actions"][: args.exec_steps]
            infer_ms.append(1000.0 * (time.perf_counter() - t_inf))
            if "infer_ms" in reply:
                server_ms.append(float(reply["infer_ms"]))
            if step == 0:
                save_frames(imgs, f"ep{tries}")
            for a in chunk:
                a = a.astype(np.float32)
                target[:7] += np.clip(np.clip(a[:7], LO, HI) - target[:7], -args.max_step_rad, args.max_step_rad)
                target[7] = a[7]
                apply(target[:7], target[7])
                step += 1
                inside, c = cube_in_box(box_xy)
                max_z = max(max_z, float(c[2]))
                if inside and first_in is None:
                    first_in = step
                if step >= steps or not simulation_app.is_running():
                    break
        for _ in range(20):
            simulation_app.update()
        ok, c = cube_in_box(box_xy)
        tries += 1
        wins += ok
        print(f"[yellow] episode {tries - 1}: cube {np.round(c_xy, 3)} box {np.round(box_xy, 3)} end {np.round(c, 3)} -> "
              f"{'SUCCESS' if ok else 'fail'}   [{wins}/{tries}]")
        if results:
            results.write(json.dumps({
                "episode": tries - 1, "seed": ep_seed, "success": ok, "steps": step,
                "seconds": round(time.time() - t_ep, 2), "first_in_box_step": first_in,
                "cube_lifted": max_z > CUBE / 2 + 0.03, "cube_max_z": round(max_z, 4),
                "end_dist_to_box": round(float(np.linalg.norm(c[:2] - box_xy)), 4),
                "cube_start": np.round(c_xy, 4).tolist(), "cube_yaw_deg": round(float(np.degrees(yaw)), 1),
                "box": np.round(box_xy, 4).tolist(), "cube_end": np.round(c.astype(np.float64), 4).tolist(),
                "ready_err_mm": round(1000 * float(err), 1), "n_infer": len(infer_ms),
                "infer_ms_mean": round(float(np.mean(infer_ms)), 2) if infer_ms else None,
                "infer_ms_max": round(float(np.max(infer_ms)), 2) if infer_ms else None,
                "server_infer_ms_mean": round(float(np.mean(server_ms)), 2) if server_ms else None,
            }) + "\n")
            results.flush()
    print(f"[yellow] evaluation done: {wins}/{tries} cubes in the box")
    sock.close()
    if results:
        results.close()
    if args.keep_open and not args.headless:
        while simulation_app.is_running():
            simulation_app.update()
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[yellow] total {time.time() - t0:.1f}s")
