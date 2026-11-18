"""YellowArm "stack three cubes" in Isaac Sim 6: scripted-expert data generation and SmolVLA policy evaluation.

Order (fixed, so the prompt is meaningful): blue at the bottom, green in the middle, red on top.
  Expert demos : D:\\Isaac\\isaacsim\\python.bat yellowarm_stack_sim.py --expert --record DIR --episodes 300 --headless
  Policy eval  : D:\\Isaac\\isaacsim\\python.bat yellowarm_stack_sim.py --port 5556 [--headless --episodes 20]
  Probe        : ... --probe --save-frames DIR

Same robot setup, cameras and state/action format as yellowarm_sim.py (state 8 = joint1..7 + gripper opening 0..1,
action 8 = joint targets + gripper command; camera1 front, camera2 wrist, camera3 side), so checkpoints trained there
can be fine-tuned here. Every episode randomizes the three cube positions/yaws and the arm's start pose.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--task", default="stack the green cube on the blue cube and the red cube on the green cube")
parser.add_argument("--episodes", type=int, default=0, help="0 = run until the window is closed")
parser.add_argument("--episode-seconds", type=float, default=16.0)
parser.add_argument("--control-hz", type=float, default=25.0)
parser.add_argument("--physics-hz", type=float, default=100.0)
parser.add_argument("--exec-steps", type=int, default=8)
parser.add_argument("--max-step-rad", type=float, default=0.3)
parser.add_argument("--img-size", type=int, default=256)
parser.add_argument("--port", type=int, default=5556)
parser.add_argument("--save-frames", default="")
parser.add_argument("--probe", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--keep-open", action="store_true")
parser.add_argument("--cube-x", type=float, nargs=2, default=[0.24, 0.46])
parser.add_argument("--cube-y", type=float, nargs=2, default=[-0.25, 0.22])
parser.add_argument("--cube-yaw", type=float, default=45.0, help="+/- degrees")
parser.add_argument("--min-sep", type=float, default=0.11, help="minimum distance between cube centres")
parser.add_argument("--ready-jitter", type=float, default=0.08)
parser.add_argument("--cube-friction", type=float, default=1.5)
parser.add_argument("--expert", action="store_true")
parser.add_argument("--record", default="")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--dart-noise", type=float, default=0.0,
                    help="expert only: std (rad) of correlated joint noise added to the EXECUTED arm command; "
                         "the recorded action stays the clean expert command (DART)")
parser.add_argument("--dart-tau", type=float, default=5.0, help="correlation time of the DART noise, in control steps")
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/
sys.path.insert(0, HERE)
from wire import HOST, recv_msg, send_msg  # noqa: E402

from mhh_paths import YELLOWARM_USD as _USD  # noqa: E402
USD = _USD.as_posix()
ROBOT = "/World/yellowarm"
ARM = [f"yellowarm_joint{i}" for i in range(1, 8)]
FINGERS = ["yellowarm_joint8", "yellowarm_joint9"]
F_OPEN, F_CLOSED = -0.022, 0.015
TCP_OFFSET = 0.085
CAM_NAMES = ("camera1", "camera2", "camera3")
CUBE = 0.04
# stacking order: bottom -> top
CUBES = [("blue", (0.12, 0.30, 0.88)), ("green", (0.15, 0.70, 0.22)), ("red", (0.90, 0.15, 0.10))]
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


def quat_yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def down_quat(yaw):
    return quat_mul(yaw_quat(yaw), np.array([0.0, 0.0, 1.0, 0.0]))


def wrap90(a):  # cube symmetry: equivalent grasp yaw in [-45, 45) deg
    return ((a + np.pi / 4) % (np.pi / 2)) - np.pi / 4


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def make_camera(path, eye, target, up=(0.0, 0.0, 1.0), focal=18.0):
    cam = RtxCamera(path, translations=np.array(eye, float), orientations=look_at_quat(eye, target, up))
    c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    c.GetFocalLengthAttr().Set(focal)
    c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(args.img_size, args.img_size), annotators=["rgb"])


def build_scene():
    stage_utils.create_new_stage()
    SimulationManager.set_physics_dt(1.0 / args.physics_hz)
    DomeLight("/World/Light").set_intensities(1200)
    GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, -0.75])
    stage_utils.add_reference_to_stage(usd_path=USD, path=ROBOT)
    root = omni.usd.get_context().get_stage().GetPrimAtPath(ROBOT)
    root.GetAttribute("xformOp:orient").Set(Gf.Quatd(1, 0, 0, 0))
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
    grip = RigidBodyMaterial("/World/PhysicsMaterials/grippy")
    grip.set_friction_coefficients(static_frictions=args.cube_friction, dynamic_frictions=args.cube_friction)
    grip.set_combine_modes(frictions="max")
    for i, (name, color) in enumerate(CUBES):
        box(f"/World/cube_{name}", (CUBE, CUBE, CUBE), (0.3, 0.6 + 0.1 * i, 0.02), color, dynamic=True).apply_physics_materials(grip)
    return {
        "camera1": make_camera("/World/cam_front", eye=(1.05, 0.0, 0.62), target=(0.32, 0.0, 0.0)),
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
    cams = build_scene()
    robot = Articulation(ROBOT)
    cubes = [RigidPrim(f"/World/cube_{n}") for n, _ in CUBES]
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

    def cube_pose(i):
        p, q = cubes[i].get_world_poses()
        return p.numpy()[0].astype(np.float64), q.numpy()[0].astype(np.float64)

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
        Jt = J.copy()
        Jt[:3] = J[:3] - np.cross(r[None, :], J[3:].T).T
        dq_err = quat_mul(goal_q, np.array([q7[0], -q7[1], -q7[2], -q7[3]]))
        err = np.concatenate([goal_p - (p7 + r), dq_err[1:] * np.sign(dq_err[0])])
        dq = Jt.T @ np.linalg.solve(Jt @ Jt.T + 0.05 ** 2 * np.eye(6), err)
        return np.clip(q[ai] + np.clip(dq, -0.15, 0.15), LO, HI)

    def stacked():
        ps = [cube_pose(i)[0] for i in range(3)]
        ok = (abs(ps[0][2] - 0.02) < 0.01 and abs(ps[1][2] - 0.06) < 0.012 and abs(ps[2][2] - 0.10) < 0.014
              and np.linalg.norm(ps[1][:2] - ps[0][:2]) < 0.02 and np.linalg.norm(ps[2][:2] - ps[1][:2]) < 0.02)
        return bool(ok), ps

    HOME = np.zeros(len(names), np.float32)
    HOME[fi] = F_OPEN

    def reset():
        for i, c in enumerate(cubes):  # park the cubes while the arm moves
            c.set_world_poses(positions=np.array([[0.0, 0.8 + 0.1 * i, 0.05]]))
        robot.set_dof_positions(HOME)
        robot.set_dof_velocities(np.zeros_like(HOME))
        robot.set_dof_position_targets(HOME)
        for _ in range(3):
            simulation_app.update()
        ready = np.array([0.32, 0.0, 0.22]) + rng.uniform(-args.ready_jitter, args.ready_jitter, 3)
        ready_q = down_quat(rng.uniform(-0.4, 0.4))
        for _ in range(int(3.0 * args.control_hz)):
            apply(ik_target(ready, ready_q), 1.0)
        xy = []
        while len(xy) < 3:
            c = np.array([rng.uniform(*args.cube_x), rng.uniform(*args.cube_y)])
            if all(np.linalg.norm(c - o) > args.min_sep for o in xy):
                xy.append(c)
        yaws = np.deg2rad(rng.uniform(-args.cube_yaw, args.cube_yaw, 3))
        for c, p, yw in zip(cubes, xy, yaws):
            c.set_world_poses(positions=np.array([[p[0], p[1], CUBE / 2 + 0.001]]), orientations=yaw_quat(yw)[None])
            c.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        for _ in range(10):
            simulation_app.update()
        return xy, yaws

    if args.probe:
        xy, yaws = reset()
        for _ in range(60):
            simulation_app.update()
        imgs = grab(cams)
        print(f"[stack] probe: cubes {np.round(xy, 3).tolist()} tcp {np.round(tcp_pose()[0], 3)}")
        save_frames(imgs, "probe")
        timeline.stop()
        simulation_app.close()
        return

    tries = wins = saved = 0
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        saved = len([f for f in os.listdir(args.record) if f.endswith(".npz")])

    if args.expert:
        rec = None
        dart = np.zeros(len(LO))
        dart_a = np.exp(-1.0 / max(args.dart_tau, 1e-6))
        dart_on = [False]  # noise only during free-space motion; grasp / place / release stay clean

        def step(arm, grip):
            if rec is not None:
                rec["state"].append(read_state())
                for n, img in zip(CAM_NAMES, grab(cams)):
                    rec[n].append(img)
                rec["action"].append(np.concatenate([arm, [grip]]).astype(np.float32))
            if args.dart_noise > 0 and not dart_on[0]:
                dart[:] = 0.0
            elif args.dart_noise > 0:
                # Ornstein-Uhlenbeck noise: the arm drifts off the expert path and ik_target (which starts from
                # the current joints) labels the next steps with the correction back
                dart[:] = dart_a * dart + np.sqrt(1 - dart_a ** 2) * rng.normal(0, args.dart_noise, len(dart))
                arm = np.clip(arm + dart, LO, HI)
            apply(arm, grip)

        def move(goal_p, goal_q, grip, speed, settle=0, noise=0.0):
            p0 = tcp_pose()[0]
            n = max(1, int(np.ceil(np.linalg.norm(goal_p - p0) / (speed / args.control_hz))))
            bend = rng.normal(0, noise, 3) if noise else np.zeros(3)
            for k in range(1, n + 1):
                s = k / n
                step(ik_target(p0 + (goal_p - p0) * s + bend * np.sin(np.pi * s), goal_q), grip)
            for _ in range(settle):
                step(ik_target(goal_p, goal_q), grip)

        def hold(goal_p, goal_q, grip, n):
            for _ in range(n):
                step(ik_target(goal_p, goal_q), grip)

        def pick_and_place(src, dst, level):
            """Pick cube `src`, put it on cube `dst`; `level` = 1 (middle) or 2 (top)."""
            sp, sq = cube_pose(src)
            g_yaw = wrap90(quat_yaw(sq))
            gq = down_quat(g_yaw)
            dart_on[0] = True
            move(np.array([sp[0], sp[1], 0.12]), gq, 1.0, 0.30, noise=0.025)
            dart_on[0] = False
            if args.dart_noise > 0:  # recover from the DART drift above the cube, then descend vertically
                hold(np.array([sp[0], sp[1], 0.12]), gq, 1.0, 10)
            grasp = np.array([sp[0], sp[1], 0.018])
            move(grasp, gq, 1.0, 0.15, settle=4)
            hold(grasp, gq, 0.0, 10)
            lift_z = 0.08 + 0.04 * level + 0.04
            dart_on[0] = True
            move(np.array([sp[0], sp[1], lift_z]), gq, 0.0, 0.20)
            dart_on[0] = False
            # where the held cube sits relative to the TCP, in the gripper frame (it is never perfectly centred)
            tp, tq = tcp_pose()
            off = quat_rotate(np.array([tq[0], -tq[1], -tq[2], -tq[3]]), cube_pose(src)[0] - tp)
            if args.debug:
                print(f"[dbg] lifted src {src}: cube {np.round(cube_pose(src)[0], 3)} tcp {np.round(tp, 3)} "
                      f"offset(gripper frame) {np.round(off, 3)} grip {read_state()[7]:.2f}")
            dq = cube_pose(dst)[1]
            # rotate the held cube onto the lower cube's faces: same relative yaw change for gripper and cube
            pq = down_quat(g_yaw + wrap90(quat_yaw(dq) - quat_yaw(cube_pose(src)[1])))

            def tcp_for(cube_target):  # TCP position that puts the held cube's centre at cube_target
                return cube_target - quat_rotate(pq, off)

            dp = cube_pose(dst)[0]
            dart_on[0] = True
            move(tcp_for(np.array([dp[0], dp[1], lift_z - 0.03])), pq, 0.0, 0.25, noise=0.02)
            dart_on[0] = False
            dp = cube_pose(dst)[0]  # re-read: correct for any drift of the lower cube
            rest = np.array([dp[0], dp[1], CUBE / 2 + CUBE * level + 0.004])  # held cube 4 mm above its resting place
            # closed-loop hover: the cube can slip in the fingers while carrying/rotating, so re-measure and correct
            for _ in range(3):
                tp, tq = tcp_pose()
                off[:] = quat_rotate(np.array([tq[0], -tq[1], -tq[2], -tq[3]]), cube_pose(src)[0] - tp)
                move(tcp_for(rest + np.array([0.0, 0.0, 0.03])), pq, 0.0, 0.10, settle=5)
            move(tcp_for(rest), pq, 0.0, 0.08, settle=6)
            if args.debug:
                print(f"[dbg] before release src {src}: tcp err {1000 * np.linalg.norm(tcp_pose()[0] - tcp_for(rest)):.1f} mm, "
                      f"held cube {np.round(cube_pose(src)[0], 3)} target {np.round(rest, 3)}")
            hold(tcp_for(rest), pq, 1.0, 8)
            place_z = tcp_for(rest)[2]
            move(tcp_for(rest) + np.array([0.0, 0.0, 0.08]), pq, 1.0, 0.20)
            if args.debug:
                print(f"[dbg] placed src {src} on {dst}: place_z {place_z:.3f} cubes " + str([np.round(cube_pose(i)[0], 3).tolist() for i in range(3)]))
                if args.save_frames:
                    save_frames(grab(cams), f"dbg_t{tries}_src{src}")

        while simulation_app.is_running() and (not args.episodes or saved < args.episodes):
            tries += 1
            xy, yaws = reset()
            dart[:] = 0.0
            rec = {"state": [], "action": [], **{n: [] for n in CAM_NAMES}}
            pick_and_place(1, 0, 1)  # green onto blue
            pick_and_place(2, 1, 2)  # red onto green
            ep, rec = rec, None
            for _ in range(30):
                simulation_app.update()
            ok, ps = stacked()
            T = len(ep["action"])
            print(f"[expert] try {tries}: cubes {np.round(xy, 3).tolist()} -> heights {[round(p[2], 3) for p in ps]} "
                  f"{T} frames -> {'SUCCESS' if ok else 'fail'}")
            if ok and args.record:
                np.savez_compressed(os.path.join(args.record, f"episode_{saved:04d}.npz"),
                                    state=np.stack(ep["state"]), action=np.stack(ep["action"]),
                                    **{n: np.stack(ep[n]) for n in CAM_NAMES}, task=args.task, fps=args.control_hz)
            saved += ok
        print(f"[expert] done: {saved} episodes saved, {tries} tries")
        timeline.stop()
        simulation_app.close()
        return

    # ---------------------------------------- policy evaluation ----------------------------------------
    sock = socket.create_connection((HOST, args.port), timeout=300)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"cmd": "info"})
    print(f"[stack] connected to policy server: {recv_msg(sock)}")
    steps = int(args.episode_seconds * args.control_hz)
    two = 0
    while simulation_app.is_running() and (not args.episodes or tries < args.episodes):
        xy, yaws = reset()
        send_msg(sock, {"cmd": "reset"})
        recv_msg(sock)
        target = read_state()
        step = 0
        while step < steps and simulation_app.is_running():
            imgs = grab(cams)
            send_msg(sock, {"cmd": "infer", "state": read_state(), "images": imgs, "cam_names": list(CAM_NAMES),
                            "task": args.task})
            chunk = recv_msg(sock)["actions"][: args.exec_steps]
            if step == 0:
                save_frames(imgs, f"ep{tries}")
            for a in chunk:
                a = a.astype(np.float32)
                target[:7] += np.clip(np.clip(a[:7], LO, HI) - target[:7], -args.max_step_rad, args.max_step_rad)
                target[7] = a[7]
                apply(target[:7], target[7])
                step += 1
                if step >= steps or not simulation_app.is_running():
                    break
        for _ in range(30):
            simulation_app.update()
        ok, ps = stacked()
        green_on_blue = abs(ps[1][2] - 0.06) < 0.012 and np.linalg.norm(ps[1][:2] - ps[0][:2]) < 0.02
        tries += 1
        wins += ok
        two += green_on_blue
        print(f"[stack] episode {tries - 1}: heights {[round(p[2], 3) for p in ps]} green-on-blue {green_on_blue} -> "
              f"{'SUCCESS' if ok else 'fail'}   [{wins}/{tries} full stacks, {two}/{tries} green on blue]")
    print(f"[stack] evaluation done: {wins}/{tries} full 3-cube stacks, {two}/{tries} green on blue")
    sock.close()
    if args.keep_open and not args.headless:
        while simulation_app.is_running():
            simulation_app.update()
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[stack] total {time.time() - t0:.1f}s")
