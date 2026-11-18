"""Pick-and-lift with the dual YellowArm + ORCA hand robot (Isaac Sim 6, Isaac python).

First ORCA task of the training pipeline (P3 seed): a cube lies on the table in front of the robot; the RIGHT hand
picks it top-down with the power-grasp synergy and lifts it. Same structure as the gripper sims:

  --expert --record DIR     scripted expert (IK + grasp synergy), saves successful episodes as episode_XXXX.npz
  --port P                  policy evaluation over wire.py (policy_server.py), one JSON line per episode
  (no headless) --expert    watch it in the Isaac Sim GUI (add --loop to keep going)

Data format (embodiment yellowarm_two_orca_v1, action space 'arm_synergy_v1', 18-D, left then right):
  per side [yellowarm_joint1..7, wrist, grasp]   grasp 0 = open hand ... 1 = closed power grasp (dual_orca_kin.grasp_pose)
  state  18 = measured joints (rad) + measured grasp closure;  action 18 = targets, same order
  state_full 48 = all joints in embodiment order (kept in the npz, not used by the converter)
  camera1 = head, camera2 = left wrist, camera3 = right wrist (256x256 RGB), 25 Hz control, 100 Hz physics.

Physics comes from dual_orca_rig.py (PhysX self-collision off, real effort/velocity limits, ORCA servo model).
"""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--expert", action="store_true")
parser.add_argument("--record", default="", help="expert: save successful episodes (.npz) here")
parser.add_argument("--episodes", type=int, default=0, help="expert: successful episodes to record; eval: episodes")
parser.add_argument("--tries", type=int, default=0, help="expert: stop after this many tries")
parser.add_argument("--seed", type=int, default=0, help="episode i uses seed + i")
parser.add_argument("--loop", action="store_true", help="GUI: keep running expert episodes")
parser.add_argument("--control-hz", type=float, default=25.0)
parser.add_argument("--physics-hz", type=float, default=100.0)
parser.add_argument("--img-size", type=int, default=256)
parser.add_argument("--grasp-dz", type=float, default=None, help="grasp-centre height offset (m; default: planner)")
parser.add_argument("--grasp-dy", type=float, default=0.0, help="grasp-centre offset along the robot front (-Y) (m)")
parser.add_argument("--dart-noise", type=float, default=0.0, help="expert: DART noise on arm targets (rad)")
parser.add_argument("--dart-tau", type=float, default=5.0)
parser.add_argument("--results-json", default="")
parser.add_argument("--save-frames", default="", help="debug: save camera frames of the first try here")
parser.add_argument("--live-dir", default="", help="write live.jpg + status.json here for dual_orca_monitor.py")
parser.add_argument("--live-every", type=int, default=3, help="--live-dir: write every N control steps")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--cam-probe", action="store_true", help="render head-camera candidates at the grasp pose")
parser.add_argument("--port", type=int, default=0)
parser.add_argument("--exec-steps", type=int, default=8)
parser.add_argument("--episode-seconds", type=float, default=12.0)
parser.add_argument("--max-step-rad", type=float, default=0.3)
args, _ = parser.parse_known_args()
sys.stdout.reconfigure(line_buffering=True)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.materials import OmniPbrMaterial, RigidBodyMaterial  # noqa: E402
from isaacsim.core.experimental.objects import Cube, DistantLight, DomeLight, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dual_orca_kin as K  # noqa: E402
import dual_orca_rig as rig  # noqa: E402
from embodiment import Embodiment  # noqa: E402
from mhh_paths import DUAL_URDF, DUAL_USD_DIR, dual_import_usd  # noqa: E402

E = Embodiment()
KIN = K.Kin()
NAMES = E.action_space("joint_full")  # 48, embodiment order
SIDES = K.SIDES
CAM_NAMES = ("camera1", "camera2", "camera3")
SYN_NAMES = [f"{s}_{j}" for s in SIDES for j in K.ARM + ["wrist", "grasp"]]  # 18
TASK = "pick up the {color} cube"
COLORS = {"red": (0.85, 0.1, 0.1), "green": (0.1, 0.7, 0.2), "blue": (0.1, 0.25, 0.85), "yellow": (0.9, 0.8, 0.1)}

# scene geometry + expert planning live in dual_orca_pick_plan (numpy, testable without Isaac)
import dual_orca_pick_plan as P  # noqa: E402

TABLE_TOP, TABLE_X, TABLE_Y, CUBE, CUBE_MASS = P.TABLE_TOP, P.TABLE_X, P.TABLE_Y, P.CUBE, P.CUBE_MASS
FRICTION = 1.0
HOME = dict(zip(NAMES, E.home(NAMES)))


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def look_at_R(eye, target, up):
    """USD camera rotation (columns: right, up, backward; the camera looks along -Z)."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), -f], 1)


def mat_to_quat(R):
    q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()  # Gf is row-vector: pass R^T
    return np.array([q.GetReal(), *q.GetImaginary()])


def usd_file():
    p = dual_import_usd()
    if p is None:
        sys.exit("no robot USD: run run_dual_probe.bat --import first")
    return str(p)


def make_camera(path, pos, quat, fov_deg):
    cam = RtxCamera(path, translations=np.array(pos, float), orientations=np.array(quat, float))
    c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    ap = c.GetHorizontalApertureAttr().Get()
    c.GetVerticalApertureAttr().Set(ap)
    c.GetFocalLengthAttr().Set(float(ap / (2 * np.tan(np.deg2rad(fov_deg) / 2))))
    c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(args.img_size, args.img_size), annotators=["rgb"])


class Scene:
    def __init__(self):
        stage_utils.create_new_stage()
        SimulationManager.set_physics_dt(1.0 / args.physics_hz)
        self.stage = stage = omni.usd.get_context().get_stage()
        DomeLight("/World/dome").set_intensities(1100.0)
        DistantLight("/World/sun").set_intensities(1500.0)
        GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, P.FLOOR_Z])
        stage_utils.add_reference_to_stage(usd_path=usd_file(), path="/World/robot")
        self.root, _ = rig.apply_rig(stage, "/World/robot", E)
        # table
        tw, td = TABLE_X[1] - TABLE_X[0], TABLE_Y[0] - TABLE_Y[1]
        th = TABLE_TOP - P.FLOOR_Z
        c = Cube("/World/table", sizes=1.0, translations=np.array([0.0, (TABLE_Y[0] + TABLE_Y[1]) / 2, P.FLOOR_Z + th / 2]),
                 scales=np.array([tw, td, th]))
        GeomPrim("/World/table", apply_collision_apis=True)
        m = OmniPbrMaterial("/World/Materials/table")
        m.set_input_values("diffuse_color_constant", [0.55, 0.45, 0.33])
        c.apply_visual_materials(m)
        ped = Cube("/World/pedestal", sizes=1.0, translations=np.array([0.0, 0.0, P.FLOOR_Z / 2]),
                   scales=np.array([*P.PEDESTAL, -P.FLOOR_Z]))
        GeomPrim("/World/pedestal", apply_collision_apis=True)
        mp = OmniPbrMaterial("/World/Materials/pedestal")
        mp.set_input_values("diffuse_color_constant", [0.3, 0.3, 0.33])
        ped.apply_visual_materials(mp)
        # cube
        Cube("/World/cube", sizes=1.0, translations=np.array([0.2, -0.37, TABLE_TOP + CUBE / 2]),
             scales=np.array([CUBE, CUBE, CUBE]))
        g = GeomPrim("/World/cube", apply_collision_apis=True)
        RigidPrim("/World/cube")
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath("/World/cube")).CreateMassAttr(CUBE_MASS)
        pm = RigidBodyMaterial("/World/PhysicsMaterials/cube")
        pm.set_friction_coefficients(static_frictions=[FRICTION], dynamic_frictions=[FRICTION])
        pm.set_combine_modes(frictions="max")
        g.apply_physics_materials(pm)
        self.cube_mat = OmniPbrMaterial("/World/Materials/cube")
        self.cube_mat.set_input_values("diffuse_color_constant", list(COLORS["red"]))
        Cube("/World/cube").apply_visual_materials(self.cube_mat)
        # cameras: head (fixed), one on each ORCA tower looking along the fingers
        head = np.array([0.0, -0.20, 1.45])  # head above and slightly in front of the shoulders (chosen with --cam-probe;
        # during the grasp the hand itself hides the cube from any overhead view: camera3 sees it)
        self.cams = {"camera1": make_camera("/World/head_cam", head, look_at_quat(head, (0.16, -0.40, TABLE_TOP)), 62.0)}
        if args.save_frames or args.live_dir:  # outside view for humans (debug frames / live monitor; not recorded)
            eye = np.array([0.75, -0.80, 0.62])
            self.cams["dbg"] = make_camera("/World/dbg_cam", eye, look_at_quat(eye, (0.18, -0.38, 0.45)), 45.0)
        W0 = KIN.fk({})
        for s, cn in (("left", "camera2"), ("right", "camera3")):
            tower = next(p for p in stage.Traverse()
                         if p.GetPath().pathString.startswith("/World/robot") and p.GetName() == f"{s}_tower")
            c0, R0 = KIN.palm(s, {}, W0)
            fdir, pn = R0[:, 0], R0[:, 1]
            pinky = R0[:, 2] * (-1.0 if s == "right" else 1.0)  # grasp-frame z points to the thumb side on the right
            eye = c0 - 0.12 * fdir + 0.10 * pinky - 0.03 * pn  # beside the pinky, behind the knuckles
            Mw = np.eye(4)
            Mw[:3, :3], Mw[:3, 3] = look_at_R(eye, c0 + 0.03 * fdir + 0.02 * pn, -pn), eye
            Ml = np.linalg.inv(W0[f"{s}_tower"]) @ Mw
            self.cams[cn] = make_camera(f"{tower.GetPath()}/cam_wrist", Ml[:3, 3], mat_to_quat(Ml[:3, :3]), 80.0)


class Robot:
    def __init__(self, scene):
        self.scene = scene
        self.art = Articulation(scene.root)
        self.cube = RigidPrim("/World/cube")
        self.tl = omni.timeline.get_timeline_interface()
        self.tl.play()
        simulation_app.update()
        dof = list(self.art.dof_names)
        self.order = [dof.index(n) for n in NAMES]
        self.ndof = len(dof)
        self.decimation = max(1, round(args.physics_hz / args.control_hz))
        self.idx = {n: i for i, n in enumerate(NAMES)}
        self.target = np.array([HOME[n] for n in NAMES])

    def q_full(self):
        return self.art.get_dof_positions().numpy()[0][self.order].astype(np.float64)

    def _send(self, q48, teleport=False):
        full = np.zeros(self.ndof)
        full[self.order] = q48
        if teleport:
            self.art.set_dof_positions(full)
            self.art.set_dof_velocities(np.zeros_like(full))
        self.art.set_dof_position_targets(full)

    @staticmethod
    def synergy_to_full(a18, base48):
        q = base48.copy()
        for k, s in enumerate(SIDES):
            a = a18[9 * k: 9 * k + 9]
            for j, n in enumerate(KIN.chain(s)):
                q[NAMES.index(n)] = a[j]
            for n, v in K.grasp_pose(a[8], s).items():
                q[NAMES.index(n)] = v
        lo, hi = E.lower_upper(NAMES)
        return np.clip(q, lo, hi)

    def read_state(self):
        q = self.q_full()
        out = []
        for s in SIDES:
            arm = [q[self.idx[n]] for n in KIN.chain(s)]
            out += arm + [K.grasp_closure({n: q[i] for n, i in self.idx.items()}, s)]
        return np.array(out, np.float32), q.astype(np.float32)

    def apply(self, a18):
        self.target = self.synergy_to_full(np.asarray(a18, float), self.target)
        self._send(self.target)
        for _ in range(self.decimation):
            simulation_app.update()

    def teleport(self, a18):
        self.target = self.synergy_to_full(np.asarray(a18, float), np.array([HOME[n] for n in NAMES]))
        self._send(self.target, teleport=True)

    def cube_pose(self):
        p, q = self.cube.get_world_poses()
        return p.numpy()[0].astype(np.float64), q.numpy()[0].astype(np.float64)

    def place_cube(self, pos, yaw):
        q = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        self.cube.set_world_poses(positions=np.array([pos]), orientations=np.array([q]))
        self.cube.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))

    def grab(self):
        out = []
        for n in CAM_NAMES:
            d, _ = self.scene.cams[n].get_data("rgb")
            img = np.zeros((args.img_size, args.img_size, 3), np.uint8) if d is None else d.numpy()[..., :3]
            out.append(np.ascontiguousarray(img.astype(np.uint8)))
        return out


# ------------------------------------------------ planning ------------------------------------------------
def left_rest():
    return [HOME[n] for n in KIN.chain("left")]


def ready_pose():
    return P.ready_pose()


# ------------------------------------------------ episodes ------------------------------------------------
def new_episode(rng):
    for _ in range(50):
        pos, yaw = P.sample_layout(rng)
        color = str(rng.choice(sorted(COLORS)))
        plan, why = P.plan_episode(pos, args.control_hz, P.GRASP_DZ if args.grasp_dz is None else args.grasp_dz,
                                  args.grasp_dy)
        if plan is not None:
            return {"pos": pos, "yaw": yaw, "color": color, "plan": plan, "task": TASK.format(color=color)}
        if args.debug:
            print("[plan] rejected", np.round(pos, 3), why)
    return None


def start_action():
    return np.r_[left_rest(), 0.0, ready_pose(), 0.0]


def setup_episode(robot, ep):
    robot.scene.cube_mat.set_input_values("diffuse_color_constant", list(COLORS[ep["color"]]))
    robot.teleport(start_action())
    robot.place_cube(ep["pos"], ep["yaw"])
    a = start_action()
    for _ in range(8):
        robot.apply(a)
    robot.place_cube(ep["pos"], ep["yaw"])  # again: the hand may have nudged it while settling
    for _ in range(4):
        robot.apply(a)
    st, _ = robot.read_state()
    err = float(np.abs(st[:8] - a[:8]).max())  # idle left arm: must hold its pose (contact = scene bug)
    if err > 0.03:
        print(f"[sim] WARNING: left arm {err:.3f} rad off its rest pose after reset")


def check_success(robot, ep, z0):
    p, _ = robot.cube_pose()
    c, _ = KIN.palm("right", dict(zip(NAMES, robot.q_full())))
    lifted = float(p[2] - z0)
    near = float(np.linalg.norm(p - c))
    return lifted > 0.06 and near < 0.10, {"lift_m": round(lifted, 4), "cube_to_hand_m": round(near, 4)}


class Live:
    """--live-dir: the latest view (outside camera + the 3 policy cameras) and a status line for the web monitor.
    Files are written to a temp name and renamed, so the monitor never reads a half-written file."""

    def __init__(self, robot):
        self.dir = args.live_dir
        self.robot = robot
        self.status = {"mode": "expert" if args.expert else "eval", "pid": os.getpid(), "seed": None, "try": 0,
                       "wins": 0, "phase": "start", "step": 0, "recent": []}
        self.n = 0
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def _replace(self, tmp, dst):
        for _ in range(5):
            try:
                os.replace(tmp, dst)
                return
            except PermissionError:  # the monitor is reading it right now
                time.sleep(0.02)

    def frame(self, imgs=None, force=False, **st):
        if not self.dir:
            return
        self.status.update(st)
        self.n += 1
        if not force and self.n % max(1, args.live_every):
            return
        from PIL import Image

        imgs = imgs if imgs is not None else self.robot.grab()
        row = list(imgs)
        if "dbg" in self.robot.scene.cams:
            d, _ = self.robot.scene.cams["dbg"].get_data("rgb")
            if d is not None:
                row = [d.numpy()[..., :3].astype(np.uint8)] + row
        tmp = os.path.join(self.dir, "live.tmp.jpg")
        Image.fromarray(np.concatenate(row, 1)).save(tmp, quality=80)
        self._replace(tmp, os.path.join(self.dir, "live.jpg"))
        self.write_status()

    def write_status(self):
        if not self.dir:
            return
        self.status["time"] = time.time()
        tmp = os.path.join(self.dir, "status.tmp.json")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.status, f)
        self._replace(tmp, os.path.join(self.dir, "status.json"))

    def result(self, rec):
        self.status["recent"] = (self.status["recent"] + [rec])[-12:]
        self.write_status()


LIVE = None


def run_expert(robot, ep, rng, record, frames_dir=""):
    plan = ep["plan"]
    rec = {"state": [], "state_full": [], "action": [], **{n: [] for n in CAM_NAMES}} if record or frames_dir else None
    left = np.r_[left_rest(), 0.0]
    dart = np.zeros(8)
    da = np.exp(-1.0 / max(args.dart_tau, 1e-6))
    last = {"q": ready_pose().copy()}
    k = {"t": 0}

    def step(q8, g, noisy=False):
        a = np.r_[left, q8, g].astype(np.float32)
        imgs = None
        if rec is not None:
            st, sf = robot.read_state()
            imgs = robot.grab()
            rec["state"].append(st)
            rec["state_full"].append(sf)
            rec["action"].append(a)
            for n, img in zip(CAM_NAMES, imgs):
                rec[n].append(img)
            if frames_dir and k["t"] % 10 == 0:
                from PIL import Image

                os.makedirs(frames_dir, exist_ok=True)
                extra = []
                if "dbg" in robot.scene.cams:
                    d, _ = robot.scene.cams["dbg"].get_data("rgb")
                    if d is not None:
                        extra = [d.numpy()[..., :3].astype(np.uint8)]
                Image.fromarray(np.concatenate(imgs + extra, 1)).save(os.path.join(frames_dir, f"f{k['t']:04d}.png"))
        qa = q8
        if args.dart_noise > 0 and noisy:
            dart[:] = da * dart + np.sqrt(1 - da ** 2) * rng.normal(0, args.dart_noise, 8)
            qa = q8 + dart
        elif args.dart_noise > 0:
            dart[:] = 0.0
        robot.apply(np.r_[left, qa, g])
        last["q"] = q8
        k["t"] += 1
        if LIVE:
            LIVE.frame(imgs, step=k["t"], grasp=round(float(g), 2))

    gp = K.G_PRESHAPE
    n_tr = len(plan["transit"])
    for i, q in enumerate(plan["transit"]):  # pre-shape the thumb while the hand is high above the table
        step(q, gp * min(1.0, (i + 1) / max(1, n_tr - 2)), noisy=True)
    for _ in range(4):
        step(last["q"], gp)
    for q in plan["down"]:
        step(q, gp)
    for i in range(1, 16):  # close the fingers over 0.6 s
        step(last["q"], gp + (1 - gp) * i / 15)
    for _ in range(8):
        step(last["q"], 1.0)
    if args.debug:
        qf = robot.q_full()
        c_now, _ = KIN.palm("right", dict(zip(NAMES, qf)))
        print(f"[dbg] closed: hand centre {np.round(c_now, 3)} planned {np.round(plan['grasp_point'], 3)} "
              f"cube {np.round(robot.cube_pose()[0], 3)} grasp state {robot.read_state()[0][17]:.2f}")
    for q in plan["up"]:
        step(q, 1.0)
    for _ in range(10):
        step(last["q"], 1.0)
    if args.debug:
        qf = robot.q_full()
        q8 = np.array([qf[NAMES.index(n)] for n in KIN.chain("right")])
        c_now, _ = KIN.palm("right", dict(zip(NAMES, qf)))
        c_plan, _ = KIN.palm("right", dict(zip(KIN.chain("right"), plan["up"][-1])))
        print(f"[dbg] end: hand centre {np.round(c_now, 3)} planned {np.round(c_plan, 3)} "
              f"|dq| max {np.abs(q8 - plan['up'][-1]).max():.3f} rad; cube {np.round(robot.cube_pose()[0], 3)}")
    return rec


def save_episode(rec, path, ep, info):
    tmp = os.path.join(os.path.dirname(path), "tmp_" + os.path.basename(path))  # write + rename: a killed worker
    # never leaves a truncated episode_*.npz (the tmp name does not match that pattern)
    np.savez_compressed(tmp, state=np.asarray(rec["state"], np.float32), state_full=np.asarray(rec["state_full"], np.float32),
                        action=np.asarray(rec["action"], np.float32),
                        **{n: np.asarray(rec[n], np.uint8) for n in CAM_NAMES}, task=ep["task"], fps=args.control_hz,
                        names=np.array(SYN_NAMES), meta=json.dumps(info))
    os.replace(tmp, path)


def gui_camera():
    if args.headless:
        return
    try:
        from isaacsim.core.rendering_manager import ViewportManager

        ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[1.3, -1.6, 1.25], target=[0.1, -0.3, 0.55])
    except Exception as e:
        print(f"[sim] viewport camera not set: {e!r}")


def expert():
    global LIVE
    scene = Scene()
    robot = Robot(scene)
    LIVE = Live(robot)
    gui_camera()
    ready_pose()
    if args.record:
        os.makedirs(args.record, exist_ok=True)
    results = open(args.results_json, "a", encoding="utf-8") if args.results_json else None
    seeds_done = []
    for f in sorted(os.listdir(args.record)) if args.record else []:
        if not (f.startswith("episode_") and f.endswith(".npz")) or ".tmp" in f:
            continue
        try:
            with np.load(os.path.join(args.record, f)) as z:
                seeds_done.append(json.loads(str(z["meta"]))["seed"])
        except Exception as e:  # truncated file from a killed run: set it aside, never train on it
            os.replace(os.path.join(args.record, f), os.path.join(args.record, f + ".bad"))
            print(f"[expert] WARNING: unreadable {f} ({e!r}) renamed to .bad")
    n_have = len(seeds_done)
    tries = wins = 0
    if n_have:  # resume: continue after the last recorded seed (never re-record a seed)
        last_seed = max(seeds_done)
        tries = max(0, last_seed + 1 - args.seed)
        wins = n_have
        print(f"[expert] resume: {n_have} episodes in {args.record}, last seed {last_seed}")
    t_all = time.time()
    while simulation_app.is_running():
        if not args.loop and ((args.episodes and wins >= args.episodes) or (args.tries and tries >= args.tries)):
            break
        if not args.loop and not args.episodes and not args.tries and tries >= 1:
            break
        seed = args.seed + tries
        rng = np.random.default_rng(seed)
        t0 = time.time()
        ep = new_episode(rng)
        tries += 1
        if ep is None:
            print(f"[expert] seed {seed}: no feasible layout")
            continue
        setup_episode(robot, ep)
        z0 = robot.cube_pose()[0][2]
        LIVE.frame(force=True, seed=seed, task=ep["task"], phase="expert", step=0, wins=wins, **{"try": tries},
                   recorded=n_have, target=args.episodes, dart_noise=args.dart_noise)
        rec = run_expert(robot, ep, rng, bool(args.record), args.save_frames if tries == 1 else "")
        ok, chk = check_success(robot, ep, z0)
        wins += ok
        info = {"seed": seed, "success": bool(ok), **chk, "cube_xy": np.round(ep["pos"][:2], 4).tolist(),
                "cube_yaw_deg": round(float(np.degrees(ep["yaw"])), 1), "color": ep["color"], "task": ep["task"],
                "steps": len(rec["action"]) if rec else None, "seconds": round(time.time() - t0, 1),
                "dart_noise": args.dart_noise, "grasp_dz": args.grasp_dz, "grasp_dy": args.grasp_dy}
        if ok and args.record:
            path = os.path.join(args.record, f"episode_{n_have:04d}.npz")
            save_episode(rec, path, ep, info)
            n_have += 1
        print(f"[expert] try {tries - 1} seed {seed}: {'SUCCESS' if ok else 'fail'} {json.dumps(chk)} "
              f"({time.time() - t0:.1f}s) [{wins}/{tries}]")
        LIVE.status.update(wins=wins, recorded=n_have, phase="reset")
        LIVE.result({"seed": seed, "success": bool(ok), **chk})
        if results:
            results.write(json.dumps(info) + "\n")
            results.flush()
    print(f"[expert] done: {wins}/{tries} successful, {time.time() - t_all:.0f}s")


def evaluate():
    import socket

    from wire import HOST, recv_msg, send_msg

    global LIVE
    scene = Scene()
    robot = Robot(scene)
    LIVE = Live(robot)
    gui_camera()
    ready_pose()
    sock = socket.create_connection((HOST, args.port), timeout=300)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"cmd": "info"})
    print(f"[eval] connected to policy server: {recv_msg(sock)}")
    results = open(args.results_json, "a", encoding="utf-8") if args.results_json else None
    steps = int(round(args.episode_seconds * args.control_hz))
    lo48, hi48 = E.lower_upper(NAMES)
    lo = np.array([lo48[NAMES.index(n)] if n in NAMES else 0.0 for n in SYN_NAMES])
    hi = np.array([hi48[NAMES.index(n)] if n in NAMES else 1.0 for n in SYN_NAMES])
    tries = wins = 0
    while simulation_app.is_running() and (not args.episodes or tries < args.episodes):
        seed = args.seed + tries
        rng = np.random.default_rng(seed)
        t0 = time.time()
        ep = new_episode(rng)
        tries += 1
        if ep is None:
            continue
        setup_episode(robot, ep)
        z0 = robot.cube_pose()[0][2]
        send_msg(sock, {"cmd": "reset", "seed": seed})
        recv_msg(sock)
        tgt = start_action().astype(np.float64)
        step, infer_ms, max_lift = 0, [], 0.0
        LIVE.frame(force=True, seed=seed, task=ep["task"], phase="policy", step=0, wins=wins, **{"try": tries},
                   target=args.episodes, exec_steps=args.exec_steps)
        while step < steps and simulation_app.is_running():
            st, _ = robot.read_state()
            imgs = robot.grab()
            t_inf = time.perf_counter()
            send_msg(sock, {"cmd": "infer", "state": st, "images": imgs, "cam_names": list(CAM_NAMES),
                            "task": ep["task"], "exec_steps": args.exec_steps})
            reply = recv_msg(sock)
            infer_ms.append(1000.0 * (time.perf_counter() - t_inf))
            chunk = np.clip(np.asarray(reply["actions"], np.float64), lo, hi)
            for a in chunk[: args.exec_steps]:
                tgt = tgt + np.clip(a - tgt, -args.max_step_rad, args.max_step_rad)
                for gi in (8, 17):
                    tgt[gi] = float(np.clip(a[gi], 0.0, 1.0))
                robot.apply(tgt)
                step += 1
                max_lift = max(max_lift, float(robot.cube_pose()[0][2] - z0))
                LIVE.frame(step=step, grasp=round(float(tgt[17]), 2), lift=round(max_lift, 3),
                           infer_ms=round(infer_ms[-1], 1))
                if step >= steps:
                    break
        for _ in range(10):
            robot.apply(tgt)
        ok, chk = check_success(robot, ep, z0)
        wins += ok
        print(f"[eval] episode {tries - 1} seed {seed}: {'SUCCESS' if ok else 'fail'} {json.dumps(chk)} "
              f"max lift {max_lift:.3f} ({time.time() - t0:.1f}s) [{wins}/{tries}]")
        LIVE.status.update(wins=wins, phase="reset")
        LIVE.result({"seed": seed, "success": bool(ok), **chk})
        if results:
            results.write(json.dumps({"episode": tries - 1, "seed": seed, "success": bool(ok), "steps": step,
                                      "seconds": round(time.time() - t0, 1), "exec_steps": args.exec_steps,
                                      "episode_seconds": args.episode_seconds, "max_lift_m": round(max_lift, 4),
                                      "n_infer": len(infer_ms),
                                      "infer_ms_mean": round(float(np.mean(infer_ms)), 2) if infer_ms else None,
                                      "infer_ms_max": round(float(np.max(infer_ms)), 2) if infer_ms else None,
                                      "task": ep["task"], **chk}) + "\n")
            results.flush()
    print(f"[eval] done: {wins}/{tries} successful")
    sock.close()


def cam_probe():
    from PIL import Image
    from isaacsim.core.experimental.prims import XformPrim

    scene = Scene()
    robot = Robot(scene)
    ep = new_episode(np.random.default_rng(300))
    setup_episode(robot, ep)
    out = os.path.join(os.path.dirname(HERE), "logs", "dual_pick", "cam_probe")
    os.makedirs(out, exist_ok=True)
    cands = [(-0.12, -0.08, 1.40), (0.0, -0.10, 1.60), (0.0, -0.20, 1.45), (-0.25, -0.15, 1.35), (0.0, -0.32, 1.30),
             (-0.15, -0.25, 1.50), (0.10, -0.15, 1.55), (0.0, 0.0, 1.75)]
    cam = XformPrim("/World/head_cam")
    for tag, q8 in (("reach", ep["plan"]["down"][-1]), ("start", P.ready_pose())):
        robot.teleport(np.r_[left_rest(), 0.0, q8, K.G_PRESHAPE])
        for _ in range(5):
            robot.apply(np.r_[left_rest(), 0.0, q8, K.G_PRESHAPE])
        imgs = []
        for eye in cands:
            cam.set_world_poses(positions=np.array([eye]), orientations=np.array([look_at_quat(eye, (0.16, -0.38, TABLE_TOP))]))
            for _ in range(4):
                simulation_app.update()
            imgs.append(robot.grab()[0])
        Image.fromarray(np.concatenate(imgs, 1)).save(os.path.join(out, f"head_{tag}.png"))
    print("[cam] candidates", cands, "->", out)


def main():
    if args.cam_probe:
        cam_probe()
    elif args.port:
        evaluate()
    elif args.expert:
        expert()
    else:
        sys.exit("choose --expert or --port")
    if not args.headless and args.loop:
        while simulation_app.is_running():
            simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
