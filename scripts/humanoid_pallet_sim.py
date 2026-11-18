"""Humanoid (2 YellowArm + fixed head camera) parts-to-pallet task in Isaac Sim 6 (PLAN.md G2).

  Expert demos : D:\\Isaac\\isaacsim\\python.bat scripts\\humanoid_pallet_sim.py --expert --level 1 --record DIR --episodes N --headless
  Expert check : ... --expert --level 1 --tries 50 --results-json results\\expert_L1.json --headless   (nothing recorded)
  Probe        : ... --probe --level 1 --probe-n 4 --save-frames logs\\pallet_probe   (first-frame grid of the 3 cameras)
  FOV probe    : ... --fov-probe 45 55 65 --save-frames logs\\humanoid_probe\\fov      (head camera at several FOVs)
  Policy eval  : env\\python.exe scripts\\eval_harness.py --sim-script scripts\\humanoid_pallet_sim.py --checkpoint CKPT
                 --episodes 20 --seed 200000 --exec-steps 8 --episode-seconds 24 --tag TAG -- --level 1 --split train
                 (the harness starts policy_server.py and runs this script with --port; see evaluate())

Body: the approved configuration in scripts/data/humanoid.json "selected" (forward45, mount plates 0.45 m above the
table, 0.55 m apart, torso 0.25 x 0.18 m, 5 cm from the table). Arms keep the URDF efforts, the real moteus velocity
limits and the USD gripper (29.6 N, 0.1 m/s). Pallet and parts are placed only inside the shared workspace of the two
arms; all planning (arm choice, IK, timing, arm-vs-arm clearance) is in humanoid_pallet_plan.py (numpy only).
Pallet: 0..--prefill-max parts already lie in random cells; the target cell is a random free cell the chosen arm can
reach in its working IK branch (with forward45 and the URDF joint-1 limits that is the arm's own outer column: cells
0/3 right arm, 2/5 left arm; the middle column only ever holds prefilled parts).
Splits (--split, PLAN 2.6): train | heldout_color (target purple) | heldout_shape (target hex nut; in both the other
parts on the table and in the pallet are training parts) | wide (target outside the training
region, still reachable and in the head camera's view). Seeds: data generation 0..99999 (only split train may be
recorded), eval test 100000-100049, validation 200000+; a seed gives the same layout with and without --dr.
Domain randomization (--dr, default on; --no-dr = nominal scene): light, table colour/texture, head-camera mounting
error, part friction and mass; ranges in scripts/data/domain_rand.json, values saved in each episode's metadata.
State (16) = left joint1..7 + gripper opening (0 closed .. 1 open), then the right arm. Action (16), same order.
Cameras 256x256: camera1 = head, camera2 = left wrist, camera3 = right wrist. Control 25 Hz, physics 100 Hz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--level", type=int, default=1, choices=[1, 2])
parser.add_argument("--split", default="train", choices=["train", "heldout_color", "heldout_shape", "wide"],
                    help="train: training parts and positions; heldout_color / heldout_shape: unseen colour / shape; "
                         "wide: target part outside the training region (eval only, see humanoid_pallet_plan)")
parser.add_argument("--episodes", type=int, default=0, help="expert: stop after this many SUCCESSFUL episodes")
parser.add_argument("--tries", type=int, default=0, help="expert: stop after this many tries (successful or not)")
parser.add_argument("--seed", type=int, default=0,
                    help="episode/try i uses seed + i (resumable, reproducible). Data generation: seeds 0..99999; "
                         "eval test 100000-100049, validation 200000+ (humanoid_pallet_plan.SEED_*)")
parser.add_argument("--control-hz", type=float, default=25.0)
parser.add_argument("--physics-hz", type=float, default=100.0)
parser.add_argument("--img-size", type=int, default=256)
parser.add_argument("--save-frames", default="")
parser.add_argument("--probe", action="store_true")
parser.add_argument("--probe-n", type=int, default=4)
parser.add_argument("--fov-probe", type=float, nargs="*", default=None)
parser.add_argument("--expert", action="store_true")
parser.add_argument("--record", default="", help="save successful expert episodes (.npz) here")
parser.add_argument("--results-json", default="", help="expert: one JSON line per try")
parser.add_argument("--prefill-max", type=int, default=3,
                    help="0..N parts (uniform) already lie in random pallet cells at the start; the target cell is a random free one")
parser.add_argument("--dr", action=argparse.BooleanOptionalAction, default=True,
                    help="domain randomization (light, table look, head-camera mounting error, part friction / mass; "
                         "ranges in scripts/data/domain_rand.json). On by default for data generation and eval")
parser.add_argument("--start-jitter", type=float, default=0.03, help="+/- m around each arm's waiting pose at reset")
parser.add_argument("--dart-noise", type=float, default=0.0,
                    help="std (rad) of OU noise added to the executed joint targets of the working arm during free-space "
                         "motion; the recorded action stays the clean expert command (DART)")
parser.add_argument("--dart-tau", type=float, default=5.0, help="correlation time of the DART noise, in control steps")
parser.add_argument("--debug", action="store_true")
# policy evaluation (PLAN G5; driven by scripts/eval_harness.py --sim-script scripts/humanoid_pallet_sim.py)
parser.add_argument("--port", type=int, default=0, help="policy eval: policy server port (0 = no policy eval)")
parser.add_argument("--exec-steps", type=int, default=8, help="policy eval: actions executed per chunk before re-planning")
parser.add_argument("--episode-seconds", type=float, default=24.0,
                    help="policy eval: episode length (PLAN 5.2: 2 x the median expert episode of the level)")
parser.add_argument("--max-step-rad", type=float, default=0.3, help="policy eval: max joint-target change per control step")
parser.add_argument("--smooth", action="store_true",
                    help="policy eval: anti-jitter layer of scripts/action_smoother.py (chunk crossfade + joint "
                         "velocity / acceleration limits like the moteus trajectory limits); off = raw chunks")
parser.add_argument("--blend-steps", type=int, default=8, help="--smooth: chunk crossfade length (steps)")
parser.add_argument("--filter-window", type=int, default=0,
                    help="--smooth: in-chunk Savitzky-Golay low-pass window (steps, order 2; 0 = off)")
parser.add_argument("--accel-scale", type=float, default=6.0, help="--smooth: joint acceleration limit = this x v_max (1/s)")
parser.add_argument("--save-video", default="",
                    help="policy eval: write annotated JPEG frames (overview camera + the 3 policy cameras) per episode to "
                         "DIR/seed_<seed>/; scripts/frames_to_mp4.py turns them into mp4 (env python)")
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
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics, Vt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import humanoid_kin as hk  # noqa: E402
import humanoid_pallet_plan as HP  # noqa: E402

assert tuple(parser._option_string_actions["--split"].choices) == HP.SPLITS
W = HP.World(ctrl_hz=args.control_hz)
CFG, SEL, PARTS = W.cfg, W.sel, W.parts
USD = hk.URDF.replace("yellow_arm.urdf", "yellow_arm/yellow_arm.usd")
ARM = hk.ARM_JOINTS
FINGERS = ["yellowarm_joint8", "yellowarm_joint9"]
F_OPEN, F_CLOSED = CFG["arm_limits"]["gripper"]["open"], CFG["arm_limits"]["gripper"]["closed"]
CAM_NAMES = ("camera1", "camera2", "camera3")
DR = json.load(open(os.path.join(HERE, "data", "domain_rand.json"), encoding="utf-8"))
TEX_DIR = os.path.join(HERE, "data", "textures")
TEXTURES = sorted(f for f in os.listdir(TEX_DIR) if f.startswith("table_") and f.endswith(".png"))
DOME_NOMINAL = 1200.0
POOL_COPIES = 3
PARK_ORIGIN = np.array([-1.5, -2.0])  # unused parts wait on the floor behind the robot
TABLE = CFG["table"]


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def head_cam_quat(pitch_deg, d_rot_deg=(0.0, 0.0, 0.0)):
    """World orientation of the head camera (USD camera: looks along -Z, +Y up), pitch down from horizontal, with
    mounting errors (roll about forward, pitch about left, yaw about up; deg) applied in the camera frame -- the same
    convention as humanoid_reach_probe.camera_need."""
    p = np.deg2rad(pitch_deg)
    fwd, left, up = np.array([np.cos(p), 0, -np.sin(p)]), np.array([0.0, 1, 0]), np.array([np.sin(p), 0, np.cos(p)])
    r, pt, y = np.deg2rad(d_rot_deg)
    C = np.stack([fwd, left, up], 1) @ hk.rot_axis((1, 0, 0), r) @ hk.rot_axis((0, 1, 0), pt) @ hk.rot_axis((0, 0, 1), y)
    Rcam = np.stack([-C[:, 1], C[:, 2], -C[:, 0]], 1)  # camera x = right, y = up, z = backwards
    return hk.mat_to_quat(Rcam)


def make_camera(path, pos, quat, fov_deg=None, focal=None):
    cam = RtxCamera(path, translations=np.array(pos, float), orientations=np.array(quat, float))
    c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    ap = c.GetHorizontalApertureAttr().Get()
    c.GetVerticalApertureAttr().Set(ap)
    c.GetFocalLengthAttr().Set(float(focal if focal else ap / (2 * np.tan(np.deg2rad(fov_deg) / 2))))
    c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(args.img_size, args.img_size), annotators=["rgb"])


def prism_mesh(stage, path, poly, length, axis):
    """Convex prism: polygon `poly` (n, 2) extruded by `length`, centred; axis 'z' (poly in xy) or 'x' (poly in yz)."""
    n = len(poly)
    pts = []
    for s in (-0.5, 0.5):
        for a, b in poly:
            pts.append((a, b, s * length) if axis == "z" else (s * length, a, b))
    faces, counts = [], []
    for i in range(n):
        j = (i + 1) % n
        faces += [i, j, n + j, n + i]
        counts.append(4)
    faces += list(range(n - 1, -1, -1))
    counts.append(n)
    faces += list(range(n, 2 * n))
    counts.append(n)
    if axis == "x":  # the yz polygon extruded along x flips the winding
        faces = [faces[k] for k in range(len(faces))]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray(faces))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(True)
    lo, hi = np.min(pts, 0), np.max(pts, 0)
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    return m


# ================================================== scene ==================================================
class Scene:
    def __init__(self, fov_deg=None):
        stage_utils.create_new_stage()
        SimulationManager.set_physics_dt(1.0 / args.physics_hz)
        self.dome = DomeLight("/World/Light")
        self.dome.set_intensities(DOME_NOMINAL)
        self.sun = DistantLight("/World/Sun")  # domain randomization only (off: intensity 0)
        self.sun.set_intensities(0.0)
        for lt in (self.dome, self.sun):
            lt.set_enabled_color_temperatures(bool(args.dr))
        GroundPlane("/World/ground", sizes=30.0, positions=[0.0, 0.0, -TABLE["height"]])
        self.stage = stage = omni.usd.get_context().get_stage()
        self.mats = {}
        h, sep = SEL["mount_height"], SEL["separation"]
        edge = SEL["table_edge_x"]
        self.table_mat = OmniPbrMaterial("/World/Materials/table")
        self.table_mat.set_input_values("diffuse_color_constant", [0.55, 0.45, 0.33])
        self.table_mat.set_input_values("project_uvw", [True])  # Cube prims have no useful UVs: world-space projection
        self.box("/World/table", (TABLE["depth"], TABLE["width"], TABLE["height"]),
                 (edge + TABLE["depth"] / 2, 0, -TABLE["height"] / 2), (0.55, 0.45, 0.33), mat=self.table_mat)
        td, tw, top = SEL["torso_depth"], SEL["torso_width"], h + 0.10
        self.box("/World/torso", (td, tw - 0.01, top + TABLE["height"]), (0, 0, (top - TABLE["height"]) / 2),
                 CFG["torso"]["color"])
        bar = HP.P.BAR
        self.box("/World/shoulder_bar", (2 * bar["half_x"], sep + 2 * bar["past_plate"], bar["z1"] - bar["z0"]),
                 (0, 0, h + (bar["z0"] + bar["z1"]) / 2), CFG["torso"]["color"])
        hc = SEL["head_camera"]
        head_x = hc["head_box_center_x"]
        if args.dr:  # the head box is visual only: keep its front face behind the camera at the largest mounting error
            head_x = min(head_x, hc["pos"][0] - 0.06 - DR["head_camera"]["d_pos_m"] - 0.005)
        self.box("/World/head", (0.12, 0.14, 0.14), (head_x, 0, hc["pos"][2]), (0.25, 0.25, 0.28),
                 collide=False)
        # arms
        lim = CFG["arm_limits"]
        self.arm_paths = {}
        for side in HP.SIDES:
            path = f"/World/arm_{side}"
            stage_utils.add_reference_to_stage(usd_path=USD, path=path)
            Rm, pm, _ = W.mount[side]
            p = stage.GetPrimAtPath(path)
            p.GetAttribute("xformOp:orient").Set(Gf.Quatd(*map(float, hk.mat_to_quat(Rm))))
            p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*map(float, pm)))
            for k, jn in enumerate(ARM):  # URDF effort, real moteus velocity (PhysX: deg/s for revolute joints)
                jp = next(x for x in stage.Traverse() if x.GetPath().pathString.startswith(path) and x.GetName() == jn)
                jp.GetAttribute("drive:angular:physics:maxForce").Set(float(lim["effort_nm"][k]))
                jp.GetAttribute("physxJoint:maxJointVelocity").Set(float(np.degrees(lim["velocity_used_rad_s"][k])))
            self.arm_paths[side] = path
        # pallet: static tray, moved every episode
        pl = PARTS["pallet"]
        UsdGeom.Xform.Define(stage, "/World/pallet")
        L, Wd = W.pallet_size
        wall, wh, base, col = pl["wall"], pl["wall_height"], pl["base"], pl["color"]
        self.box("/World/pallet/base", (L, Wd, base), (0, 0, base / 2), col)
        for r in range(pl["rows"] + 1):
            u = -L / 2 + wall / 2 + r * W.pitch
            self.box(f"/World/pallet/wall_u{r}", (wall, Wd, wh), (u, 0, base + wh / 2), col)
        for c in range(pl["cols"] + 1):
            v = -Wd / 2 + wall / 2 + c * W.pitch
            self.box(f"/World/pallet/wall_v{c}", (L, wall, wh), (0, v, base + wh / 2), col)
        # parts pool: every shape x size, POOL_COPIES copies; colours are set per episode
        grip = RigidBodyMaterial("/World/PhysicsMaterials/part")
        grip.set_friction_coefficients(static_frictions=PARTS["friction"], dynamic_frictions=PARTS["friction"])
        grip.set_combine_modes(frictions="max")
        self.pool, self.pool_paths, self.pool_mat = [], [], []
        k = 0
        for shape in PARTS["shapes"]:
            for scale in PARTS["sizes"]:
                for c in range(POOL_COPIES):
                    part = HP.Part(PARTS, shape, scale, "red")
                    path = f"/World/parts/{shape}_{int(round(scale * 100))}_{c}"
                    self.make_part(path, part, grip)
                    self.pool.append((shape, scale))
                    self.pool_paths.append(path)
                    k += 1
        # cameras
        self.part_phys = grip
        q_head = head_cam_quat(hc["pitch_deg"])
        self.cams = {
            "camera1": make_camera("/World/head_cam", hc["pos"], q_head, fov_deg=fov_deg or hc["fov_deg"]),
            "camera2": make_camera(f"{self.arm_paths['L']}/yellowarm_link7/cam_wrist", (0.10, 0.0, 0.03),
                                   look_at_quat((0.10, 0.0, 0.03), (0.0, 0.0, 0.17), (1.0, 0.0, 0.0)), focal=11.0),
            "camera3": make_camera(f"{self.arm_paths['R']}/yellowarm_link7/cam_wrist", (0.10, 0.0, 0.03),
                                   look_at_quat((0.10, 0.0, 0.03), (0.0, 0.0, 0.17), (1.0, 0.0, 0.0)), focal=11.0),
        }

    def material(self, key, color):
        if key not in self.mats:
            self.mats[key] = OmniPbrMaterial(f"/World/Materials/{key}")
            self.mats[key].set_input_values("diffuse_color_constant", list(map(float, color)))
        return self.mats[key]

    def box(self, path, size, pos, color, collide=True, mat=None):
        c = Cube(path, sizes=1.0, translations=np.array(pos, float), scales=np.array(size, float))
        if collide:
            GeomPrim(path, apply_collision_apis=True)
        c.apply_visual_materials(mat or self.material("c_" + "_".join(f"{int(round(v * 1000))}" for v in color), color))
        return c

    def make_part(self, path, part, phys_mat):
        sp = part.spec
        park = self.park_pos(len(self.pool_paths), part)
        if sp["kind"] == "box":
            vis = Cube(path, sizes=1.0, translations=park, scales=part.dims)
        else:
            n = 24 if sp["kind"] == "cylinder" else 6
            if sp["kind"] == "cylinder":
                r = part.dims[1] / 2
                ang = np.arange(n) * 2 * np.pi / n
            else:  # hex: corners on local x, flats across local y
                r = part.dims[0] / 2
                ang = np.arange(n) * np.pi / 3
            poly = np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
            axis = sp.get("axis", "z")
            if axis == "x":  # polygon in (y, z) for a lying cylinder
                length = part.dims[0]
            else:
                length = part.dims[2]
            prism_mesh(self.stage, path, poly, length, axis)
            UsdGeom.XformCommonAPI(self.stage.GetPrimAtPath(path)).SetTranslate(Gf.Vec3d(*map(float, park)))
            vis = None
        g = GeomPrim(path, apply_collision_apis=True)
        if sp["kind"] != "box":
            g.set_collision_approximations("convexHull")
        RigidPrim(path)
        UsdPhysics.MassAPI.Apply(self.stage.GetPrimAtPath(path)).CreateMassAttr(float(part.mass))
        g.apply_physics_materials(phys_mat)
        mat = OmniPbrMaterial(f"/World/Materials/part_{len(self.pool_paths)}")
        mat.set_input_values("diffuse_color_constant", [0.8, 0.8, 0.8])
        if vis is not None:
            vis.apply_visual_materials(mat)
        else:
            XformPrim(path).apply_visual_materials(mat)
        self.pool_mat.append(mat)

    def park_pos(self, k, part):
        return np.array([PARK_ORIGIN[0] - 0.15 * (k % 5), PARK_ORIGIN[1] + 0.15 * (k // 5),
                         -TABLE["height"] + part.height / 2 + 0.002])

    def set_pallet(self, center, yaw):
        p = self.stage.GetPrimAtPath("/World/pallet")
        api = UsdGeom.XformCommonAPI(p)
        api.SetTranslate(Gf.Vec3d(float(center[0]), float(center[1]), 0.0))
        api.SetRotate(Gf.Vec3f(0.0, 0.0, float(np.degrees(yaw))))


# ================================================== runtime ==================================================
class Robot:
    def __init__(self, scene):
        self.scene = scene
        self.arts = {s: Articulation(scene.arm_paths[s]) for s in HP.SIDES}
        self.parts = RigidPrim(scene.pool_paths)
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        simulation_app.update()
        names = self.arts["L"].dof_names
        self.ai = [names.index(j) for j in ARM]
        self.fi = [names.index(j) for j in FINGERS]
        self.ctrl = self.ai + self.fi
        self.ndof = len(names)
        self.decimation = max(1, round(args.physics_hz / args.control_hz))
        self.mass0 = self.parts.get_masses().numpy().reshape(-1).astype(np.float64)
        self.inertia0 = self.parts.get_inertias().numpy().reshape(len(self.mass0), -1).astype(np.float64)
        self.head_cam = XformPrim("/World/head_cam")

    # ---- arms ----
    def q(self, side):
        return self.arts[side].get_dof_positions().numpy()[0].astype(np.float64)

    def read_state(self):
        out = []
        for s in HP.SIDES:
            q = self.q(s)
            g = np.clip((F_CLOSED - q[self.fi].mean()) / (F_CLOSED - F_OPEN), 0.0, 1.0)
            out.append(np.concatenate([q[self.ai], [g]]))
        return np.concatenate(out).astype(np.float32)

    def teleport(self, side, q7, grip=1.0):
        full = np.zeros(self.ndof)
        full[self.ai] = q7
        f = F_CLOSED + (F_OPEN - F_CLOSED) * grip
        full[self.fi] = f
        a = self.arts[side]
        a.set_dof_positions(full)
        a.set_dof_velocities(np.zeros_like(full))
        a.set_dof_position_targets(full)

    def apply(self, targets):
        """targets: {side: (q7, grip 0..1)} -> one control step."""
        for s, (q7, g) in targets.items():
            f = F_CLOSED + (F_OPEN - F_CLOSED) * float(np.clip(g, 0.0, 1.0))
            self.arts[s].set_dof_position_targets(np.concatenate([q7, [f, f]]), dof_indices=self.ctrl)
        for _ in range(self.decimation):
            simulation_app.update()

    # ---- parts ----
    def part_pose(self, k):
        p, q = self.parts.get_world_poses(indices=[k])
        return p.numpy()[0].astype(np.float64), q.numpy()[0].astype(np.float64)

    def part_vel(self, k):
        lv, av = self.parts.get_velocities(indices=[k])
        return float(np.linalg.norm(lv.numpy()[0])), float(np.linalg.norm(av.numpy()[0]))

    def place_part(self, k, pos, quat):
        self.parts.set_world_poses(positions=np.array([pos], float), orientations=np.array([quat], float), indices=[k])
        self.parts.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)), indices=[k])

    def park_all(self):
        n = len(self.scene.pool_paths)
        pos = np.stack([self.scene.park_pos(k, HP.Part(PARTS, *self.scene.pool[k], "red")) for k in range(n)])
        self.parts.set_world_poses(positions=pos, orientations=np.tile([1.0, 0, 0, 0], (n, 1)))
        self.parts.set_velocities(np.zeros((n, 3)), np.zeros((n, 3)))

    def grab(self):
        out = []
        for n in CAM_NAMES:
            d, _ = self.scene.cams[n].get_data("rgb")
            img = np.zeros((args.img_size, args.img_size, 3), np.uint8) if d is None else d.numpy()[..., :3]
            out.append(np.ascontiguousarray(img.astype(np.uint8)))
        return out


def hsv_to_rgb(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def sample_dr(seed, n_pool):
    """Domain-randomization values of one episode (None with --no-dr). Own random stream (seed, 2025): the layout of a
    seed does not depend on --dr."""
    if not args.dr:
        return None
    rng = np.random.default_rng([seed, 2025])
    L, T, C, P = DR["light"], DR["table"], DR["head_camera"], DR["parts"]

    def u(r):
        return float(rng.uniform(*r))

    return {
        "dome_intensity": u(L["dome_intensity"]), "sun_intensity": u(L["distant_intensity"]),
        "sun_elevation_deg": u(L["distant_elevation_deg"]), "sun_azimuth_deg": u(L["distant_azimuth_deg"]),
        "color_temperature_k": u(L["color_temperature_k"]),
        "table_rgb": [round(c, 3) for c in hsv_to_rgb(u(T["hue"]), u(T["saturation"]), u(T["value"]))],
        "table_texture": TEXTURES[int(rng.integers(len(TEXTURES)))] if rng.random() < T["texture_prob"] else "",
        "table_roughness": u(T["roughness"]), "texture_scale": u(T["texture_scale"]),
        "texture_rotate_deg": u((0.0, 180.0)), "texture_translate": rng.uniform(0.0, 1.0, 2).round(3).tolist(),
        "cam_d_pos_m": rng.uniform(-C["d_pos_m"], C["d_pos_m"], 3).round(4).tolist(),
        "cam_d_rot_deg": rng.uniform(-C["d_rot_deg"], C["d_rot_deg"], 3).round(2).tolist(),
        "part_friction": rng.uniform(*P["friction"], n_pool).round(3).tolist(),
        "part_mass_scale": rng.uniform(*P["mass_scale"], n_pool).round(3).tolist(),
    }


def apply_dr(robot, dr):
    """Set the scene to the DR values of one episode (before the parts are placed). No-op with --no-dr."""
    if dr is None:
        return
    import warp as wp

    sc = robot.scene
    sc.dome.set_intensities(dr["dome_intensity"])
    sc.sun.set_intensities(dr["sun_intensity"])
    for lt in (sc.dome, sc.sun):
        lt.set_color_temperatures(dr["color_temperature_k"])
    e, a = np.deg2rad(dr["sun_elevation_deg"]), np.deg2rad(dr["sun_azimuth_deg"])
    # a distant light shines along its -Z: point +Z at the sun (elevation e, azimuth a)
    Rs = hk.rot_axis((0, 0, 1), a) @ hk.rot_axis((0, 1, 0), np.pi / 2 - e)
    sc.sun.set_world_poses(orientations=hk.mat_to_quat(Rs)[None])
    m, tex = sc.table_mat, dr["table_texture"]
    m.set_input_values("diffuse_texture", [os.path.join(TEX_DIR, tex).replace(os.sep, "/") if tex else ""])
    m.set_input_values("diffuse_color_constant", dr["table_rgb"])
    m.set_input_values("diffuse_tint", dr["table_rgb"] if tex else [1.0, 1.0, 1.0])
    m.set_input_values("reflection_roughness_constant", [dr["table_roughness"]])
    m.set_input_values("texture_scale", [dr["texture_scale"]] * 2)
    m.set_input_values("texture_rotate", [dr["texture_rotate_deg"]])
    m.set_input_values("texture_translate", dr["texture_translate"])
    hc = SEL["head_camera"]
    robot.head_cam.set_world_poses(positions=(np.array(hc["pos"]) + dr["cam_d_pos_m"])[None],
                                   orientations=head_cam_quat(hc["pitch_deg"], dr["cam_d_rot_deg"])[None])
    k = np.asarray(dr["part_mass_scale"])
    robot.parts.set_masses((robot.mass0 * k)[:, None])
    robot.parts.set_inertias(robot.inertia0 * k[:, None])
    # per-part friction at runtime through the PhysX view (the USD material is shared by all parts)
    view = robot.parts._physics_rigid_body_view
    props = view.get_material_properties()  # (parts, shapes, 3: static, dynamic, restitution)
    pm = props.numpy().copy()
    mu = np.asarray(dr["part_friction"], np.float32)[:, None]
    pm[:, :, 0], pm[:, :, 1] = mu, mu
    view.set_material_properties(wp.array(pm, dtype=props.dtype, device=props.device),
                                 wp.array(np.arange(len(pm), dtype=np.int32), dtype=wp.int32, device=props.device))


def part_rest_quat(part, yaw):
    return yaw_quat(yaw)  # every part rests on its local -z face (the lying cylinder's axis is its local x)


class Episode:
    """One layout: parts (pool indices), poses, pallet pose, target, task string, nominal plan."""


def new_episode(robot, rng, seed):
    """Sample parts + layout + start poses until the expert has a valid nominal plan. Returns (Episode, n_rejected)."""
    rejected = 0
    while True:
        n = 1 if args.level == 1 else int(rng.integers(3, 5))
        parts, tgt = W.sample_task_parts(rng, args.split, n)
        lay = W.sample_layout(rng, parts, tgt, split=args.split)
        if lay is None:
            rejected += 1
            continue
        pc, pyaw, poses = lay
        fill = W.sample_prefill(rng, parts, int(rng.integers(args.prefill_max + 1)), pc, pyaw)
        # pool slots: (shape, scale) copies, for the table parts then the prefilled ones
        used, slots = {}, []
        for p in parts + [f[0] for f in fill]:
            key = (p.shape, p.scale)
            c = used.get(key, 0)
            used[key] = c + 1
            slots.append(None if c >= POOL_COPIES else
                         [k for k, ps in enumerate(robot.scene.pool) if ps == key][c])
        if any(s is None for s in slots):
            rejected += 1
            continue
        q_start = {}
        for s in HP.SIDES:
            tcp = W.fk_tcp(s, W.park[s])[0][0] + rng.uniform(-args.start_jitter, args.start_jitter, 3)
            q_start[s] = W.ik(s, tcp, float(rng.uniform(-0.3, 0.3)), q_ref=W.park[s])
        if any(v is None for v in q_start.values()):
            rejected += 1
            continue
        xy, yaw = poses[tgt]
        filled = {f[1] for f in fill}
        free = [k for k in range(PARTS["pallet"]["rows"] * PARTS["pallet"]["cols"]) if k not in filled]
        plan = W.choose_and_plan(parts[tgt], xy, yaw, pc, pyaw, free, q_start, rng=rng, fill=fill)
        if plan is None:
            rejected += 1
            continue
        ep = Episode()
        ep.parts, ep.slots, ep.poses, ep.target = parts, slots[:len(parts)], poses, tgt
        ep.fill, ep.fill_slots = fill, slots[len(parts):]
        ep.pallet_c, ep.pallet_yaw, ep.q_start, ep.plan = pc, pyaw, q_start, plan
        ep.task = f"put the {parts[tgt].describe()} on the pallet"
        ep.dr = sample_dr(seed, len(robot.scene.pool))
        return ep, rejected


def setup_episode(robot, ep):
    robot.park_all()
    apply_dr(robot, ep.dr)
    robot.scene.set_pallet(ep.pallet_c, ep.pallet_yaw)
    for p, k in zip(ep.parts, ep.slots):
        robot.scene.pool_mat[k].set_input_values("diffuse_color_constant", list(map(float, p.rgb)))
    for s in HP.SIDES:
        robot.teleport(s, ep.q_start[s], 1.0)
    for _ in range(3):
        simulation_app.update()
    for p, k, (xy, yaw) in zip(ep.parts, ep.slots, ep.poses):
        robot.place_part(k, (xy[0], xy[1], p.height / 2 + 0.001), part_rest_quat(p, yaw))
    base = PARTS["pallet"]["base"]
    for (p, _, xy, yaw), k in zip(ep.fill, ep.fill_slots):
        robot.scene.pool_mat[k].set_input_values("diffuse_color_constant", list(map(float, p.rgb)))
        robot.place_part(k, (xy[0], xy[1], base + p.height / 2 + 0.001), part_rest_quat(p, yaw))
    hold = {s: (ep.q_start[s], 1.0) for s in HP.SIDES}
    for _ in range(8):  # let parts settle and the RTX cameras catch up (not recorded)
        robot.apply(hold)


def check_success(robot, ep):
    """Target part inside a pallet cell that was free at the start, at rest, not tipped over; no other table part on
    the pallet (L2). The prefilled parts do not count."""
    k = ep.slots[ep.target]
    part = ep.parts[ep.target]
    pos, quat = robot.part_pose(k)
    lv, av = robot.part_vel(k)
    cell = W.cell_of(pos[:2], ep.pallet_c, ep.pallet_yaw)
    pl = PARTS["pallet"]
    low = pos[2] < pl["base"] + pl["wall_height"] + part.height / 2 + 0.01
    still = lv < 0.02 and av < 0.5
    upright = part.tilt_ok(quat)
    others = []
    for i, (p, kk) in enumerate(zip(ep.parts, ep.slots)):
        if i == ep.target:
            continue
        po = robot.part_pose(kk)[0]
        if W.dist_to_pallet(po[:2], ep.pallet_c, ep.pallet_yaw)[0] < 1e-6:
            others.append(i)
    was_free = cell is not None and cell not in {f[1] for f in ep.fill}
    ok = was_free and low and still and upright and not others
    return ok, {"cell": cell, "cell_was_free": bool(was_free), "low": bool(low), "still": bool(still), "upright": bool(upright), "others_on_pallet": others,
                "end_pos": np.round(pos, 4).tolist(), "lin_vel": round(lv, 4), "ang_vel": round(av, 3)}


GRIP_LOST = 0.05  # gripper opening (1 = 75 mm) below which the closed fingers hold nothing


def run_expert(robot, ep, rng, record):
    """Execute the plan; returns the recorded frames (or None) and info."""
    plan = ep.plan
    s = plan["side"]
    o = "R" if s == "L" else "L"
    hold_o = (ep.q_start[o], 1.0)
    rec = {"state": [], "action": [], **{n: [] for n in CAM_NAMES}} if record else None
    dart = np.zeros(7)
    dart_a = np.exp(-1.0 / max(args.dart_tau, 1e-6))
    last = {"q": ep.q_start[s].copy()}
    # smallest gripper opening of the working arm while it is commanded closed after the grasp: ~0 = the part slipped
    # out during the carry (an episode can still end in the cell by luck; such demos are not recorded)
    grip = {"watch": False, "min": 1.0}
    gi = 7 if s == "L" else 15

    def step(q7, g, noisy=False):
        if grip["watch"] and g == 0.0:
            grip["min"] = min(grip["min"], float(robot.read_state()[gi]))
        if rec is not None:
            rec["state"].append(robot.read_state())
            for n, img in zip(CAM_NAMES, robot.grab()):
                rec[n].append(img)
            a = {s: (q7, g), o: hold_o}
            rec["action"].append(np.concatenate([np.concatenate([a[x][0], [a[x][1]]]) for x in HP.SIDES]).astype(np.float32))
        qa = q7
        if args.dart_noise > 0 and noisy:
            dart[:] = dart_a * dart + np.sqrt(1 - dart_a ** 2) * rng.normal(0, args.dart_noise, 7)
            qa = np.clip(q7 + dart, W.lo, W.hi)
        elif args.dart_noise > 0:
            dart[:] = 0.0
        robot.apply({s: (qa, g), o: hold_o})
        last["q"] = q7

    def run(traj, g, noisy=False):
        for q7 in traj:
            step(q7, g, noisy)

    def hold(g, n):
        for _ in range(n):
            step(last["q"], g)

    pk = plan["pick"]

    def dbg(tag, q_plan):
        if args.debug:
            q_now = robot.q(s)[robot.ai]
            print(f"[dbg] {tag}: tcp {np.round(W.fk_tcp(s, q_now)[0][0], 3)} planned {np.round(W.fk_tcp(s, q_plan)[0][0], 3)} "
                  f"|dq| max {np.abs(q_now - q_plan).max():.3f} rad (joint {int(np.abs(q_now - q_plan).argmax()) + 1})")

    if args.debug:
        print(f"[dbg] plan: arm {s} transit {pk['transit_kind']} {len(pk['transit'])} steps, yaw {np.degrees(pk['yaw']):.0f}, "
              f"margin {pk['margin']:.2f}, carry {plan['place']['carry_kind']}")
    run(pk["transit"], 1.0, noisy=True)
    dbg("after transit", pk["transit"][-1])
    if args.dart_noise > 0:
        hold(1.0, 6)  # recover from the drift above the part before going down
    run(pk["down"], 1.0)
    dbg("after down", pk["down"][-1])
    hold(0.0, 15)  # close (fingers at 0.1 m/s need ~0.4 s)
    grip["watch"] = True
    # lift with the nominal plan, then measure where the part sits in the gripper and re-plan the place
    pp = plan["place"]
    run(pp["lift"], 0.0)
    q_now = robot.q(s)[robot.ai]
    tcp, Rl = W.fk_tcp(s, q_now)
    k = ep.slots[ep.target]
    ppos, pquat = robot.part_pose(k)
    off_g = Rl[0].T @ (ppos - tcp[0])
    replanned = W.plan_place(s, ep.parts[ep.target], None, off_g, HP.yaw_of(pquat), pk["yaw"], ep.pallet_c, ep.pallet_yaw,
                             plan["cell"], q_from=last["q"])
    held = np.linalg.norm(ppos[:2] - tcp[0][:2]) < 0.04 and ppos[2] > 0.05
    if replanned is not None and not W.release_clear(replanned, ep.fill):
        replanned = None  # the re-planned release would touch a prefilled part: keep the nominal one
    if replanned is not None:
        pp = replanned
    if args.debug:
        print(f"[dbg] after lift: part {np.round(ppos, 3)} offset_mm {np.round(1000 * off_g, 1)} held {held} "
              f"replanned {replanned is not None} carry {pp['carry_kind']} {len(pp['carry'])} steps")
    run(pp["carry"], 0.0, noisy=True)
    dbg("after carry", pp["carry"][-1])
    if args.debug:
        print(f"[dbg] after carry: part {np.round(robot.part_pose(k)[0], 3)}")
    run(pp["lower"], 0.0)
    if args.debug:
        print(f"[dbg] after lower: part {np.round(robot.part_pose(k)[0], 3)}")
    hold(pp["release_open"], 12)
    run(pp["up"], pp["release_open"])
    run(pp["home"], 1.0, noisy=True)
    hold(1.0, 4)
    for _ in range(20):  # settle (not recorded)
        robot.apply({s: (last["q"], 1.0), o: hold_o})
    info = {"held_after_lift": bool(held), "replanned": replanned is not None, "grip_min_open": round(grip["min"], 3),
            "grasp_offset_mm": np.round(1000 * off_g, 1).tolist()}
    return rec, info


def episode_meta(ep, info=None):
    p = ep.plan
    return {"level": args.level, "split": args.split, "task": ep.task, "dart_noise": args.dart_noise,
            "parts": [dict(pt.meta(), xy=np.round(xy, 4).tolist(), yaw_deg=round(float(np.degrees(yw)), 1))
                      for pt, (xy, yw) in zip(ep.parts, ep.poses)],
            "prefill": [dict(pt.meta(), cell=c, xy=np.round(xy, 4).tolist(), yaw_deg=round(float(np.degrees(yw)), 1))
                        for pt, c, xy, yw in ep.fill],
            "target": ep.target, "target_cell": p["cell"], "arm": p["side"], "arm_reach": p["reach"],
            "grasp_yaw_deg": round(float(np.degrees(p["pick"]["yaw"])), 1),
            "ik_margin": round(p["pick"]["margin"], 3), "arm_clearance_m": round(p["arm_clearance"], 3),
            "dr": dr_meta(ep),
            "pallet_center": np.round(ep.pallet_c, 4).tolist(), "pallet_yaw_deg": round(float(np.degrees(ep.pallet_yaw)), 2),
            **(info or {})}


def dr_meta(ep):
    """The DR values of an episode for the npz / results metadata: per-part friction and mass only for the parts in use."""
    if ep.dr is None:
        return None
    d = {k: v for k, v in ep.dr.items() if k not in ("part_friction", "part_mass_scale")}
    for key in ("part_friction", "part_mass_scale"):
        d[key] = [ep.dr[key][k] for k in list(ep.slots) + list(ep.fill_slots)]
    return d


def warm_textures(robot):
    """Load every table texture once, so no recorded first frame shows a texture still streaming in."""
    if not args.dr:
        return
    for tex in TEXTURES:
        robot.scene.table_mat.set_input_values("diffuse_texture", [os.path.join(TEX_DIR, tex).replace(os.sep, "/")])
        for _ in range(20):
            simulation_app.update()


# ================================================== modes ==================================================
def fov_probe(fovs):
    """Head camera at each FOV, nominal and at the two worst mounting errors of humanoid.json, over green tiles that
    mark the shared workspace. Arms at their waiting poses."""
    from PIL import Image, ImageDraw

    scene = Scene()
    hc = SEL["head_camera"]
    errs = [{"d_pos": [0, 0, 0], "d_rot_deg": [0, 0, 0]}] + hc["worst_errors"][:2]
    cams = {}
    for f in fovs:
        for e, er in enumerate(errs):
            cams[(f, e)] = make_camera(f"/World/fov_{int(f)}_{e}", np.array(hc["pos"]) + np.array(er["d_pos"]),
                                       head_cam_quat(hc["pitch_deg"], er["d_rot_deg"]), fov_deg=f)
    S = W.ws["shared"]
    X, Y = np.meshgrid(W.ws["xs"], W.ws["ys"], indexing="ij")
    for i, (x, y) in enumerate(zip(X[S], Y[S])):
        scene.box(f"/World/tiles/t{i}", (W.ws_step - 0.003, W.ws_step - 0.003, 0.002), (x, y, 0.001), (0.2, 0.75, 0.25),
                  collide=False)
    robot = Robot(scene)
    robot.park_all()
    scene.set_pallet((5.0, 5.0), 0.0)
    for s in HP.SIDES:
        robot.teleport(s, W.park[s])
    for _ in range(80):
        robot.apply({s: (W.park[s], 1.0) for s in HP.SIDES})
    out = args.save_frames or os.path.join(os.path.dirname(HERE), "logs", "humanoid_probe", "fov")
    os.makedirs(out, exist_ok=True)
    tile = args.img_size
    grid = Image.new("RGB", (len(errs) * (tile + 8) + 8, len(fovs) * (tile + 28) + 30), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    d.text((8, 6), f"head camera pitch {hc['pitch_deg']} deg at {hc['pos']}; green = shared workspace; columns: "
                   f"nominal, worst mounting errors", fill=(0, 0, 0))
    for r, f in enumerate(fovs):
        for e, er in enumerate(errs):
            dd, _ = cams[(f, e)].get_data("rgb")
            img = Image.fromarray(dd.numpy()[..., :3].astype(np.uint8))
            img.save(os.path.join(out, f"head_fov{int(f)}_{['nominal', 'err1', 'err2'][e]}.png"))
            x0, y0 = 8 + e * (tile + 8), 30 + r * (tile + 28)
            grid.paste(img, (x0, y0 + 20))
            lab = f"FOV {f:g}" + ("" if e == 0 else f"  dpos {er['d_pos']} drot {er['d_rot_deg']}")
            d.text((x0, y0 + 4), lab, fill=(0, 0, 0))
    grid.save(os.path.join(out, "fov_grid.png"))
    print(f"[fov] saved {out}\\fov_grid.png")


def probe():
    from PIL import Image, ImageDraw

    scene = Scene()
    robot = Robot(scene)
    warm_textures(robot)
    out = args.save_frames or os.path.join(os.path.dirname(HERE), "logs", "pallet_probe")
    os.makedirs(out, exist_ok=True)
    tile = args.img_size
    grid = Image.new("RGB", (3 * (tile + 6) + 6, args.probe_n * (tile + 40) + 6), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    for i in range(args.probe_n):
        rng = np.random.default_rng(args.seed + i)
        ep, rej = new_episode(robot, rng, args.seed + i)
        setup_episode(robot, ep)
        for _ in range(30):
            robot.apply({s: (ep.q_start[s], 1.0) for s in HP.SIDES})
        imgs = robot.grab()
        y0 = 6 + i * (tile + 40)
        d.text((6, y0), f"L{args.level} seed {args.seed + i}: {ep.task} | arm {ep.plan['side']} -> cell {ep.plan['cell']} "
                        f"| rejected layouts {rej}", fill=(0, 0, 0))
        for c, img in enumerate(imgs):
            Image.fromarray(img).save(os.path.join(out, f"L{args.level}_s{args.seed + i}_{CAM_NAMES[c]}.png"))
            grid.paste(Image.fromarray(img), (6 + c * (tile + 6), y0 + 16))
        print(f"[probe] seed {args.seed + i}: {json.dumps(episode_meta(ep))}")
    grid.save(os.path.join(out, f"probe_L{args.level}.png"))
    print(f"[probe] saved {os.path.join(out, f'probe_L{args.level}.png')}")


def die(msg):
    """Exit with status 2 (sys.exit inside the Kit app is swallowed and the process returns 0)."""
    print(msg, file=sys.stderr, flush=True)
    os._exit(2)


def expert():
    scene = Scene()
    robot = Robot(scene)
    warm_textures(robot)
    saved = tries = wins = 0
    if args.record and args.split != "train":
        die(f"[expert] --record with --split {args.split}: only the train split may become training data")
    seed0 = args.seed
    if args.record:
        os.makedirs(args.record, exist_ok=True)
        done = sorted(f for f in os.listdir(args.record) if f.endswith(".npz"))
        saved = len(done)
        if done:  # resume after the last recorded seed (never record a seed twice)
            last = max(json.loads(str(np.load(os.path.join(args.record, f))["meta"]))["seed"] for f in done)
            seed0 = max(args.seed, last + 1)
            print(f"[expert] resume: {saved} episodes in {args.record}, last seed {last} -> continuing at seed {seed0}")
    results = None
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        results = open(args.results_json, "a", encoding="utf-8")
    t_all = time.time()
    while simulation_app.is_running():
        if args.episodes and saved >= args.episodes:
            break
        if args.tries and tries >= args.tries:
            break
        seed = seed0 + tries
        if args.record and seed >= HP.SEED_DATA_END:
            die(f"[expert] seed {seed} >= {HP.SEED_DATA_END}: the eval seeds (test {HP.SEED_TEST}, validation "
                     f"{HP.SEED_VAL_START}+) must never be recorded as training data")
        rng = np.random.default_rng(seed)
        t0 = time.time()
        ep, rej = new_episode(robot, rng, seed)
        setup_episode(robot, ep)
        rec, info = run_expert(robot, ep, rng, record=bool(args.record))
        ok, chk = check_success(robot, ep)
        tries += 1
        wins += ok
        T = len(rec["action"]) if rec else None
        meta = episode_meta(ep, {**info, **chk, "seed": seed, "rejected_layouts": rej})
        print(f"[expert] try {tries - 1} seed {seed}: {ep.task!r} arm {ep.plan['side']} cell {ep.plan['cell']} "
              f"-> {'SUCCESS' if ok else 'fail'} {json.dumps(chk)} ({time.time() - t0:.1f}s) [{wins}/{tries}]")
        keep = ok and info["grip_min_open"] > GRIP_LOST
        if ok and not keep:
            print(f"[expert] seed {seed}: success but the part slipped out of the gripper during the carry "
                  f"(min opening {info['grip_min_open']}) - not recorded")
        if results:
            results.write(json.dumps({"try": tries - 1, "seed": seed, "success": bool(ok), "kept": bool(keep), "frames": T,
                                      "seconds": round(time.time() - t0, 1), **meta}) + "\n")
            results.flush()
        if keep and args.record:
            path = os.path.join(args.record, f"episode_{saved:04d}.npz")
            with open(path + ".tmp", "wb") as fh:  # write then rename: a killed worker leaves no broken .npz
                np.savez_compressed(fh, state=np.stack(rec["state"]), action=np.stack(rec["action"]),
                                    **{n: np.stack(rec[n]) for n in CAM_NAMES}, task=ep.task, fps=args.control_hz,
                                    level=args.level, meta=json.dumps(meta))
            os.replace(path + ".tmp", path)
            saved += 1
    print(f"[expert] done: {wins}/{tries} successful, {saved} episodes saved, {time.time() - t_all:.0f}s")
    if results:
        results.close()


class VideoWriter:
    """Annotated frames of a policy episode: a square overview camera (768 x 768, front right, above the table) over
    the 3 policy cameras (256 x 256 each), as JPEGs in DIR/seed_<seed>/ plus info.json. One frame per control step
    (25 fps). (A non-square CameraSensor resolution came back as 512 x 512 here, hence square.)"""

    OV_W = OV_H = 768

    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        eye, target = np.array([1.20, -0.55, 0.85]), np.array([0.38, 0.0, 0.08])
        cam = RtxCamera("/World/video_cam", translations=eye, orientations=look_at_quat(eye, target))
        c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath("/World/video_cam"))
        ap = c.GetHorizontalApertureAttr().Get()
        c.GetVerticalApertureAttr().Set(ap)
        c.GetFocalLengthAttr().Set(float(ap / (2 * np.tan(np.deg2rad(55.0) / 2))))
        c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
        self.cam = CameraSensor(cam, resolution=(self.OV_W, self.OV_H), annotators=["rgb"])

    def start(self, seed, ep):
        self.dir = os.path.join(self.out, f"seed_{seed}")
        os.makedirs(self.dir, exist_ok=True)
        for f in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, f))
        self.n, self.seed, self.ep = 0, seed, ep

    def _compose(self, imgs, step, banner=None):
        from PIL import Image, ImageDraw

        d, _ = self.cam.get_data("rgb")
        ov = np.zeros((self.OV_H, self.OV_W, 3), np.uint8) if d is None else d.numpy()[..., :3].astype(np.uint8)
        canvas = Image.new("RGB", (self.OV_W, self.OV_H + args.img_size), (0, 0, 0))
        canvas.paste(Image.fromarray(ov).resize((self.OV_W, self.OV_H)), (0, 0))
        for i, img in enumerate(imgs):
            canvas.paste(Image.fromarray(img), (i * args.img_size, self.OV_H))
        dr = ImageDraw.Draw(canvas)
        dr.rectangle((0, 0, self.OV_W, 22), fill=(0, 0, 0))
        dr.text((6, 5), f"L{args.level} {args.split} seed {self.seed} | {self.ep.task!r} | t {step / args.control_hz:5.2f} s"
                        f" | exec {args.exec_steps}{' | smooth' if args.smooth else ''}", fill=(255, 255, 255))
        for i, lab in enumerate(("camera1 head", "camera2 left wrist", "camera3 right wrist")):
            dr.text((i * args.img_size + 4, self.OV_H + 3), lab, fill=(255, 255, 0))
        if banner:
            ok = banner.startswith("SUCCESS")
            dr.rectangle((0, self.OV_H - 34, self.OV_W, self.OV_H), fill=(0, 120, 0) if ok else (150, 0, 0))
            dr.text((8, self.OV_H - 26), banner, fill=(255, 255, 255))
        return canvas

    def frame(self, imgs, step, banner=None):
        self._compose(imgs, step, banner).save(os.path.join(self.dir, f"{self.n:05d}.jpg"), quality=90)
        self.n += 1

    def finish(self, imgs, step, ok, chk):
        banner = ("SUCCESS" if ok else "FAIL") + f": cell {chk['cell']} free {chk['cell_was_free']} upright " \
                 f"{chk['upright']} still {chk['still']} others_on_pallet {chk['others_on_pallet']}"
        for _ in range(int(args.control_hz * 1.5)):  # hold the end state with the verdict for 1.5 s
            self.frame(imgs, step, banner)
        json.dump({"seed": self.seed, "task": self.ep.task, "success": bool(ok), "frames": self.n, "fps": args.control_hz,
                   "level": args.level, "split": args.split, "exec_steps": args.exec_steps, **chk},
                  open(os.path.join(self.dir, "info.json"), "w", encoding="utf-8"))


def jitter_metrics(q, hz, thr_mrad=0.5):
    """Shaking of the executed joint targets q (T, 14): mean / p99 |second difference| (mrad) and joint velocity sign
    flips per second per joint, counting steps that move > thr_mrad (as scripts/jitter_offline.py)."""
    if len(q) < 3:
        return {"jerk_mean_mrad": None, "jerk_p99_mrad": None, "rev_per_s": None}
    j = np.abs(q[2:] - 2 * q[1:-1] + q[:-2]) * 1000
    v = np.diff(q, axis=0) * 1000
    flip = (np.sign(v[1:]) != np.sign(v[:-1])) & (np.abs(v[1:]) > thr_mrad) & (np.abs(v[:-1]) > thr_mrad)
    return {"jerk_mean_mrad": round(float(j.mean()), 3), "jerk_p99_mrad": round(float(np.percentile(j, 99)), 2),
            "rev_per_s": round(float(flip.sum() / q.shape[1] / (len(q) / hz)), 3)}


def evaluate():
    """Policy evaluation: episode i = seed + i, the same layout / DR / start pose as the expert sees for that seed (the
    expert's nominal plan only decides that the layout is feasible; it is never executed). The policy drives all 16
    targets for --episode-seconds; joint targets are clipped to the URDF limits and to --max-step-rad per step, the
    gripper to 0..1. Scored with the expert's check_success after a short settle. --episodes = episodes run (successful
    or not), unlike --expert. One JSON line per episode in --results-json."""
    import socket

    from wire import HOST, recv_msg, send_msg

    if args.seed < HP.SEED_DATA_END:
        print(f"[eval] WARNING: seed {args.seed} is in the data-generation range (< {HP.SEED_DATA_END}); use the test "
              f"seeds {HP.SEED_TEST} or validation {HP.SEED_VAL_START}+")
    scene = Scene()
    video = VideoWriter(args.save_video) if args.save_video else None  # its camera must exist before the timeline plays
    robot = Robot(scene)
    warm_textures(robot)
    if not args.headless:  # GUI demo: look at the robot and the table from the front right
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=[1.45, -0.95, 0.95], target=[0.35, 0.0, 0.15])
        except Exception as e:  # cosmetic only
            print(f"[eval] viewport camera not set: {e!r}")
    sock = socket.create_connection((HOST, args.port), timeout=300)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    send_msg(sock, {"cmd": "info"})
    print(f"[eval] connected to policy server: {recv_msg(sock)}")
    results = None
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        results = open(args.results_json, "a", encoding="utf-8")
    steps = int(round(args.episode_seconds * args.control_hz))
    gi = {"L": 7, "R": 15}
    smoother = None
    if args.smooth:
        from action_smoother import ActionSmoother

        smoother = ActionSmoother(list(range(7)) + list(range(8, 15)), np.concatenate([W.vmax, W.vmax]),
                                  1.0 / args.control_hz, accel_scale=args.accel_scale, blend_steps=args.blend_steps,
                                  filter_window=args.filter_window)
        print(f"[eval] anti-jitter on: crossfade {args.blend_steps} steps, acceleration limit {args.accel_scale} x v_max, "
              f"in-chunk filter window {args.filter_window}")
    tries = wins = 0
    t_all = time.time()
    while simulation_app.is_running() and (not args.episodes or tries < args.episodes):
        seed = args.seed + tries
        rng = np.random.default_rng(seed)
        t0 = time.time()
        ep, rej = new_episode(robot, rng, seed)
        setup_episode(robot, ep)
        send_msg(sock, {"cmd": "reset", "seed": seed})
        recv_msg(sock)
        st = robot.read_state()
        tgt = {s: [st[gi[s] - 7:gi[s]].astype(np.float64), float(st[gi[s]])] for s in HP.SIDES}
        k = ep.slots[ep.target]
        z0 = robot.part_pose(k)[0][2]
        step, infer_ms, server_ms = 0, [], []
        max_lift, first_in_cell, arm_dmin = 0.0, None, np.inf
        grip_min = {s: 1.0 for s in HP.SIDES}  # smallest gripper command per arm: which arm(s) tried to grasp
        q_hist = []  # executed joint targets (14), for the jitter metrics
        if smoother:
            smoother.reset(np.concatenate([tgt[s][0] for s in HP.SIDES]))
        if video:
            video.start(seed, ep)
        while step < steps and simulation_app.is_running():
            st = robot.read_state()
            imgs = robot.grab()
            if video:
                video.frame(imgs, step)
            t_inf = time.perf_counter()
            send_msg(sock, {"cmd": "infer", "state": st, "images": imgs, "cam_names": list(CAM_NAMES), "task": ep.task,
                            "exec_steps": args.exec_steps})
            reply = recv_msg(sock)
            infer_ms.append(1000.0 * (time.perf_counter() - t_inf))
            if "infer_ms" in reply:
                server_ms.append(float(reply["infer_ms"]))
            chunk = np.asarray(reply["actions"], np.float64)
            for s in HP.SIDES:  # URDF limits first, so neither the crossfade nor the limiter ever aims outside them
                chunk[:, gi[s] - 7:gi[s]] = np.clip(chunk[:, gi[s] - 7:gi[s]], W.lo, W.hi)
            seq = smoother.new_chunk(chunk, args.exec_steps) if smoother else chunk[: args.exec_steps]
            for a in seq:
                if smoother:
                    a = smoother.step(a)
                for s in HP.SIDES:
                    q_cmd = np.clip(a[gi[s] - 7:gi[s]], W.lo, W.hi)
                    tgt[s][0] = tgt[s][0] + np.clip(q_cmd - tgt[s][0], -args.max_step_rad, args.max_step_rad)
                    tgt[s][1] = float(np.clip(a[gi[s]], 0.0, 1.0))
                    grip_min[s] = min(grip_min[s], tgt[s][1])
                robot.apply({s: (tgt[s][0], tgt[s][1]) for s in HP.SIDES})
                q_hist.append(np.concatenate([tgt[s][0] for s in HP.SIDES]))
                step += 1
                if video and step % args.exec_steps:  # the chunk's first frame is written above, with the infer
                    video.frame(robot.grab(), step)
                pos = robot.part_pose(k)[0]
                max_lift = max(max_lift, float(pos[2] - z0))
                if first_in_cell is None and W.cell_of(pos[:2], ep.pallet_c, ep.pallet_yaw) is not None:
                    first_in_cell = step
                if step % 5 == 0:
                    arm_dmin = min(arm_dmin, float(W.arm_distance("L", robot.q("L")[robot.ai], "R",
                                                                    robot.q("R")[robot.ai]).min()))
                if step >= steps or not simulation_app.is_running():
                    break
        for _ in range(20):  # settle with the last targets (as after the expert's episode)
            robot.apply({s: (tgt[s][0], tgt[s][1]) for s in HP.SIDES})
        ok, chk = check_success(robot, ep)
        jit = jitter_metrics(np.asarray(q_hist), args.control_hz)
        if video:
            video.finish(robot.grab(), step, ok, chk)
        tries += 1
        wins += ok
        print(f"[eval] episode {tries - 1} seed {seed}: {ep.task!r} expert arm {ep.plan['side']} -> "
              f"{'SUCCESS' if ok else 'fail'} {json.dumps(chk)} lift {max_lift:.3f} m ({time.time() - t0:.1f}s) "
              f"[{wins}/{tries}]")
        if results:
            results.write(json.dumps({
                "episode": tries - 1, "seed": seed, "success": bool(ok), "steps": step,
                "seconds": round(time.time() - t0, 1), "exec_steps": args.exec_steps,
                "episode_seconds": args.episode_seconds, "smooth": bool(args.smooth),
                "blend_steps": args.blend_steps if args.smooth else None,
                "accel_scale": args.accel_scale if args.smooth else None,
                "filter_window": args.filter_window if args.smooth else None, "max_lift_m": round(max_lift, 4),
                "lifted": max_lift > 0.03, "first_in_cell_step": first_in_cell,
                "grip_min_cmd": {s: round(v, 3) for s, v in grip_min.items()},
                "arm_distance_min_m": round(arm_dmin, 4) if np.isfinite(arm_dmin) else None,
                "n_infer": len(infer_ms),
                "infer_ms_mean": round(float(np.mean(infer_ms)), 2) if infer_ms else None,
                "infer_ms_max": round(float(np.max(infer_ms)), 2) if infer_ms else None,
                "server_infer_ms_mean": round(float(np.mean(server_ms)), 2) if server_ms else None, **jit,
                **episode_meta(ep, {**chk, "rejected_layouts": rej})}) + "\n")
            results.flush()
    print(f"[eval] done: {wins}/{tries} successful, {time.time() - t_all:.0f}s")
    sock.close()
    if results:
        results.close()


def main():
    if args.fov_probe:
        fov_probe(args.fov_probe)
    elif args.probe:
        probe()
    elif args.expert:
        expert()
    elif args.port:
        evaluate()
    else:
        sys.exit("choose --expert, --probe, --fov-probe or --port (policy evaluation)")
    simulation_app.close()


if __name__ == "__main__":
    main()
