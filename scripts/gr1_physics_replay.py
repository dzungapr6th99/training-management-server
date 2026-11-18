"""Physics replay of a GR1 -> YellowArm+ORCA retarget (gr1_retarget.py output) in Isaac Sim 6, with the NVIDIA task.

Task (Isaac-PickPlace-GR1T2, rescaled by the retarget layout): a steering wheel (NVIDIA asset, scale 0.75 as in the
demo) lies on the table in front of the left hand; the left hand picks it up and hands it to the right hand, which
puts it down in the target area on the right. Success = NVIDIA's task_done_pick_place mapped into our frame: object
inside the mapped x/y target box, below the mapped max height, slower than 0.2 m/s, right wrist retracted.

The joint trajectory (48 joints, 25 Hz) is sent as position targets (arm: near-rigid drives, fingers: ORCA servo
model, see dual_orca_rig.py); contacts and the object are simulated, so the grasp can really fail.

  D:\\Isaac\\isaacsim\\python.bat scripts\\gr1_physics_replay.py --demo demo_0 [--no-headless] [--record DIR]
      [--save-frames DIR] [--live-dir DIR]
Writes a result line to logs/gr1_physics_replay.jsonl; with --record an episode_XXXX.npz (state/action 48-D,
camera1 head, camera2/3 wrists, task, fps, names, meta) in the same format as dual_orca_pick_sim.
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--demo", default="demo_0")
ap.add_argument("--traj", default=None, help="retarget npz (default datasets/external/gr1/retarget/<demo>.npz)")
ap.add_argument("--no-headless", action="store_true")
ap.add_argument("--physics-hz", type=float, default=100.0)
ap.add_argument("--control-hz", type=float, default=25.0)
ap.add_argument("--img-size", type=int, default=256)
ap.add_argument("--obj-scale", type=float, default=0.75)
ap.add_argument("--friction", type=float, default=1.0)
ap.add_argument("--record", default="", help="directory for episode_XXXX.npz (only successful replays)")
ap.add_argument("--save-frames", default="", help="directory for outside-view PNGs every 5 steps")
ap.add_argument("--live-dir", default="", help="live monitor status/images (dual_orca_monitor)")
ap.add_argument("--hold-s", type=float, default=1.0, help="extra seconds holding the last pose before the check")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.no_headless})

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
from mhh_paths import ROOT, dual_import_usd  # noqa: E402

E = Embodiment()
KIN = K.Kin()
NAMES = E.action_space("joint_full")
GR1 = ROOT / "datasets" / "external" / "gr1"
WHEEL_USD = GR1 / "assets" / "steering_wheel.usd"
CAM_NAMES = ("camera1", "camera2", "camera3")
TASK = "pick up the steering wheel, pass it to the right hand and put it in the target area"
# NVIDIA task_done_pick_place (GR1 frame): object x in (0.40, 0.85), y in (0.35, 0.60), z < 1.10, |v| < 0.2,
# right wrist (right_hand_roll_link) x < 0.26
GR1_SUCCESS = dict(min_x=0.40, max_x=0.85, min_y=0.35, max_y=0.60, max_z=1.10, min_vel=0.20, right_wrist_max_x=0.26)

traj = np.load(args.traj or GR1 / "retarget" / f"{args.demo}.npz")
q_traj = traj["q"].astype(np.float64)
assert [str(n) for n in traj["names"]] == NAMES, "retarget joint order != embodiment.yaml joint_full"
obj_traj = traj["obj_pos"].astype(np.float64)
objq0 = traj["obj_quat_wxyz"][0].astype(np.float64)
lay = json.loads(str(traj["layout"]))
Sv, Cg, Co = np.array(lay["scale_xyz"]), np.array(lay["C_gr1"]), np.array(lay["C_ours"])
mp = lambda p: Co + Sv * (np.asarray(p, float) - Cg)  # noqa: E731
lo_xy, hi_xy = mp([GR1_SUCCESS["min_x"], GR1_SUCCESS["min_y"], 0.0]), mp([GR1_SUCCESS["max_x"], GR1_SUCCESS["max_y"], 0.0])
max_z = mp([0, 0, GR1_SUCCESS["max_z"]])[2]
wrist_max_x = mp([GR1_SUCCESS["right_wrist_max_x"], 0, 0])[0]
WHEEL_BOTTOM = 0.0048 * args.obj_scale  # steering_wheel.usd bbox z min (m, unscaled) below its origin
TABLE_TOP = float(obj_traj[0, 2] - WHEEL_BOTTOM)
FLOOR_Z = min(-0.2, TABLE_TOP - 0.8)


def look_at_R(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), -f], 1)


def mat_to_quat(R):
    q = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


def make_camera(path, pos, quat, fov_deg, size):
    cam = RtxCamera(path, translations=np.array(pos, float), orientations=np.array(quat, float))
    c = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    a = c.GetHorizontalApertureAttr().Get()
    c.GetVerticalApertureAttr().Set(a)
    c.GetFocalLengthAttr().Set(float(a / (2 * np.tan(np.deg2rad(fov_deg) / 2))))
    c.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    return CameraSensor(cam, resolution=(size, size), annotators=["rgb"])


# ------------------------------------------------ scene ------------------------------------------------
stage_utils.create_new_stage()
SimulationManager.set_physics_dt(1.0 / args.physics_hz)
stage = omni.usd.get_context().get_stage()
DomeLight("/World/dome").set_intensities(1100.0)
DistantLight("/World/sun").set_intensities(1500.0)
GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, FLOOR_Z])
stage_utils.add_reference_to_stage(usd_path=str(dual_import_usd()), path="/World/robot")
root, _ = rig.apply_rig(stage, "/World/robot", E)

# table: covers the object's whole mapped path and the target area, front edge kept clear of the robot base
xs = np.r_[obj_traj[:, 0], lo_xy[0], hi_xy[0]]
ys = np.r_[obj_traj[:, 1], lo_xy[1], hi_xy[1]]
tx0, tx1 = xs.min() - 0.25, xs.max() + 0.25
ty0, ty1 = max(0.18, ys.min() - 0.20), ys.max() + 0.25
th = TABLE_TOP - FLOOR_Z
table = Cube("/World/table", sizes=1.0, translations=np.array([(tx0 + tx1) / 2, (ty0 + ty1) / 2, FLOOR_Z + th / 2]),
             scales=np.array([tx1 - tx0, ty1 - ty0, th]))
GeomPrim("/World/table", apply_collision_apis=True)
mt = OmniPbrMaterial("/World/Materials/table")
mt.set_input_values("diffuse_color_constant", [0.55, 0.45, 0.33])
table.apply_visual_materials(mt)
# target area marker (visual only, 2 mm proud of the table)
tgt = Cube("/World/target", sizes=1.0, translations=np.array([(lo_xy[0] + hi_xy[0]) / 2, (lo_xy[1] + hi_xy[1]) / 2,
                                                                 TABLE_TOP + 0.001]),
           scales=np.array([hi_xy[0] - lo_xy[0], hi_xy[1] - lo_xy[1], 0.002]))
mg = OmniPbrMaterial("/World/Materials/target")
mg.set_input_values("diffuse_color_constant", [0.15, 0.55, 0.2])
tgt.apply_visual_materials(mg)
ped = Cube("/World/pedestal", sizes=1.0, translations=np.array([0.0, 0.0, FLOOR_Z / 2]),
           scales=np.array([0.34, 0.34, -FLOOR_Z]))
GeomPrim("/World/pedestal", apply_collision_apis=True)

# steering wheel (rigid body is the Geometry/sm_steeringwheel_a01_01 child of the asset)
obj_root = "/World/object"
stage_utils.add_reference_to_stage(usd_path=str(WHEEL_USD), path=obj_root)
xf = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(obj_root))
xf.SetScale(Gf.Vec3f(args.obj_scale, args.obj_scale, args.obj_scale))
body_path = next(str(p.GetPath()) for p in stage.Traverse()
                 if p.GetPath().pathString.startswith(obj_root) and p.HasAPI(UsdPhysics.RigidBodyAPI))
pm = RigidBodyMaterial("/World/PhysicsMaterials/wheel")
pm.set_friction_coefficients(static_frictions=[args.friction], dynamic_frictions=[args.friction])
pm.set_combine_modes(frictions="max")
for p in stage.Traverse():
    if p.GetPath().pathString.startswith(obj_root) and p.HasAPI(UsdPhysics.CollisionAPI):
        GeomPrim(str(p.GetPath())).apply_physics_materials(pm)

# cameras: head above the shoulders looking at the table (robot faces +Y), wrists on the ORCA towers
head = np.array([0.0, 0.18, 1.50])
look = np.array([obj_traj[:, 0].mean(), obj_traj[:, 1].mean(), TABLE_TOP])
cams = {"camera1": make_camera("/World/head_cam", head, mat_to_quat(look_at_R(head, look)), 70.0, args.img_size)}
W0 = KIN.fk({})
for s, cn in (("left", "camera2"), ("right", "camera3")):
    tower = next(p for p in stage.Traverse()
                 if p.GetPath().pathString.startswith("/World/robot") and p.GetName() == f"{s}_tower")
    c0, R0 = KIN.palm(s, {}, W0)
    fdir, pn = R0[:, 0], R0[:, 1]
    side_ax = R0[:, 2] * (-1.0 if s == "right" else 1.0)
    eye = c0 - 0.12 * fdir + 0.10 * side_ax - 0.03 * pn
    Mw = np.eye(4)
    Mw[:3, :3], Mw[:3, 3] = look_at_R(eye, c0 + 0.03 * fdir + 0.02 * pn, -pn), eye
    Ml = np.linalg.inv(W0[f"{s}_tower"]) @ Mw
    cams[cn] = make_camera(f"{tower.GetPath()}/cam_wrist", Ml[:3, 3], mat_to_quat(Ml[:3, :3]), 80.0, args.img_size)
if args.save_frames or args.live_dir:
    eye = np.array([look[0] + 0.2, look[1] + 1.7, 1.55])
    cams["dbg"] = make_camera("/World/dbg_cam", eye, mat_to_quat(look_at_R(eye, look + [0, -0.25, 0.15])), 55.0, 480)

# ------------------------------------------------ run ------------------------------------------------
art = Articulation(root)
wheel = RigidPrim(body_path)
tl = omni.timeline.get_timeline_interface()
tl.play()
simulation_app.update()
dof = list(art.dof_names)
order = [dof.index(n) for n in NAMES]
dec = max(1, round(args.physics_hz / args.control_hz))


def send(q48, teleport=False):
    full = np.zeros(len(dof))
    full[order] = q48
    if teleport:
        art.set_dof_positions(full)
        art.set_dof_velocities(np.zeros_like(full))
    art.set_dof_position_targets(full)


def q_now():
    return art.get_dof_positions().numpy()[0][order].astype(np.float64)


def wheel_pose():
    p, q = wheel.get_world_poses()
    v, _ = wheel.get_velocities()
    return p.numpy()[0].astype(np.float64), q.numpy()[0].astype(np.float64), v.numpy()[0].astype(np.float64)


def grab():
    out = []
    for n in CAM_NAMES:
        d, _ = cams[n].get_data("rgb")
        img = np.zeros((args.img_size, args.img_size, 3), np.uint8) if d is None else d.numpy()[..., :3]
        out.append(np.ascontiguousarray(img.astype(np.uint8)))
    return out


live = None
if args.live_dir:
    from PIL import Image

    os.makedirs(args.live_dir, exist_ok=True)

    def live(step, extra):
        d, _ = cams["dbg"].get_data("rgb")
        if d is not None:
            ims = [d.numpy()[..., :3].astype(np.uint8)] + [np.asarray(Image.fromarray(i).resize((240, 240)))
                                                           for i in grab()]
            top = ims[0]
            row = np.concatenate([np.concatenate(ims[1:3], 1), np.concatenate([ims[3], np.zeros_like(ims[3])], 1)], 0)
            canvas = np.concatenate([top, row], 1) if top.shape[0] == row.shape[0] else top
            tmp = os.path.join(args.live_dir, "tmp_live.jpg")
            Image.fromarray(canvas).save(tmp, quality=80)
            os.replace(tmp, os.path.join(args.live_dir, "live.jpg"))
        st = {"time": time.time(), "phase": "gr1_replay", "demo": args.demo, "step": step, "steps": len(q_traj),
              "task": TASK, **extra}
        tmp = os.path.join(args.live_dir, "tmp_status.json")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(st, fh)
        os.replace(tmp, os.path.join(args.live_dir, "status.json"))

send(q_traj[0], teleport=True)
wheel.set_world_poses(positions=obj_traj[0:1] + [0, 0, 0.002], orientations=objq0[None])
wheel.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
for _ in range(20):  # settle with the start pose held
    send(q_traj[0])
    simulation_app.update()
p0, _, _ = wheel_pose()
print(f"[replay] {args.demo}: {len(q_traj)} steps, table top {TABLE_TOP:.3f}, wheel at {np.round(p0, 3)} "
      f"(planned {np.round(obj_traj[0], 3)}), target box x {lo_xy[0]:.2f}..{hi_xy[0]:.2f} y {lo_xy[1]:.2f}..{hi_xy[1]:.2f}",
      flush=True)
rec = {"state": [], "action": [], **{n: [] for n in CAM_NAMES}} if args.record else None
track_err, max_lift, t0 = [], 0.0, time.time()
for i, q in enumerate(q_traj):
    if rec is not None:
        rec["state"].append(q_now().astype(np.float32))
        rec["action"].append(q.astype(np.float32))
        for n, im in zip(CAM_NAMES, grab()):
            rec[n].append(im)
    send(q)
    for _ in range(dec):
        simulation_app.update()
    p, _, _ = wheel_pose()
    max_lift = max(max_lift, float(p[2] - p0[2]))
    track_err.append(float(np.linalg.norm(p - obj_traj[i])))
    if args.save_frames and i % 5 == 0:
        from PIL import Image

        os.makedirs(args.save_frames, exist_ok=True)
        d, _ = cams["dbg"].get_data("rgb")
        if d is not None:
            Image.fromarray(d.numpy()[..., :3].astype(np.uint8)).save(os.path.join(args.save_frames, f"f{i:04d}.png"))
    if live and i % 3 == 0:
        live(i, {"lift": round(max_lift, 3), "obj_track_err": round(track_err[-1], 3)})
for _ in range(int(args.hold_s * args.control_hz)):
    send(q_traj[-1])
    for _ in range(dec):
        simulation_app.update()
p, _, v = wheel_pose()
W = KIN.fk(dict(zip(NAMES, q_now())))
wrist_x = float(W["right_yellowarm_link7"][0, 3])
chk = {"in_x": bool(lo_xy[0] < p[0] < hi_xy[0]), "in_y": bool(lo_xy[1] < p[1] < hi_xy[1]), "low": bool(p[2] < max_z),
       "still": bool(np.all(np.abs(v[:2]) < GR1_SUCCESS["min_vel"])), "wrist_back": bool(wrist_x < wrist_max_x)}
ok = all(chk.values())
res = {"demo": args.demo, "success": ok, "checks": chk, "obj_final": np.round(p, 3).tolist(),
       "obj_planned_final": np.round(obj_traj[-1], 3).tolist(), "max_lift_m": round(max_lift, 3),
       "obj_track_err_mean_m": round(float(np.mean(track_err)), 3), "obj_track_err_max_m": round(float(np.max(track_err)), 3),
       "wall_s": round(time.time() - t0, 1), "traj": str(args.traj or f"retarget/{args.demo}.npz")}
print("[replay] result", json.dumps(res), flush=True)
with open(ROOT / "logs" / "gr1_physics_replay.jsonl", "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), **res}) + "\n")
if live:
    live(len(q_traj), {"result": ok, **chk})
if rec is not None and ok:
    os.makedirs(args.record, exist_ok=True)
    k = len([f for f in os.listdir(args.record) if f.startswith("episode_") and f.endswith(".npz")])
    tmp = os.path.join(args.record, f"tmp_episode_{k:04d}.npz")
    np.savez_compressed(tmp, state=np.array(rec["state"]), action=np.array(rec["action"]),
                        **{n: np.array(rec[n]) for n in CAM_NAMES}, task=TASK, fps=args.control_hz,
                        names=np.array(NAMES), meta=json.dumps({"source": "gr1_retarget", **res}))
    os.replace(tmp, os.path.join(args.record, f"episode_{k:04d}.npz"))
    print("[replay] recorded", os.path.join(args.record, f"episode_{k:04d}.npz"))
tl.stop()
simulation_app.close()
