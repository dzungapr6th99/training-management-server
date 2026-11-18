"""Side-by-side kinematic replay: NVIDIA GR1T2 demo (left) vs its retarget on YellowArm + ORCA (right), Isaac Sim 6.

Joint angles are written directly every frame (gravity off, no contacts): this checks how the MOTION looks, not
whether the grasp works (that is the physics replay). Object = orange sphere at the (mapped) object centre, table =
grey slab under the object's start height.

  D:\\Isaac\\isaacsim\\python.bat scripts\\gr1_replay_view.py --demo demo_0 [--no-headless]
Frames -> logs/gr1_view/<demo>/f%04d.png, then (env python) scripts/gr1_replay_view.py --encode --demo demo_0
-> logs/gr1_view/<demo>.mp4 (3 x 2 grid: front view / side view / left-hand close-up, GR1 | YellowArm).
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--demo", default="demo_0")
ap.add_argument("--no-headless", action="store_true")
ap.add_argument("--img", type=int, default=400)
ap.add_argument("--every", type=int, default=1, help="render every n-th frame")
ap.add_argument("--encode", action="store_true", help="(env python) encode the PNG frames to mp4")
args, _ = ap.parse_known_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mhh_paths import ROOT  # noqa: E402

OUTD = ROOT / "logs" / "gr1_view" / args.demo
GR1 = ROOT / "datasets" / "external" / "gr1"
GR1_X = -1.6  # GR1 stands 1.6 m to the left of YellowArm (both face +Y)

if args.encode:
    import av
    import numpy as np
    from PIL import Image

    files = sorted(OUTD.glob("f*.png"))
    out = OUTD.parent / f"{args.demo}.mp4"
    c = av.open(str(out), "w")
    st = c.add_stream("libx264", rate=25 // args.every)
    im0 = np.asarray(Image.open(files[0]))
    st.width, st.height, st.pix_fmt = im0.shape[1], im0.shape[0], "yuv420p"
    st.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}
    for f in files:
        for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(Image.open(f))[..., :3].copy(), format="rgb24")):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()
    print("[view] ->", out, len(files), "frames")
    sys.exit(0)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": not args.no_headless})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import Cube, DistantLight, DomeLight, GroundPlane, Sphere  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, GeomPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

import dual_orca_rig as rig  # noqa: E402
from embodiment import Embodiment  # noqa: E402
from gr1_retarget import h5  # noqa: E402
from mhh_paths import dual_import_usd  # noqa: E402

E = Embodiment()
r = np.load(GR1 / "retarget" / f"{args.demo}.npz")
q_ours, t_ours, obj_ours = r["q"], r["t"], r["obj_pos"]
names_ours = [str(n) for n in r["names"]]
h5py = h5()
f = h5py.File(GR1 / "dataset_annotated_gr1.hdf5", "r")
d = f["data"][args.demo]
q_gr1 = d["states/articulation/robot/joint_position"][()]
root_gr1 = d["states/articulation/robot/root_pose"][()]
obj_gr1 = d["obs/object_pos"][()]
dt_gr1 = 6 / 120.0

stage_utils.create_new_stage()
SimulationManager.set_physics_dt(0.01)
stage = omni.usd.get_context().get_stage()
UsdPhysics.Scene.Define(stage, "/World/physicsScene").CreateGravityMagnitudeAttr(0.0)
DomeLight("/World/dome").set_intensities(1200.0)
DistantLight("/World/sun").set_intensities(1800.0)
GroundPlane("/World/ground", sizes=20.0, positions=[0.0, 0.0, -0.6])
stage_utils.add_reference_to_stage(usd_path=str(GR1 / "usd" / "GR1T2_fourier_hand_6dof.usd"), path="/World/gr1")
stage_utils.add_reference_to_stage(usd_path=str(dual_import_usd()), path="/World/robot")
root_ours, _ = rig.apply_rig(stage, "/World/robot", E)
root_g = next(str(p.GetPath()) for p in stage.Traverse()
              if p.GetPath().pathString.startswith("/World/gr1") and p.HasAPI(UsdPhysics.ArticulationRootAPI))


def slab(path, centre, size, color):
    c = Cube(path, sizes=1.0, positions=np.array(centre, float), scales=np.array(size, float))
    UsdGeom.Gprim(stage.GetPrimAtPath(path)).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return c


def ball(path, r_, color):
    s = Sphere(path, radii=r_)
    UsdGeom.Gprim(stage.GetPrimAtPath(path)).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return GeomPrim(path)


off_g = np.array([GR1_X, 0.0, 0.0])
slab("/World/table_gr1", obj_gr1[0] + off_g + [0.25, 0.0, -0.035], [1.0, 0.6, 0.03], (0.55, 0.55, 0.58))
slab("/World/table_ours", obj_ours[0] + [0.35, 0.0, -0.035 * 1.5], [1.4, 0.9, 0.04], (0.55, 0.55, 0.58))
WHEEL = GR1 / "assets" / "steering_wheel.usd"


class Wheel:
    """Visual-only steering wheel (scale 0.75 as in the NVIDIA task): rigid body and collisions disabled, pose set
    directly every frame."""

    def __init__(self, path):
        stage_utils.add_reference_to_stage(usd_path=str(WHEEL), path=path)
        for p in stage.Traverse():
            if not p.GetPath().pathString.startswith(path):
                continue
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr(False)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
        x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        x.ClearXformOpOrder()
        self.t = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        self.o = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.75, 0.75, 0.75))

    def set(self, pos, q_wxyz):
        self.t.Set(Gf.Vec3d(*[float(v) for v in pos]))
        self.o.Set(Gf.Quatd(float(q_wxyz[0]), Gf.Vec3d(*[float(v) for v in q_wxyz[1:]])))


wheel_g, wheel_o = Wheel("/World/wheel_gr1"), Wheel("/World/wheel_ours")
objq_gr1 = d["obs/object_rot"][()]
objq_ours = r["obj_quat_wxyz"]


def look_R(eye, target, up=(0, 0, 1)):
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    rr = np.cross(fwd, up)
    rr /= np.linalg.norm(rr)
    return np.stack([rr, np.cross(rr, fwd), -fwd], 1)


def cam(path, eye, target, fov=55.0):
    R = look_R(np.array(eye, float), target)
    qd = Gf.Matrix3d(*R.T.flatten().tolist()).ExtractRotation().GetQuat()
    c = RtxCamera(path, translations=np.array(eye, float), orientations=np.array([qd.GetReal(), *qd.GetImaginary()]))
    uc = UsdGeom.Camera(stage.GetPrimAtPath(path))
    a = uc.GetHorizontalApertureAttr().Get()
    uc.GetVerticalApertureAttr().Set(a)
    uc.GetFocalLengthAttr().Set(float(a / (2 * np.tan(np.deg2rad(fov) / 2))))
    uc.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))  # default near plane (~1 m) clips the close-ups
    return CameraSensor(c, resolution=(args.img, args.img), annotators=["rgb"])


# same relative views of both robots (GR1 coordinates scaled by the retarget layout scale)
sc = np.array(json.loads(str(r["layout"]))["scale_xyz"])
C_g, C_o = np.array(json.loads(str(r["layout"]))["C_gr1"]), np.array(json.loads(str(r["layout"]))["C_ours"])
views = {"front": (np.array([-1.0, 1.6, 0.35]), np.array([-0.1, 0.35, -0.25])),  # relative to shoulder centre
         "side": (np.array([-1.7, 0.30, 0.10]), np.array([0.0, 0.30, -0.25]))}
cams = {}
for v, (e, t) in views.items():
    cams[f"gr1_{v}"] = cam(f"/World/cam_gr1_{v}", C_g + off_g + e / sc, C_g + off_g + t / sc)
    cams[f"ours_{v}"] = cam(f"/World/cam_ours_{v}", C_o + e, C_o + t)

# close-up of the left hand: 0.45 m (GR1: / scale) in front-left-above of the knuckles, following the hand
import gr1_retarget as GRT  # noqa: E402

_o = GRT.Ours()
_names = json.loads((GR1 / "gr1_names.json").read_text(encoding="utf-8"))
_gd = GRT.GR1Demo(d, _names)


def hand_g(a, u):
    return (1 - u) * _gd.frame("left", a)[0] + u * _gd.frame("left", a + 1)[0] + off_g


def hand_o(q48):
    return _o.frame("left", _o.k.fk(dict(zip(names_ours, q48))))[0]


# close-up of the left hand at the grasp (first frame where GR1 lifts the wheel by 1 cm), fixed cameras
lift = obj_gr1[:, 2] - obj_gr1[0, 2]
kg = int(np.argmax(lift > 0.01))
ko = int(np.argmin(np.abs(t_ours - kg * dt_gr1)))
Bg_grasp, Bo_grasp = hand_g(kg, 0.0), hand_o(q_ours[ko])
off = np.array([-0.28, 0.22, 0.20])
cams["gr1_close"] = cam("/World/cam_gr1_close", Bg_grasp + off / sc.mean(), Bg_grasp, 50.0)
cams["ours_close"] = cam("/World/cam_ours_close", Bo_grasp + off, Bo_grasp, 50.0)

art_g, art_o = Articulation(root_g), Articulation(root_ours)
tl = omni.timeline.get_timeline_interface()
tl.play()
app.update()
dof_o = list(art_o.dof_names)
order_o = [dof_o.index(n) for n in names_ours]
OUTD.mkdir(parents=True, exist_ok=True)
for old in OUTD.glob("f*.png"):
    old.unlink()


def put(art, q):
    q = np.asarray(q, np.float32)[None]
    art.set_dof_positions(q)
    art.set_dof_velocities(np.zeros_like(q))
    art.set_dof_position_targets(q)


for i in range(0, len(t_ours), args.every):
    tt = float(t_ours[i])
    a = min(int(tt / dt_gr1), len(q_gr1) - 2)
    u = tt / dt_gr1 - a
    qg = (1 - u) * q_gr1[a] + u * q_gr1[a + 1]
    rp = root_gr1[a].copy()
    art_g.set_world_poses(positions=(rp[:3] + off_g)[None], orientations=rp[3:7][None])
    put(art_g, qg)
    full = np.zeros(len(dof_o), np.float32)
    full[order_o] = q_ours[i]
    put(art_o, full)
    og = (1 - u) * obj_gr1[a] + u * obj_gr1[a + 1]
    wheel_g.set(og + off_g, objq_gr1[a])
    wheel_o.set(obj_ours[i], objq_ours[i])
    for _ in range(2):
        app.update()
    imgs = {}
    for k, c in cams.items():
        dd, _ = c.get_data("rgb")
        imgs[k] = dd.numpy()[..., :3].astype(np.uint8) if dd is not None else np.zeros((args.img, args.img, 3), np.uint8)
    grid = np.concatenate([np.concatenate([imgs["gr1_front"], imgs["ours_front"]], 1),
                           np.concatenate([imgs["gr1_side"], imgs["ours_side"]], 1),
                           np.concatenate([imgs["gr1_close"], imgs["ours_close"]], 1)], 0)
    Image.fromarray(grid).save(OUTD / f"f{i:04d}.png")
    if i % 25 == 0:
        print(f"[view] {args.demo} frame {i}/{len(t_ours)}", flush=True)
print("[view] frames ->", OUTD)
tl.stop()
app.close()
