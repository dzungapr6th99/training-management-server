"""Bring-up probe for the dual YellowArm+ORCA robot (robots/yellowarm_orca_dual) in Isaac Sim 6.

  --import   convert YellowArm_Two_ORCA.urdf -> USD (robots/yellowarm_orca_dual/usd/), fixed base, then continue
  (default)  load that USD, apply dual_orca_rig (PhysX self-collision off, real drive limits), check dof order
             against embodiment.yaml, then
             A) gravity OFF: drive every joint lower -> home -> upper -> home (left/right counterpart together) and
                report tracking error: checks joint mapping, limits, drives, contacts -- not motor strength;
             B) gravity ON: hold the home pose, report the drift (motor strength at rest);
             and save pictures of the robot from the front and the back.

Run with run_dual_probe.bat (Isaac python). Output: logs/dual_probe/report.json + view_*.png
"""
import argparse
import json
import sys

from mhh_paths import DUAL_URDF, DUAL_USD_DIR, ROOT, dual_import_usd

ap = argparse.ArgumentParser()
ap.add_argument("--import", dest="do_import", action="store_true")
ap.add_argument("--no-headless", dest="headless", action="store_false")
ap.add_argument("--max-updates", type=int, default=1500, help="max app updates per target (stops earlier once settled)")
ap.add_argument("--merge-fixed", action="store_true")
ap.add_argument("--self-collision", action="store_true", help="enable PhysX self-collision (P1.3); off by default")
ap.add_argument("--only", default="", help="comma-separated substrings: test only matching joints")
args, _ = ap.parse_known_args()
sys.stdout.reconfigure(line_buffering=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import DistantLight, DomeLight, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from pxr import Gf, UsdPhysics  # noqa: E402

import dual_orca_rig as rig  # noqa: E402
from embodiment import Embodiment  # noqa: E402

E = Embodiment()
NAMES = E.action_space("joint_full")
OUT = ROOT / "logs" / "dual_probe"
OUT.mkdir(parents=True, exist_ok=True)
TOL = 0.05  # rad

if args.do_import:
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    DUAL_USD_DIR.mkdir(parents=True, exist_ok=True)
    cfg = URDFImporterConfig(urdf_path=str(DUAL_URDF), usd_path=str(DUAL_USD_DIR), fix_base=True,
                             merge_fixed_joints=args.merge_fixed, collision_from_visuals=False,
                             allow_self_collision=False, joint_target_type="position")
    usd_file = URDFImporter(cfg).import_urdf()
    print("[import] USD:", usd_file)
else:
    usd_file = dual_import_usd()
    if usd_file is None:
        sys.exit("no USD yet: run with --import first")
usd_file = str(usd_file)
print("[probe] USD:", usd_file)

stage_utils.create_new_stage()
SimulationManager.set_physics_dt(0.01)
stage = omni.usd.get_context().get_stage()
GroundPlane("/World/ground", positions=[0.0, 0.0, -0.5])  # fingertips hang 2 cm above z=0
DomeLight("/World/dome").set_intensities(1200.0)
DistantLight("/World/sun").set_intensities(2000.0)
stage_utils.add_reference_to_stage(usd_path=usd_file, path="/World/robot")
root, nd = rig.apply_rig(stage, "/World/robot", E, self_collision=args.self_collision)
print("[probe] articulation root:", root, "| drives set on", nd, "joints | self-collision:", args.self_collision)


def physics_scene():
    p = next((p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
    return UsdPhysics.Scene(p) if p is not None else UsdPhysics.Scene.Define(stage, "/World/physicsScene")


def set_gravity(on):
    physics_scene().CreateGravityMagnitudeAttr(9.81 if on else 0.0)
    physics_scene().CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))


set_gravity(False)
art = Articulation(root)
tl = omni.timeline.get_timeline_interface()
tl.play()
app.update()
dof = list(art.dof_names)
missing = [n for n in NAMES if n not in dof]
extra = [n for n in dof if n not in NAMES]
print("[probe] dof count:", len(dof), "| missing in USD:", missing, "| extra in USD:", extra)
if missing:
    sys.exit("joint names do not match embodiment.yaml")
order = [dof.index(n) for n in NAMES]  # dof index of each embodiment joint


def step(n):
    for _ in range(n):
        app.update()


def settle(tgt, ii, max_updates):
    """Step until joints ii are within TOL/2 of tgt and stopped moving (or max_updates). Returns updates used."""
    prev, k = q_emb(), 0
    while k < max_updates:
        step(10)
        k += 10
        q = q_emb()
        if np.abs(q[ii] - tgt[ii]).max() < TOL / 2 and np.abs(q - prev).max() < 1e-4:
            break
        prev = q
    return k


def q_emb():
    return art.get_dof_positions().numpy()[0][order].astype(np.float64)


def drive(q):
    full = np.zeros(len(dof))
    full[order] = q
    art.set_dof_position_targets(full)


def teleport(q):
    full = np.zeros(len(dof))
    full[order] = q
    art.set_dof_positions(full)
    art.set_dof_velocities(np.zeros_like(full))
    art.set_dof_position_targets(full)


home = E.home(NAMES)
lo, hi = E.lower_upper(NAMES)
teleport(home)
step(30)
base0 = art.get_world_poses()[0].numpy()[0]
report = {"usd": usd_file, "dof_names": dof, "missing": missing, "extra": extra, "home": home.tolist(),
          "self_collision": args.self_collision, "max_updates_per_target": args.max_updates, "tol_rad": TOL, "joints": {}}

# ---------- A) gravity off: limits / mapping / drives ----------
pairs = [(n, n.replace("left_", "right_", 1)) for n in NAMES if n.startswith("left_")]
if args.only:
    keys = args.only.split(",")
    pairs = [p for p in pairs if any(k in p[0] for k in keys)]
t0 = __import__("time").time()
for pair in pairs:
    ii = [NAMES.index(n) for n in pair]
    res = {n: {} for n in pair}
    for tag in ("lower", "upper"):
        tgt = home.copy()
        for i in ii:
            tgt[i] = lo[i] + 0.02 if tag == "lower" else hi[i] - 0.02
        drive(tgt)
        used = settle(tgt, ii, args.max_updates)
        q = q_emb()
        for n, i in zip(pair, ii):
            res[n][tag] = {"target": float(tgt[i]), "reached": float(q[i]), "err": float(abs(q[i] - tgt[i])),
                           "updates": used}
        others = [k for k in range(len(NAMES)) if k not in ii]
        drive(home)
        settle(home, ii, args.max_updates)
        q = q_emb()
        for n in pair:
            res[n][f"after_{tag}_others_err_max"] = float(np.abs(q[others] - home[others]).max())
        for n, i in zip(pair, ii):
            res[n][f"back_home_after_{tag}_err"] = float(abs(q[i] - home[i]))
    for n in pair:
        r = res[n]
        r["ok"] = bool(max(r["lower"]["err"], r["upper"]["err"], r["back_home_after_lower_err"],
                           r["back_home_after_upper_err"]) < TOL)
        report["joints"][n] = r
        print(f"[joint] {'OK ' if r['ok'] else 'BAD'} {n:24s} lo {r['lower']['target']:+.3f}->{r['lower']['reached']:+.3f}"
              f"  hi {r['upper']['target']:+.3f}->{r['upper']['reached']:+.3f}"
              f"  back {max(r['back_home_after_lower_err'], r['back_home_after_upper_err']):.3f}"
              f"  upd {r['lower']['updates']}/{r['upper']['updates']}")
print(f"[probe] limit test {__import__('time').time() - t0:.0f}s")
report["bad_tracking"] = [n for n, r in report["joints"].items() if not r["ok"]]
print("[probe] joints that fail the no-gravity limit test:", report["bad_tracking"])

# ---------- B) gravity on: hold home ----------
teleport(home)
set_gravity(True)
step(300)
q = q_emb()
err = np.abs(q - home)
report["gravity_hold_home"] = {n: float(e) for n, e in zip(NAMES, err)}
worst = int(np.argmax(err))
print(f"[probe] gravity hold home: max err {err.max():.4f} rad at {NAMES[worst]}")
base1 = art.get_world_poses()[0].numpy()[0]
report["base_moved_m"] = float(np.linalg.norm(base1 - base0))
print("[probe] base moved (must be 0):", report["base_moved_m"])
report["nan"] = bool(not np.all(np.isfinite(q)))

# ---------- pictures ----------
def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    qq = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([qq.GetReal(), *qq.GetImaginary()])


views = {"view_pos_y": (0.0, 2.2, 1.0), "view_neg_y": (0.0, -2.2, 1.0), "view_side": (2.2, 0.6, 1.1)}
sensors = {}
for k, eye in views.items():
    cam = RtxCamera(f"/World/{k}", translations=np.array(eye), orientations=look_at_quat(eye, (0.0, 0.0, 0.55)))
    sensors[k] = CameraSensor(cam, resolution=(800, 800), annotators=["rgb"])
step(20)
from PIL import Image  # noqa: E402

for k, s in sensors.items():
    d, _ = s.get_data("rgb")
    if d is not None:
        Image.fromarray(d.numpy()[..., :3].astype(np.uint8)).save(OUT / f"{k}.png")
json.dump(report, open(OUT / "report.json", "w"), indent=1)
print("[probe] report:", OUT / "report.json")
app.close()
