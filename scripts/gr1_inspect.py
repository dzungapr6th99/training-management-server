"""Identify the joint / link order of the NVIDIA Isaac Lab Mimic GR1T2 dataset and check it against the GR1 USD.

The Mimic dataset (datasets/external/gr1/dataset_annotated_gr1.hdf5, Isaac-PickPlace-GR1T2-Abs-Mimic-v0) stores
robot joint positions (54) and every link state (55 x 13) but no names. Here the GR1 USD is loaded, the recorded joint
positions are written into it, and the simulated link positions are compared with the recorded ones. If they match
(mm level), dof_names / link_names are the dataset order.

  D:\\Isaac\\isaacsim\\python.bat scripts\\gr1_inspect.py [--headless]
Output: datasets/external/gr1/gr1_names.json
"""
import argparse
import json
import sys

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true", default=True)
ap.add_argument("--demo", default="demo_0")
args, _ = ap.parse_known_args()
app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from mhh_paths import ISAAC_SIM_DIR, ROOT  # noqa: E402

# h5py ships inside an Isaac Sim extension: use it from there instead of pip-installing into Isaac python
sys.path.insert(0, str(ISAAC_SIM_DIR / "exts" / "isaacsim.replicator.episode_recorder" / "pip_prebundle"))
import h5py  # noqa: E402

GR1 = ROOT / "datasets" / "external" / "gr1"
USD = GR1 / "usd" / "GR1T2_fourier_hand_6dof.usd"

f = h5py.File(GR1 / "dataset_annotated_gr1.hdf5", "r")
d = f["data"][args.demo]
q_rec = d["states/articulation/robot/joint_position"][()]
links_rec = d["obs/robot_links_state"][()]
root_rec = d["states/articulation/robot/root_pose"][()]

stage_utils.create_new_stage()
SimulationManager.set_physics_dt(0.01)
stage = omni.usd.get_context().get_stage()
scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
scene.CreateGravityMagnitudeAttr(0.0)
stage_utils.add_reference_to_stage(usd_path=str(USD), path="/World/gr1")
root = next(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI))
print("[gr1] articulation root:", root)
art = Articulation(root)
tl = omni.timeline.get_timeline_interface()
tl.play()
app.update()
dof, links = list(art.dof_names), list(art.link_names)
print(f"[gr1] {len(dof)} dofs, {len(links)} links (dataset: {q_rec.shape[1]} joints, {links_rec.shape[1]} links)")
view = art._physics_articulation_view  # tensor view: link transforms (pos + quat xyzw), world frame

errs = []
for t in np.linspace(1, len(q_rec) - 1, 6).astype(int):  # frame 0 = before the env reset settled
    art.set_world_poses(positions=root_rec[t:t + 1, :3], orientations=root_rec[t:t + 1, 3:7])
    art.set_dof_positions(q_rec[t:t + 1])
    art.set_dof_velocities(np.zeros_like(q_rec[t:t + 1]))
    art.set_dof_position_targets(q_rec[t:t + 1])
    app.update()
    tr = view.get_link_transforms()
    tr = tr.numpy() if hasattr(tr, "numpy") else np.asarray(tr)
    sim = tr.reshape(-1, len(links), 7)[0, :, :3]
    rec = links_rec[t, :, :3]
    e = np.linalg.norm(sim - rec, axis=1)
    errs.append(e)
    print(f"[gr1] frame {t}: link position error mean {1000 * e.mean():.1f} mm, max {1000 * e.max():.1f} mm "
          f"({links[int(e.argmax())]})")
errs = np.array(errs)
hand = np.array([n[:2] in ("L_", "R_") for n in links])
arm_err = errs[:, ~hand]
print(f"[gr1] body/arm links: mean {1000 * arm_err.mean():.1f} mm, max {1000 * arm_err.max():.1f} mm | "
      f"finger links: mean {1000 * errs[:, hand].mean():.1f} mm (fingers have mimic joints, settle after 1 update)")
out = {"dataset": "dataset_annotated_gr1.hdf5", "usd": str(USD.relative_to(ROOT)), "dof_names": dof,
       "link_names": links, "check_demo": args.demo, "link_err_mean_mm": round(1000 * float(errs.mean()), 2),
       "link_err_max_mm": round(1000 * float(errs.max()), 2), "arm_err_max_mm": round(1000 * float(arm_err.max()), 2), "order_ok": bool(arm_err.mean() < 0.005 and arm_err.max() < 0.02)}
(GR1 / "gr1_names.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("[gr1] order_ok:", out["order_ok"], "->", GR1 / "gr1_names.json")
tl.stop()
app.close()
