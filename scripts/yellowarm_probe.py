"""Probe the YellowArm USD in Isaac Sim 6: link poses at zero config, gripper direction, side-view render."""
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import DomeLight, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from pxr import Gf  # noqa: E402

from mhh_paths import YELLOWARM_USD as _USD  # noqa: E402
USD = _USD.as_posix()
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUT, exist_ok=True)


def look_at_quat(eye, target, up=(0, 0, 1)):
    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


stage_utils.create_new_stage()
DomeLight("/World/Light").set_intensities(1200)
GroundPlane("/World/ground", sizes=10.0)
stage_utils.add_reference_to_stage(usd_path=USD, path="/World/yellowarm")
import omni.usd
from pxr import UsdGeom
_p = omni.usd.get_context().get_stage().GetPrimAtPath("/World/yellowarm")
_p.GetAttribute("xformOp:orient").Set(Gf.Quatd(1, 0, 0, 0))  # upright: base Z = world up
_p.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))   # base on the ground (table top in the task scene)
cams = {n: CameraSensor(RtxCamera(f"/World/cam_{n}", translations=np.array(e, float), orientations=look_at_quat(e, t)),
                        resolution=(480, 480), annotators=["rgb"])
        for n, e, t in (("side", (0.35, -1.6, 0.5), (0.35, 0, 0.25)), ("front", (1.6, 0.4, 0.8), (0.3, 0, 0.2)))}
robot = Articulation("/World/yellowarm")
omni.timeline.get_timeline_interface().play()
app.update()
names = robot.dof_names
print("[probe] dofs", names)
print("[probe] limits", np.round(np.concatenate([x.numpy() for x in robot.get_dof_limits()], 0), 3).tolist())
links = robot.link_names
print("[probe] links", links)


def settle(q, n=60):
    robot.set_dof_positions(q)
    robot.set_dof_velocities(np.zeros_like(q))
    robot.set_dof_position_targets(q)
    for _ in range(n):
        app.update()


def link_pos(name):
    return RigidPrim(f"/World/yellowarm/{name}").get_world_poses()[0].numpy()[0]


q = np.zeros(len(names), np.float32)
settle(q)
for ln in links:
    print(f"[probe] zero pose {ln}: {np.round(link_pos(ln), 3)}")
i8, i9 = names.index("yellowarm_joint8"), names.index("yellowarm_joint9")
for a, b in ((-0.022, -0.022), (0.015, 0.015), (-0.022, 0.015), (0.015, -0.022)):
    q2 = q.copy(); q2[i8], q2[i9] = a, b
    settle(q2, 30)
    gap = np.linalg.norm(link_pos("yellowarm_link8") - link_pos("yellowarm_link9"))
    print(f"[probe] joint8={a:+.3f} joint9={b:+.3f}: finger link distance {gap * 1000:.1f} mm, pos read {np.round(robot.get_dof_positions().numpy()[0][[i8, i9]], 4)}")
settle(q)
for _ in range(120):
    app.update()
from PIL import Image  # noqa: E402

for n, s in cams.items():
    d, _ = s.get_data("rgb")
    if d is not None:
        Image.fromarray(d.numpy()[..., :3]).save(os.path.join(OUT, f"yellow_upright_{n}.png"))
print("[probe] saved renders to", OUT)
app.close()
