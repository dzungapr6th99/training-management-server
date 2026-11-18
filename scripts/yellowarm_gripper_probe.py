"""YellowArm gripper probe: close-up renders at both finger extremes + fingertip bounding boxes (upright mount)."""
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.experimental.objects import DomeLight, GroundPlane  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, RigidPrim  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from mhh_paths import YELLOWARM_USD as _USD  # noqa: E402
USD = _USD.as_posix()
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def look_at_quat(eye, target, up=(0, 0, 1)):
    q = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(*up)).GetInverse().ExtractRotationQuat()
    return np.array([q.GetReal(), *q.GetImaginary()])


stage_utils.create_new_stage()
DomeLight("/World/Light").set_intensities(1200)
GroundPlane("/World/ground", sizes=10.0)
stage_utils.add_reference_to_stage(usd_path=USD, path="/World/yellowarm")
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/yellowarm")
root.GetAttribute("xformOp:orient").Set(Gf.Quatd(1, 0, 0, 0))
root.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0))
cam = CameraSensor(RtxCamera("/World/cam_close", translations=np.array([0.78, 0.0, 0.40]),
                             orientations=look_at_quat((0.78, 0.0, 0.40), (0.66, 0.0, 0.27))),
                   resolution=(480, 480), annotators=["rgb"])
robot = Articulation("/World/yellowarm")
omni.timeline.get_timeline_interface().play()
app.update()
names = robot.dof_names
i8, i9 = names.index("yellowarm_joint8"), names.index("yellowarm_joint9")
from PIL import Image  # noqa: E402

for tag, v in (("neg", -0.022), ("pos", 0.015)):
    q = np.zeros(len(names), np.float32)
    q[i8] = q[i9] = v
    robot.set_dof_positions(q)
    robot.set_dof_position_targets(q)
    for _ in range(90):
        app.update()
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=False)
    for ln in ("yellowarm_link7", "yellowarm_link8", "yellowarm_link9"):
        rng = bc.ComputeWorldBound(stage.GetPrimAtPath(f"/World/yellowarm/{ln}")).ComputeAlignedRange()
        print(f"[grip] fingers={v:+.3f} {ln} bbox min {np.round(list(rng.GetMin()), 4)} max {np.round(list(rng.GetMax()), 4)}")
    l7 = RigidPrim("/World/yellowarm/yellowarm_link7").get_world_poses()
    print(f"[grip] link7 pose {np.round(l7[0].numpy()[0], 4)} quat {np.round(l7[1].numpy()[0], 4)}")
    d, _ = cam.get_data("rgb")
    Image.fromarray(d.numpy()[..., :3]).save(os.path.join(OUT, f"grip_{tag}.png"))
app.close()
