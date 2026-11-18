"""Bake the GUI-ready USD files of the dual YellowArm+ORCA robot (Isaac python; run_dual_orca_bake.bat).

Writes next to the imported asset (robots/yellowarm_orca_dual/usd/):
  YellowArm_Two_ORCA_sim.usda   robot = imported URDF asset + dual_orca_rig physics (self-collision off, real drive
                                limits, ORCA servo model) + home pose as initial state and drive targets
  dual_orca_pick_scene.usda     PhysicsScene 100 Hz, floor, lights, table + cube of dual_orca_pick_sim, the robot
Open dual_orca_pick_scene.usda in Isaac Sim (File > Open) and press Play: the robot holds the home pose.
Only relative asset paths are written, so the folder can be moved. Re-run after re-importing the URDF.
"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

import dual_orca_pick_plan as P  # noqa: E402
import dual_orca_rig as rig  # noqa: E402
from embodiment import Embodiment  # noqa: E402
from mhh_paths import DUAL_URDF, DUAL_USD_DIR, dual_import_usd  # noqa: E402

E = Embodiment()
NAMES = E.action_space("joint_full")
asset = dual_import_usd()  # newest import of the current URDF
if not asset.exists():
    sys.exit(f"missing {asset}: run run_dual_probe.bat --import first")
sim_path = DUAL_USD_DIR / f"{DUAL_URDF.stem}_sim.usda"
scene_path = DUAL_USD_DIR / "dual_orca_pick_scene.usda"

# ---------------- robot layer ----------------
st = Usd.Stage.CreateNew(str(sim_path))
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(st, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
robot = st.DefinePrim("/robot", "Xform")
robot.GetReferences().AddReference(f"./{DUAL_URDF.stem}/{DUAL_URDF.stem}.usda")
st.SetDefaultPrim(robot)
root, n = rig.apply_rig(st, "/robot", E)
rig.set_targets_attr(st, "/robot", dict(zip(NAMES, E.home(NAMES))))
st.GetRootLayer().customLayerData = {"generated_by": "smolvla-openarm/scripts/dual_orca_bake.py"}
st.Save()
print(f"[bake] {sim_path} (articulation root {root}, {n} drives)")

# ---------------- scene ----------------
sc = Usd.Stage.CreateNew(str(scene_path))
UsdGeom.SetStageUpAxis(sc, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(sc, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(sc, 1.0)
world = UsdGeom.Xform.Define(sc, "/World")
sc.SetDefaultPrim(world.GetPrim())
ps = UsdPhysics.Scene.Define(sc, "/World/physicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(9.81)
try:
    from pxr import PhysxSchema

    PhysxSchema.PhysxSceneAPI.Apply(ps.GetPrim()).CreateTimeStepsPerSecondAttr(100)
except Exception as e:  # noqa: BLE001
    print("[bake] PhysxSceneAPI not set:", e)
UsdLux.DomeLight.Define(sc, "/World/dome").CreateIntensityAttr(1100.0)
sun = UsdLux.DistantLight.Define(sc, "/World/sun")
sun.CreateIntensityAttr(1500.0)
UsdGeom.XformCommonAPI(sun).SetRotate(Gf.Vec3f(-40, 0, 30))


def material(path, rgb):
    m = UsdShade.Material.Define(sc, path)
    sh = UsdShade.Shader.Define(sc, f"{path}/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def box(path, size, center, rgb, rigid=False, mass=None):
    c = UsdGeom.Cube.Define(sc, path)
    c.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(c)
    api.SetTranslate(Gf.Vec3d(*map(float, center)))
    api.SetScale(Gf.Vec3f(*map(float, size)))
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    if rigid:
        UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
        UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(float(mass))
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(material(f"/World/Looks/{c.GetPrim().GetName()}", rgb))
    return c


box("/World/floor", (20.0, 20.0, 0.02), (0, 0, P.FLOOR_Z - 0.01), (0.45, 0.47, 0.5))
box("/World/pedestal", (*P.PEDESTAL, -P.FLOOR_Z), (0, 0, P.FLOOR_Z / 2), (0.3, 0.3, 0.33))
tx, ty = P.TABLE_X, P.TABLE_Y
th = P.TABLE_TOP - P.FLOOR_Z
box("/World/table", (tx[1] - tx[0], ty[0] - ty[1], th), (0.0, (ty[0] + ty[1]) / 2, P.FLOOR_Z + th / 2),
    (0.55, 0.45, 0.33))
cube = box("/World/cube", (P.CUBE,) * 3, (0.21, -0.39, P.TABLE_TOP + P.CUBE / 2), (0.1, 0.25, 0.85), rigid=True,
           mass=P.CUBE_MASS)
pm = UsdShade.Material.Define(sc, "/World/PhysicsMaterials/cube")
pmat = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
pmat.CreateStaticFrictionAttr(1.0)
pmat.CreateDynamicFrictionAttr(1.0)
UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
r = sc.DefinePrim("/World/robot", "Xform")
r.GetReferences().AddReference(f"./{sim_path.name}")
cam = UsdGeom.Camera.Define(sc, "/World/view_cam")
cam.CreateFocalLengthAttr(18.0)
eye, tgt = Gf.Vec3d(1.3, -1.6, 1.25), Gf.Vec3d(0.1, -0.3, 0.55)
cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, tgt, Gf.Vec3d(0, 0, 1)).GetInverse())
sc.GetRootLayer().customLayerData = {"generated_by": "smolvla-openarm/scripts/dual_orca_bake.py"}
sc.Save()
print(f"[bake] {scene_path}")

if "--verify" in sys.argv:  # what the GUI does: open the scene, press Play, wait 4 s; the robot must hold home
    import numpy as np
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    from mhh_paths import ROOT

    omni.usd.get_context().open_stage(str(scene_path))
    for _ in range(20):
        app.update()
    stage = omni.usd.get_context().get_stage()
    art_root = rig.find_articulation_root(stage, "/World/robot").GetPath().pathString
    shot = RtxCamera("/World/verify_cam", translations=np.array([1.3, -1.6, 1.25]),
                     orientations=np.array([1.0, 0, 0, 0]))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.3, -1.6, 1.25), Gf.Vec3d(0.1, -0.3, 0.55), Gf.Vec3d(0, 0, 1)).GetInverse()
    q = m.ExtractRotationQuat()
    shot.set_world_poses(orientations=np.array([[q.GetReal(), *q.GetImaginary()]]))
    sens = CameraSensor(shot, resolution=(960, 720), annotators=["rgb"])
    omni.timeline.get_timeline_interface().play()
    app.update()
    art = Articulation(art_root)
    dof = list(art.dof_names)
    home = np.array([E.home(NAMES)[NAMES.index(n)] for n in dof])
    for _ in range(240):
        app.update()
    qd = art.get_dof_positions().numpy()[0]
    err = np.abs(qd - home)
    print(f"[verify] {len(dof)} dofs, max |q - home| after Play {err.max():.4f} rad ({dof[int(err.argmax())]}), "
          f"finite {bool(np.isfinite(qd).all())}")
    d, _ = sens.get_data("rgb")
    if d is not None:
        from PIL import Image

        out = ROOT / "logs" / "dual_probe" / "baked_scene_play.png"
        Image.fromarray(d.numpy()[..., :3].astype(np.uint8)).save(out)
        print("[verify] picture:", out)
    print("[verify]", "OK" if err.max() < 0.05 and np.isfinite(qd).all() else "FAILED")
app.close()
