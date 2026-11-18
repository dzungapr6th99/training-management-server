"""Physics setup of the dual YellowArm+ORCA robot in Isaac Sim 6 (shared by probe, GUI viewer and task sims).

The URDF importer (URDF USD Converter v0.1.3) leaves the asset unusable as-is:
  * allow_self_collision=False only writes `newton:selfCollisionEnabled`; there is no PhysxArticulationAPI, so PhysX
    runs WITH self-collision (default) -> neighbouring-but-not-adjacent links (tower/palm/forearm) push each other.
  * drive stiffness/damping are 0 -> every joint is limp under gravity.
  * arm maxJointVelocity is the URDF value (90 deg/s), not the real moteus setup used by the gripper humanoid sim.
apply_rig() authors all of that on a stage (runtime, or into a USD layer with --bake in dual_orca_bake.py).

Sources (do not change without updating the source):
  limits: embodiment.limits = URDF + embodiment.yaml sim_overrides (J1 range moved so the robot faces +Y, arm torques
         industrial class); overridden joint limits are written to the USD joints here (the importer used the URDF).
  arm  : same drive model as the gripper humanoid (yellow_arm.usd: near-rigid position drive), maxForce = effort,
         maxJointVelocity = scripts/data/humanoid.json arm_limits.velocity_used_rad_s (real moteus velocity limits).
  hand : design/orcahand/urdf_usd/orcahand_description/assets/mjcf/options.xml (ORCA team's servo model), max torque overridden by
         embodiment.yaml sim_overrides.hand_max_torque_nm (19 kg*cm servos = 1.86 N*m):
         kp 2.0 Nm/rad, forcerange +-1 Nm, damping 0.1 Nm*s/rad, armature 0.001 kg*m^2.
USD angular drive units are per DEGREE (stiffness Nm/deg, damping Nm*s/deg), hence the pi/180 factors.
"""
import json
import math
import os

from pxr import PhysxSchema, Sdf, UsdPhysics

HERE = os.path.dirname(os.path.abspath(__file__))
D2R = math.pi / 180.0
_ARM_LIM = json.load(open(os.path.join(HERE, "data", "humanoid.json"), encoding="utf-8"))["arm_limits"]

ARM_STIFFNESS = 17453292716032.0  # per deg, = yellow_arm.usd (gripper humanoid)
ARM_DAMPING = 1745.3292236328125  # per deg, = yellow_arm.usd
ARM_VEL_RAD_S = [float(v) for v in _ARM_LIM["velocity_used_rad_s"]]  # joint1..7
HAND_KP = 2.0  # Nm/rad   (ORCA MJCF)
HAND_KD = 0.1  # Nm*s/rad (ORCA MJCF)
HAND_FMAX = 1.0  # Nm     (ORCA MJCF forcerange)
HAND_ARMATURE = 0.001  # kg*m^2 (ORCA MJCF)


def is_arm(name):
    return "yellowarm_joint" in name


def arm_index(name):
    return int(name.rsplit("joint", 1)[1]) - 1


def find_articulation_root(stage, under):
    return next((p for p in stage.Traverse()
                 if p.GetPath().pathString.startswith(under) and p.HasAPI(UsdPhysics.ArticulationRootAPI)), None)


def apply_rig(stage, under, embodiment, self_collision=False, pos_iters=32, vel_iters=4):
    """Author articulation + drive settings on every robot joint below `under`. Returns (root_path, n_joints)."""
    names = set(embodiment.action_space("joint_full"))
    root = find_articulation_root(stage, under)
    if root is None:
        raise RuntimeError(f"no ArticulationRootAPI below {under}")
    pa = PhysxSchema.PhysxArticulationAPI.Apply(root)
    pa.CreateEnabledSelfCollisionsAttr(bool(self_collision))
    pa.CreateSolverPositionIterationCountAttr(int(pos_iters))
    pa.CreateSolverVelocityIterationCountAttr(int(vel_iters))
    n = 0
    for p in stage.Traverse():
        if not p.GetPath().pathString.startswith(under) or p.GetName() not in names:
            continue
        jn = p.GetName()
        lim = embodiment.limits[jn]
        if jn in embodiment.overrides.get("joint_limits_rad", {}):
            rj = UsdPhysics.RevoluteJoint(p)
            rj.CreateLowerLimitAttr(float(math.degrees(lim["lower"])))
            rj.CreateUpperLimitAttr(float(math.degrees(lim["upper"])))
        dr = UsdPhysics.DriveAPI.Apply(p, "angular")
        dr.CreateTypeAttr("force")
        pj = PhysxSchema.PhysxJointAPI.Apply(p)
        if is_arm(jn):
            dr.CreateStiffnessAttr(ARM_STIFFNESS)
            dr.CreateDampingAttr(ARM_DAMPING)
            dr.CreateMaxForceAttr(float(lim["effort"]))
            pj.CreateMaxJointVelocityAttr(float(math.degrees(ARM_VEL_RAD_S[arm_index(jn)])))
        else:
            dr.CreateStiffnessAttr(HAND_KP * D2R)
            dr.CreateDampingAttr(HAND_KD * D2R)
            dr.CreateMaxForceAttr(float(embodiment.overrides.get("hand_max_torque_nm", HAND_FMAX)))
            pj.CreateArmatureAttr(HAND_ARMATURE)
        n += 1
    return root.GetPath().pathString, n


def set_targets_attr(stage, under, q_by_name):
    """Write drive targets (rad) into the USD (used for the baked home pose: Play in the GUI holds this pose)."""
    for p in stage.Traverse():
        if p.GetPath().pathString.startswith(under) and p.GetName() in q_by_name:
            deg = float(math.degrees(q_by_name[p.GetName()]))
            UsdPhysics.DriveAPI.Apply(p, "angular").CreateTargetPositionAttr(deg)
            # initial joint position (the importer already applied PhysicsJointStateAPI:angular)
            attr = p.GetAttribute("state:angular:physics:position")
            if not attr:
                attr = p.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float)
            attr.Set(deg)
