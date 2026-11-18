"""Static knowledge the indexer cannot infer from files: which robots/tasks exist and how datasets map onto them.
Everything numeric about data (episodes, frames, success rates) is read from disk, never written here."""

# id -> metadata. Paths are relative to design/<design_dir>/. state/action dims are what the pipeline's datasets use.
ROBOTS = {
    "humanoid_yellowarm": dict(
        name="Humanoid 2×YellowArm + gripper", kind="humanoid", status="active",
        description="Nửa thân trên: 2 cánh tay YellowArm (7 khớp) + gripper mỗi tay, camera đầu cố định + 2 camera cổ tay. "
                    "Robot của tác vụ pallet (G2–G5).",
        state_dim=16, action_dim=16, cameras=["camera1", "camera2", "camera3"], design_dir="yellowarm",
        urdf="urdf_usd/yellowarm_description/robots/yellow_arm.urdf",
        usd="urdf_usd/yellowarm_description/robots/yellow_arm/yellow_arm.usd",
        notes="State/action 16 = tay trái (joint1..7 + gripper) rồi tay phải. Thông số thân: scripts/data/humanoid.json. "
              "Chưa có thư mục design riêng (sẽ có ở Phase 3); URDF/USD hiển thị là của một cánh tay YellowArm."),
    "yellowarm": dict(
        name="YellowArm (một tay + gripper)", kind="arm", status="active",
        description="Cánh tay YellowArm 7 khớp + gripper, tác vụ cube-in-box/stack cũ (G0–G1).",
        state_dim=8, action_dim=8, cameras=["camera1", "camera2", "camera3"], design_dir="yellowarm",
        urdf="urdf_usd/yellowarm_description/robots/yellow_arm.urdf",
        usd="urdf_usd/yellowarm_description/robots/yellow_arm/yellow_arm.usd", notes="Gripper mở tối đa ~75 mm."),
    "yellowarm_orca_dual": dict(
        name="Hai YellowArm + hai ORCA Hand", kind="humanoid", status="active",
        description="Hai cánh tay YellowArm gắn hai bàn tay ORCA (16 khớp tay mỗi bên), tác vụ dual_orca_pick.",
        state_dim=18, action_dim=18, cameras=["camera1", "camera2", "camera3"], design_dir="yellowarm_orca_dual",
        urdf="YellowArm_Two_ORCA.urdf", usd="usd/YellowArm_Two_ORCA_sim.usda", usd_ros="usd/YellowArm_Two_ORCA_ros.usda",
        notes="Dataset dùng state/action 18 chiều (arm + synergy bàn tay). Embodiment: design/yellowarm_orca_dual/embodiment.yaml."),
    "yellowarm_orca_left": dict(
        name="YellowArm + ORCA (tay trái)", kind="arm_hand", status="design",
        description="Một YellowArm + ORCA Hand trái, 24 khớp. Mới có URDF, chưa import/Play trong Isaac.",
        state_dim=None, action_dim=None, cameras=[], design_dir="yellowarm_orca_left",
        urdf="urdf_usd/yellowarm_orca_left.urdf", usd=None, notes=None),
    "yellowarm_orca_right": dict(
        name="YellowArm + ORCA (tay phải)", kind="arm_hand", status="design",
        description="Một YellowArm + ORCA Hand phải, 24 khớp. Mới có URDF, chưa import/Play trong Isaac.",
        state_dim=None, action_dim=None, cameras=[], design_dir="yellowarm_orca_right",
        urdf="urdf_usd/yellowarm_orca_right.urdf", usd=None, notes=None),
    "openarm": dict(
        name="OpenArm bimanual", kind="humanoid", status="deprecated",
        description="Hai tay OpenArm, tác vụ cube-in-box đầu tiên. Đã bỏ.",
        state_dim=9, action_dim=8, cameras=["base", "wrist_left", "wrist_right"], design_dir="openarm",
        urdf=None, usd=None, notes="Thư mục design/openarm đang rỗng."),
    "wheeled_humanoid": dict(
        name="Humanoid trên đế bánh xe", kind="concept", status="design",
        description="Thiết kế khái niệm: thân humanoid trên đế có bánh. Chưa có dataset hay model.",
        state_dim=None, action_dim=None, cameras=[], design_dir="wheeled_humanoid",
        urdf="wheeled_humanoid.urdf", usd=None, notes=None),
    "orcahand": dict(
        name="ORCA Hand (bộ phận)", kind="component", status="component",
        description="Bàn tay ORCA v1/v2 (CAD + URDF). Dùng để ghép với YellowArm.",
        state_dim=None, action_dim=None, cameras=[], design_dir="orcahand",
        urdf="urdf_usd/orcahand_description/v1/models/urdf/orcahand_right.urdf", usd=None, notes=None),
}

# dataset robot_type (from meta/info.json) -> robot id
ROBOT_TYPE_TO_ROBOT = {
    "humanoid_yellowarm_isaac": "humanoid_yellowarm",
    "yellowarm_isaac": "yellowarm",
    "yellowarm_two_orca_isaac": "yellowarm_orca_dual",
    "openarm_bimanual_isaac": "openarm",
}

TASKS = {
    "humanoid_pallet": dict(
        name="Gắp linh kiện đặt vào pallet", robot_id="humanoid_yellowarm", status="active",
        description="Gắp một linh kiện khác hình dạng/màu sắc và đặt vào một ô trống của pallet 2×3. L1: lệnh theo hình+màu, "
                    "L2: có thêm vật nhiễu/ô đã có vật.",
        levels=[1, 2], splits=["train", "heldout_color", "heldout_shape", "wide"],
        seeds={"train/dataset": "job seeds 0–70000", "validation": "200000+", "test": "100000–100049"},
        sim_script="scripts/humanoid_pallet_sim.py"),
    "dual_orca_pick": dict(
        name="Gắp vật bằng hai tay ORCA", robot_id="yellowarm_orca_dual", status="active",
        description="Tác vụ gắp vật với robot hai tay YellowArm + ORCA (action 18 chiều).",
        levels=[], splits=[], seeds={}, sim_script="scripts/dual_orca_pick_sim.py"),
    "cube_in_box": dict(
        name="Cube-in-box (một tay)", robot_id="yellowarm", status="deprecated",
        description="Gắp khối vào hộp, tác vụ G0–G1. Đã thay bằng pallet.", levels=[], splits=[], seeds={},
        sim_script="scripts/yellowarm_sim.py"),
    "stack3": dict(
        name="Xếp chồng 3 khối", robot_id="yellowarm", status="deprecated",
        description="Tác vụ xếp chồng. Đã bỏ, không eval.", levels=[], splits=[], seeds={},
        sim_script="scripts/legacy/yellowarm_stack_sim.py"),
    "openarm_cube_in_box": dict(
        name="Cube-in-box (OpenArm)", robot_id="openarm", status="deprecated",
        description="Tác vụ đầu tiên trên OpenArm bimanual. Đã bỏ.", levels=[], splits=[], seeds={},
        sim_script="scripts/legacy/openarm_smolvla_sim.py"),
}

# sim script basename -> task id (for eval results)
SIM_TO_TASK = {
    "humanoid_pallet_sim.py": "humanoid_pallet",
    "dual_orca_pick_sim.py": "dual_orca_pick",
    "yellowarm_sim.py": "cube_in_box",
    "yellowarm_stack_sim.py": "stack3",
    "openarm_smolvla_sim.py": "openarm_cube_in_box",
    "openarm_bimanual_sim.py": "openarm_cube_in_box",
}


def task_for_dataset(name: str) -> str | None:
    if name.startswith("yellowarm_pallet"):
        return "humanoid_pallet"
    if name.startswith("dual_orca_pick"):
        return "dual_orca_pick"
    if name.startswith("yellowarm_cube_in_box"):
        return "cube_in_box"
    if name.startswith("yellowarm_stack3"):
        return "stack3"
    if name.startswith("openarm_cube_in_box"):
        return "openarm_cube_in_box"
    return None
