# scripts/ — bản đồ theo chức năng

Các script gọi nhau bằng tên file (`import wire`, `scripts/eval_harness.py`…) và chạy bằng hai Python khác nhau
(`env\python.exe` và Python của Isaac Sim), nên giữ phẳng ở đây. Thư mục `legacy/` là các tác vụ đã bỏ.
Mọi đường dẫn lấy từ `mhh_paths.py`.

| Nhóm | File |
|---|---|
| **Dùng chung** | `mhh_paths.py` (đường dẫn), `wire.py` (giao thức sim ⇄ policy), `embodiment.py`, `humanoid_kin.py`, `action_smoother.py` |
| **Sim + expert (Isaac)** | `humanoid_pallet_sim.py` + `humanoid_pallet_plan.py` (pallet humanoid), `dual_orca_pick_sim.py` + `dual_orca_pick_plan.py` + `dual_orca_kin.py` + `dual_orca_rig.py` + `dual_orca_bake.py`, `yellowarm_sim.py` (một tay) |
| **Sinh dữ liệu** | `pallet_pipeline.py`, `dual_orca_pipeline.py` (chạy không giám sát), `npz_to_lerobot.py` |
| **Kiểm tra dữ liệu** | `verify_npz.py`, `verify_dataset.py`, `verify_humanoid.py`, `pallet_dataset_stats.py`, `expert_review.py` (+ `expert_review_page.html`), `view_episode.py` |
| **Train** | `train_pipeline.py` (ACT, SmolVLA, GR00T, π0.5) |
| **Eval** | `eval_harness.py` (một lượt, ghi `results/<tag>.json`), `eval_pipeline.py` (nhiều lượt), `collect_metrics.py`, `make_report.py`, `jitter_offline.py` |
| **Phục vụ policy** | `policy_server.py`, `run_server.bat` |
| **Quay video demo** | `frames_to_mp4.py`, `concat_mp4.py`, `demo_active_arm.py`, `make_table_textures.py` |
| **Thiết kế / thăm dò robot** (sẽ chuyển sang `design/` khi cần) | `yellowarm_probe.py`, `yellowarm_gripper_probe.py`, `render_yellowarm_views.py`, `humanoid_reach_probe.py`, `dual_robot_probe.py`, `gr1_*.py`, `dual_orca_monitor.py` (+ page) |
| **Bảo trì** | `fix_last_junctions.bat`, `fix_moved_paths.py` |
| **`legacy/`** | Stack, OpenArm, OpenVLA và các script bring-up một lần (`stack_*`, `openarm_*`, `openvla_server.py`, `run_all/run_sim/run_bimanual_all/run_openvla_all/eval_finetuned/gen_data/verify_pathfix*.bat`, `RUN_1_probe.bat`, `00_probe.ps1`) |
| **`data/`** | `humanoid.json`, `parts.json`, `domain_rand.json`, bản cache `humanoid_park.json`, `humanoid_*.npz`, `qualia_cube_in_box.json`, `textures/` |

Script trong `legacy/` tự thêm `scripts/` vào `sys.path` và tính gốc dự án từ vị trí file, nên chạy được từ chỗ cũ lẫn chỗ mới.
