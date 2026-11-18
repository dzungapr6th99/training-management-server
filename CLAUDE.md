# training-management-server (trước đây `smolvla-openarm`, đổi tên 2026-11-15) — bối cảnh cho Claude

Dự án Denso Hackathon 2026: fine-tune nhiều model điều khiển robot (VLA và baseline) trên **humanoid nửa thân trên
gồm 2 cánh tay YellowArm + camera đầu cố định** trong Isaac Sim 6.0.1, cho tác vụ **gắp linh kiện khác hình dạng/màu
sắc đặt vào pallet**, rồi so sánh chúng một cách công bằng. Chưa có phần cứng (mới có bản thiết kế tay): mọi thông số
trong sim phải bám sát những gì lắp được ngoài đời, không nới giới hạn khớp/lực cho dễ.
Kế hoạch đang thực hiện: xem `PLAN.md` (làm theo thứ tự, đánh dấu [x] khi xong, ghi kết quả vào mục "Nhật ký").

## Môi trường (Windows, RTX 4070 Ti SUPER 16 GB)
- Isaac Sim python: `D:\Isaac\isaacsim\python.bat` (Isaac Sim KHÔNG nằm trong repo; đổi vị trí bằng biến môi trường
  `ISAAC_SIM_DIR`) — KHÔNG pip install gì vào đây.
- Env train/inference: `env\python.exe` trong thư mục này (Python 3.12, torch cu128, lerobot 0.6.2 editable từ
  `vendor\lerobot`, khai báo bằng `.pth` đường dẫn tương đối). Lệnh train: `env\python.exe -m lerobot.scripts.lerobot_train ...`
  (KHÔNG dùng `env\Scripts\*.exe`: launcher của conda env ghi cứng đường dẫn python cũ nên hỏng sau khi di chuyển).
- Luôn đặt `HF_HOME=<thư mục này>\models\hf_home`, `HF_HUB_OFFLINE=1` và `HF_DATASETS_CACHE=D:\hfds` khi chạy policy/train/verify
  (đường dẫn cache dataset phải NGẮN: tên cache lock dài quá 260 ký tự sẽ lỗi `WinError 206`; `mhh_paths.HF_DS_CACHE`).
- Hai tiến trình tách biệt: `scripts/policy_server.py` (env) ⇄ TCP ⇄ script sim (Isaac python). Giao thức ở
  `scripts/wire.py` (stdlib-only, dùng chung cả hai phía). Lệnh: `info`, `reset`, `infer`.
- Đường dẫn: KHÔNG ghi cứng. Mọi script lấy đường dẫn từ `scripts/mhh_paths.py` (suy ra từ vị trí file; chạy
  `python scripts\mhh_paths.py` để kiểm tra). Repo có thể nằm ở bất kỳ đâu; chỉ Isaac Sim ở ngoài (`ISAAC_SIM_DIR`).
  Script mới phải `from mhh_paths import ...`, không tự viết `D:\...`. Các `.bat` dùng `%ROOT%` và `%ISAAC_SIM_DIR%`.
- Sau khi di chuyển thư mục (2026-11-03): `scripts\fix_last_junctions.bat` tạo lại junction `checkpoints\last`;
  `env\python.exe scripts\fix_moved_paths.py [--apply]` sửa `train_config.json` cũ (chỉ cần khi `--resume`).
- Sau khi đổi tên thành `training-management-server` (2026-11-15): đã chạy lại `fix_last_junctions.bat` và `fix_moved_paths.py --apply`
  (24 `train_config.json`, có `.bak`). Robot/CAD nằm ở `..\design\<robot>\` (xem `mhh_paths.py`), `openvla-7b` ở `models\openvla-7b`.
- `vendor\lerobot\...\train_utils.py` đã được vá: `checkpoints\last` là directory junction (Windows chặn symlink).

## Định dạng humanoid (từ G2 trở đi)
- State 16 = tay trái (joint1..7 + gripper 0..1) rồi tay phải; Action 16 cùng thứ tự.
- Camera: `camera1` đầu, `camera2` cổ tay trái, `camera3` cổ tay phải. Đầu cố định.
- Thông số thân trong `scripts/data/humanoid.json`.

## Định dạng YellowArm một tay (cube-in-box cũ, dùng cho G0–G1)
- State 8 = joint1..7 (rad) + độ mở gripper (0 đóng … 1 mở). Action 8 = joint1..7 target + lệnh gripper.
- Camera 256×256 RGB: `camera1` front, `camera2` wrist, `camera3` side. Điều khiển 25 Hz, physics 100 Hz.
- Gripper mở tối đa ~75 mm → mọi vật thể phải có cạnh kẹp ≤ ~60 mm.
- USD robot: `..\design\yellowarm\urdf_usd\yellowarm_description\robots\yellow_arm\yellow_arm.usd` (xem `scripts/mhh_paths.py`).
- Episode thô: `episode_XXXX.npz` (`state`, `action`, `camera1..3`, `task`, `fps`) → `scripts/npz_to_lerobot.py`
  (`--raw --out --repo-id --robot-type yellowarm_isaac --overwrite`, encode NVENC) → LeRobot v3.

## Thư mục
- `scripts/yellowarm_sim.py` — cube-in-box (expert `--expert --record DIR`, eval `--port`). Mẫu để phát triển tác vụ pallet.
- `scripts/legacy/yellowarm_stack_sim.py`, `scripts/legacy/stack_*pipeline.py` (cùng `openarm_*`, `openvla_server.py` và các bat bring-up; bản đồ ở `scripts/README.md`) — tác vụ xếp chồng **đã bỏ, không làm tiếp, không eval**.
  Chỉ dùng làm nguồn tham khảo code: cài đặt DART (`--dart-noise/--dart-tau`) và mẫu pipeline chạy không giám sát.
- `scripts/humanoid_pallet_sim.py` (Isaac) + `scripts/humanoid_pallet_plan.py` (numpy, chạy được bằng env python,
  `--selftest`) — tác vụ pallet humanoid (G2). Cache trong `scripts/data/`: `humanoid_park.json`,
  `humanoid_parts_region.npz` — xoá khi đổi cấu hình thân.
- `scripts/verify_npz.py`, `scripts/verify_dataset.py` — kiểm tra dữ liệu.
- `datasets/raw/*` (npz), `datasets/lerobot/*`, `outputs/<run>/checkpoints/<step>/pretrained_model`, `logs/`.
  **Từ 2026-11-15 `datasets/raw/` đang rỗng**: dữ liệu thô pallet và dual_orca đã chuyển sang `E:\backup\My-Humanoid-Hand_groupD_2026-11-15\` (xem `DISK_PLAN.md`).
  Train/eval chỉ cần `datasets/lerobot/*`; `verify_dataset.py`, `verify_npz.py`, `expert_review.py` và sinh lại dataset cần chép ngược (robocopy từ E,
  rồi `_archive\raw_merge_restore\restore_raw_merges.py` dựng lại các thư mục gộp bằng hard link). Sinh dữ liệu mới vẫn ghi vào đây.
- `models/smolvla_base_yellowarm` — điểm xuất phát SmolVLA cho YellowArm (base đã đổi sang 8-D).

## Quy tắc làm việc
- Việc chạy lâu (sinh dữ liệu, train, eval) chạy nền, ghi log vào `logs/`, rồi theo dõi log; không chặn terminal hàng giờ.
- GPU 16 GB: không chạy train song song với eval Isaac. Sinh dữ liệu headless chạy được 2 worker song song.
- Không xoá dataset, checkpoint hay file `.npz` khi chưa hỏi người dùng.
- Mọi kết quả eval phải ghi ra `results/*.json` qua `scripts/eval_harness.py` (sau khi xong Giai đoạn 0) — không chép
  tay con số từ log.
- So sánh model chỉ hợp lệ khi cùng dataset, cùng bộ seed test, cùng `--episode-seconds`. `--exec-steps` được chọn riêng
  cho từng model trên seed validation (200000+), không bao giờ chọn trên seed test (100000–100049).
- Báo cáo trung thực: luôn ghi số episode và khoảng tin cậy Wilson 95%; ghi rõ điều kiện chạy.
