# Training Studio (MVP)

Web app quản lý dữ liệu training của `training-management-server`: robots, tasks, datasets, các lần train, model, kết quả eval.
Kế hoạch REORG_PLAN.md: S1 + S2 (đọc dữ liệu có sẵn) và **S3 tối thiểu: chạy job train từ app**. Chưa có job sinh data, eval, deploy.

## Chạy
```bash
training-management-server\studio.bat
```
Mở <http://127.0.0.1:8700>. Lần đầu cần cài môi trường (xem đầu `studio.bat`):
`py -3.12 -m venv backend\.venv`, `backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt`,
rồi `cd frontend && npm install && npm run build`. Backend dùng một venv riêng, không đụng `env\` (môi trường train) hay Python của Isaac Sim.

Phát triển giao diện: `cd frontend && npm run dev` (cổng 5175, proxy `/api` sang 8700).

## Dữ liệu lấy từ đâu (app không sửa file nào trong số này)
| Trang | Nguồn |
|---|---|
| Robots | `design/<robot>/` (URDF/USD, embodiment) + danh sách trong `backend/app/registry.py` |
| Tasks | `backend/app/registry.py` (level, split, khoảng seed) |
| Datasets | `datasets/lerobot/*/meta/info.json`, thống kê độ phủ `results/dataset_stats.json` |
| Training | `outputs/*/checkpoints`, `results/train_*.json`, loss đọc từ `logs/train_<run>*.log` |
| Models | từng checkpoint `outputs/<run>/checkpoints/<step>/pretrained_model/config.json` |
| Eval | `results/*.json` do `eval_harness.py` ghi (số k/n và Wilson 95% giữ nguyên, không tính lại) |

Bấm **Quét lại dữ liệu** ở trang Tổng quan sau khi có checkpoint hoặc kết quả mới.

## Điều app tự kiểm tra
- **So sánh eval:** chỉ coi là so sánh được khi cùng level, cùng khoảng seed và cùng `episode_seconds` (trang Eval đánh dấu dòng khác điều kiện).
  Wilson trong file được tính lại để phát hiện lệch.
- **Gán cho robot:** ghi vào `backend/data/studio.db` (chỉ bảng `assignment`, có lịch sử). Server từ chối gán dataset/model khi
  số chiều state/action hoặc camera không khớp robot, hoặc dataset là bản cũ đã thay thế.

## Cài lên máy khác (từ git)
Hướng dẫn cài đặt và bảng biến `.env` đầy đủ: [`README_STUDIO.md`](README_STUDIO.md).

Commit hai thư mục `backend/` và `frontend/` (đã có `.gitignore`: bỏ `.venv/`, `data/`, `node_modules/`, `dist/`) cùng `studio.bat`, `studio.sh`.
Trên máy mới, sau khi clone **giữ đúng bố cục** `training-management-server/{backend,frontend,studio.bat,studio.sh}`:
```bash
studio.bat            # Windows       (Linux/macOS: ./studio.sh)
```
Lần đầu script tự: tạo venv và cài `backend/requirements.txt` → build frontend nếu chưa có `frontend/dist` (cần Node 18+) → **tạo database**
`backend/data/studio.db` (`python -m app.init_db`) → chạy app ở http://127.0.0.1:8700. Cần Python 3.10+ và internet ở lần đầu.
- `studio.bat initdb` chỉ tạo/làm mới database; `studio.bat resetdb` xóa database rồi tạo lại (mất chỉnh sửa robot/model, việc gán, lịch sử job).
- Chưa có dữ liệu cũng chạy được: danh sách robot/task mặc định vẫn nạp, các trang dữ liệu để trống. Khi có `datasets/lerobot/`, `outputs/`, `results/`,
  `design/` ở chỗ cấu hình trong `.env` (`STUDIO_*_DIR`, `STUDIO_DESIGN_DIR`; mặc định nằm trong `training-management-server/`) thì bấm "Quét lại dữ liệu".
- Cấu hình: chép `training-management-server/.env.example` thành `.env` rồi sửa (host/port, thư mục dữ liệu, `STUDIO_TRAIN_PYTHON`, cache HF, ngưỡng GPU...). `studio.bat`/`studio.sh`, backend và `vite.config.js` cùng đọc file này; biến môi trường có sẵn được ưu tiên hơn `.env`; để trống = mặc định như trước. `.env` không commit. Chính: `STUDIO_HOST` (mặc định 127.0.0.1), `STUDIO_PORT` (8700), `STUDIO_DB`, `STUDIO_VENV`. Windows giới hạn đường dẫn 260 ký tự: clone vào thư mục ngắn (vd `C:\studio`)
  hoặc đặt `STUDIO_VENV=C:\studio-venv`, nếu không `pip install pyarrow` có thể lỗi.
- App không có đăng nhập: giữ `127.0.0.1`. Chạy job train chỉ có trên máy Windows có `env\python.exe`; máy khác chỉ xem/quản lý dữ liệu.
- Tùy chọn: `tools/package_studio.py` tạo một file zip chứa app + bản dữ liệu nhẹ (kết quả, metadata dataset, cấu hình model).

## Xem chi tiết dữ liệu (trang Datasets → chọn dataset → Episodes)
- Danh sách episode (câu lệnh, số frame, giây), lọc theo câu lệnh hoặc số episode.
- Chọn một episode để xem **video của cả 3 camera đồng bộ** (phát/dừng, thanh tua, tốc độ 0.5–4×) và **biểu đồ state/action** theo từng nhóm
  (tay trái, tay phải, gripper) với đường đứt chỉ thời điểm đang xem.
- Nguồn: `meta/episodes` (đoạn video của từng camera), file mp4 trong `videos/`, và `data/*.parquet` (state/action). LeRobot v3 gộp nhiều episode
  vào một mp4 dài (khoảng 1 giờ, 200 MB); trình phát tua tới đoạn của episode và dừng ở cuối. Server trả HTTP Range nên tua nhanh, không tải cả file.
- Video là H.264 256×256 nên phát được trong trình duyệt thường. Chỉ đọc, không ghi gì.

## Quản lý robot và model (thêm / sửa / xóa)
**Robot** (trang Robots → "+ Thêm robot", hoặc mở robot → Sửa / Xóa):
- Mỗi robot cần **ba bản mô hình cho Isaac Sim**: `URDF` và `USD` là **bắt buộc**, `USD có ROS` (stage có đồ thị ROS 2 bridge) là **tùy chọn, có thể để trống**.
  Server kiểm tra file có thật trong `design/<thư mục>/`, đúng đuôi (`.urdf` / `.usd*`), nằm trong dự án; thiếu URDF hoặc USD thì không lưu được.
  Robot có sẵn nhưng còn thiếu bản nào thì được gắn nhãn "thiếu" trong danh sách và vẫn sửa được các trường khác.
- Biểu mẫu gợi ý file sẵn có theo từng thư mục trong `design/` (danh sách quét tối đa 4 cấp thư mục).
- **Xóa** không xóa file: robot tự tạo bị gỡ khỏi studio, robot có sẵn (trong `registry.py`) được ẩn và khôi phục được ("hiện cả robot đã xóa" → Khôi phục).
  URDF/USD/CAD, dataset và model không bị đụng tới; việc gán dataset/model của robot đó bị bỏ.
- Sửa robot có sẵn được lưu thành bản ghi đè trong database, `registry.py` không bị sửa; ID không đổi được sau khi tạo.

**Model** (trang Models → "+ Đăng ký model", hoặc mở model):
- Đăng ký thêm một thư mục model LeRobot (có `config.json`, thường `outputs/<run>/checkpoints/<step>/pretrained_model`); app đọc loại model, số chiều state/action, camera, chunk từ file.
- Sửa tên hiển thị, ghi chú, trạng thái (cuối / trung gian / tự đăng ký / lưu trữ), robot và dataset. Với checkpoint trên đĩa, phần chỉnh sửa là bản ghi đè trong database.
- Xóa = gỡ khỏi studio (model tự đăng ký) hoặc ẩn (checkpoint trên đĩa, khôi phục được); **checkpoint trên đĩa không bao giờ bị xóa**. Việc chọn model cho robot bị bỏ.
- **Chọn model cho robot:** mở một model → bảng "Chọn model này cho robot" liệt kê mọi robot kèm kết quả kiểm tra tương thích (state/action, camera, robot của model); chỉ robot khớp mới bấm chọn được. Cũng chọn được từ trang chi tiết robot.
- Dữ liệu người dùng nằm ở các bảng `robot_user`, `model_user`, `assignment` trong `backend/data/studio.db`, không bị xóa khi quét lại.

## Chạy train từ app (trang Jobs)
Chọn dataset (chỉ dataset hợp lệ có robot), ACT hoặc SmolVLA (base theo số chiều state: 16-D `models/smolvla_base_humanoid`, 8-D `models/smolvla_base_yellowarm`), số step, batch, tên run → **Chạy train**.
- Chạy bằng `env\python.exe -m lerobot.scripts.lerobot_train` (cùng cờ như `train_pipeline.py`), với `HF_HOME`, `HF_HUB_OFFLINE=1`, `HF_DATASETS_CACHE=D:\hfds`.
- Một worker, mỗi lần một job (đóng vai trò khoá GPU); từ chối chạy nếu GPU còn dưới 4 GiB trống. Có nút Huỷ (kill cả cây tiến trình) và xem log trực tiếp.
- **App sẽ ghi** (chỉ khi bạn bấm chạy): `outputs/<run>/`, `results/train_<run>.json` (cùng trường như `train_pipeline.py`) và `logs/studio/job_<id>.log`. Không ghi đè run cũ (trùng tên bị từ chối).
- Xong job tự quét lại dữ liệu nên run/checkpoint mới hiện ngay trong Training và Models.
- Job đang chạy khi server tắt sẽ bị đánh dấu "bị ngắt" (tiến trình train có thể vẫn chạy tiếp, kiểm tra bằng Task Manager).

## Giới hạn hiện tại
- Chưa có: job sinh data/eval/deploy, wizard tạo policy, gắn nhãn/xoá episode, so sánh nhiều episode cạnh nhau, video của lần eval (đang nằm ở `logs/videos/`).
- Train chỉ hỗ trợ ACT và SmolVLA; GR00T/π0.5 vẫn chạy qua `scripts/train_pipeline.py`.
- Danh sách robot/task mặc định nằm trong `registry.py`, chưa đọc từ `robot.yaml` (Phase 1 chưa tạo file này); robot thêm/sửa trong app nằm trong database. Robot `humanoid_yellowarm` chưa có thư mục design riêng.
- Database là SQLite thuần (`sqlite3`), chưa dùng SQLAlchemy/Alembic như kế hoạch ban đầu. Bảng index được dựng lại mỗi lần quét; chỉ `assignment` và `assignment_history` là dữ liệu người dùng.
- Một số lần train cũ không có `results/train_*.json` nên step/thời gian/loss hiển thị "—" hoặc "không rõ", không đoán.
