# Training Studio — cài đặt và cấu hình

Web app quản lý dữ liệu training: robot (URDF/USD), task, dataset LeRobot, các lần train, model, kết quả eval, và chạy job train.
App độc lập: mọi đường dẫn và tham số nằm trong file `.env`, không phụ thuộc vào vị trí repo.
Hướng dẫn dùng từng trang: [`STUDIO.md`](STUDIO.md).

## Thành phần

```
training-management-server/
├── backend/          FastAPI + SQLite (backend/app/config.py đọc .env)
├── frontend/         React + Vite (build ra frontend/dist)
├── studio.bat        chạy trên Windows
├── studio.sh         chạy trên Linux/macOS
├── .env.example      mẫu cấu hình (commit)
└── .env              cấu hình của máy này (KHÔNG commit)
```

## Yêu cầu

| | Bắt buộc | Ghi chú |
|---|---|---|
| Python 3.10+ | có | tạo venv riêng cho backend, không đụng env train hay Python của Isaac Sim |
| Internet lần đầu | có | `pip install -r backend/requirements.txt` |
| Node 18+ | chỉ khi chưa có `frontend/dist` | build giao diện |
| Môi trường train (Python có `lerobot` + `torch`) + GPU NVIDIA | chỉ để chạy job train | thiếu thì app chỉ để xem/quản lý dữ liệu |

## Cài nhanh

Windows:
```bat
cd training-management-server
copy .env.example .env
notepad .env
studio.bat
```

Linux/macOS:
```bash
cd training-management-server
cp .env.example .env
nano .env
./studio.sh
```

Mở <http://127.0.0.1:8700> (hoặc host/port đặt trong `.env`). Lần chạy đầu, script tự:
1. tạo venv (`STUDIO_VENV`, mặc định `backend/.venv`) và cài `backend/requirements.txt`;
2. build frontend nếu thiếu `frontend/dist`;
3. tạo database (`STUDIO_DB`, mặc định `backend/data/studio.db`) và quét dữ liệu.

Lệnh khác:

| Lệnh | Việc |
|---|---|
| `studio.bat initdb` / `./studio.sh initdb` | chỉ tạo/làm mới database và quét lại dữ liệu |
| `studio.bat resetdb` / `./studio.sh resetdb` | xoá database rồi tạo lại (mất chỉnh sửa robot/model, việc gán, lịch sử job) |

Sau khi đổi `.env`, khởi động lại app rồi bấm **Quét lại dữ liệu** ở trang Tổng quan.

## Cấu hình `.env`

Quy tắc:
- Mỗi dòng `KEY=VALUE`, **không dùng dấu nháy**, chú thích để ở dòng riêng bắt đầu bằng `#` (`studio.bat` không bỏ được dấu nháy hay chú thích cuối dòng).
- Để trống hoặc xoá dòng = dùng mặc định.
- Đường dẫn tương đối tính từ thư mục `training-management-server/`.
- Biến môi trường có sẵn được ưu tiên hơn `.env`. Muốn dùng file khác: đặt `STUDIO_ENV_FILE`.

### Web server

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `STUDIO_HOST` | `127.0.0.1` | địa chỉ lắng nghe. App **không có đăng nhập**: chỉ dùng `0.0.0.0` trong mạng tin cậy |
| `STUDIO_PORT` | `8700` | cổng |
| `STUDIO_VENV` | `backend/.venv` | venv của backend. Windows: đặt ngắn (vd `C:\studio-venv`) để tránh giới hạn 260 ký tự khi cài pyarrow |
| `STUDIO_DB` | `backend/data/studio.db` | database SQLite |
| `STUDIO_FRONTEND_DIST` | `frontend/dist` | giao diện đã build |
| `STUDIO_DEV_PORT` | `5175` | cổng `npm run dev` |
| `STUDIO_API_URL` | `http://STUDIO_HOST:STUDIO_PORT` | API mà `npm run dev` proxy `/api` sang |

### Dữ liệu

| Biến | Mặc định | Nội dung |
|---|---|---|
| `STUDIO_DESIGN_DIR` | `design` | mô tả robot: `<design>/<design_dir>/` chứa URDF/USD của từng robot |
| `STUDIO_DATASETS_DIR` | `datasets/lerobot` | dataset LeRobot v3 (`<tên>/meta/info.json`, `data/`, `videos/`) |
| `STUDIO_OUTPUTS_DIR` | `outputs` | các lần train: `<run>/checkpoints/<step>/pretrained_model` |
| `STUDIO_RESULTS_DIR` | `results` | `*.json` của eval và tóm tắt train |
| `STUDIO_LOGS_DIR` | `logs` | log train (`train_<run>*.log`); log job của app ở `<logs>/studio/` |
| `STUDIO_MODELS_DIR` | `models` | base model và cache HF |
| `STUDIO_HUMANOID_JSON` | `scripts/data/humanoid.json` | thông số thân humanoid (không có cũng chạy) |

Thư mục nào không tồn tại thì trang tương ứng để trống, app vẫn chạy.

### Isaac Sim

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `STUDIO_ISAAC_SIM_DIR` | `$ISAAC_SIM_DIR`, nếu không có thì `isaacsim` | thư mục cài Isaac Sim (có `python.bat` / `python.sh`) |
| `STUDIO_ISAAC_SIM_PYTHON` | `<STUDIO_ISAAC_SIM_DIR>/python.bat` | Python import được `isaacsim`. Dùng env từ `setup_train_env` thì trỏ vào Python của env đó |

Hiện app chỉ có job train (không cần Isaac Sim). Đường dẫn này dành cho job sinh dữ liệu/eval chạy trong sim.
Không cài `lerobot` hay gói nào khác vào bản Isaac Sim cài sẵn (`python.bat`). Muốn có một Python chứa cả hai, tạo env riêng như mục dưới.

## Cài môi trường train (torch + Isaac Sim + lerobot)

**Tự động:** mỗi lần chạy `studio.bat` / `studio.sh`, script kiểm tra Python train. Thiếu torch, isaacsim hoặc lerobot thì tự cài phần còn thiếu
rồi mới mở app (lần đầu tải nhiều GB). Python được kiểm tra:
- `STUDIO_TRAIN_PYTHON` nếu đã đặt và có thật: cài phần thiếu **vào chính env đó** (ví dụ env có torch + lerobot nhưng chưa có isaacsim);
- nếu không: venv `STUDIO_TRAIN_ENV` (mặc định `env_isaacsim/`), chưa có thì tạo mới. Cài xong app tự dùng venv này cho job train.

Đặt `STUDIO_AUTO_SETUP_TRAIN_ENV=0` để chỉ kiểm tra, không cài (máy chỉ xem dữ liệu, hoặc env đang dùng cho việc khác không muốn bị đổi).
Cài thất bại thì app vẫn mở, chỉ báo môi trường train chưa sẵn sàng. `python.bat` / `python.sh` của bản Isaac Sim cài sẵn không bao giờ bị cài thêm gói.

**Chạy tay:** `setup_train_env.bat` (Linux: `./setup_train_env.sh`) cài phần còn thiếu; `setup_train_env.bat check` chỉ báo cáo. Các gói:

| Gói | Phiên bản mặc định | Biến đổi phiên bản |
|---|---|---|
| Python | 3.12 (bắt buộc với Isaac Sim 6) | |
| torch / torchvision | 2.11.0 / 0.26.0, bản CUDA 12.8 | `STUDIO_TORCH_VERSION`, `STUDIO_TORCHVISION_VERSION`, `STUDIO_TORCH_INDEX` (cu130 cho driver CUDA 13) |
| isaacsim `[all,extscache]` | 6.0.1.0 (từ `pypi.nvidia.com`) | `STUDIO_ISAACSIM_VERSION` |
| lerobot `[smolvla]` | 0.6.2: bản trong `vendor/lerobot` nếu có (đã vá junction `checkpoints\last` cho Windows), nếu không lấy từ PyPI | |

torch được ghim bằng file constraint để isaacsim/lerobot không đổi phiên bản torch.

Trước khi chạy:
- Cài Python 3.12 (python.org) và driver NVIDIA. Linux cần GLIBC 2.35+.
- Windows: bật long paths (`LongPathsEnabled`), hoặc đặt `STUDIO_TRAIN_ENV` ngắn như `C:\mhh_env`. Script sẽ cảnh báo nếu long paths đang tắt.
- Cần internet; tải nhiều GB (Isaac Sim là phần lớn nhất).

```bat
setup_train_env.bat
```

`studio.bat` tự dùng env vừa cài. Muốn cố định thì chép hai dòng script in ra vào `.env`:
```ini
STUDIO_TRAIN_PYTHON=<env>\Scripts\python.exe
STUDIO_ISAAC_SIM_PYTHON=<env>\Scripts\python.exe
```
Lỗi giữa chừng thì chạy lại: các gói đã cài được giữ, script chỉ cài phần còn thiếu.

Lần đầu `import isaacsim`, Isaac Sim hỏi chấp nhận NVIDIA Omniverse EULA. Script không tự chấp nhận thay bạn; muốn chạy không hỏi thì tự đặt
`OMNI_KIT_ACCEPT_EULA=YES` sau khi đã đọc điều khoản. Job train của app không import `isaacsim` nên không bị chặn bởi bước này.

### Job train

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `STUDIO_TRAIN_PYTHON` | `env/python.exe` (Windows), `env/bin/python` (Linux) | Python có `lerobot`. Không có file này thì nút chạy train bị tắt |
| `STUDIO_HF_HOME` | `<models>/hf_home` | cache Hugging Face |
| `STUDIO_HF_HUB_OFFLINE` | `1` | không tải gì từ Hub khi train |
| `STUDIO_HF_DATASETS_CACHE` | `<ổ đĩa>\hfds` | cache dataset. Windows: phải là đường dẫn **ngắn**, nếu không lỗi `WinError 206` |
| `STUDIO_SMOLVLA_BASE_16D` | `<models>/smolvla_base_humanoid` | base SmolVLA cho state 16 chiều |
| `STUDIO_SMOLVLA_BASE_8D` | `<models>/smolvla_base_yellowarm` | base SmolVLA cho state 8 chiều |
| `STUDIO_POLICY_DEVICE` | `cuda` | thiết bị train ACT |
| `STUDIO_GPU_MIN_FREE_GIB` | `4` | GPU trống ít hơn số này thì không chạy job |
| `STUDIO_TRAIN_NUM_WORKERS` | `4` | số worker DataLoader |
| `STUDIO_TRAIN_MAX_STEPS` / `STUDIO_TRAIN_MAX_BATCH` | `200000` / `64` | giới hạn trên form Jobs |

Mỗi lần app chạy một job (đóng vai trò khoá GPU). Job ghi vào `<outputs>/<run>/`, `<results>/train_<run>.json`, `<logs>/studio/job_<id>.log`,
không ghi đè run đã có.

## Ví dụ `.env`

Máy chỉ xem dữ liệu, dữ liệu để ở ổ khác:
```ini
STUDIO_DESIGN_DIR=E:\robot-data\design
STUDIO_DATASETS_DIR=E:\robot-data\datasets\lerobot
STUDIO_OUTPUTS_DIR=E:\robot-data\outputs
STUDIO_RESULTS_DIR=E:\robot-data\results
STUDIO_LOGS_DIR=E:\robot-data\logs
```

Máy train Windows (Isaac Sim ở ổ D, env train có sẵn):
```ini
STUDIO_DESIGN_DIR=../design
STUDIO_ISAAC_SIM_DIR=D:\Isaac\isaacsim
STUDIO_TRAIN_PYTHON=D:\envs\lerobot\python.exe
STUDIO_HF_DATASETS_CACHE=D:\hfds
```

Máy Linux, mở cho mạng nội bộ:
```ini
STUDIO_HOST=0.0.0.0
STUDIO_PORT=8700
STUDIO_TRAIN_PYTHON=/opt/lerobot-env/bin/python
STUDIO_HF_DATASETS_CACHE=/tmp/hfds
```

## Robot và file URDF/USD

- Mỗi robot trỏ tới một thư mục `<STUDIO_DESIGN_DIR>/<design_dir>/`. Đường dẫn URDF, USD, USD có ROS lưu **tương đối với thư mục đó**,
  nên chuyển cả thư mục design sang chỗ khác chỉ cần sửa `STUDIO_DESIGN_DIR`.
- Thêm/sửa robot ở trang Robots. URDF và USD bắt buộc, phải là file có thật trong thư mục của robot. Xoá robot không xoá file.

## Chuyển sang máy khác

1. Chép hoặc clone `training-management-server/` gồm `backend/`, `frontend/`, `studio.bat`, `studio.sh`, `setup_train_env.*`, `.env.example`
   (không cần `backend/.venv`, `backend/data/`, `frontend/node_modules/`).
2. Chép dữ liệu cần xem (design, datasets, outputs, results, logs) vào bất kỳ đâu.
3. Tạo `.env` từ `.env.example`, trỏ các `STUDIO_*_DIR` vào chỗ vừa chép.
4. Chạy `studio.bat` / `./studio.sh`.

Hoặc đóng gói sẵn một zip nhẹ (app + metadata, không có video/weights):
```bat
backend\.venv\Scripts\python.exe tools\package_studio.py --out D:\packages
```
Zip chứa `.env.example`, không bao giờ chứa `.env` của máy đóng gói.

## Sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `pip install` lỗi `No such file or directory ... pyarrow` | đường dẫn quá dài: giải nén vào thư mục ngắn hoặc đặt `STUDIO_VENV=C:\studio-venv` |
| Train lỗi `WinError 206` | đặt `STUDIO_HF_DATASETS_CACHE` ngắn, vd `D:\hfds` |
| Trang Jobs báo thiếu môi trường train | chạy `setup_train_env.bat`, hoặc đặt `STUDIO_TRAIN_PYTHON` trỏ tới Python có `lerobot` |
| Job báo GPU không đủ bộ nhớ trống | tắt tiến trình khác đang dùng GPU, hoặc hạ `STUDIO_GPU_MIN_FREE_GIB` |
| Trang trống sau khi đổi `.env` | khởi động lại app, bấm **Quét lại dữ liệu** |
| Chỉ thấy API, không có giao diện | thiếu `frontend/dist`: cài Node 18+ rồi chạy lại script |
| Cổng 8700 đã bị dùng | đổi `STUDIO_PORT` |
