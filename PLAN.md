# Kế hoạch: Parts-to-Pallet benchmark trên YellowArm

**Mục tiêu:** fine-tune nhiều model trên cùng một robot và cùng một tác vụ gắp linh kiện vào pallet, rồi so sánh chúng:
tỉ lệ thành công, khả năng tổng quát hoá, chi phí train và độ trễ suy luận.
**Robot:** humanoid nửa thân trên, gồm 2 cánh tay YellowArm gắn ở vai và 1 camera trên đầu cố định. **Chưa có phần
cứng**, mới có bản thiết kế 2 tay. Nếu kết quả trong sim đạt ngưỡng ở G7 thì mới lắp robot thật. Vì vậy sim vừa là nơi
train và so sánh model, vừa là công cụ thử thiết kế phần cứng: mọi thông số thân, vai và đầu phải là tham số chỉnh được,
và phải bám sát những gì lắp được ngoài đời.
**Hạn nộp:** 10/10/2026. Nếu thiếu thời gian, ưu tiên làm theo thứ tự: G0 → G1 → G2 (chỉ L1 và L2) → G3 → G4 (chỉ ACT
và SmolVLA) → G5 → G6.

Cách dùng: mỗi lần mở Claude trong terminal, bảo nó "đọc CLAUDE.md và PLAN.md, làm task chưa xong tiếp theo". Làm xong
task nào thì đánh dấu [x] và ghi một dòng vào mục **Nhật ký** ở cuối file.

---

## Kết quả đã có (mốc tham chiếu, SmolVLA)
| Tác vụ | Checkpoint | Điều kiện | Kết quả |
|---|---|---|---|
| YellowArm cube-in-box | `outputs/smolvla_yellowarm_cube_in_box/checkpoints/030000` | trong vùng train, exec 8, seed 777 | 18/20 |
| như trên | 20k / 25k / 30k | exec 16 | 13 / 14 / 12 trên 20 |
| như trên | 30k | vùng rộng hơn, exec 8 | 11/20 |
| OpenArm cube-in-box | `outputs/smolvla_cube_in_box_isaac/checkpoints/020000` | seed 123 | 18/20 (lưu trữ, không làm tiếp) |

Bài học rút ra: (1) phải fine-tune bằng dữ liệu sinh trong Isaac; (2) `exec-steps` ảnh hưởng ngang việc chọn checkpoint;
(3) tổng quát hoá ra ngoài vùng train còn yếu; (4) 20 episode là quá ít để so sánh.

---

## G0 — Chuẩn hoá đánh giá (1–2 ngày)
- [x] **0.1** Thêm `--results-json PATH` vào `yellowarm_sim.py`: ghi mỗi episode một dòng JSONL
  (`seed`, `success`, chỉ số phụ, số step, thời gian). Giữ nguyên dòng log `evaluation done` như cũ.
- [x] **0.2** Viết `scripts/eval_harness.py` (chạy bằng env python). Đầu vào: `--checkpoint --sim-script --port --episodes
  --seed --exec-steps --tag [extra sim args]`. Việc cần làm: khởi động policy server, chờ port mở, chạy sim headless,
  **tự retry tối đa 2 lần nếu server chết** (`ConnectionResetError` từng làm hỏng một lần eval trước đây), dừng server,
  ghi `results/<tag>.json` gồm cấu hình, từng episode, tỉ lệ thành công, Wilson CI 95%, thời gian chạy, độ trễ infer
  trung bình (đo phía sim).
  - *Xong khi:* chạy được trên checkpoint cube-in-box 30k với 5 episode và tạo ra JSON hợp lệ.
- [x] **0.3** Lập mốc chuẩn: cube-in-box 30k, **50 episode**, seed 1000, exec 8 → `results/baseline_cubebox_30k.json`.

## G1 — Policy server dùng chung cho mọi model (1 ngày)
- [x] **1.1** Tổng quát hoá `policy_server.py`: đọc `config.json` của checkpoint để lấy `type` (smolvla / act /
  diffusion / pi0 / pi05), rồi nạp bằng factory của lerobot. Phải dùng đúng pre/post-processor đã lưu cùng checkpoint.
  Model không nhận câu lệnh (ACT, Diffusion) thì bỏ qua trường `task`. Giữ nguyên giao thức `wire.py`.
- [x] **1.2** Mỗi phản hồi `infer` trả thêm `infer_ms`. Lệnh `info` trả thêm `policy_type`, `n_action_steps`, VRAM đang
  dùng.
- [x] **1.3** Server không được giả định state/action 8 chiều: đọc số chiều từ checkpoint (humanoid sẽ là 16).
  Selftest qua được với SmolVLA (checkpoint 30k). Train smoke một checkpoint ACT 200 step trên
  `datasets/lerobot/yellowarm_cube_in_box`, rồi selftest qua được với checkpoint đó.
  - *Xong khi:* `eval_harness.py` chạy được 5 episode với cả checkpoint SmolVLA lẫn ACT.

## G2 — Humanoid và scene Parts-to-Pallet (4–6 ngày)
Tạo `scripts/humanoid_pallet_sim.py`, phát triển từ `yellowarm_sim.py` (không lấy từ bản xếp chồng). Thêm chức năng DART
(nhiễu quanh quỹ đạo expert); có thể chép phần cài đặt `--dart-noise/--dart-tau` sẵn có trong `yellowarm_stack_sim.py`,
chỉ lấy code, không làm gì thêm với tác vụ xếp chồng.

**Định dạng mới (cố định cho mọi model từ G2 trở đi):**
- State 16 = tay trái (joint1..7 + độ mở gripper 0..1) rồi tay phải (cùng thứ tự). Action 16, cùng thứ tự.
- Camera 256×256: `camera1` = đầu, `camera2` = cổ tay trái, `camera3` = cổ tay phải. Bỏ camera front và side, vì
  humanoid thật không có.
- Đầu cố định, không có khớp cổ.

- [x] **2.0 Thân humanoid (làm trước tiên).** Dựng thân cứng (hình hộp/trụ đơn giản), gắn 2 YellowArm từ USD hiện có
  tại 2 vai. Đưa vào `scripts/data/humanoid.json` các tham số: chiều cao vai so với mặt bàn, khoảng cách hai vai, góc
  gắn tay (thẳng xuống / nằm ngang / nghiêng), khoảng cách từ thân tới mép bàn, vị trí và góc nghiêng camera đầu, FOV.
  Giới hạn khớp, tốc độ và lực lấy đúng theo URDF/thiết kế YellowArm, không nới ra cho dễ.
  - Viết `scripts/humanoid_reach_probe.py`: lấy lưới điểm trên mặt bàn, với mỗi cấu hình thân chạy IK gắp từ trên xuống
    cho từng tay. Xuất bản đồ vùng với tới (điểm nào tay nào gắp được) và ảnh từ camera đầu. Thử ít nhất 3 cấu hình
    gắn tay, chọn cấu hình cho **vùng làm việc chung của hai tay rộng nhất** mà camera đầu vẫn nhìn thấy toàn bộ vùng đó.
  - Ghi cấu hình đã chọn và lý do vào Nhật ký, lưu ảnh vào `logs/humanoid_probe/`. **Dừng lại để người dùng duyệt
    cấu hình trước khi làm tiếp**, vì đây cũng là thông số để lắp robot thật.
  - **Đã duyệt 2026-10-05: forward45**, mặt gắn cao 0.45 m, hai mặt gắn cách 0.55 m, khe thân–bàn 0.05 m, roll 90° cả
    hai tay, thân 0.25 × 0.18 m. Bản đồ vùng với tới của cấu hình này: `scripts/data/humanoid_workspace.npz`.
- [x] **2.1** Thư viện vật thể trong `scripts/data/parts.json`. Hình dạng: `box` (40 mm), `flat_box` (60×40×20),
  `cyl_up` (Ø40×50), `cyl_lying` (Ø30×60), `hex_prism` (dạng đai ốc, mesh), và `long_bar` (200×30×30, dành cho L4).
  Mỗi hình có 3 cỡ (±20%), cạnh kẹp không quá 60 mm. Màu: red, green, blue, yellow, orange, purple. Gắn nhãn
  `train` / `heldout` cho từng màu và hình, ví dụ heldout gồm màu `purple` và hình `hex_prism`.
- [x] **2.2** Pallet là khay lưới 2×3 ô, mỗi ô có thành thấp khoảng 15 mm, đặt trước thân trong vùng hai tay cùng với
  tới. **Pallet chỉ đặt trong vùng chung** (vùng xanh lá của `map_forward45`, tức mặt nạ `shared` trong
  `scripts/data/humanoid_workspace.npz`): phần trong của cả 6 ô phải nằm trong vùng chung. Vật chỉ đặt trong "vùng đặt
  vật" = phần vùng chung mà expert gắp được không phải đảo khuỷu (`humanoid_parts_region.npz`, 0.084/0.138 m²). Vị trí và góc xoay pallet ngẫu nhiên trong
  một vùng nhỏ. Thành công khi tâm vật nằm trong ô đích, vật đã nằm
  yên và không bị lật quá 30° (riêng `cyl_lying` thì kiểm tra theo trục của nó).
- [x] **2.3** Expert: **chọn tay theo bản đồ với tới** (vùng chỉ một tay tới được thì dùng tay đó; vùng chung thì chọn
  tay có cấu hình IK xa giới hạn khớp hơn). Tay không làm việc về tư thế chờ, không đi vào vùng của tay đang làm. Gắp từ trên xuống, kẹp dọc trục ngắn nhất, góc yaw lấy từ tư thế vật (dùng tính đối xứng như `wrap90`), độ
  cao gắp tuỳ hình. Có kiểm tra va chạm giữa hai tay. Chỉ ghi lại episode thành công. Lưu thêm metadata vào npz:
  `level`, danh sách vật (hình, cỡ, màu), ô đích, tay đã dùng.
- [x] **2.4** Các mức độ khó, chọn bằng `--level`. **Vật ở L1/L2 chỉ đặt trong vùng chung** (như pallet). Cắt phạm vi
  2026-10-05: chỉ làm L1 và L2; L3, L4 bỏ (giữ mô tả để tham khảo).
  - **L1:** trên bàn 1 vật ngẫu nhiên, đặt vào ô trống bất kỳ. Câu lệnh: "put the {color} {shape} on the pallet".
  - **L2:** trên bàn 3–4 vật, câu lệnh chỉ định 1 vật. Thành công chỉ khi đúng vật đó vào pallet.
  - ~~**L3:**~~ (bỏ) trên bàn 3 vật, gắp hết lần lượt vào các ô. Câu lệnh: "put all parts on the pallet". Tính số vật đặt đúng
    trên 3.
  - ~~**L4:**~~ (bỏ) việc cần hai tay. Hoặc (a) `long_bar` phải hai tay cùng nâng, hoặc
    (b) vật nằm ở phía một tay nhưng pallet ở ngoài tầm tay đó, nên phải chuyền sang tay kia.
- [x] **2.5** Domain randomization, có cờ bật/tắt: cường độ và hướng ánh sáng, màu/texture mặt bàn, lệch camera đầu
  ±2 cm và ±3° (mô phỏng sai số lắp ráp), ma sát và khối lượng vật. Mặc định bật khi sinh dữ liệu và khi eval.
- [x] **2.6** Thêm tham số `--split train|heldout_color|heldout_shape|wide`. Dùng seed riêng: khi sinh dữ liệu seed lấy
  từ 0 trở lên, khi eval dùng seed cố định 100000–100049.
- [x] **2.7** Tạo `models/smolvla_base_humanoid`: bản `smolvla_base` đổi sang state/action 16 chiều, làm giống cách đã
  tạo `smolvla_base_yellowarm`.
  - *Xong khi:* expert chạy 50 lần thử, tỉ lệ thành công ≥ 90% ở L1 và ≥ 80% ở L2. Lưu ảnh lưới frame đầu
    (`--probe`) của mỗi mức vào `logs/pallet_probe/` để người dùng xem.

## G3 — Sinh dữ liệu (1–2 ngày, chạy qua đêm)
- [x] **3.1** Viết `scripts/pallet_pipeline.py` theo mẫu `stack_dart_pipeline.py`. Chạy 2 worker headless song song,
  mỗi mức sinh 300 episode: 200 episode sạch và 100 episode có DART (noise 0.02–0.03). Sau đó gộp bằng hard link và
  convert sang LeRobot.
- [x] **3.2** Chạy `verify_npz.py` và `verify_dataset.py` trên từng tập. Thống kê độ phủ: số lượng theo hình × màu, phân
  bố vị trí, độ dài episode. Ghi vào `results/dataset_stats.json`.
- ~~**3.3**~~ (bỏ 2026-10-05, cắt phạm vi) Tạo thêm các tập con 50 / 150 episode của L1 để vẽ đường cong theo lượng dữ liệu (dùng
  `--max-episodes` của converter).
  - *Xong khi:* có `datasets/lerobot/yellowarm_pallet_L1`, `_L2` (và bản gộp `_all` = L1+L2). **Mục tiêu: sinh dữ liệu
    xong trong đêm 7/10/2026.**

## G4 — Train ma trận model (khoảng 1 tuần, train lần lượt qua đêm)
Mọi model train trên **cùng một tập** (`_all`, gồm L1+L2). Cắt phạm vi 2026-10-05: chỉ ACT và SmolVLA. Với mỗi lần train ghi lại: số step, thời gian
thực, VRAM đỉnh, loss cuối. Lưu vào `results/train_<model>.json`.
- [x] **4.1 ACT** train từ đầu (`--policy.type=act`), mặc định của lerobot, khoảng 50k–100k step. Là baseline nhanh.
- [x] **4.2 SmolVLA** đi từ `models/smolvla_base_humanoid`, 40k step, batch 8 (cùng công thức như trước).
- ~~**4.3 Diffusion Policy**~~ (bỏ) train từ đầu (`--policy.type=diffusion`).
- ~~**4.4 π0 hoặc π0.5 (lerobot)**~~ (bỏ; 2026-10-10 thử lại theo yêu cầu, dừng ở smoke test — xem Nhật ký): trước hết **smoke test 200 step** để đo VRAM. Nếu hết bộ nhớ (OOM) thì thử
  đóng băng vision encoder/backbone, bật gradient checkpointing, batch 1–2. Nếu vẫn không chạy được trên 16 GB thì
  dừng lại, ghi lý do vào Nhật ký và báo người dùng (phương án khác: thuê GPU cloud). Không tự ý cài thêm thư viện lớn.
- [x] **4.5 GR00T N1.7** (người dùng yêu cầu 2026-10-10): `outputs/groot_pallet_all`, 20k step, LLM + vision + DiT đóng
  băng (chỉ projector embodiment + VL-LayerNorm, 529M tham số), bf16 — xem Nhật ký. Chưa eval (5.1/5.2). Ghi chú cũ: chỉ làm khi người dùng đồng ý. Đây là repo riêng, cần env riêng, có thể phải chạy
  trong WSL. Kiểm tra yêu cầu VRAM trước.
- ~~**4.6**~~ (bỏ) Đường cong theo lượng dữ liệu: train ACT và SmolVLA trên L1 với 50 / 150 / 300 episode.
- [ ] **4.7 GR00T — sửa lỗi cấu hình và mở một phần DiT** (người dùng yêu cầu 2026-10-10, sau chẩn đoán ghi trong Nhật
  ký cùng ngày). Làm lần lượt 4.7.1 → 4.7.6, xong bước nào đánh dấu bước đó. **Không xoá** `outputs/groot_pallet_all`
  (bản cũ, giữ làm mốc so sánh). Ổ D còn khoảng 90 GB, mỗi run GR00T 20k với 4 checkpoint tốn khoảng 43 GB → các run
  mới dùng `--save_freq=10000` (2 checkpoint); trước mỗi run kiểm tra dung lượng trống ≥ 30 GB, thiếu thì dừng và hỏi
  người dùng (không tự xoá checkpoint).
  - [ ] **4.7.1 Vá embodiment id.** Hiện `embodiment_id.json` của checkpoint N1.7 không có key `new_embodiment`, nên
    `embodiment_mapping.get(self.embodiment_tag, 0)` trong `vendor/lerobot/src/lerobot/policies/groot/processor_groot.py`
    âm thầm trả về **id 0** (slot `simpler_env_google`). Sửa **chỉ ở nhánh dựng processor từ base model để train**
    (quanh dòng `embodiment_mapping = (checkpoint_assets.embodiment_mapping ...)`, ~dòng 1157): nếu `embodiment_tag`
    không có trong mapping thì thêm vào với id 10 (`N1_7_EMBODIMENT_MAPPING` dùng 10 cho `new_embodiment`; kiểm tra id
    10 chưa bị key nào khác dùng, nếu bị thì lấy id nhỏ nhất chưa dùng < `max_num_embodiments`=32) và in cảnh báo rõ
    ràng. Không sửa nhánh nạp processor đã lưu: checkpoint cũ `groot_pallet_all` đã train với id 0 nên khi nạp lại
    phải vẫn ra id 0. *Kiểm tra:* (a) nạp preprocessor của `outputs/groot_pallet_all/checkpoints/020000` → id 0;
    (b) dựng processor mới từ base → id 10. Ghi kết quả vào Nhật ký.
  - [ ] **4.7.2 Sửa stats quantile.** `datasets/lerobot/yellowarm_pallet_all/meta/stats.json` có q01/q99 sai (gần bằng
    min/max toàn tập), q10/q50/q90 sai hẳn; ví dụ joint7 tay phải (`action[14]`) ghi q01/q99 = −3.46/2.38 nhưng thật là
    −1.21/2.14, joint6 tay trái (`action[5]`) ghi −1.38/1.54, thật là 1.00/1.54. GR00T normalize min-max theo q01/q99
    (`normalize_min_max=true`, `use_percentiles=true`, `clip_outliers=true`) nên bị ảnh hưởng; ACT/SmolVLA dùng mean/std
    (mean/std đúng) — xác nhận lại bằng cách đọc `normalization_mapping` trong config hai checkpoint đó và ghi vào Nhật ký.
    Việc cần làm: (a) viết `scripts/compute_quantiles.py` (env python, pandas/pyarrow) tính chính xác q01/q10/q50/q90/q99
    cho `action` và `observation.state` trên toàn bộ parquet của `_all` → `results/quantiles_pallet_all.json`;
    (b) thêm trường `quantile_stats_path: str | None = None` vào `GrootConfig` (`configuration_groot.py`); khi dựng
    processor để train, nếu trường này có giá trị thì ghi đè các key q01/q10/q50/q90/q99 của `action` và
    `observation.state` trong stats trước khi đưa vào bước `groot_n1_7_pack_inputs_v1`. **Không sửa
    `meta/stats.json` gốc** của dataset. (c) Không bắt buộc: tìm vì sao lerobot tính quantile sai (gộp L1+L2? gộp theo
    episode?) và ghi một dòng vào Nhật ký. *Kiểm tra:* đọc file `policy_preprocessor_step_2_*.safetensors` và
    `policy_postprocessor_step_0_*.safetensors` của run smoke ở 4.7.3, q01/q99 phải khớp `results/quantiles_pallet_all.json`.
  - [ ] **4.7.3 Train lại GR00T, giữ nguyên công thức cũ + hai bản vá.** Thêm run `groot_pallet_all_fix` vào
    `train_pipeline.py`: giống `groot_pallet_all` (DiT đóng băng, bf16, `model_params_fp32=false`, batch 8, 20k step) +
    `--policy.quantile_stats_path=...` + `--save_freq=10000`. Trước tiên smoke 200 step để kiểm tra 4.7.1(b) và 4.7.2;
    sau đó chạy đủ 20k (khoảng 2.4 h). Ghi `results/train_groot_pallet_all_fix.json`.
  - [ ] **4.7.4 Kiểm tra offline có hệ thống (chưa phải eval).** Viết `scripts/offline_action_check.py` thay cho phép
    thử 12 frame trước đây: ≥ 200 frame cố định (seed cố định, rải đều các episode), so action đầu chunk với action
    dataset; in theo từng chiều: MAE (rad), sai số có dấu (bias), và MAE trong thang đã chuẩn hoá (chia cho
    (q99−q01)/2 lấy từ `results/quantiles_pallet_all.json`). Với GR00T chạy 3 lần lấy mẫu khác seed để đo phần dao động do
    lấy mẫu flow. Chạy cho: ACT 100k, SmolVLA 40k, GR00T cũ 20k, GR00T fix 20k, mốc "giữ nguyên state". Ghi
    `results/offline_check.json`. Không chạy song song với train trên GPU.
  - [ ] **4.7.5 Mở K block cuối của DiT.** **Trước khi chạy:** nhắc người dùng tắt CUDA Sysmem Fallback (NVIDIA Control
    Panel → Manage 3D settings → CUDA - Sysmem Fallback Policy → Prefer No Sysmem Fallback) để tràn VRAM báo lỗi ngay
    thay vì chậm đi hàng chục lần như lần thử 1 (~160 s/step); chờ người dùng xác nhận đã tắt. Thêm trường
    `tune_top_dit_layers: int = 0` vào `GrootConfig`, truyền vào `set_trainable_parameters` trong `groot_n1_7.py`: khi
    `tune_diffusion_model=false` và K > 0 thì mở `requires_grad` cho K phần tử cuối của
    `self.model.transformer_blocks` cùng `proj_out_1`, `proj_out_2` (DiT ở `action_head/cross_attention_dit.py`); in số
    block DiT, số tham số train được. Smoke 50 step cho K = 2, 4, 8 (dừng tăng K khi OOM), batch 8, ghi peak VRAM bằng
    `torch.cuda.max_memory_allocated()` và s/step. Chọn K lớn nhất có peak ≤ 14.5 GB và tốc độ ≥ 1 step/s. Nếu cả K=2
    cũng không vừa batch 8 thì thử batch 4 + `--accelerator.gradient_accumulation.steps=2` (kiểm tra đúng tên cờ trong
    `vendor/lerobot/src/lerobot/configs/train.py`). Train run `groot_pallet_all_dit<K>` 20k step với hai bản vá 4.7.1–4.7.2,
    `--save_freq=10000` → `results/train_groot_pallet_all_dit<K>.json`; chạy lại 4.7.4 cho checkpoint mới.
  - [ ] **4.7.6 Eval chính thức** cho bản GR00T tốt nhất theo 4.7.4 (fix hoặc dit<K>): làm 5.1 (chọn `exec-steps` trên
    seed validation, thử {4, 8, 16, 32}) rồi 5.2 như các model khác. Trong bảng kết quả ghi rõ cấu hình train của GR00T
    (DiT đóng băng / mở K block, số tham số train) vì công thức khác ACT/SmolVLA.
  - *Không làm nếu người dùng chưa đồng ý:* LoRA cho DiT bằng `peft` (lưu ý: `lora_rank`/`lora_full_model` trong
    `GrootConfig` của lerobot 0.6.2 **không được code nào dùng**, đặt giá trị cũng không có tác dụng), hoặc thuê GPU
    cloud ≥ 40 GB để mở toàn bộ DiT.

## G5 — Đánh giá ma trận (2–3 ngày)
- [x] **5.1 Chọn `exec-steps` cho từng model** (số action thực thi từ mỗi chunk trước khi hỏi lại model). Không ép
  chung một giá trị, vì mỗi model có độ dài chunk khác nhau (SmolVLA 50, ACT thường 100, Diffusion ~8–16). Với mỗi
  model, thử các giá trị {4, 8, 16} (thêm 32 nếu chunk của model ≥ 50) ở L1, split `train`, **20 episode trên bộ seed
  validation riêng 200000–200019**, tuyệt đối không dùng seed test. Chọn giá trị có tỉ lệ thành công cao nhất; nếu
  bằng nhau thì chọn giá trị lớn hơn (ít lần gọi model hơn, nhẹ hơn cho phần cứng thật). Ghi lựa chọn vào
  `results/exec_choice.json`. Mỗi lần thử đều đi qua `eval_harness.py`.
- [x] **5.2 Đánh giá chính thức:** với mỗi model × mỗi mức (L1/L2) × mỗi split (`train`,
  `heldout_color`, `heldout_shape`, `wide`): chạy 50 episode trên seed test 100000–100049, dùng `exec-steps` đã chọn ở
  5.1, `--episode-seconds` = 2 × độ dài trung vị episode expert của mức đó. ACT bỏ qua L2 vì không nhận câu lệnh; ghi
  "N/A" vào bảng. Bảng kết quả phải có cột `exec-steps` và độ trễ infer, kèm kiểm tra: với 25 Hz, mỗi
  lần infer phải xong trong `exec-steps / 25` giây, nếu không robot thật sẽ bị khựng.
- [x] **5.3** Lưu video 3 episode tiêu biểu (2 thành công, 1 thất bại) cho mỗi model ở L1 và L2 (`--save-frames`, ghép
  thành mp4).

## G6 — Tổng hợp báo cáo (1 ngày)
- [x] **6.1** Viết `scripts/make_report.py`: đọc `results/*.json`, xuất `results/summary.csv` và các biểu đồ: tỉ lệ thành
  công kèm Wilson CI theo model × mức; tỉ lệ heldout so với in-dist; chi phí (giờ train,
  VRAM, ms/infer).
- [x] **6.2** Viết `results/REPORT.md` tóm tắt kết quả, nêu rõ điều kiện chạy và hạn chế của thí nghiệm (chỉ là mô
  phỏng, số episode, vật thể đơn giản hoá).

## G7 — Quyết định lắp robot thật và chuyển sang robot thật
**Ngưỡng để quyết định lắp** (đề xuất, người dùng chốt lại): model tốt nhất đạt ≥ 80% ở L1 và ≥ 60% ở L2 trên seed
test, **khi bật domain randomization**, và tỉ lệ ở split `wide` không tụt quá một nửa so với in-dist. Độ trễ infer phải
đủ nhanh cho phần cứng sẽ dùng (đo trên Jetson nếu có).
- [ ] **7.1** Xuất "gói thiết kế" từ sim cho việc lắp ráp: cấu hình thân đã chọn ở 2.0 (chiều cao vai, khoảng cách hai
  vai, góc gắn tay, khoảng cách tới bàn), vị trí, góc và FOV của camera đầu, vùng làm việc, kích thước pallet. Lưu vào
  `results/hardware_spec.md`.
  - **Việc còn mở (từ 2.0):** khối lượng khâu đang lấy từ USD, chưa có số CAD thật. Khi có số CAD: chạy lại kiểm tra tải
    (`mass_check`, đã thử ×1.5/×2 bằng numpy: vùng chung giữ được, đỉnh 85% effort ở ×2) **và** kiểm chứng giữ tư thế trong
    Isaac với khối lượng thật trước khi chốt motor/hộp số. Va chạm giữa hai tay được kiểm tra trong expert (2.3).
  - **Việc còn mở (từ 2.4, 2026-10-06):** với forward45 và giới hạn khớp 1 của URDF ([-4.57, 1.43] rad, lệch tâm), ở
    nhánh IK làm việc mỗi tay chỉ với tới cột ô phía mình (tay trái chạm giới hạn j1 ở y ≈ +0.04 m); cột giữa của pallet
    (y ≈ 0) chỉ tới được bằng cách đảo nhánh, mà đường đảo nhánh va thân/bàn. Ghi vào gói thiết kế: hoặc chấp nhận
    (pallet 2 cột ngoài), hoặc xoay vị trí 0 của khớp 1 khi lắp để giới hạn đối xứng hơn quanh tư thế làm việc.
- [ ] **7.2** Tự viết đánh giá trung thực về rủi ro sim → thật: những gì sim đã đơn giản hoá (ma sát, độ cứng gripper,
  độ rơ khớp, độ trễ của camera và servo, ánh sáng thật). Ghi vào cuối `results/REPORT.md`.
- Sau khi lắp (người dùng làm, Claude chỉ hỗ trợ code): đặt camera đúng như gói thiết kế; chạy model tốt nhất ở L1
  qua `YellowArmSim/Jetson/ws_receiver_arm_v3.py`, mở rộng cho 2 tay. Nếu chênh lệch sim–thật lớn: thu 30–50 demo
  thật bằng teleop, rồi fine-tune thêm (sim + thật).
- Claude không gửi lệnh chuyển động tới robot thật khi chưa được người dùng xác nhận.

---
## Nhật ký
<!-- Mỗi dòng: YYYY-MM-DD — task — kết quả/ghi chú -->
- 2026-10-04 — dọn dẹp — xoá checkpoint trung gian, npz thô và zip Isaac (~53 GB). Giữ: cubebox 30k, OpenArm 20k.
- 2026-10-04 — bỏ hẳn tác vụ xếp chồng 3 cube khỏi kế hoạch.
- 2026-10-04 — exec-steps chọn riêng cho từng model trên seed validation 200000+ (mục 5.1), không chọn trên seed test.
- 2026-10-04 — chuyển sang humanoid nửa thân trên (2 YellowArm + camera đầu cố định), state/action 16 chiều. Chưa có
  phần cứng; sim dùng để quyết định có lắp hay không. G0–G1 vẫn làm trên cube-in-box một tay.
- 2026-10-04 — 0.1 — `yellowarm_sim.py --results-json`: mỗi episode 1 dòng JSONL (append, flush) gồm seed, success, steps, seconds, first_in_box_step, cube_lifted, end_dist_to_box, n_infer, infer_ms_mean/max. Eval giờ seed theo episode (seed+i) để harness resume được → không tái lập đúng chuỗi seed 777 cũ. Thử 30k, 2 ep seed 1000 exec 8: 2/2, infer ~260 ms (đo phía sim).
- 2026-10-04 — 0.2 — `scripts/eval_harness.py` xong: khởi động server, chạy sim headless, retry tối đa 2 lần và resume tại seed chưa xong, ghi `results/<tag>.json` (config gồm exec_steps/episode_seconds, từng episode, Wilson 95%, infer ms phía sim). Thử 30k, 5 ep seed 1000–1004, exec 8, cố ý giết server sau ep đầu → tự khởi động lại, 5/5 (Wilson [0.57, 1.0]), infer ~259 ms → `results/test_harness_30k.json` (file thử, không phải mốc chuẩn).
- 2026-10-04 — 0.3 — mốc chuẩn cube-in-box 30k, 50 ep, seed 1000–1049, exec 8, episode 10 s, vùng train mặc định: **36/50 = 72%, Wilson 95% [58%, 83%]**, infer ~256 ms → `results/baseline_cubebox_30k.json`. Thấp hơn 18/20 cũ (20 ep quá ít, khớp bài học 4). 14 ca hỏng: 9 không nhấc được khối, 5 nhấc rồi rơi/đặt trượt. Seed 1000 và 1004 thành công ở lần thử 0.2 nhưng hỏng lần này → policy không tất định (lấy mẫu nhiễu flow-matching), nên cùng seed chỉ cố định layout chứ không cố định kết quả. Ep seed 1014 có infer hơi cao do lỡ chạy selftest GPU song song ~10 s.
- 2026-10-04 — 1.1 — `policy_server.py` đọc `type` từ config.json, nạp lớp qua `get_policy_class` và processor đã lưu qua `make_pre_post_processors`; chỉ gửi `task` khi preprocessor có bước Tokenizer; Diffusion đẩy lịch sử quan sát theo từng request (cách nhau exec-steps bước, khác lúc train). Selftest qua: SmolVLA 30k, `smolvla_base_yellowarm` (8-D và thích ứng 16-D), `smolvla_base` 6→8, ACT khởi tạo ngẫu nhiên (uses_task=False, chunk 100, ~18 ms). **Diffusion chưa thử được: env thiếu gói `diffusers`** — chờ người dùng đồng ý cài (cần cho 4.3).
- 2026-10-04 — 1.2 — `infer` trả thêm `infer_ms` (giữ `ms` cũ); `info` trả `policy_type`, `n_action_steps`, `state_dim/action_dim`, `uses_task`, `vram_gib`, `vram_peak_gib`. Sim ghi `server_infer_ms_mean`; harness lưu `server_info`, `vram_peak_gib`. Thử 2 ep SmolVLA 30k: round-trip 263 ms vs server 261 ms (overhead TCP ~2 ms), VRAM đỉnh 0.9 GiB.
- 2026-10-04 — 1.3 — server lấy state/action dim từ checkpoint (đã thử thích ứng 16-D ở 1.1). Train smoke ACT 200 step trên `yellowarm_cube_in_box` (batch 8, ~13.7 step/s, VRAM train 1.6 GB, loss 15.5→3.7) → `outputs/act_smoke_cubebox/checkpoints/000200`. Harness 5 ep seed 1000–1004 exec 8: ACT 200 step 0/5 (đúng dự kiến, chưa nhấc được khối lần nào; infer ~19 ms, VRAM 0.2 GB) → `results/smoke_act_200.json`; SmolVLA 30k 4/5 (infer ~265 ms) → `results/smoke_smolvla_30k.json`. Ước tính ACT 100k step ≈ 2 h. G1 xong (trừ Diffusion chờ `diffusers`).
- 2026-10-04 — env — người dùng đồng ý: cài `diffusers==0.39.0` (kèm importlib_metadata, zipp; không đổi gói nào khác,
  freeze trước/sau ở `logs/env_freeze_*_diffusers.txt`). Sửa server cho Diffusion (bỏ `action=None` khỏi batch trước khi
  đẩy vào hàng đợi). Selftest Diffusion khởi tạo ngẫu nhiên: chunk 32, **~700 ms/infer, 1.2 GB** (100 bước DDPM mặc
  định) → ở 25 Hz cần exec ≥ 18; khi làm 4.3 cân nhắc DDIM ~10 bước. Giữ lại `results/test_*.json` theo ý người dùng.
- 2026-10-04 — 2.0 (CHỜ DUYỆT) — `scripts/humanoid_kin.py` (FK/IK numpy theo lô, khớp Isaac 0.00 mm), `scripts/data/humanoid.json`,
  `scripts/humanoid_reach_probe.py` (pha A numpy sweep 480 cấu hình: 4 kiểu gắn × roll × cao vai × khoảng cách vai × khe
  thân–bàn; gắp từ trên xuống ở 3 yaw, độ cao gắp 3.5 cm và tiếp cận 15 cm; va chạm theo hộp collider USD từng khâu với
  bàn/thân/thanh vai; mô-men trọng lực ≤ 0.8 × effort URDF. Pha B Isaac: lực URDF + tốc độ moteus thật, giữ 30 tư thế/tay).
  Kết quả (vùng chung 2 tay | kiểm chứng vật lý <10 mm): **forward45 0.141 m² | 60/60 (max 3.4 mm)**; tilt45 0.068 | 59/60;
  down 0.056 | 46/60; lateral (kiểu gắn của USD gốc) 0.015 | 48/48. **Đề xuất forward45**: gốc tay hướng ra trước và chúc
  45°, roll 90° cả hai tay, mặt gắn cao 0.45 m trên mặt bàn, hai mặt gắn cách nhau 0.55 m trên thanh vai, thân 0.25 × 0.18 m,
  khe thân–bàn 0.05 m (bằng 0 cho cùng vùng chung, chọn 5 cm cho an toàn). Vùng chung x 0.22–0.70 m tính từ tâm thân,
  y ±0.25 m; hợp hai tay 0.68 m². Camera đầu cao 0.70 m trên bàn, pitch 60°, vùng chung chỉ cần FOV 34°; với FOV 90° hộp vai
  che hai mép ảnh. Tải tĩnh lớn nhất 42% effort (j1, j2, vật 0.15 kg). Phát hiện: khâu tay rộng 14–19 cm → với kiểu treo, tâm
  mặt gắn phải cách mép thân ≥ 0.115 m; khối lượng khâu lấy từ USD (URDF chỉ có số giữ chỗ) — cần số CAD thật; USD gốc nới lực
  j1 200 Nm, j2–3 100 Nm, sim humanoid dùng 22.8/29.6 Nm. Ảnh/bản đồ: `logs/humanoid_probe/` (map_*.png, *_head_hi.png,
  *_overview.png, reach_results.json, render_report.json). Bản v1 (kiểm tra theo đường tâm, sai) và v2 (thân quá rộng) lưu ở
  thư mục con để đối chiếu.
- 2026-10-05 — 2.0 (vẫn CHỜ DUYỆT) — theo yêu cầu người dùng: sơ đồ 4 kiểu gắn `logs/humanoid_probe/mounts_schematic.png`,
  ghép ảnh Isaac `mounts_isaac_side_by_side.png`. Kiểm tra tải forward45 (h 0.45, sep 0.55, gap 0.05, roll 90) với khối lượng
  khâu USD ×1.5/×2 (vật 0.15 kg giữ nguyên, chỉ mô-men trọng lực tĩnh): vùng chung 0.141 | 0.141 | 0.136 m² (ngưỡng 0.8×effort),
  mô-men cần lớn nhất 44% | 65% | 85% effort (j1/j2), ×2 ở ngưỡng 1.0×effort vẫn 0.141 m². → `logs/humanoid_probe/mass_check/`.
- 2026-10-05 — 2.0 DUYỆT — người dùng chốt forward45 (cao 0.45 m, cách 0.55 m, khe 0.05 m, roll 90°, thân 0.25×0.18 m).
  Ghi `humanoid.json` "selected" (`humanoid_reach_probe.py --phase select`) + `scripts/data/humanoid_workspace.npz`.
  **Sửa lỗi probe:** dấu phía thân của tay phải lấy từ `axis[1]` (= −0.0 với forward45/down) nên tay phải bị kiểm tra
  va chạm thân/thanh vai ở sai phía. Chạy lại: forward45 vẫn đứng đầu, vùng chung 0.141 → **0.1375 m²** (tay trái/phải
  0.404/0.404 m², giờ đối xứng); down 0.056 → 0.046; tilt45, lateral không đổi. Bản cũ: `logs/humanoid_probe/v3_right_sign_bug/`.
  Kiểm tra tải chạy lại: ×1/×1.5/×2 → vùng chung 0.1375/0.1375/0.1319 m², mô-men cần lớn nhất 44/65/85% effort.
- 2026-10-05 — FOV camera đầu — camera ở (0.105, 0, 0.70), pitch 63.5°. Cần 40.1° khi lắp đúng, 50.5° ở trường hợp xấu
  nhất khi lệch ±2 cm/±3° (729 tổ hợp, tính cả góc ô lưới và vật cao 6 cm) → **chọn 55°** trong {45, 55, 65}. Render:
  `logs/humanoid_probe/fov/fov_grid.png` (45° chạm mép ảnh khi lệch, 55° còn lề).
- 2026-10-05 — cắt phạm vi (người dùng, hạn 10/10) — G2 chỉ L1, L2; G4 chỉ ACT, SmolVLA; bỏ L3, L4, Diffusion, π0, GR00T,
  đường cong lượng dữ liệu (3.3, 4.6). Lý do: không đủ thời gian; ưu tiên một so sánh ACT vs SmolVLA đủ số episode và
  đúng quy trình thay vì nhiều model làm dở. Mục tiêu sinh dữ liệu xong đêm 7/10.
- 2026-10-05 — 2.1/2.2/2.3 — `scripts/data/parts.json` (5 hình × 3 cỡ × 6 màu; heldout: purple, hex_prism; bỏ long_bar
  vì L4 bị cắt; orange đổi sang [0.90, 0.25, 0.00] vì bản cũ trông như yellow), `scripts/humanoid_pallet_plan.py` (numpy:
  layout, chọn tay, IK, định thời theo giới hạn tốc độ moteus ×0.85, kiểm tra va chạm 2 tay), `scripts/humanoid_pallet_sim.py`
  (Isaac: `--expert/--probe/--fov-probe`, DART, npz 16-D + `meta`). Pallet 2×3 ô 80 mm, ±1 cm/±5° (lớn nhất để phần trong
  6 ô vẫn trong vùng chung), tâm (0.515, 0.01). Tư thế chờ: hai tay vươn ra góc xa ngoài vùng chung, cách ≥ 6 cm với tay kia
  ở mọi cấu hình làm việc (`humanoid_park.json`). Gắp sát thân (x < 0.40 m) phải đảo khuỷu (6–11 s, quét sang tay kia) nên
  vật chỉ đặt trong "vùng đặt vật" 0.084 m² (61% vùng chung, `humanoid_parts_region.npz`, ảnh
  `logs/pallet_probe/workspace_regions.png`). Lần chạy 1: 48/50 — 2 ca hỏng do transit nội suy khớp không kiểm tra va chạm
  thân ở điểm giữa (tay kẹt); sửa (kiểm tra cả đường, ưu tiên transit Cartesian). **Expert L1 sau sửa: 50/50, Wilson 95%
  [0.93, 1.00]**, seed 0–49, split train, chưa bật domain randomization → `results/expert_L1_50.json` (bản cũ `_v1`).
  Tay trái/phải 25/25, ~12 s/episode (300 frame), ~35 s/lần thử khi không ghi ảnh, ~60 s/episode khi ghi với 2 worker,
  46 MB/episode. Probe: `logs/pallet_probe/probe_L1.png`, `probe_L2.png`. Thử ghi 2 episode: `datasets/raw/test_humanoid_L1`.
  Tóm tắt bằng lời thường:
  1. Robot giờ đặt được vật vào pallet một cách tự động gần như không trượt (50/50 lần ở mức dễ nhất).
  2. Hai tay chia nhau làm: vật bên trái do tay trái gắp, bên phải do tay phải; tay kia chờ ở góc xa, không vướng.
  3. Vùng sát ngực robot khó gắp với kiểu gắn tay đã chọn, nên tạm thời không đặt vật ở đó — robot thật cũng sẽ gặp đúng
     giới hạn này (ghi lại cho gói thiết kế 7.1).
  4. Camera đầu chọn góc nhìn 55°: đủ rộng để vẫn thấy hết vùng làm việc khi lắp lệch vài cm/vài độ.
  5. Một lỗi nhỏ trong công cụ đo vùng với tới đã được sửa; kết luận chọn kiểu gắn không đổi.
- 2026-10-06 — 2.4 — L1/L2 đã có sẵn trong `humanoid_pallet_sim.py --level 1|2` (L2: 3–4 vật, mỗi cặp hình–màu
  xuất hiện tối đa 1 lần nên câu lệnh luôn chỉ đúng 1 vật; thành công khi vật đích vào ô và không vật nào khác nằm trên
  pallet). Selftest numpy L2, 200 bố cục: 157 có plan (15 không xếp được, 28 không có plan → sim tự lấy mẫu lại), trung vị
  289 step (11.6 s). **Expert L2 Isaac: 49/50, Wilson 95% [0.90, 1.00]**, seed 0–49, split train, chưa bật domain
  randomization → `results/expert_L2_50.json`, log `logs/expert_L2_50.log`. 3 vật 21 lần, 4 vật 29 lần; tay phải/trái
  26/24; ~41 s/lần thử. Ca hỏng duy nhất (seed 9, con lăn vàng ở x 0.365, sát mép trong vùng đặt vật, margin khớp 0.03):
  chạy lại với `--debug` (thêm dòng in sau lift/carry/lower) → vật tuột trong pha carry. Ở đây đường Cartesian không khả
  thi nên planner nội suy khớp, gripper không giữ hướng thẳng xuống và hình trụ nằm lăn ra. Không sửa: chỉ ghi episode
  thành công nên lỗi chỉ tốn thời gian sinh dữ liệu. **Điểm cần người dùng biết:** ô đích = ô trống gần vật nhất, mà
  pallet luôn trống lúc đầu → expert gần như chỉ dùng ô 2 (tay trái) và ô 0 (tay phải), hiếm khi ô 3 (L1 25/22/3,
  L2 24/25/1); ô 1, 4, 5 không bao giờ dùng. Hợp lệ theo tiêu chí ("ô trống bất kỳ") và giúp expert nhất quán, nhưng dữ
  liệu không đa dạng về ô đích; đổi (ví dụ đặt sẵn vật vào vài ô, hoặc chọn ô ngẫu nhiên) chỉ khi người dùng muốn.
- 2026-10-06 — 2.4 (bổ sung, theo yêu cầu người dùng) — pallet có sẵn 0–3 vật (đều) ở ô ngẫu nhiên (`--prefill-max 3`;
  khác cặp hình–màu với mọi vật trên bàn nên câu lệnh L2 không mơ hồ; xoay theo trục pallet ±8°, footprint cách thành
  ≥ 3 mm) và ô đích là ô trống ngẫu nhiên (`choose_and_plan(rng=...)`, thử ô kế tiếp theo thứ tự ngẫu nhiên nếu không lập
  được plan). Planner kiểm tra ngón/cổ tay lúc thả không chạm vật có sẵn (`release_clear`). Thành công đòi ô đích **vốn
  trống**; vật có sẵn không tính là "vật khác trên pallet". **Phát hiện:** cột giữa (ô 1, 4) không bao giờ là ô đích —
  giới hạn khớp 1 (xem việc còn mở ở 7.1), không nới. Ô đích thực tế: 0/3 (tay phải), 2/5 (tay trái).
- 2026-10-06 — 2.5 — domain randomization, mặc định bật (`--dr`, tắt bằng `--no-dr` = cảnh danh nghĩa cũ). Khoảng trong
  `scripts/data/domain_rand.json`: dome 500–1800 + đèn mặt trời 0–2500 (cao 30–85°, hướng bất kỳ), nhiệt màu 4000–7500 K;
  mặt bàn màu HSV ngẫu nhiên, 60% có texture (6 texture sinh bằng `scripts/make_table_textures.py`, chiếu theo toạ độ thế
  giới, tỉ lệ/xoay/dịch ngẫu nhiên); camera đầu lệch ±2 cm, ±3° mỗi trục; ma sát từng vật 0.7–1.2 (qua PhysX view lúc chạy)
  và khối lượng ×0.8–1.25 (quán tính theo). Màu vật không đổi (là tên vật trong câu lệnh). DR lấy từ luồng ngẫu nhiên riêng
  (seed, 2025) nên bố cục của một seed giống nhau khi bật/tắt DR; giá trị DR ghi vào `meta["dr"]`. Sửa: khi bật DR lùi khối
  "đầu" (chỉ hiển thị) để camera lệch ra sau không nằm trong khối (ảnh đen ở lần probe đầu). Probe: `logs/pallet_probe/dr/`
  (`probe_L1.png`, `probe_L2.png`).
  **Expert có DR + vật có sẵn + ô ngẫu nhiên, seed 0–49, split train:** L1 **49/50, Wilson 95% [0.90, 1.00]**; L2 **47/50,
  [0.84, 0.98]** → `results/expert_L1_50_dr.json`, `results/expert_L2_50_dr.json` (bản không DR, ô gần nhất giữ ở
  `expert_L*_50.json`). Ô đích L1 0/2/3/5 = 13/14/11/12, L2 13/11/12/14; tay trái/phải 26/24, 25/25; ~65 s/lần thử khi
  chạy 2 tiến trình song song. 4 ca hỏng: 3 là con lăn nằm tuột/lăn khi gắp (vật nằm lại gần chỗ cũ), 1 khối dẹt nhỏ
  (16 mm) lật khi mang với ma sát 0.72 — chạy lại cùng seed với `--debug` thì thành công (vật lý không tất định). Không
  sửa: expert chỉ ghi episode thành công.
- 2026-10-06 — 2.6 — `--split train|heldout_color|heldout_shape|wide` trong `humanoid_pallet_sim.py` /
  `humanoid_pallet_plan.py` (`SPLITS`, `sample_task_parts`, `_wide_region`). **heldout_color / heldout_shape:** chỉ vật
  ĐÍCH mang màu purple / hình hex nut; vật khác trên bàn và vật đặt sẵn trong pallet là vật train (bản đầu lấy toàn bộ
  vật từ tập heldout → L2 treo vì chỉ có 4–5 tổ hợp (hình, màu) khác nhau cho tới 7 vật). **wide:** tâm vật đích nằm
  ngoài vùng đặt vật lúc train (0.084 m²), trong vành ≤ 10 cm quanh nó (`WIDE_RING`), tay nào đó với tới và nằm trọn
  trong ảnh camera đầu với mọi góc lệch lắp của DR (hộp 80×80×50 mm, FOV − 1°): mặt nạ 0.123 m²; vật nhiễu L2 ở vùng train
  hoặc vành; pallet giữ nguyên (đã ở jitter lớn nhất). Planner chỉ có plan cho ~1/3 bố cục `wide` (sim tự lấy mẫu lại):
  bản đồ độ phủ (`logs/plan_wide_coverage.log`) cho thấy vật `wide` thực tế nằm ở hai bên ngoài và góc xa; dải sát thân
  (cần đảo khuỷu) và quanh pallet không bao giờ có plan. **Seed:** `SEED_DATA_END = 100000`, test 100000–100049,
  validation 200000+ (hằng số trong `humanoid_pallet_plan.py`); `--record` từ chối (exit ≠ 0) khi split ≠ train hoặc seed
  ≥ 100000 (đã thử cả hai). Split `train` giữ nguyên chuỗi ngẫu nhiên cũ (seed 0–49 cho cùng bố cục như 2.5).
  **Kiểm tra expert (có DR, seed test 100000–100019, 20 lần thử mỗi bộ, không ghi):** wide L1 **20/20 [0.84, 1.00]**,
  heldout_shape L1 **19/20 [0.76, 0.99]** (1 đai ốc vào đúng ô nhưng nghiêng > 30°), wide L2 **20/20 [0.84, 1.00]**,
  heldout_color L2 **20/20 [0.84, 1.00]** → `results/expert_L{1,2}_20_{split}.json`, log `logs/expert_L*_20_*.log`.
  Probe: `logs/pallet_probe/split_{wide,heldout_color,heldout_shape}/`.
- 2026-10-06 — 2.7 — `models/smolvla_base_humanoid`: cùng cách với `smolvla_base_yellowarm` (bản đó giống hệt
  `smolvla_base` từng byte ở trọng số và processor, chỉ khác `input_features`/`output_features` trong `config.json`):
  chép `model.safetensors` + 4 file processor, `config.json` đổi `observation.state` và `action` sang [16] (max_state_dim /
  max_action_dim = 32 nên vẫn vừa padding; 3 camera camera1..3 giữ nguyên). Selftest `policy_server.py --checkpoint
  models/smolvla_base_humanoid --selftest`: nạp thẳng 16-D (`adapted=False`), chunk (50, 16), ~280 ms/lần infer (lần đầu
  1.0 s), VRAM đỉnh 0.90 GiB → `logs/selftest_smolvla_base_humanoid.log`. Thống kê chuẩn hoá trong processor vẫn là của
  so100 (6-D) như bản yellowarm; khi fine-tune lerobot lấy thống kê từ dataset.
- 2026-10-06 — 3.1 — `scripts/pallet_pipeline.py`: 2 worker Isaac headless (lệch 60 s), mỗi worker 4 job lần lượt, theo
  dõi log (treo 20 phút → kill và chạy lại, tối đa 5 lần), gộp bằng hard link, convert, rồi chạy kiểm tra 3.2. Job (split
  train, có DR, vật có sẵn 0–3): mỗi mức 2 × 100 sạch (seed 0 / 10000, L2: 40000 / 50000) + 50 DART noise 0.02 + 50 DART 0.03
  (seed 20000 / 30000, L2: 60000 / 70000). Sửa thêm trước khi chạy: `--record` resume theo seed lớn nhất trong meta (không
  ghi trùng seed), ghi npz nguyên tử (`.tmp` rồi `os.replace`), **lọc vật tuột**: episode thành công chỉ được ghi khi độ mở
  gripper tay làm việc không xuống dưới 0.05 sau khi kẹp (lúc smoke test DART có 1 ca vật tuột khỏi tay giữa chừng rồi rơi
  đúng vào ô; ghi trường `kept` trong jsonl), meta có `dart_noise`. Chạy 15:06–20:48 (~47 s/episode/worker), không worker
  nào treo hay phải chạy lại. Expert trên mọi lần thử (Wilson 95%): L1 sạch 104/105 [0.95, 1.00] và 104/106 [0.93, 0.99];
  L1 DART 50/51 [0.90, 1.00], 53/54 [0.90, 1.00]; L2 sạch 101/104 [0.92, 0.99], 100/103 [0.92, 0.99]; L2 DART 52/54
  [0.88, 0.99], 51/53 [0.87, 0.99]. 15 lần thành công bị loại vì vật tuột (L1 11, L2 4). Kết quả: `datasets/raw/
  humanoid_pallet_{L1,L2,all}` (hard link tới 8 thư mục job) → `datasets/lerobot/yellowarm_pallet_L1` (300 ep, 89 569
  frame), `_L2` (300 ep, 89 558 frame), `_all` (600 ep, 179 127 frame), robot_type `humanoid_yellowarm_isaac`, ~20 phút/300
  ep khi convert. Converter đặt tên chiều 16-D (`left_joint1..7`, `left_gripper`, … ; action thêm `_target`).
  **Lỗi phát hiện khi kiểm tra và đã sửa:** bộ encode video kiểu streaming của lerobot âm thầm BỎ frame khi hàng đợi
  (mặc định 30 frame) đầy → 5/1800 đoạn video của `_all` (L1 2, L2 2) ngắn hơn episode 1–2 frame, ảnh lệch so với
  state/action (34–43/255 ở frame bị lệch). Sửa `npz_to_lerobot.py` (`encoder_queue_maxsize=4000`, đủ cả một episode),
  `verify_humanoid.py` thêm kiểm tra số frame từng đoạn video (bắt đúng lỗi trên tập cũ), convert lại cả 3 tập: 0 frame bị
  bỏ. Tập cũ KHÔNG xoá, đổi tên `datasets/lerobot/yellowarm_pallet_{L1,L2,all}_dropframes` (chờ người dùng quyết định).
- 2026-10-06 — 3.2 — `verify_npz.py` / `verify_dataset.py` có nhánh humanoid 16-D (`scripts/verify_humanoid.py`; `--lerobot`
  so dataset LeRobot với npz). Cả 3 tập qua hết (exit 0, log `logs/pallet_verify_{npz,dataset}_*.log`): không giá trị
  state/action nào ngoài giới hạn URDF; action là target khớp tuyệt đối (sai lệch theo bám 0.003 rad); mỗi episode đúng 1 tay
  đóng gripper, tay chờ đứng yên (dịch 0.000 rad); gripper kẹp vào vật (độ mở 0.31–0.64, không ca nào kẹp hụt); không
  trajectory/seed trùng; LeRobot khớp số episode/frame, state/action trùng tuyệt đối, ảnh lệch ≤ 3.3/255 (nén video), 0 đoạn
  video sai số frame. Thống kê `scripts/pallet_dataset_stats.py` → `results/dataset_stats.json`, ảnh vị trí
  `logs/pallet_probe/dataset_positions.png`: độ dài episode trung vị 297 frame (11.9 s), 265–517; đủ 20 tổ hợp hình × màu train
  (mỗi tổ hợp 6–24 ep vật đích mỗi mức); hình box/cyl_lying/cyl_up/flat_box = 161/115/185/139, màu blue/green/orange/red/
  yellow = 146/107/111/96/140; cỡ ×0.8/1.0/1.2 đều; ô đích 0/2/3/5 = 191/106/187/116; vật có sẵn 0–3 đều; vị trí vật đích x
  0.36–0.56 m, y ±0.24 m (27 ô lưới 2.5 cm); yaw đều; DR phủ hết khoảng cấu hình, 60% có texture. **Mất cân bằng tay:**
  trái 222 / phải 378 — tay được chọn đúng theo phía vật (y > 0 → tay trái, 100%), nhưng chỉ 37% vật đích nằm bên trái;
  có lẽ do bố cục bên trái ít khi lập được plan hơn nên bị lấy mẫu lại (CHƯA kiểm chứng). Seed test dùng cùng bộ lấy
  mẫu nên so sánh model vẫn công bằng; chưa sửa.
- 2026-10-07 — kiểm chứng mất cân bằng tay: giả thuyết "lấy mẫu lại vì plan bên trái hỏng" **SAI**. Lệch có ngay ở
  `sample_layout`, trước khi lập plan (3000 bố cục numpy: vật đích bên trái L1 36.6%, L2 38.0%, khớp 37% của dataset);
  bố cục bị loại chỉ 182 lần cho 600 episode. Vùng đặt vật `parts` gần đối xứng (68/66 ô). Nguyên nhân: tâm pallet danh
  định `_find_pallet_nominal` = (0.515, **+0.010**) (tâm vùng chung lệch +2.4 mm + lưới 5 mm), pallet rộng 0.264 m theo y,
  vật đích phải cách pallet ≥ 54.5 mm → dải đặt vật bên trái hẹp hơn bên phải ~2 cm trên dải chỉ rộng ~6–8 cm. Rất nhạy:
  tâm pallet y = −0.005 / 0 / +0.005 / +0.010 → trái 61.6% / 52.2% / 43.4% / 36.3%; y = 0 vẫn vừa vùng chung với jitter.
  Chưa sửa (đổi tâm pallet làm dataset hiện tại khác phân phối với seed test).
- 2026-10-08 — 4.1/4.2 — `scripts/train_pipeline.py` trên `yellowarm_pallet_all` (600 ep, giữ nguyên lệch tay 37/63),
  batch 8, exit 0 cả hai. **ACT** (`outputs/act_pallet_all`): 100k step, train 0–20k ngày 2026-10-06 rồi resume 20k–100k
  ngày 2026-10-07 (1.83 h cho đoạn resume, ~12.5 step/s), loss cuối 0.045, mem log 1.6 GB, checkpoint 20k/40k/60k/80k/100k.
  **SmolVLA** (`outputs/smolvla_pallet_all`, từ `models/smolvla_base_humanoid`): 40k step trong 2.43 h (4.56 step/s), loss
  2.52 → 0.027, mem log 3.57 GB, checkpoint mỗi 10k → `results/train_{act,smolvla}_pallet_all.json`. Log cảnh báo
  libtorchcodec không nạp được là vô hại (dataset đọc bằng `pyav`). Chưa eval: `humanoid_pallet_sim.py` chưa có chế độ eval
  policy (`--port/--exec-steps/--episode-seconds`) cho `eval_harness.py`.
- 2026-10-08 — G5 (chuẩn bị) — `humanoid_pallet_sim.py` có chế độ eval policy (`--port`, gọi qua `eval_harness.py
  --sim-script scripts/humanoid_pallet_sim.py ... -- --level N --split S`): mỗi seed cùng bố cục/DR/tư thế đầu như expert,
  policy điều khiển 16 kênh (kẹp giới hạn URDF, ≤ 0.3 rad/bước), chấm bằng `check_success` của expert; jsonl ghi thêm
  nhấc vật, bước vào ô, lệnh gripper nhỏ nhất từng tay, khoảng cách hai tay, tay expert. `--save-video DIR` (JPEG,
  camera tổng quan 768² + 3 camera policy) → `scripts/frames_to_mp4.py`, `scripts/concat_mp4.py`. `scripts/eval_pipeline.py`
  chạy 5.1 + 5.2 không giám sát (2 worker). `--episode-seconds` = 2 × trung vị expert từ `results/dataset_stats.json`
  (L1 23.76 s, L2 23.84 s).
  **5.1 ACT, KHÔNG chống rung** (seed 200000–200019, L1 train): exec 4/8/16/32 → 2, 4, 6, 7 trên 20 → chọn 32
  (`results/val_act_pallet_all_L1_exec*.json`, `results/exec_choice.json` khoá `act_pallet_all`); 5.2 tương ứng đã DỪNG
  giữa chừng (không có kết quả). Phân tích ACT exec 8: 10/20 kẹp hụt, 4 không kẹp, 2 đặt vào ô đã có vật; vật bên trái
  0/6, bên phải 4/14; mọi lần nhấc được đều mang tới ô 3 (expert chọn ô trống ngẫu nhiên, câu lệnh không nói ô → policy
  học một mode). SmolVLA 40k exec 8 cùng 20 seed, không chống rung: 4/20 [0.08, 0.42] (`results/demo_smolvla_L1_val20_exec8.json`,
  chạy cùng lúc với 2 worker ACT nên infer 520 ms; lúc GPU rảnh ~265 ms), 4 ca thành công đều vật bên phải.
  **Rung:** đo offline trên quan sát của dataset (exec 8): độ giật p99 ACT 25–38 mrad, SmolVLA 57–82, expert 3–4; bước nhảy
  ở chỗ nối chunk ACT 27–39, SmolVLA 46–69 mrad. Thêm `scripts/action_smoother.py` (không cần train): crossfade 8 bước ở
  chỗ nối chunk + giới hạn vận tốc/gia tốc khớp kiểu moteus (v_max phần cứng, a_max = 6 v_max, có feed-forward và hãm).
  Offline: p99 giảm 3.5–4.5×, bước nối giảm ~½, lệch so với lệnh 3–10 mrad, action expert đi qua lệch < 15 mrad (0 lúc
  kẹp/thả). Cờ `--smooth` (mặc định tắt). Smoke: ACT seed 200000 exec 32 và SmolVLA seed 200002 exec 8 đều thành công
  (`logs/videos/smooth_test/`). Chạy lại 5.1 + 5.2 cho cả hai model với `--smooth` (tên `*_smooth`, `logs/eval_smooth_chain.cmd`).
- 2026-10-08 — G5 ACT (`--smooth`) — 5.1 exec 4/8/16/32 → 3/6/9/7 trên 20 (raw: 2/4/6/7) → chọn 16. 5.2 L1, 50 ep seed
  test 100000–100049, 23.76 s, exec 16, smooth: train **14/50 [0.17, 0.42]**, heldout_color **13/50 [0.16, 0.40]**,
  heldout_shape **14/50 [0.17, 0.42]**, wide **0/50 [0.00, 0.07]** (50/50 không nhấc được vật: vật ngoài vùng train).
  Theo tay (train): trái 3/18, phải 11/32; lỗi chính là không nhấc được vật (20/50). Infer ~29 ms. Không có 5.2 raw
  của ACT để so (đã dừng). SmolVLA smooth chạy tiếp trong cùng chain.
- 2026-10-09 — G5 SmolVLA (`--smooth`) — 5.1 exec 4/8/16/32 → 2/1/3/1 trên 20 → chọn 16 (infer ~335 ms < 16/25 = 0.64 s).
  5.2 L1 seed test, 23.76 s, exec 16: train **12/50 [0.14, 0.37]**, heldout_color **7/50 [0.07, 0.26]**. Chuỗi bị
  ngắt lúc 01:08 ngày 09/10 (máy tắt/khởi động lại, sim bị kill mã 0x40010004, các lượt L2 sau đó thoát 0xC000026B;
  nguyên nhân chưa xác định) khi heldout_shape/wide mới xong 18/50. Thêm `--resume` cho `eval_harness.py` và
  `eval_pipeline.py` (giữ jsonl của lượt bị ngắt, chỉ dùng khi cấu hình giữ nguyên; đã kiểm tra exec 16, 23.76 s,
  smooth khớp) → chạy tiếp 17:54 (`logs/eval_smolvla_resume.cmd`). 5.3 chuẩn bị ở `logs/g5_videos.cmd` (seed validation).
  Thêm `scripts/expert_review.py` + trang xem 600 video expert `logs/expert_review/index.html`; phát hiện state ≈
  action bước trước (lệch 0–3 mrad) → state đang gần như là lệnh khớp, cần xem lại khi so với robot thật.
- 2026-10-09 — 5.1/5.2 XONG (cả hai model với `--smooth`, exec 16, seed test 100000–100049, 50 ep/ô, Wilson 95%;
  L1 23.76 s, L2 23.84 s). Nguồn: `results/{act,smolvla}_pallet_all_smooth_L*_*.json`.
  | model | L1 train | L1 heldout_color | L1 heldout_shape | L1 wide | L2 train | L2 heldout_color | L2 heldout_shape | L2 wide | infer |
  | ACT | 14/50 [.17,.42] | 13/50 [.16,.40] | 14/50 [.17,.42] | 0/50 [0,.07] | N/A | N/A | N/A | N/A | ~29 ms |
  | SmolVLA | 12/50 [.14,.37] | 7/50 [.07,.26] | 9/50 [.10,.31] | 0/50 [0,.07] | 3/50 [.02,.16] | 6/50 [.06,.24] | 5/50 [.04,.21] | 0/50 [0,.07] | 337–381 ms |
  Ngân sách infer ở exec 16 / 25 Hz = 640 ms → cả hai đạt (SmolVLA đo khi 2 worker dùng chung GPU). Phân tích lỗi:
  wide 0/50 cả hai model, không nhấc được vật lần nào (vật ngoài vùng train). L1 hai model nhấc được 23–33/50 nhưng chỉ
  đặt đạt 7–14 → khâu đặt (low/still/upright/ô trống) là nút cổ chai. SmolVLA L2 đặt **nhầm vật nhiễu** lên pallet
  10/9/6/3 ep (train/color/shape/wide), nhấc đúng vật đích chỉ 9–15/50 → chưa bám câu lệnh. Tay trái kém ở mọi ô
  (vd. SmolVLA L2 trái 1/72 tổng, phải 13/128), khớp lệch dữ liệu 37/63.
- 2026-10-10 — 5.3 XONG — video tiêu biểu (seed validation, `--smooth`, exec 16; 2 thành công + 1 thất bại theo lỗi
  chính của 5.2): `logs/videos/g5_act_L1_representative.mp4` (200004 ✓, 200005 ✓, 200006 ✗ không nhấc được vật),
  `g5_smolvla_L1_representative.mp4` (200022 ✓, 200030 ✓, 200029 ✗ vào đúng ô trống nhưng vật nghiêng > 30°),
  `g5_smolvla_L2_representative.mp4` (200001 ✓, 200017 ✓, 200000 ✗ đặt nhầm vật nhiễu lên ô 0). ACT L2: không có (N/A,
  ACT không nhận câu lệnh). 4 seed SmolVLA L1 chọn từ 5.1 đều trượt khi quay lại (policy ngẫu nhiên) nên quay thêm
  200020–200031 (2/12, `results/demo_g5_smolvla_L1_more.json`); lượt L2 thêm 200010–200025 bị ngắt ở 15/16 (không có
  file results, đủ ca thành công nên không chạy lại; thư mục frame dở `g5_smolvla_L2/seed_200025` để nguyên). Các số
  demo này chỉ để chọn video, không dùng để so sánh model.
- 2026-10-10 — 4.4/4.5 (người dùng yêu cầu train thêm π0.5 và GR00T) — tải `lerobot/pi05_base`, `nvidia/GR00T-N1.7-3B`,
  tokenizer `google/paligemma-3b-pt-224` và processor `nvidia/Cosmos-Reason2-2B` (gated, token HF của người dùng lưu trong
  `models/hf_home`) ; cài peft/dm-tree/timm/decord vào `env` (torch, transformers giữ nguyên). `train_pipeline.py` thêm run
  `pi05_pallet_all`, `groot_pallet_all`, cờ `--runs/--steps`. **Hai lỗi Windows khiến lerobot âm thầm train từ trọng số
  ngẫu nhiên** (log chỉ in "Returning model without loading pretrained weights"): repo id bị đổi thành `lerobot\pi05_base`
  → dùng đường dẫn snapshot cục bộ; π0.5 in "✓" sau khi nạp, stdout cp1252 lỗi → trọng số bị bỏ → ép `PYTHONUTF8=1`,
  pipeline tự kill run nếu gặp dòng đó.
  **π0.5 (dừng, không có model):** smoke 200 step, expert-only + freeze vision, bf16, grad checkpointing, batch 8: VRAM
  đầy 15.9/16 GB, ~4–6 s/step (20k step ≈ 22 h) → người dùng cho dừng ở ~130/200 step.
  **GR00T N1.7 (xong):** lần 1 mặc định lerobot (chỉ đóng băng LLM + vision) còn 1.62B tham số train ở fp32 → tràn RAM,
  ~160 s/step, dừng ở step 11 (log `logs/train_groot_pallet_all_try1_dit.log`). Lần 2 thêm `tune_diffusion_model=false`,
  `model_params_fp32=false` (công thức GPU nhỏ của Isaac-GR00T): 529M tham số train, batch 8, 20k step trong 2.38 h
  (2.33 step/s), mem log 12.7 GB, loss 1.276 → 0.027, exit 0, checkpoint 5k/10k/15k/20k (43 GB) →
  `results/train_groot_pallet_all.json`. **Lưu ý so sánh:** GR00T không train DiT nên không cùng công thức với ACT/SmolVLA
  (train toàn bộ / expert).
  **Kiểm tra offline (chưa phải eval):** 12 frame ngẫu nhiên của `_all`, MAE action đầu chunk so với action dataset:
  GR00T 20k **0.042 rad** (có câu lệnh; 0.051 nếu không), SmolVLA 40k 0.008, ACT 100k 0.008, mốc "giữ nguyên state"
  0.018 → GR00T fit kém hơn hẳn, sai nhiều nhất ở joint7 hai tay (0.13 rad). Infer GR00T ~194 ms (chunk 40).
  `policy_server.py` sửa: GR00T đọc "task" nhưng không có bước "Tokenizer" nên trước đây server không gửi câu lệnh.
- 2026-10-10 — chẩn đoán GR00T (Claude Cowork, chỉ đọc file, không chạy GPU) — hai lỗi cấu hình đã xác nhận, đều nhất
  quán giữa train và infer nên model chạy được nhưng chất lượng kém: (1) embodiment id âm thầm = 0 (slot
  `simpler_env_google`) vì `embodiment_id.json` của N1.7 thiếu `new_embodiment`; (2) q01/q99 trong `meta/stats.json`
  của `_all` sai (≈ min/max), GR00T normalize min-max theo q01/q99 nên khoảng chuẩn hoá quá rộng. Đổi sai số kiểm tra
  offline về thang chuẩn hoá: joint7 ≈ 0.045–0.058, trung bình mọi khớp ≈ 0.045 → model sai đều mọi khớp, joint7 chỉ
  bị khuếch đại vì khoảng rộng nhất (không có lỗi riêng ở joint7). Nguyên nhân chính còn lại nghi là DiT đóng băng.
  `lora_rank` trong `GrootConfig` không được dùng ở đâu. Thêm task 4.7. Backup PLAN cũ: `logs/PLAN.backup_2026-10-10_groot.md`.
- 2026-10-11 — G6 XONG (người dùng yêu cầu làm G6 trước, tạm dừng 4.7) — `scripts/make_report.py` (stdlib, env không có
  matplotlib nên vẽ SVG trực tiếp): đọc eval 5.2, `val_*`, `exec_choice.json`, `train_*`, `expert_*` (JSONL), `dataset_stats.json`
  → `results/summary.csv` (12 ô, kèm phân loại kết cục: vật nhiễu / không nhấc / nhấc không vào ô / vào ô đặt hỏng, và
  thành công theo tay), `summary_train.csv`, `summary_tables.md` (7 bảng, gồm kiểm tra ngưỡng G7), `figures/{success_rates,
  generalization,cost}.svg` (tooltip `<title>`, sáng/tối); chèn bảng vào `results/REPORT.md` giữa `BEGIN/END:tables`.
  Giờ train ACT = 0.47 h (0–20k, tqdm log gốc) + 1.82 h resume = 2.29 h. `results/REPORT.md`: tóm tắt, thiết lập, phân tích,
  hạn chế (sim, vật đơn giản, 50 ep/ô, 1 lần train/model, công thức train khác nhau, độ trễ đo khi 2 worker). Kết luận: chưa
  model nào đạt ngưỡng G7 (L1 28%/24%, L2 6%, wide 0/50); GR00T chưa eval (chờ 4.7). 7.2 (rủi ro sim→thật) chưa viết.
- 2026-10-11 — chống rung SmolVLA (ngoài kế hoạch, theo yêu cầu: video SmolVLA `--smooth` vẫn rung) — đo offline
  `scripts/jitter_offline.py` (10 episode sạch, exec 16, mrad, `results/jitter_offline_smolvla*.json`): rung chủ yếu nằm
  TRONG chunk, không phải ở chỗ nối (đảo chiều vận tốc khớp 7.8/s với `--smooth`, expert 0.01/s). Nhiễu cố định mỗi
  episode: không tác dụng. RTC (lerobot RTCProcessor, guided): đổi ý giữa chunk 17.8 → 3.3, nhưng lệch so với expert
  11.8 → 16.6–21 và infer +90 ms → bỏ. Lọc Savitzky–Golay bậc 2 trong chunk (`action_smoother.py`, numpy, cờ sim
  `--filter-window`): cửa sổ 11 → giật TB 4.42 → 1.21, đảo chiều 7.8 → 0.9/s, lệch 13.5 → 12.8. Vòng kín, seed validation
  200000–200019, L1 train, exec 16, smooth: fw0 4/20 [0.08, 0.42] (giật TB 4.44, đảo chiều 8.9/s), **fw11 7/20 [0.18, 0.57]
  (1.08, 1.1/s)**, fw21 2/20 [0.03, 0.30] (0.65, 0.4/s; quá mượt) → chọn fw11. Sim ghi thêm `jerk_mean_mrad`,
  `jerk_p99_mrad`, `rev_per_s` mỗi episode; server có `--noise fixed`/`--rtc-horizon` (không dùng), harness `--server-args`.
  Đang chạy `logs/fw11_chain.sh`: SmolVLA 8 ô test với fw11 (`smolvla_pallet_all_smooth_fw11_*`), ACT validation fw11
  (`val_act_smooth_fw11_L1`, ACT fw0 = 9/20), rồi video demo mới `logs/videos/fw11_smolvla_L{1,2}`.
- 2026-10-11 — dừng `fw11_chain.sh` theo yêu cầu (chỉ cần video): ACT validation fw11 xong 6/20 (fw0 9/20 → ACT giữ fw0);
  SmolVLA test fw11 dở (L2 train 18/50 episode, chạy tiếp được bằng `--resume`). Video `logs/fw11_videos.sh` (seed
  validation, fw11): L1 2/12 đạt, L2 3/30 đạt → `demo_giam_khao/` chỉ còn video thành công (SmolVLA thay bằng bản fw11).
- 2026-10-12 — thêm mẫu demo SmolVLA L1 fw11 (`logs/fw11_videos_more.sh`, seed 200012–200041): 4/30 đạt → demo L1 có
  6 video đạt (tổng 6/42 episode L1 để quay video).
- 2026-10-12 — `demo_giam_khao/0_thiet_ke_YellowArm/`: 6 ảnh chỉ có cánh tay YellowArm (2 ảnh CAD có sẵn trong
  `YellowArmSim/CAD_export` + `CAD_manufacturing/inspection`, 4 góc nhìn render mới từ mesh URDF bằng
  `scripts/render_yellowarm_views.py`).
