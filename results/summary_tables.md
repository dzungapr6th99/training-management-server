<!-- sinh tự động bởi scripts/make_report.py — không sửa tay -->
## Bảng 1 — Tỉ lệ thành công (seed test, Wilson 95%)

| model | exec | chunk | L1 in-dist | L1 heldout màu | L1 heldout hình | L1 wide | L2 in-dist | L2 heldout màu | L2 heldout hình | L2 wide |
|---|---|---|---|---|---|---|---|---|---|---|
| ACT | 16 | 100 | 14/50 (28%) [0.17, 0.42] | 13/50 (26%) [0.16, 0.40] | 14/50 (28%) [0.17, 0.42] | 0/50 (0%) [0.00, 0.07] | N/A | N/A | N/A | N/A |
| SmolVLA | 16 | 50 | 12/50 (24%) [0.14, 0.37] | 7/50 (14%) [0.07, 0.26] | 9/50 (18%) [0.10, 0.31] | 0/50 (0%) [0.00, 0.07] | 3/50 (6%) [0.02, 0.16] | 6/50 (12%) [0.06, 0.24] | 5/50 (10%) [0.04, 0.21] | 0/50 (0%) [0.00, 0.07] |

Seed: 100000–100049; episode-seconds: L1 23.76 s, L2 23.84 s; chống rung (`--smooth`): có; mọi ô hoàn tất: có.
N/A = không chạy (ACT không nhận câu lệnh nên không có L2).

## Bảng 2 — Tổng quát hoá: tỉ lệ split / tỉ lệ in-dist cùng mức

| model | mức | in-dist | heldout màu | heldout hình | wide |
|---|---|---|---|---|---|
| ACT | L1 | 28% | 26% (×0.93) | 28% (×1.00) | 0% (×0.00) |
| SmolVLA | L1 | 24% | 14% (×0.58) | 18% (×0.75) | 0% (×0.00) |
| SmolVLA | L2 | 6% | 12% (×2.00) | 10% (×1.67) | 0% (×0.00) |

Tỉ số tính trên điểm ước lượng; với k nhỏ (≤ 15/50) khoảng tin cậy của tỉ số rất rộng — đọc kèm Bảng 1.

## Bảng 3 — Phân loại kết cục (mỗi episode đúng 1 loại)

| model | mức | split | thành công | vật nhiễu lên pallet | không nhấc được vật đích | nhấc nhưng không vào ô | vào ô nhưng đặt hỏng | nhấc được vật đích | thành công tay trái | tay phải |
|---|---|---|---|---|---|---|---|---|---|---|
| ACT | L1 | in-dist | 14 | 0 | 20 | 6 | 10 | 30/50 | 3/18 | 11/32 |
| ACT | L1 | heldout màu | 13 | 0 | 17 | 4 | 16 | 33/50 | 5/14 | 8/36 |
| ACT | L1 | heldout hình | 14 | 0 | 27 | 5 | 4 | 23/50 | 3/18 | 11/32 |
| ACT | L1 | wide | 0 | 0 | 50 | 0 | 0 | 0/50 | 0/29 | 0/21 |
| SmolVLA | L1 | in-dist | 12 | 0 | 21 | 9 | 8 | 29/50 | 5/18 | 7/32 |
| SmolVLA | L1 | heldout màu | 7 | 0 | 25 | 5 | 13 | 25/50 | 3/14 | 4/36 |
| SmolVLA | L1 | heldout hình | 9 | 0 | 27 | 4 | 10 | 23/50 | 3/18 | 6/32 |
| SmolVLA | L1 | wide | 0 | 0 | 50 | 0 | 0 | 0/50 | 0/29 | 0/21 |
| SmolVLA | L2 | in-dist | 3 | 10 | 29 | 2 | 6 | 11/50 | 0/18 | 3/32 |
| SmolVLA | L2 | heldout màu | 6 | 9 | 28 | 1 | 6 | 15/50 | 1/16 | 5/34 |
| SmolVLA | L2 | heldout hình | 5 | 6 | 35 | 2 | 2 | 9/50 | 0/14 | 5/36 |
| SmolVLA | L2 | wide | 0 | 3 | 47 | 0 | 0 | 0/50 | 0/24 | 0/26 |

Thứ tự xét: thành công → có vật khác trên pallet → không nhấc vật đích (> 3 cm) → chưa từng vào ô → còn lại (vào ô nhưng nghiêng/chưa yên/ô đã có vật/không còn trong ô lúc kết thúc).

## Bảng 4 — Chi phí và độ trễ

| run | model | khởi tạo | step | batch | giờ train | step/s | VRAM train (log, GB) | loss đầu → cuối | ghi chú |
|---|---|---|---|---|---|---|---|---|---|
| act_pallet_all | ACT | scratch (lerobot defaults) | 100000 | 8 | 2.29 | 12.11 | 1.60 | 0.109 → 0.0454 | 2 đoạn: 0–20345 (0.47 h) + resume (1.82 h) |
| smolvla_pallet_all | SmolVLA | models/smolvla_base_humanoid | 40000 | 8 | 2.43 | 4.56 | 3.57 | 2.516 → 0.0272 |  |
| groot_pallet_all | GR00T N1.7 | nvidia/GR00T-N1.7-3B (LLM + vision + DiT frozen, bf16) | 20000 | 8 | 2.38 | 2.33 | 12.73 | 1.276 → 0.027 |  |
| pi05_pallet_all_smoke | π0.5 | lerobot/pi05_base (expert-only fine-tune) | 200 | 8 | 0.03 | 2.25 | – | – | không hoàn tất (exit 1) |

| model | exec | ngân sách exec/25 Hz | infer server TB (ms) | infer phía sim TB (ms) | infer phía sim max (ms) | đạt ngân sách (TB) | VRAM infer đỉnh (GiB) |
|---|---|---|---|---|---|---|---|
| ACT | 16 | 640 ms | 26 | 29 | 318 | có | 0.22 |
| SmolVLA | 16 | 640 ms | 355 | 358 | 1297 | có (max vượt) | 0.91 |

Độ trễ là trung bình có trọng số theo episode trên mọi ô của Bảng 1; đo khi 2 worker eval dùng chung GPU (RTX 4070 Ti SUPER) nên cao hơn lúc GPU rảnh. VRAM train là số `mem` lerobot in ra trong log, không phải đỉnh `nvidia-smi`.

## Bảng 5 — Chọn exec-steps (seed validation, L1 in-dist, 20 ep)

| run | exec 4 | exec 8 | exec 16 | exec 32 | chọn |
|---|---|---|---|---|---|
| act_pallet_all | 2/20 | 4/20 | 6/20 | 7/20 | 32 |
| act_pallet_all_smooth (--smooth) | 3/20 | 6/20 | 9/20 | 7/20 | 16 |
| smolvla_pallet_all_smooth (--smooth) | 2/20 | 1/20 | 3/20 | 1/20 | 16 |

Run không có `--smooth` chỉ để tham khảo (5.2 tương ứng đã dừng giữa chừng, không có kết quả).

## Bảng 6 — Expert lập trình sẵn (tham chiếu trần, KHÔNG cùng bộ seed với Bảng 1)

| file | mức | split | DR | seed | thành công |
|---|---|---|---|---|---|
| expert_L1_20_heldout_shape | L1 | heldout_shape | có | 100000–100019 | 19/20 [0.76, 0.99] |
| expert_L1_20_wide | L1 | wide | có | 100000–100019 | 20/20 [0.84, 1.00] |
| expert_L1_50 | L1 | train | không | 0–49 | 50/50 [0.93, 1.00] |
| expert_L1_50_dr | L1 | train | có | 0–49 | 49/50 [0.90, 1.00] |
| expert_L2_20_heldout_color | L2 | heldout_color | có | 100000–100019 | 20/20 [0.84, 1.00] |
| expert_L2_20_wide | L2 | wide | có | 100000–100019 | 20/20 [0.84, 1.00] |
| expert_L2_50 | L2 | train | không | 0–49 | 49/50 [0.90, 1.00] |
| expert_L2_50_dr | L2 | train | có | 0–49 | 47/50 [0.84, 0.98] |

Dataset `_all`: 600 episode, 179127 frame; tay trái/phải 222/378.

## Bảng 7 — Ngưỡng G7 (đề xuất trong PLAN.md)

| model | L1 in-dist ≥ 80% | L2 in-dist ≥ 60% | wide ≥ ½ in-dist (L1) | infer TB ≤ ngân sách |
|---|---|---|---|---|
| ACT | không (28%) | N/A | không (0% vs 28%) | ĐẠT (29 ≤ 640 ms) |
| SmolVLA | không (24%) | không (6%) | không (0% vs 24%) | ĐẠT (358 ≤ 640 ms) |
