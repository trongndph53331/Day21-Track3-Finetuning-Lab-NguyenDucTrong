# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Trọng · **MSSV**: PH53331 · **Ngày**: 21/08/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4, 14.6 GB`

> Số liệu lấy trực tiếp từ `results/`; đánh giá dùng đủ 50 mẫu target và 15 mẫu regression, không đặt `EVAL_LIMIT`.

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 là 98; mức gợi ý tối thiểu 256, cấu hình T4 giữ 1024 để tương thích tier và có dư địa |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch / 30 bước mỗi run |

**Template có giữ khối `<think>` không?** Có. Artifact xác nhận thẻ mở và nội dung reasoning còn nguyên, nên template an toàn để train trên trace.

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.7578 | 0.000 | 3380.5 |
| (b) base + optimized prompt | 0.765 | 0.7578 | 1.000 | 1076.1 |
| (c) LoRA fine-tune | 0.965 | 0.5222 | 1.000 | 1494.8 |

**(b) có thật sự mạnh hơn (a) không?** Có: target tăng từ 0.000 lên 0.765, format từ 0 lên 1, đồng thời latency giảm khoảng 68%. Tôi không sửa `OPTIMIZED_PROMPT`; hash `719e74d3b6232053` khớp prompt chuẩn. Vì vậy fine-tune được so với một baseline prompt mạnh, không phải đối chứng bị làm yếu.

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | 0.965 | 959.9 | 12.01 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | 0.5364 | 0.965 | 845.4 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5704 | 0.000 | 999.0 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 1104.2 | 7.09 |

**4.1 — Rank so với vị trí adapter.** `attn_only` hòa `correct` trên target ở 0.965, dù số tham số gần như bằng nhau (chênh khoảng 0.025%). Thứ tự theo train loss lại khác: `attn_only` có loss thấp hơn, 0.5364 so với 0.6263, nhưng không tạo thêm điểm target. Trên bài toán hẹp này, tăng rank lên 283 có thể bù phần nào cho phạm vi module hẹp; loss thấp hơn không chứng minh tổng quát tốt hơn, và phép đo không cho phép kết luận rằng chỉ vị trí hoặc chỉ rank luôn thắng.

**4.2 — Learning rate sai.** `wrong_lr` chỉ giảm LR từ 1e-4 xuống 1e-5 nhưng kết thúc ở loss 1.5704, cao hơn nhiều 0.6263 của cấu hình đúng, và target rơi về 0 với format bằng 0. Ở cùng 30 bước, LR kiểu full fine-tuning khiến LoRA học quá chậm và chưa sinh JSON đúng. Nếu chỉ nhìn loss mà không biết LR và ngân sách bước, tôi có thể kết luận sai rằng LoRA hoặc dữ liệu không học được; thực tế biến gây lỗi là bước cập nhật quá nhỏ.

**4.3 — QLoRA.** QLoRA giảm peak VRAM từ 12.01 xuống 7.09 GB, tiết kiệm **40.97%**. Giá phải trả là target giảm 0.025, loss tăng lên 0.7058, train tăng 144.3 giây và latency suy luận tăng 368.9 ms. T4 vẫn chứa được fp16 ở 12.01 GB, nên số đo ủng hộ khuyến nghị không dùng QLoRA cho run chính; tuy vậy, nó vẫn hợp lý khi giới hạn bộ nhớ quan trọng hơn mức giảm nhỏ về chất lượng và tốc độ.

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.200` · `regression Δ = -0.236` · `valid_trace_rate = 0.00`

Fine-tune cải thiện rõ tác vụ chính: target tăng từ 0.765 lên 0.965 và giữ format rate 1.0. Tuy nhiên, regression giảm từ 0.7578 xuống 0.5222, mất 0.2356 điểm, vượt xa dung sai 0.02. Vì thế FAILED là đúng và không nên nới ngưỡng để làm đẹp kết quả. Tập 250 ticket chuyên biệt đã kéo model quá mạnh về phân phối JSON triage, gây quên năng lực chung. `valid_trace_rate=0` không phải bằng chứng duy nhất vì corpus mục tiêu chủ yếu cần JSON ngắn, nhưng nó cho thấy run chưa giữ reasoning trace hợp lệ. Hướng sửa là trộn 1–5% replay data tổng quát, giữ nguyên eval đóng băng, rồi đo lại target và regression. Chỉ khi target vẫn tăng và regression nằm trong dung sai thì model mới vượt cổng triển khai.

## 6. Định tính — có cả ca thua

NB5 lưu prediction từng mẫu của fine-tune nhưng không lưu nguyên văn prediction (b); cột (b) ghi điểm aggregate 0.765 thay vì bịa output không có trong artifact.

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Trả chuột không dây, “đã quá hạn” | `doi_tra/cao/chuột không dây/tich_cuc` | aggregate 0.765 | đúng 4/4 (1.00) | ✅ FT thắng |
| 2 | Hoàn tiền ốp lưng, “sớm nhé” | `hoan_tien/trung_binh/ốp lưng điện thoại/tieu_cuc` | aggregate 0.765 | đúng 4/4 (1.00) | ✅ FT thắng |
| 3 | Bình giữ nhiệt, “chưa thấy tiền” | `hoan_tien/thap/bình giữ nhiệt/tich_cuc` | aggregate 0.765 | urgency `trung_binh` (0.75) | ❌ **FT thua 1 trường** |
| 4 | Nồi chiên thiếu phụ kiện | `san_pham_loi/thap/nồi chiên không dầu/trung_tinh` | aggregate 0.765 | urgency `trung_binh` (0.75) | ❌ **FT thua 1 trường** |
| 5 | Áo khoác bị lỗi, “khi nào tiện” | `san_pham_loi/thap/áo khoác gió/tich_cuc` | aggregate 0.765 | urgency `trung_binh` (0.75) | ❌ **FT thua 1 trường** |

Các ca thua có mẫu chung: intent, product và sentiment đều đúng, nhưng model đẩy `urgency=thap` lên `trung_binh`. Những câu lịch sự hoặc sự cố nhẹ như “chưa thấy tiền”, “thiếu phụ kiện”, “khi nào tiện” chưa đủ để suy ra mức khẩn cao hơn. Đây là lỗi hiệu chỉnh ranh giới urgency, không phải lỗi parse JSON hay nhận diện loại yêu cầu.

## 7. Kết luận & điều tôi học được

Tôi **không deploy** adapter này ở trạng thái hiện tại. Nó rất tốt trên tác vụ ticket: tăng 0.200 điểm so với prompt tối ưu, tạo JSON hợp lệ 100%, và đạt target 0.965. Dù vậy, chỉ nhìn target sẽ che giấu mức mất 0.2356 điểm regression, trong khi cổng chỉ cho phép giảm 0.02. Đòn bẩy mạnh nhất quan sát được là learning rate: đổi 1e-4 thành 1e-5 làm target từ 0.965 về 0 và phá format trong cùng 30 bước. Mask cũng là điều kiện nền tảng vì nó giữ 58.51% token prompt ngoài loss, nhưng proof cho thấy mask hiện tại đúng. Vị trí adapter không khác biệt target khi rank được nâng để khớp ngân sách, còn QLoRA chủ yếu đổi chất lượng và thời gian để lấy 40.97% VRAM. Vấn đề triển khai còn lại là độ phủ dữ liệu: tập chuyên biệt giúp triage nhưng gây catastrophic forgetting. Bước tiếp theo là thêm 1–5% replay data tổng quát, bổ sung ví dụ phân biệt `thap` và `trung_binh`, train lại với LR 1e-4, rồi giữ nguyên cổng đóng băng để tránh tự đánh lừa bằng cách chỉnh metric sau khi thấy kết quả.

**Ba điều tôi học được**:

1. Prompt tối ưu tự nó tăng target 0.765 và giảm latency mạnh; fine-tune phải so với (b), không phải baseline naive.
2. Với LoRA 30 bước, LR 1e-5 quá nhỏ: target 0 và format 0 dù các cấu hình khác giống run đúng.
3. Tiết kiệm 4.92 GB bằng QLoRA không miễn phí: target giảm 0.025, train và inference đều chậm hơn trên T4.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1%, 3%, 5% replay data, oversample mẫu urgency thấp, giữ seed/eval/30 bước cố định, rồi chọn checkpoint theo cổng target + regression thay vì train loss.

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
