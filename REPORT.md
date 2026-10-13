Họ tên: Nguyễn Văn Quốc Việt
`dataset_version`: <điền, lấy từ data/splits/meta.json>

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Bảng sinh tự động bằng `python -m lab.summarize --write` (đọc thẳng các file bằng chứng).

<!-- TABLE:START -->
| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | | | | | | | | |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |
<!-- TABLE:END -->

Ảnh minh hoạ: `report/figures/` (xanh = box đúng, đỏ = box dự đoán); chú thích ở `report/figures/*_captions.md`.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Số đếm bằng `describe_architecture()` (meta device):

| Khối | Tham số |
|---|---|
| ViT (patch embed + 24 block) | <điền> M |
| `visual.merger` | <điền> M |
| `visual.deepstack_merger_list` (đặc trưng lớp 5, 11, 17) | <điền> M |
| LLM (`language_model`, 36 lớp + embedding) | <điền> B |
| Tổng | <điền> B |

- E1: chỉ LoRA r=16 trên 252 module của LLM (q/k/v/o + gate/up/down × 36 lớp), <điền> tham số huấn luyện; ViT và projector đóng băng.
- E2: E1 + mở đầy đủ merger và deepstack merger (fp32), tổng <điền> tham số (hơn E1 ≈ <điền> M).
- E3: E2 + LoRA trên 96 module của ViT (qkv/proj + linear_fc1/fc2 × 24 block), tổng 348 module, <điền> tham số.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` của E0 = <điền>. Thống kê lỗi trên 311 ảnh test (ô "Câu 2" trong notebook): <điền>.
Hai lỗi phổ biến nhất: <điền>. Ví dụ output thô:
```
<điền>
```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
`score.py --coord pixel` coi toạ độ là pixel và đổi sang 0–1000 bằng cách nhân x với 1000/640 ≈ 1,56 và y với 1000/360 ≈ 2,78. Nếu model trả thang 0–1000 mà ta chấm như pixel, mọi box bị kéo giãn và đẩy xuống phía dưới/bên phải, nên IoU với box đúng gần như luôn < 0,5. JSON vẫn hợp lệ (`parse_valid_rate` không đổi) nhưng F1 sụp về gần 0. Đây là lỗi "âm thầm": nhìn tỉ lệ JSON hợp lệ không phát hiện được.
Kiểm chứng: chấm lại đúng file dự đoán của E1 với `--coord pixel` → F1 = <điền> (so với <điền> khi chấm đúng thang). Ngoài ra, test `test_round_trip_to_pixel` và ô kiểm tra bước 2 cho sai số quy ngược lớn nhất <điền> px trên toàn bộ train.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Trong một ví dụ huấn luyện có <điền> token, trong đó 220 token ảnh, chỉ <điền> token là đáp án (JSON + `<|im_end|>`). Prompt và token ảnh là **đầu vào** mà model luôn được cho sẵn lúc suy luận; điều ta cần học là p(đáp án | ảnh, prompt).
Nếu tính loss trên cả chuỗi: (1) phần lớn gradient đến từ việc đoán lại prompt cố định và các token `<|image_pad|>` vốn là placeholder, nên loss giảm rất nhanh nhưng không học được grounding; tín hiệu từ toạ độ bị pha loãng khoảng <điền> lần; (2) model có thể học "nói lại" prompt/chat template thay vì trả JSON, làm hỏng `parse_valid_rate`; (3) loss báo cáo không còn phản ánh chất lượng định vị.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Khi lượng tử hoá NF4, trọng số Linear được lưu dạng `Params4bit` (số nguyên 4-bit đóng gói + hằng số scale), không phải tensor số thực khả vi: không có `.grad`, và optimizer không thể cộng một bước cập nhật nhỏ vào mã 4-bit. Trên lớp 4-bit chỉ huấn luyện được adapter phụ (LoRA), không huấn luyện đầy đủ được. Merger nằm trong `visual`, nên mình dùng `llm_int8_skip_modules=["visual", "lm_head"]` để giữ toàn bộ ViT và projector ở fp16, rồi đưa riêng tham số projector lên fp32 và đặt `requires_grad=True`.
Cần fp32 vì: (a) fp16 chỉ có khoảng 3 chữ số thập phân, nên bước cập nhật lr·Adam ≈ 2e-4 cộng vào trọng số cỡ 1e-2 thường bị làm tròn mất; (b) `GradScaler.unscale_` từ chối gradient fp16 ("Attempting to unscale FP16 gradients"). Nhờ vậy E2 có thêm <điền> M tham số huấn luyện và VRAM đỉnh tăng từ <điền> lên <điền> GB.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Ảnh là frame cắt từ video, nên frame liền kề gần như trùng nhau (cùng ổ gà, cùng góc nhìn). Nếu chia ngẫu nhiên, gần như mọi ảnh test đều có "anh em sinh đôi" trong train: model chỉ cần nhớ, và F1 test đo khả năng nhớ chứ không đo khả năng tổng quát sang đoạn đường mới. Chia theo phiên (ngày + giờ) buộc test là những đoạn đường model chưa từng thấy. Mình đã kiểm tra: không phiên nào xuất hiện ở hai split.

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | <điền> | <điền> | <điền> | <điền> | <điền> |
| val | <điền> | <điền> | <điền> | <điền> | <điền> |
| test | <điền> | <điền> | <điền> | <điền> | <điền> |

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền: so F1 và CI95 của E1/E2/E3, chi phí VRAM và phút tăng thêm; kết luận setup nào đáng nhất, có vượt CI95 hay không>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 là bootstrap theo ảnh (lấy mẫu lại 311 ảnh, 1000 lần) do `score.py` tính. Độ rộng CI của các setup: E1 <điền>, E2 <điền>, E3 <điền> (khoảng ±<điền> điểm F1).
<điền: chênh lệch F1 giữa các setup so với độ rộng CI; các CI có chồng nhau không; kết luận nào còn đứng vững (vd. "fine-tune > zero-shot" chắc chắn), kết luận nào chưa (vd. E2 vs E3)>. Thêm vào đó, test chỉ gồm vài phiên quay, nên các ảnh trong cùng phiên tương quan với nhau; CI bootstrap theo ảnh còn **lạc quan** hơn thực tế.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền: F1 từng lớp; diện tích trung vị và recall theo kích thước box (ô phân tích trong notebook); số mẫu train từng lớp; nhầm lẫn ngữ nghĩa (vd. vết nứt dài mảnh, ranh giới mơ hồ khiến IoU khó đạt 0,5)>

Ảnh minh hoạ: <điền: chèn ít nhất 6 ảnh từ report/figures/ kèm chú thích>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền: biến (vd. data_size 10% vs 100% trên setup tốt nhất), F1 và CI95 hai giá trị, các thứ khác giữ nguyên, kết luận có vượt CI95 không>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
