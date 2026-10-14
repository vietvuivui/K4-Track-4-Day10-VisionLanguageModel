Họ tên: Nguyễn Văn Quốc Việt
`dataset_version`: d10-21e4070301

Môi trường: Google Colab; transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2. Mã: gói `lab/` + `lab_day10.ipynb` (notebook kèm output của lần chạy cuối).

**GPU từng setup** (do giới hạn quota Colab, lần chạy chính dùng hai loại GPU):

| Setup | GPU train | Ghi chú |
|---|---|---|
| E0 | T4 | chỉ suy luận |
| E1 | T4 | 41,4 phút |
| E2 | T4 (bước 1–100, 14,1 phút) → A100 (bước 101–300, 12,1 phút) | tiếp tục từ checkpoint bước 100; tổng 26,2 phút là thời gian **lẫn hai GPU** |
| E3 | A100 | 22,8 phút |
| E4 (10 %) | A100 | 18,0 phút |

Vì vậy cột "Phút" **không so sánh trực tiếp** được giữa các setup; VRAM đỉnh (`torch.cuda.max_memory_allocated`) gần như không phụ thuộc loại GPU nên so sánh được. So sánh tốc độ công bằng ở câu 7 dùng số phút trên **cùng một GPU**.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Bảng sinh tự động bằng `python -m lab.summarize --write` (đọc thẳng các file bằng chứng). VRAM/phút lấy từ `meta` (= `run.json` của lần train).

<!-- TABLE:START -->
| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole | parse_valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — | 0.095 | [0.066, 0.127] | 0.91 | 0.129 | 0.032 | 0.200 | 0.929 |
| E1 | LoRA LLM | 33,030,144 | 6.32 | 41.4 | 0.288 | [0.253, 0.328] | 1.05 | 0.201 | 0.209 | 0.528 | 0.990 |
| E2 | + projector | 142,135,296 | 8.14 | 26.2 | 0.332 | [0.294, 0.375] | 1.19 | 0.193 | 0.274 | 0.583 | 0.997 |
| E3 | + LoRA ViT | 148,426,752 | 8.11 | 22.8 | 0.448 | [0.407, 0.488] | 0.88 | 0.364 | 0.349 | 0.702 | 1.000 |
| E4 data_size=0.1 | data_size = 0.1 | 142,135,296 | 7.59 | 18.0 | 0.173 | [0.147, 0.202] | 2.00 | 0.099 | 0.108 | 0.431 | 0.997 |
| E4 data_size=1.0 | data_size = 1.0 | 142,135,296 | 8.14 | 26.2 | 0.332 | [0.294, 0.375] | 1.19 | 0.193 | 0.274 | 0.583 | 0.997 |
<!-- TABLE:END -->

`E4 data_size=1.0` chính là E2 (cùng file dự đoán), chấm lại dưới tên E4 để có đủ hai giá trị của biến.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Đếm bằng `describe_architecture()` (dựng model trên meta device, không tải trọng số):

| Khối | Tham số |
|---|---|
| ViT (patch embed + 24 block) | 306,2 M |
| `visual.merger` | 27,3 M |
| `visual.deepstack_merger_list` (3 merger cho đặc trưng lớp 5, 11, 17) | 81,8 M |
| → toàn bộ phần thị giác | 415,3 M |
| LLM (`language_model`: 36 lớp + embedding 151.936 × 2560) | 4.022,5 M |
| Tổng (lm_head **dùng chung** trọng số với embedding) | ≈ 4.437,8 M ≈ 4,44 B |

Bộ đếm còn in `lm_head 389,0 M` và tổng 4.826,8 M vì model dựng trên meta device chưa buộc (tie) `lm_head` với embedding; config có `tie_word_embeddings=True` nên tổng thật phải trừ 389 M.

- **E1**: LoRA r=16 trên 252 module của LLM (q/k/v/o + gate/up/down × 36 lớp) = **33.030.144** tham số. ViT và projector đóng băng.
- **E2**: E1 + mở đầy đủ `visual.merger` và `visual.deepstack_merger_list` (fp32) = **142.135.296** tham số, hơn E1 đúng **109.105.152** (= 27,3 M + 81,8 M).
- **E3**: E2 + LoRA trên 96 module của ViT (qkv/proj + linear_fc1/fc2 × 24 block), tổng 348 module = **148.426.752** tham số; LoRA ViT chỉ thêm 6.291.456 tham số.

(Số "tổng" ~2,65 B in trong log train nhỏ hơn 4,44 B vì trọng số NF4 đóng gói 2 giá trị/byte nên `numel()` chỉ còn một nửa.)

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` của E0 = **0,929** (22/311 output không parse được). Model gốc không bọc ```` ```json ````, không bịa lớp ngoài 3 lớp, không trả toạ độ > 1000: nó **hiểu đúng định dạng và thang 0–1000**. Ví dụ output thô (ảnh `vlcsnap-00200`):
```
[{"label": "manhole", "bbox_2d": [759, 640, 843, 687]}]
```
Hai lỗi phổ biến nhất:
1. **Định vị sai / box bịa (FP)**: 326 box dự đoán nhưng precision chỉ 0,132; model đoán 205 box crack mà F1 crack chỉ 0,032. Điển hình là một box "crack" rất dẹt kéo ngang cả khung hình, hoặc nằm trên nắp capo xe (`report/figures/e0_sai_1.jpg`, `e0_sai_2.jpg`). Model gốc không biết quy ước ghi nhãn của bộ dữ liệu này (crack được khoanh theo vùng).
2. **Bỏ sót (FN)**: recall 0,074; trung bình chỉ 1,05 box/ảnh trong khi test có 583/311 = 1,87 box/ảnh, 8 ảnh trả `[]`. Ổ gà nhỏ gần như bị bỏ qua (F1 pothole 0,129).
Ngoài ra 7,1 % output không parse được, khiến cả ảnh bị tính như không dự đoán.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
`score.py --coord pixel` coi toạ độ là pixel và đổi sang 0–1000 bằng cách nhân x với 1000/640 ≈ 1,56 và y với 1000/360 ≈ 2,78. Nếu model trả thang 0–1000 mà ta chấm như pixel, mọi box bị kéo giãn và đẩy xuống dưới/sang phải, nhiều box vượt khỏi khung, nên IoU với box đúng gần như luôn < 0,5. JSON vẫn hợp lệ nên đây là lỗi "âm thầm".

Kiểm chứng: chấm lại **đúng file dự đoán của E1** với `--coord pixel` → F1 = **0,000** (P = R = 0, meanIoU = 0), trong khi `parse_valid_rate` vẫn 0,990 và F1 khi chấm đúng thang là **0,288**. FP/ảnh tăng từ 1,05 lên 1,54 vì mọi box đều thành FP. Chiều ngược lại được kiểm tra khi dựng nhãn: quy box JSON 0–1000 ngược về pixel và vẽ lại trùng box YOLO gốc; sai số lớn nhất trên toàn train là 5,00 px — lớn hơn sai số làm tròn (≤ 0,64 px theo x) do một số box YOLO tràn khỏi mép ảnh và bị kẹp về [0, 1000].

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Một ví dụ huấn luyện (ảnh 9 box) dài **588 token**, trong đó **220 token ảnh** và chỉ **250 token đáp án** (JSON + `<|im_end|>\n`); 338 token còn lại (57 %) là chat template, prompt và ảnh. Mình kiểm tra bằng cách giải mã riêng các token có `labels != -100`: ra đúng chuỗi JSON đáp án + `<|im_end|>`.

Prompt và token ảnh là **đầu vào** model luôn được cho sẵn lúc suy luận; mục tiêu cần học là p(đáp án | ảnh, prompt). Nếu tính loss trên cả chuỗi: (1) hơn một nửa gradient dùng để đoán lại prompt cố định (giống hệt nhau ở mọi ví dụ, loss về ~0 rất nhanh) và các `<|image_pad|>` vốn chỉ là placeholder, nên tín hiệu từ toạ độ bị pha loãng; ảnh ít box thì đáp án chỉ vài chục token, tỉ lệ còn tệ hơn; (2) model có thể học "nói lại" template/prompt thay vì trả JSON; (3) loss trung bình giảm nhanh nhưng không phản ánh chất lượng định vị nên khó dùng để gỡ lỗi.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Lớp NF4 lưu trọng số dạng `Params4bit`: số nguyên 4-bit đóng gói (2 giá trị/byte) + hằng số scale. Đó không phải tensor số thực khả vi: không có `.grad` có nghĩa, optimizer không thể cộng một bước cập nhật nhỏ vào mã 4-bit. Trên lớp 4-bit chỉ huấn luyện được adapter (LoRA) gắn thêm.

Mình gặp đúng lỗi này: với transformers 5.19, `llm_int8_skip_modules=["visual", "lm_head"]` **không có tác dụng** — lần chạy thử đầu in `projector 54.579.200`, đúng một nửa 109 M, tức merger đã bị đóng gói 4-bit. Xử lý: ghi thêm tên đầy đủ `"model.visual"` vào danh sách bỏ qua, và sau khi nạp thì kiểm tra không còn `Linear4bit` nào trong `visual` (nếu còn thì giải lượng tử về fp16). Sau khi sửa, log in `projector 109.105.152`.

Cần fp32 vì: (a) fp16 chỉ có ~3 chữ số thập phân, bước cập nhật cỡ lr·Adam ≈ 2e-4 cộng vào trọng số cỡ 1e-2 thường bị làm tròn mất; (b) `GradScaler.unscale_` từ chối gradient fp16 — lần chạy đầu của mình báo đúng lỗi `Attempting to unscale FP16 gradients` vì LoRA được tạo ở fp16, phải đưa mọi tham số huấn luyện lên fp32. Cái giá: VRAM đỉnh tăng từ 6,32 GB (E1) lên 8,14 GB (E2), +1,82 GB — khớp ước lượng 109 M × 4 byte × 4 (trọng số fp32, gradient, hai moment AdamW) ≈ 1,75 GB.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Ảnh là frame cắt từ video, frame liền kề gần như trùng nhau (cùng ổ gà, cùng góc nhìn). Chia ngẫu nhiên thì gần như mọi ảnh test đều có "anh em sinh đôi" trong train: model chỉ cần nhớ, F1 test đo khả năng nhớ chứ không đo khả năng tổng quát sang đoạn đường mới. Khả năng học thuộc rất mạnh thấy rõ ở E4: với 142 ảnh, loss train về ~0,005–0,015 mà F1 test chỉ 0,173 — nếu test chứa frame gần trùng với train, con số này sẽ bị thổi phồng. Chia theo phiên (ngày + giờ trong tên file) buộc test là các đoạn đường chưa từng thấy. Mình đã kiểm tra: không phiên nào xuất hiện ở hai split.

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1.422 | 13 | 1.074 | 1.750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

Phân bố lớp lệch giữa các split (val chỉ 66 pothole), vì cả phiên đi cùng nhau.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
| | F1 | CI95 | FP/ảnh | VRAM | Tốc độ train (cùng GPU) |
|---|---|---|---|---|---|
| E1 | 0,288 | [0,253; 0,328] | 1,05 | 6,32 GB | T4: 41,4 phút/300 bước (8,3 s/bước) |
| E2 | 0,332 | [0,294; 0,375] | 1,19 | 8,14 GB | T4: 14,1 phút/100 bước (8,5 s/bước); A100: 12,1 phút/200 bước (3,6 s/bước) |
| E3 | **0,448** | [0,407; 0,488] | **0,88** | 8,11 GB | A100: 22,8 phút/300 bước (4,6 s/bước) |

- **Mở projector (E1 → E2)**: F1 +0,044, crack và manhole tăng (+0,065 / +0,055) nhưng pothole không (0,201 → 0,193), FP/ảnh còn tăng nhẹ. Hai CI95 chồng nhau nhiều ([0,253; 0,328] và [0,294; 0,375]), và ở lần chạy thử trước chênh lệch là +0,076 (xem câu 8) → **có xu hướng tốt hơn nhưng chưa chắc chắn**. Chi phí: +109 M tham số, +1,82 GB VRAM, tốc độ gần như không đổi trên cùng T4 (8,5 so với 8,3 s/bước).
- **Thêm LoRA ViT (E2 → E3)**: F1 +0,116, **cả ba lớp đều tăng mạnh** (pothole 0,193 → 0,364, crack 0,274 → 0,349, manhole 0,583 → 0,702), FP/ảnh giảm 1,19 → 0,88. Hai CI95 **không chồng nhau** (0,375 < 0,407) → cải thiện thật. Chi phí rất nhỏ: chỉ +6,3 M tham số, VRAM gần như không đổi (8,11 so với 8,14 GB — nhờ gradient checkpointing, gradient qua ViT không phải giữ activation), chậm hơn ~26 % mỗi bước trên cùng A100 (4,6 so với 3,6 s/bước) vì phải lan truyền ngược qua 24 block ViT.

Kết luận: **E3 đáng đồng tiền nhất**. Ổ gà nhỏ và vết nứt là đặc trưng thị giác chi tiết mà ViT tiền huấn luyện không chú ý tới; projector chỉ "dịch" lại đặc trưng có sẵn, còn LoRA trên ViT cho phép bộ mã hoá ảnh học **nhìn** ra chúng — đó là lý do pothole (lớp nhỏ nhất) tăng gần gấp đôi chỉ ở E3. Trong quá trình train E3 có một bước gradient tràn số fp16 (`grad_norm=nan` ở bước 230); GradScaler tự bỏ qua bước đó và giảm scale, loss vẫn hữu hạn.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 là bootstrap theo ảnh (lấy mẫu lại 311 ảnh, 1000 lần) do `score.py` tính. Độ rộng: E0 0,061; E1 0,075; E2 0,081; E3 0,081; E4 (10 %) 0,055 — tức khoảng **±0,04 điểm F1**.

Bằng chứng thực tế về độ dao động: mình đã chạy E1 và E2 **hai lần** với cùng config và seed 42 (lần đầu bị gián đoạn vì lỗi E3 nên chạy lại toàn bộ). Kết quả: E1 0,283 → 0,288; E2 0,359 → 0,332. Chỉ riêng tính không tất định của GPU (và E2 lần 2 chạy nửa T4 nửa A100) đã làm F1 lệch tới **0,027** — cùng cỡ nửa độ rộng CI. Lần 1 E2 hơn E1 0,076, lần 2 chỉ hơn 0,044.

Hệ quả:
- **Vững**: fine-tune > zero-shot (0,095 → ≥ 0,288); E3 > E2 (CI không chồng nhau, chênh 0,116 ≫ dao động giữa các lần chạy); 10 % < 100 % dữ liệu (0,173 vs 0,332, CI cách xa).
- **Chưa vững**: E2 > E1. Chênh lệch 0,044–0,076 tuỳ lần chạy, CI chồng nhau; cần nhiều seed hoặc bootstrap **ghép cặp** (lấy mẫu cùng tập ảnh cho hai setup rồi xét phân bố của hiệu) mới kết luận được.
- CI bootstrap theo ảnh còn **lạc quan**: test chỉ có 3 phiên quay, ảnh cùng phiên tương quan mạnh nên số mẫu độc lập thực sự ít hơn 311 nhiều; đổi sang phiên khác, F1 có thể dịch hơn ±0,04.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
F1 từng lớp: E2 pothole **0,193** / crack 0,274 / manhole 0,583; E3 pothole **0,364** / crack 0,349 / manhole 0,702. Recall theo diện tích box (tỉ lệ khung hình) trên test, đo với E2 (setup được chọn làm `BEST` khi chạy, xem câu 10):

| Lớp | Diện tích trung vị | Nhỏ < 0,5 % | Vừa 0,5–2 % | Lớn ≥ 2 % | Box train |
|---|---|---|---|---|---|
| pothole | 0,42 % | 13/68 (19 %) | 8/46 (17 %) | 2/7 | 1.074 |
| crack | 1,98 % | 1/15 (7 %) | 31/136 (23 %) | 54/148 (36 %) | 1.750 |
| manhole | 0,67 % | 20/64 (31 %) | 60/96 (63 %) | 1/3 | 691 |

- **Pothole kém nhất** (ở E1, E2) dù có 1.074 box train, nhiều hơn manhole: box nhỏ nhất (trung vị 0,42 % khung hình, 68/121 box < 0,5 %), trong khi ảnh 640×360 chỉ có 220 token — mỗi token ứng một ô 32×32 px, một ổ gà nhỏ chỉ chiếm 1–2 token. Ngữ nghĩa cũng mơ hồ: ổ gà thường nằm lẫn trong vùng nứt, model nhầm giữa pothole và crack (`e2_sai_2.jpg`, `e2_sai_3.jpg`). Đây đúng là lớp hưởng lợi nhiều nhất khi mở ViT (E3: 0,193 → 0,364).
- **Crack**: box lớn (trung vị 1,98 %) nhưng ranh giới vùng nứt do người gán nhãn tự khoanh, không có biên rõ; model khoanh một vùng rộng hơn/hẹp hơn là IoU < 0,5. Crack nhỏ gần như không bắt được (1/15).
- **Manhole tốt nhất**: hình dạng rõ, tương phản cao, kích thước ổn định → recall 63 % ở cỡ vừa, kể cả ở xa (`e2_dung_1.jpg`).
- Một lỗi đáng chú ý: ở ảnh nhiều hư hại, model **lặp lại** cùng một box "pothole" thành một cột 7 box gần trùng nhau (`e2_sai_1.jpg`: TP=0, FP=9, FN=7) — dạng lặp token của sinh tự hồi quy, mỗi box lặp đều thành FP.

Ảnh minh hoạ (xanh = box đúng, đỏ = box dự đoán; dòng dưới ảnh ghi TP/FP/FN):

| | |
|---|---|
| ![](report/figures/e2_dung_1.jpg) E2 đúng: 2 manhole (TP=2, FP=0, FN=0) | ![](report/figures/e2_dung_2.jpg) E2 đúng (TP/FP/FN dưới ảnh) |
| ![](report/figures/e2_dung_3.jpg) E2 đúng (TP/FP/FN dưới ảnh) | ![](report/figures/e2_sai_1.jpg) E2 sai: box crack trùm cả vùng + cột pothole lặp lại (TP=0, FP=9, FN=7) |
| ![](report/figures/e2_sai_2.jpg) E2 sai: nhầm pothole/crack trong vùng hư hại dày (TP=1, FP=6, FN=5) | ![](report/figures/e2_sai_3.jpg) E2 sai: box crack quá rộng, pothole lệch (TP=0, FP=5, FN=6) |
| ![](report/figures/e0_sai_1.jpg) E0 sai: box crack dẹt kéo ngang cả ảnh | ![](report/figures/e0_sai_2.jpg) E0 sai: box "crack" nằm trên capo xe |

Đường loss E1/E2/E3: `report/figures/loss_e1_e2_e3.png`.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Biến: **`data_size`**, chỉ đổi `train_fraction` của setup E2: 10 % (142 ảnh) và 100 % (1.422 ảnh). Mọi thứ khác giữ nguyên: r=16, lr 2e-4, 300 bước × 8 ảnh, seed 42. E4 dựa trên E2 vì khi cấu hình E4, E2 là setup tốt nhất đã có; E3 hoàn thành sau và tốt hơn, nhưng hết quota GPU để chạy lại E4 trên E3. Vì chỉ đổi đúng một biến so với E2, so sánh vẫn sạch; kết luận áp dụng cho setup E2.

| | F1 | CI95 | P | R | FP/ảnh | loss train cuối |
|---|---|---|---|---|---|---|
| 10 % | 0,173 | [0,147; 0,202] | 0,155 | 0,196 | 2,00 | 0,005–0,015 |
| 100 % | 0,332 | [0,294; 0,375] | 0,339 | 0,326 | 1,19 | ≈ 0,5–0,6 |

Với 10 %, 300 bước × 8 ảnh = 2.400 lượt ≈ 17 epoch trên 142 ảnh: loss train từ 0,47 (bước 100) về 0,005 (bước 290) — học thuộc — nhưng F1 test chỉ bằng một nửa và FP/ảnh gần gấp đôi: model "nhớ" bố cục box của ảnh train và vẽ nhiều box bịa. Hai CI95 cách xa nhau (0,202 < 0,294), và kết quả lặp lại gần như y hệt ở cả hai lần chạy (lần 1: 0,172; lần 2: 0,173), nên kết luận "dữ liệu nhiều hơn giúp rõ rệt" là **đáng tin** dù test chỉ 311 ảnh. Hạn chế: chỉ hai điểm (10 %, 100 %) và một seed, nên không nói được đường cong theo kích thước dữ liệu bão hoà ở đâu; và chưa kiểm chứng trên E3.

## Nhìn lại

- **Lỗi công cụ chấm không bắt được**: với transformers 5.19, `llm_int8_skip_modules=["visual"]` bị bỏ qua âm thầm — ViT và projector bị lượng tử hoá 4-bit mà không có cảnh báo. Nếu không đối chiếu số tham số projector (54,6 M thay vì 109 M) thì E0 chạy với ViT 4-bit, E2/E3 "huấn luyện" trên trọng số đóng gói và kết quả sai mà `score.py` vẫn chấm bình thường. Bài học: luôn in và đối chiếu số tham số với con số kỳ vọng.
- E3 lần đầu lỗi do môi trường (`torchao` 0.10 có sẵn trên Colab không tương thích peft khi gắn LoRA vào Linear fp16 của ViT); gỡ `torchao` là chạy được.
- Chạy lại cùng seed vẫn cho F1 lệch tới 0,027: kết quả một lần chạy không đủ để so các setup chênh nhau ít.
- Tốc độ: A100 train nhanh hơn T4 ~2,3 lần (3,6 so với 8,5 s/bước ở E2), nhưng **suy luận gần như không nhanh hơn** (~14 phút/311 ảnh trên cả hai): sinh tự hồi quy từng token với batch 4 và giải lượng tử NF4 bị giới hạn bởi độ trễ, không phải sức tính.
- Nhãn crack mơ hồ: nhiều ảnh khoanh cả vùng nứt lớn thành một box, ảnh khác khoanh từng vết; với ngưỡng IoU 0,5, F1 crack bị giới hạn bởi độ nhất quán của nhãn hơn là năng lực model.
- Điều chưa chắc: CI bootstrap theo ảnh có quá hẹp không khi test chỉ có 3 phiên; muốn chắc nên bootstrap theo phiên hoặc thêm phiên vào test. E5 (GRPO) chưa chạy vì hết quota GPU; hàm reward và test của nó có trong `lab/reward.py`, `lab/tests/test_lab_reward.py`.
