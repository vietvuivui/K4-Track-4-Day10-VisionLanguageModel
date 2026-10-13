# Việc bạn cần tự làm (Lab Day 10)

Mã đã viết xong trong `lab/` và `lab_day10.ipynb`. Phần còn lại cần **GPU Colab T4** và **số liệu thật của bạn**.

## A. Chuẩn bị (≈ 15 phút)
- [ ] `git push` repo lên GitHub (để Colab clone được). Repo phải **công khai** khi nộp.
- [ ] Tạo Colab Secrets `KAGGLE_API_TOKEN` (token `KGAT_...`) và `GH_TOKEN` (GitHub token quyền `repo`).
- [ ] Mở `lab_day10.ipynb` trên Colab → Runtime **T4 GPU** → chạy mục 0–1 (mount Drive, clone, pip, pytest, prepare_data).
  - Nếu `REPO_URL` không phải repo của bạn: sửa ở ô đầu tiên.
  - Nếu gặp `ImportError ... Torchvision`: `pip install torchvision` cùng bản torch của Colab.

## B. Chạy thí nghiệm (≈ 3 giờ GPU, chạy lần lượt các ô)
- [ ] Mục 2 + ô kiến trúc: ghi lại số tham số từng khối (câu 1).
- [ ] **E0**: predict + score → `report/evidence/e0.json`; chạy ô phân tích lỗi (câu 2).
- [ ] Mục 4: xác nhận **220 token ảnh** và phần có loss giải mã ra đúng JSON + `<|im_end|>`.
- [ ] **E1**: train → gỡ lỗi trên val 40 ảnh → chỉ khi ổn mới predict/score test.
  - Log train phải in `252 module`. E1 cần `parse_valid_rate ≥ 0.95` và F1 hơn E0 ≥ 0.10.
- [ ] **E2**, **E3**: log phải in 252 / 348 module; E2 nhiều hơn E1 ≈ 109M tham số.
- [ ] Ô "Câu 3": chấm lại E1 với `--coord pixel` để có số cho câu 3.
- [ ] **E4**: đặt `BEST = 'e1'|'e2'|'e3'` theo bảng, rồi chạy ô E4 (mặc định data_size 10% vs 100%).
- [ ] Ô ảnh minh hoạ → `report/figures/` (≥ 6 ảnh) + ô phân tích theo lớp (câu 9).
- [ ] (Tuỳ chọn +10) **E5 GRPO**. `lab/grpo.py` chưa chạy thử được trên trl 1.9.2: nếu `GRPOConfig` báo tham số lạ, sửa theo docs đúng bản.
- Mất session? Chạy lại đúng ô đó: `train --resume` tiếp tục từ checkpoint 100 bước gần nhất, `predict` bỏ qua ảnh đã làm. Dữ liệu cần chạy lại `prepare_data` (nhanh).

## C. Đưa kết quả về repo
- [ ] Trên Colab: `!python -m lab.summarize --write` (bảng tự điền vào REPORT.md).
- [ ] Commit + push từ Colab (hoặc tải về máy): `report/evidence/*.json`, `report/figures/*`, `REPORT.md`, notebook **có output**.
  - **Không** commit `data/`, `work/`, adapter, `*.pt`/`*.safetensors` (đã có trong `.gitignore`).

## D. Viết báo cáo (`REPORT.md`, 30 điểm)
- [ ] Điền `dataset_version` và mọi chỗ `<điền ...>` bằng **số của bạn** (các câu 1, 2, 3, 6 đã có sẵn phần giải thích, chỉ thiếu số).
- [ ] Tự viết câu 7, 8, 9, 10 và "Nhìn lại" dựa trên kết quả (nhìn CI95 trước khi kết luận).
- [ ] **Viết lại bằng lời của bạn**: giảng viên đối chiếu văn bản giữa các repo và có vấn đáp 5 phút. Đọc hiểu `lab/vlm.py` (mask loss, apply_setup, lưu/nạp projector).
- [ ] Chèn ≥ 6 ảnh minh hoạ có chú thích vào câu 9.

## E. Trước khi nộp
- [ ] `python scripts/check_submission.py` → 0 FAIL (cần ≥ 5 commit, đủ e0–e3 + e4, REPORT không còn `<điền`).
- [ ] Không sửa gì trong `scripts/` và `tests/` (sửa bị trừ 10 điểm).
- [ ] Xoá file `TODO.md` này nếu muốn.
