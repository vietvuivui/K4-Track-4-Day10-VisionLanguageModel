# Việc còn lại (Lab Day 10)

Đã xong: E0, E1, E2, E4 (data_size), ảnh minh hoạ, REPORT.md điền bằng số thật (trừ E3).

## 1. Chạy lại trên Colab (~1,5 giờ)
- [ ] Thêm secret `GH_TOKEN` (GitHub token quyền `repo`) — lần trước thiếu nên kết quả chưa lên GitHub.
- [ ] Trong notebook Colab, ô pip (Mục 0) thêm dòng: `!pip -q uninstall -y torchao`
- [ ] **Run all**. Setup đã train xong được bỏ qua, dự đoán đã có được bỏ qua, chỉ E3 chạy thật
      (~50 phút train + ~15 phút suy luận). Các file `report/evidence/*.json` được chấm lại từ dự đoán trên Drive.
- [ ] Kiểm tra ô E3: log in `348 module`, không có `Traceback`, có dòng `E3 [test] ...`.
- [ ] Ô "Lưu kết quả" push được lên GitHub. Dừng ở đây nếu không làm E5 (Runtime → Interrupt).
- [ ] File → Save a copy in GitHub → `lab_day10.ipynb` (notebook có output).

## 2. Hoàn thiện REPORT.md
- [ ] Điền các chỗ `<điền E3 ...>` (câu 1, 7, 8, 10 và bảng nếu chưa tự cập nhật).
- [ ] Nếu E3 tốt hơn E2 rõ rệt (vượt CI95): ghi chú trong câu 10 vì sao E4 vẫn dựa trên E2.
- [ ] Đọc lại toàn bộ, viết lại bằng lời của bạn (có vấn đáp 5 phút).

## 3. Nộp
- [ ] `python scripts/check_submission.py` → 0 FAIL, repo công khai.
