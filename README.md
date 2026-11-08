# Topic 6 Object Detection Robustness

> Thành viên mới nên bắt đầu tại [PROJECT_GUIDE.md](PROJECT_GUIDE.md). Tài liệu này
> mô tả thứ tự đọc project, vai trò từng file, cách tái lập thí nghiệm và dàn ý báo cáo/slide.

Dự án đánh giá mức suy giảm của mô hình phát hiện vật thể khi ảnh bị Gaussian noise, Salt-and-Pepper noise, sương mù tổng hợp và thiếu sáng. Pipeline giữ nguyên annotation COCO, tạo ảnh suy giảm trong bộ nhớ, chạy detector và tính COCO mAP.

## Thiết kế thực nghiệm

- Baseline: ảnh sạch, model và tham số inference cố định.
- Corruption: 4 loại, mỗi loại 5 severity.
- Nhiễu ngẫu nhiên: lặp lại bằng nhiều seed xác định.
- Recovery: preprocessing riêng theo từng loại suy giảm.
- Metric: mAP@0.5:0.95, mAP@0.5, mAP@0.75, AP small/medium/large, confusion matrix và latency.

## Cài đặt

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Nếu máy có GPU NVIDIA, nên cài PyTorch theo cấu hình CUDA phù hợp từ hướng dẫn chính thức trước khi cài project.

## Tải COCO val2017

Lệnh sau tải ảnh validation và annotation trực tiếp từ máy chủ COCO:

```powershell
.\.venv\Scripts\topic6.exe download --root data/coco
```

Dung lượng tải và giải nén vào khoảng hơn 1 GB. Archive được xóa sau khi giải nén; thêm `--keep-archives` nếu cần giữ lại.

## Kiểm thử nhanh

Smoke test chỉ chạy 8 ảnh, một severity và một repeat:

```powershell
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\topic6.exe preview --config configs/smoke_test.yaml --severity 3
.\.venv\Scripts\topic6.exe benchmark --config configs/smoke_test.yaml
.\.venv\Scripts\topic6.exe plot --metrics runs/smoke_test/metrics.csv
```

## Chạy benchmark đầy đủ

Nên chạy pilot 200 ảnh trước:

```powershell
.\.venv\Scripts\topic6.exe benchmark --config configs/pilot.yaml
.\.venv\Scripts\topic6.exe plot --metrics runs/coco_pilot/metrics.csv
```

Sau khi pilot ổn định, dùng cấu hình cuối cùng. Cấu hình này dùng một holdout 1.000 ảnh,
severity 1/3/5 và chỉ chạy recovery ở các mức đã có tín hiệu từ pilot. Cấu hình
`coco_val2017.yaml` vẫn giữ sẵn cho phép đo mở rộng trên toàn bộ 5.000 ảnh:

```powershell
.\.venv\Scripts\topic6.exe benchmark --config configs/coco_final.yaml
.\.venv\Scripts\topic6.exe plot --metrics runs/coco_topic6_final_1000/metrics.csv
```

Mặc định dùng `yolo11n.pt`. Có thể đổi `weights` trong YAML thành `yolov8n.pt`, model lớn hơn hoặc đường dẫn tới checkpoint cục bộ. Không đổi threshold giữa các condition nếu muốn so sánh công bằng.

Các cấu hình đi kèm đang dùng `device: 0` cho GPU NVIDIA. Đổi thành `device: cpu` nếu chạy trên máy không có CUDA.

Thiết lập Ultralytics được lưu trong `.ultralytics/` của project để môi trường chạy không cần quyền ghi vào roaming profile của Windows.

## Kết quả

Mỗi condition tạo:

- `predictions.json`: dự đoán theo định dạng COCO.
- `metrics.json`: metric tổng hợp.
- `class_ap.json`: AP theo lớp.
- `confusion_matrix.csv`: ma trận gồm 80 lớp và background.
- `coco_eval.txt`: báo cáo COCOeval.

`metrics.csv` tách riêng thời gian tạo corruption, recovery, inference và end-to-end.
Lệnh `plot` tạo thêm đường cong recovery gain và end-to-end latency; các số latency
không bao gồm thời gian đọc ảnh từ đĩa hay COCOeval.

File `metrics.csv` ở thư mục run là đầu vào cho script vẽ đường cong mAP theo severity.

## Cấu trúc

```text
configs/                    Cấu hình thí nghiệm
docs/SOURCES.md             Nguồn dữ liệu và nghiên cứu
src/topic6_robustness/      Mã nguồn chính
tests/                      Unit tests
data/                       Dataset, không commit
runs/                       Kết quả thực nghiệm, không commit
```

## Lưu ý nghiên cứu

- Không fine-tune model trước phép đo baseline robustness.
- Không thay đổi annotation khi tạo corruption.
- Báo cáo trung bình và độ lệch chuẩn cho phép biến đổi ngẫu nhiên.
- Đánh giá recovery cả về mAP và thời gian xử lý; ảnh đẹp hơn chưa chắc detector tốt hơn.

