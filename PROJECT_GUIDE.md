# Hướng dẫn đọc và sử dụng dự án Topic 6

## 1. Mục tiêu dự án

Dự án trả lời Topic 6: **Robustness of Object Detection Under Environmental
Noise & Synthetic Distortions**. Mô hình YOLO11n được đánh giá trên COCO val2017
khi ảnh bị Gaussian noise, Salt-and-Pepper noise, fog tổng hợp và low-light.

Câu hỏi cần trả lời trong báo cáo là:

1. Mỗi loại suy giảm làm mAP giảm như thế nào khi severity tăng?
2. Preprocessing có thể phục hồi detection hay không?
3. Recovery làm đổi false negative, false positive và thời gian xử lý như thế nào?
4. Vật thể nhỏ, trung bình và lớn chịu tác động khác nhau ra sao?

## 2. Đọc theo thứ tự này

1. Đọc `README.md` để cài môi trường và chạy lệnh cơ bản.
2. Đọc `docs/SOURCES.md` và `docs/DATASET_MANIFEST.md` để ghi phần nguồn dữ liệu,
   mô hình và tài liệu nền tảng trong báo cáo.
3. Đọc `configs/coco_final.yaml`. Đây là thiết kế benchmark cuối: 1.000 ảnh COCO,
   severity 1/3/5, seed cố định và recovery chỉ ở các mức có tín hiệu từ pilot.
4. Đọc mã theo luồng: `cli.py` → `config.py` → `experiment.py` →
   `corruptions.py` / `recovery.py` → `detector.py` → `metrics.py` → `plotting.py`.
5. Mở `runs/coco_topic6_final_1000/metrics.csv`, rồi xem các ảnh trong thư mục
   `runs/coco_topic6_final_1000/plots/`.
6. Dùng phần “Làm báo cáo và slide” bên dưới để chuyển kết quả thành nội dung nộp bài.

## 3. Cấu trúc và vai trò từng file

| Vị trí | Vai trò |
| --- | --- |
| `README.md` | Cài đặt, tải COCO, chạy smoke/pilot/final benchmark. |
| `PROJECT_GUIDE.md` | Hướng dẫn đọc project, diễn giải kết quả và chuẩn bị báo cáo/slide. |
| `requirements.txt`, `pyproject.toml` | Thư viện và package command `topic6`. |
| `configs/smoke_test.yaml` | Kiểm tra nhanh 8 ảnh, dùng để xác nhận môi trường. |
| `configs/pilot.yaml` | Pilot 200 ảnh, lặp seed để chọn strategy recovery. |
| `configs/gaussian_bilateral_pilot.yaml` | So sánh bilateral denoising cho Gaussian noise. |
| `configs/coco_final.yaml` | Benchmark cuối đã chạy và tạo kết quả công bố trong repository. |
| `configs/coco_val2017.yaml` | Cấu hình mở rộng trên toàn bộ COCO val2017 (5.000 ảnh). |
| `src/topic6_robustness/cli.py` | Các lệnh `download`, `benchmark`, `plot`, `preview`. |
| `src/topic6_robustness/config.py` | Đọc/kiểm tra YAML, severity, repeat, batch size và recovery strategy. |
| `src/topic6_robustness/coco_data.py` | Tải COCO, đọc ảnh RGB, annotation và dataset manifest. |
| `src/topic6_robustness/corruptions.py` | Sinh Gaussian, Salt-and-Pepper, fog và low-light ở severity 1–5. |
| `src/topic6_robustness/recovery.py` | NLMeans/bilateral cho Gaussian, median filter cho Salt-and-Pepper, CLAHE/gamma cho low-light và dark-channel dehazing cho fog. |
| `src/topic6_robustness/detector.py` | Adapter YOLO11n, map COCO class và chạy mini-batch GPU. |
| `src/topic6_robustness/experiment.py` | Vòng lặp benchmark, warm-up, đo thời gian và lưu condition-level result. |
| `src/topic6_robustness/metrics.py` | COCO mAP, AP theo kích thước và confusion matrix 80 class + background. |
| `src/topic6_robustness/plotting.py` | Vẽ mAP/severity, recovery gain, end-to-end latency và bảng aggregate. |
| `src/topic6_robustness/preview.py` | Ảnh minh họa clean–corrupted–recovered. |
| `tests/` | Unit test corruption, recovery, metrics, experiment schedule và plotting. |
| `runs/coco_topic6_final_1000/` | Toàn bộ evidence của benchmark cuối, đã được đưa vào repository. |

## 4. Chạy lại từ đầu

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\topic6.exe download --root data/coco
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\topic6.exe benchmark --config configs/coco_final.yaml
.\.venv\Scripts\topic6.exe plot --metrics runs/coco_topic6_final_1000/metrics.csv
```

Để xem trực quan trước khi benchmark, dùng:

```powershell
.\.venv\Scripts\topic6.exe preview --config configs/coco_final.yaml --severity 5
```

Repository gồm sẵn 5.000 ảnh COCO val2017 và annotation cần thiết qua Git LFS. Sau khi
clone, chạy `git lfs pull` nếu ảnh chưa tự tải. Các annotation train2017 không cần cho
dự án, `.venv`, cache và checkpoint model vẫn không được đưa vào repository.

## 5. Cách đọc kết quả cuối

Mở `runs/coco_topic6_final_1000/metrics.csv`:

- `map_50_95`: metric chính, càng cao càng tốt.
- `map_50`, `map_75`: mAP tại hai ngưỡng IoU.
- `map_small`, `map_medium`, `map_large`: độ nhạy theo kích thước vật thể.
- `recovered`: `True` là đã áp dụng preprocessing.
- `preprocessing_milliseconds_per_image`: thời gian tạo nhiễu + recovery.
- `inference_milliseconds_per_image`: inference YOLO riêng.
- `end_to_end_milliseconds_per_image`: preprocessing + inference; đây là latency dùng trong kết luận thực tế.
- `false_negative_unmatched`, `false_positive_unmatched`: vật thể bị bỏ sót và detection sai ở threshold confusion đã cấu hình.

Kết luận từ final run:

| Kết quả | mAP50–95 raw → recovery | Diễn giải |
| --- | --- | --- |
| Gaussian severity 3 | 0.228 → 0.288 | Bilateral phục hồi có ý nghĩa, nhưng tốn thời gian. |
| Gaussian severity 5 | 0.065 → 0.120 | Gaussian là corruption gây hại mạnh nhất. |
| Salt-and-Pepper severity 3 | 0.214 → 0.349 | Median filter phục hồi rất tốt. |
| Salt-and-Pepper severity 5 | 0.089 → 0.298 | Recovery mạnh nhất, nhưng cần theo dõi false positive. |
| Fog severity 5 | 0.327 → 0.348 | Có lợi nhẹ, chi phí dehazing cao. |
| Low-light severity 5 | 0.306 → 0.356 | Gamma + CLAHE có hiệu quả rõ rệt. |

## 6. Làm báo cáo

Đề xuất cấu trúc báo cáo:

1. **Giới thiệu**: tầm quan trọng của robustness với object detection ngoài thực tế.
2. **Mục tiêu và câu hỏi nghiên cứu**: bốn câu hỏi ở phần 1.
3. **Dữ liệu và mô hình**: COCO val2017, YOLO11n pretrained, GPU, confidence/NMS cố định.
4. **Phương pháp**: bốn corruption, severity, seed, recovery tương ứng và COCOeval.
5. **Thiết kế thực nghiệm**: pilot 200 ảnh để chọn recovery; final holdout 1.000 ảnh với severity 1/3/5.
6. **Kết quả**: dùng hai biểu đồ mAP, biểu đồ recovery gain, bảng summary và confusion matrix ở severity 5.
7. **Thảo luận**: Salt-and-Pepper/low-light phục hồi tốt; Gaussian vẫn yếu; fog trade-off latency; vật thể nhỏ khó nhất.
8. **Kết luận và hướng phát triển**: adaptive recovery, denoiser học sâu, full 5.000 ảnh và real-world weather data.

Khi ghi số liệu, nêu rõ final result dùng holdout 1.000 ảnh và một seed cố định.
Pilot lặp seed dùng để xác nhận xu hướng, không thay thế final result.

## 7. Làm slide

Gợi ý 10 slide:

1. Tên đề tài, thành viên, mục tiêu.
2. Bài toán và lý do robustness quan trọng.
3. Câu hỏi nghiên cứu và đóng góp.
4. Dataset, model và setup.
5. Pipeline clean → corruption → recovery → YOLO → COCOeval.
6. Bốn loại corruption và recovery tương ứng.
7. `map_50_95_vs_severity.png`.
8. `recovery_gain_vs_severity.png`.
9. `end_to_end_latency_vs_severity.png` cùng trade-off FP/FN.
10. Kết luận, hạn chế và hướng phát triển.

Trên slide kết quả, ưu tiên thông điệp thay vì đọc tất cả số: **Salt-and-Pepper được
phục hồi mạnh nhất; Gaussian nghiêm trọng nhất; recovery có trade-off accuracy–latency–false positive.**

## 8. Kiểm tra tính tái lập

Trước khi nộp, chạy `pytest -q`, kiểm tra `metrics.csv` và bốn file trong `plots/`.
Giữ nguyên model weight, confidence, NMS IoU, seed và COCO annotation giữa các condition
để phép so sánh công bằng.
