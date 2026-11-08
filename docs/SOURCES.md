# Nguồn dữ liệu và phương pháp

## Nguồn chính

- COCO Dataset: https://cocodataset.org/
- COCO 2017 validation images: http://images.cocodataset.org/zips/val2017.zip
- COCO 2017 train/validation annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
- COCO terms of use: https://cocodataset.org/#termsofuse
- Ultralytics validation documentation: https://docs.ultralytics.com/modes/val/
- Ultralytics detection task documentation: https://docs.ultralytics.com/tasks/detect/
- COCO API: https://github.com/cocodataset/cocoapi

## Tài liệu nghiên cứu tham khảo

- Hendrycks, D. and Dietterich, T. Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. ICLR 2019. https://arxiv.org/abs/1903.12261
- Wang, S. et al. A Survey on the Robustness of Computer Vision Models against Common Corruptions. 2023. https://arxiv.org/abs/2305.06024
- Michaelis, C. et al. Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming. 2019. https://arxiv.org/abs/1907.07484

## Lựa chọn phương pháp

- COCO val2017 được dùng vì mô hình pretrained và annotation dùng cùng hệ 80 lớp, tránh sai lệch do ánh xạ lớp.
- Ảnh suy giảm được tạo trong bộ nhớ. Bounding box và nhãn gốc không bị thay đổi.
- Gaussian noise và Salt-and-Pepper được lặp lại với nhiều seed vì đây là các phép biến đổi ngẫu nhiên.
- Metric chính là COCO mAP@[0.5:0.95]; mAP@0.5, mAP@0.75, AP theo kích thước và confusion matrix là metric bổ sung.
- Baseline ảnh sạch luôn được chạy bằng cùng model, image size, confidence và NMS threshold.

