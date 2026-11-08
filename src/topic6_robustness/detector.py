from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float

    def as_coco(self) -> dict:
        return {
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": [round(value, 3) for value in self.bbox],
            "score": round(self.score, 6),
        }


class UltralyticsDetector:
    def __init__(
        self,
        weights: str,
        category_name_to_id: dict[str, int],
        device: str | int | None,
        image_size: int,
        confidence: float,
        nms_iou: float,
    ):
        # The managed runtime cannot write to the normal roaming profile.
        # Keep Ultralytics settings inside the project workspace instead.
        os.environ.setdefault("YOLO_CONFIG_DIR", str((Path.cwd() / ".ultralytics").resolve()))
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.nms_iou = nms_iou
        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))
        self.class_to_category = {}
        missing = []
        for class_index, name in names.items():
            if name in category_name_to_id:
                self.class_to_category[int(class_index)] = category_name_to_id[name]
            else:
                missing.append(name)
        if missing:
            raise ValueError(f"Model classes do not match COCO categories: {missing}")

    def _detections_from_result(self, result, image_id: int) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_index in zip(xyxy, scores, classes, strict=True):
            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(
                Detection(
                    image_id=image_id,
                    category_id=self.class_to_category[class_index],
                    bbox=(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                    score=float(score),
                )
            )
        return detections

    def predict_batch(self, images: list[np.ndarray], image_ids: list[int]) -> list[list[Detection]]:
        if len(images) != len(image_ids):
            raise ValueError("images and image_ids must have the same length.")
        if not images:
            return []
        results = self.model.predict(
            source=images,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.nms_iou,
            device=self.device,
            batch=len(images),
            verbose=False,
        )
        return [self._detections_from_result(result, image_id) for result, image_id in zip(results, image_ids, strict=True)]

    def predict(self, image: np.ndarray, image_id: int) -> list[Detection]:
        return self.predict_batch([image], [image_id])[0]

