from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass

import numpy as np

from .detector import Detection


COCO_STAT_NAMES = (
    "map_50_95",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def evaluate_coco(coco_gt, detections: list[Detection], image_ids: list[int]) -> tuple[dict, dict[str, float], str]:
    from pycocotools.cocoeval import COCOeval

    coco_payload = [item.as_coco() for item in detections]
    if not coco_payload:
        zeros = {name: 0.0 for name in COCO_STAT_NAMES}
        return zeros, {}, "No detections were produced."

    coco_dt = coco_gt.loadRes(coco_payload)
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        evaluator.summarize()

    summary = {name: float(value) for name, value in zip(COCO_STAT_NAMES, evaluator.stats, strict=True)}
    precision = evaluator.eval["precision"]
    class_ap: dict[str, float] = {}
    for class_position, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, class_position, 0, -1]
        valid = values[values > -1]
        if valid.size:
            category_name = coco_gt.loadCats([category_id])[0]["name"]
            class_ap[category_name] = float(valid.mean())
    return summary, class_ap, stream.getvalue()


def _xywh_to_xyxy(box: list[float] | tuple[float, ...]) -> np.ndarray:
    x, y, width, height = (float(value) for value in box)
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class ConfusionAccumulator:
    category_ids: list[int]

    def __post_init__(self) -> None:
        self.index = {category_id: position for position, category_id in enumerate(self.category_ids)}
        size = len(self.category_ids) + 1
        self.background = size - 1
        self.matrix = np.zeros((size, size), dtype=np.int64)

    def update(
        self,
        ground_truth: list[dict],
        predictions: list[Detection],
        confidence: float,
        iou_threshold: float,
    ) -> None:
        predictions = [item for item in predictions if item.score >= confidence]
        candidates: list[tuple[float, int, int]] = []
        for gt_index, gt in enumerate(ground_truth):
            gt_box = _xywh_to_xyxy(gt["bbox"])
            for pred_index, prediction in enumerate(predictions):
                score = _iou(gt_box, _xywh_to_xyxy(prediction.bbox))
                if score >= iou_threshold:
                    candidates.append((score, gt_index, pred_index))
        candidates.sort(reverse=True)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        for _, gt_index, pred_index in candidates:
            if gt_index in matched_gt or pred_index in matched_pred:
                continue
            matched_gt.add(gt_index)
            matched_pred.add(pred_index)
            gt_class = self.index[int(ground_truth[gt_index]["category_id"])]
            pred_class = self.index[predictions[pred_index].category_id]
            self.matrix[gt_class, pred_class] += 1

        for gt_index, gt in enumerate(ground_truth):
            if gt_index not in matched_gt:
                self.matrix[self.index[int(gt["category_id"])], self.background] += 1
        for pred_index, prediction in enumerate(predictions):
            if pred_index not in matched_pred:
                self.matrix[self.background, self.index[prediction.category_id]] += 1

    def summary(self) -> dict[str, int]:
        true_positive = int(np.trace(self.matrix[:-1, :-1]))
        class_confusions = int(self.matrix[:-1, :-1].sum() - true_positive)
        missed = int(self.matrix[:-1, self.background].sum())
        spurious = int(self.matrix[self.background, :-1].sum())
        return {
            "true_positive": true_positive,
            "class_confusions": class_confusions,
            "false_negative_unmatched": missed,
            "false_positive_unmatched": spurious,
        }

