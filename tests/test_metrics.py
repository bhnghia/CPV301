from topic6_robustness.detector import Detection
from topic6_robustness.metrics import ConfusionAccumulator


def test_confusion_accumulator_counts_matches_and_errors() -> None:
    accumulator = ConfusionAccumulator([1, 2])
    ground_truth = [
        {"category_id": 1, "bbox": [0, 0, 10, 10]},
        {"category_id": 2, "bbox": [30, 30, 10, 10]},
    ]
    predictions = [
        Detection(1, 1, (0, 0, 10, 10), 0.9),
        Detection(1, 1, (30, 30, 10, 10), 0.8),
        Detection(1, 2, (60, 60, 5, 5), 0.7),
    ]
    accumulator.update(ground_truth, predictions, confidence=0.25, iou_threshold=0.5)
    summary = accumulator.summary()
    assert summary["true_positive"] == 1
    assert summary["class_confusions"] == 1
    assert summary["false_positive_unmatched"] == 1
    assert summary["false_negative_unmatched"] == 0

