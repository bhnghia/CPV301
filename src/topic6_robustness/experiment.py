from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .coco_data import CocoDataset
from .config import ProjectConfig
from .corruptions import apply_corruption
from .detector import Detection, UltralyticsDetector
from .metrics import ConfusionAccumulator, evaluate_coco
from .recovery import recover


@dataclass(frozen=True)
class Condition:
    corruption: str
    severity: int
    repeat: int
    recovered: bool = False
    recovery_method: str | None = None

    @property
    def name(self) -> str:
        suffix = f"_recovered_{self.recovery_method}" if self.recovered and self.recovery_method else "_recovered" if self.recovered else ""
        return f"{self.corruption}_s{self.severity}_r{self.repeat}{suffix}"


def build_conditions(config: ProjectConfig) -> list[Condition]:
    conditions: list[Condition] = []
    if config.experiment.include_clean:
        conditions.append(Condition("clean", 0, 0, False))
    for corruption in config.experiment.corruptions:
        for severity in config.experiment.severities:
            repeat_count = config.experiment.repeats_by_corruption.get(corruption, config.experiment.repeats)
            recovery_levels = config.experiment.recovery_severities.get(corruption, config.experiment.severities)
            for repeat in range(repeat_count):
                conditions.append(Condition(corruption, severity, repeat, False))
                if config.experiment.include_recovery and severity in recovery_levels:
                    conditions.append(
                        Condition(
                            corruption,
                            severity,
                            repeat,
                            True,
                            config.experiment.recovery_methods.get(corruption),
                        )
                    )
    return conditions


def _condition_image(
    clean: np.ndarray,
    condition: Condition,
    seed: int,
    image_id: int,
) -> tuple[np.ndarray, float, float]:
    if condition.corruption == "clean":
        return clean, 0.0, 0.0
    sequence = np.random.SeedSequence([seed, image_id, condition.severity, condition.repeat])
    rng = np.random.default_rng(sequence)
    corruption_started = time.perf_counter()
    corrupted = apply_corruption(clean, condition.corruption, condition.severity, rng)
    corruption_seconds = time.perf_counter() - corruption_started
    if not condition.recovered:
        return corrupted, corruption_seconds, 0.0
    recovery_started = time.perf_counter()
    recovered = recover(corrupted, condition.corruption, condition.severity, condition.recovery_method)
    recovery_seconds = time.perf_counter() - recovery_started
    return recovered, corruption_seconds, recovery_seconds


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_benchmark(config: ProjectConfig) -> Path:
    dataset = CocoDataset(config.dataset.images_dir, config.dataset.annotations, config.dataset.limit)
    detector = UltralyticsDetector(
        weights=config.model.weights,
        category_name_to_id=dataset.category_name_to_id,
        device=config.model.device,
        image_size=config.model.image_size,
        confidence=config.model.confidence,
        nms_iou=config.model.nms_iou,
    )
    # Exclude model loading, CUDA context creation, and kernel warm-up from
    # condition latency so clean and corrupted runs remain comparable.
    warmup_id = dataset.image_ids[0]
    detector.predict(dataset.load_rgb(warmup_id), warmup_id)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.write_manifest(config.output_dir / "dataset_manifest.json")

    metrics_rows: list[dict] = []
    for condition in build_conditions(config):
        condition_dir = config.output_dir / "conditions" / condition.name
        condition_dir.mkdir(parents=True, exist_ok=True)
        detections: list[Detection] = []
        confusion = ConfusionAccumulator(dataset.category_ids)
        inference_seconds = 0.0
        corruption_seconds = 0.0
        recovery_seconds = 0.0

        progress = tqdm(total=len(dataset), desc=condition.name, unit="image")
        for start_index in range(0, len(dataset.image_ids), config.model.batch_size):
            batch_ids = dataset.image_ids[start_index : start_index + config.model.batch_size]
            batch_images: list[np.ndarray] = []
            for image_id in batch_ids:
                clean = dataset.load_rgb(image_id)
                image, image_corruption_seconds, image_recovery_seconds = _condition_image(
                    clean, condition, config.seed, image_id
                )
                corruption_seconds += image_corruption_seconds
                recovery_seconds += image_recovery_seconds
                batch_images.append(image)
            started = time.perf_counter()
            batch_detections = detector.predict_batch(batch_images, batch_ids)
            inference_seconds += time.perf_counter() - started
            for image_id, image_detections in zip(batch_ids, batch_detections, strict=True):
                detections.extend(image_detections)
                confusion.update(
                    dataset.ground_truth(image_id),
                    image_detections,
                    confidence=config.experiment.confusion_confidence,
                    iou_threshold=config.experiment.confusion_iou,
                )
            progress.update(len(batch_ids))
        progress.close()

        summary, class_ap, report = evaluate_coco(dataset.coco, detections, dataset.image_ids)
        confusion_summary = confusion.summary()
        preprocessing_seconds = corruption_seconds + recovery_seconds
        end_to_end_seconds = preprocessing_seconds + inference_seconds
        image_count = max(len(dataset), 1)
        row = {
            **asdict(condition),
            "condition": condition.name,
            "image_count": len(dataset),
            "detections": len(detections),
            "corruption_seconds": corruption_seconds,
            "corruption_milliseconds_per_image": 1000.0 * corruption_seconds / image_count,
            "recovery_seconds": recovery_seconds,
            "recovery_milliseconds_per_image": 1000.0 * recovery_seconds / image_count,
            "preprocessing_seconds": preprocessing_seconds,
            "preprocessing_milliseconds_per_image": 1000.0 * preprocessing_seconds / image_count,
            "inference_seconds": inference_seconds,
            "inference_milliseconds_per_image": 1000.0 * inference_seconds / image_count,
            "end_to_end_seconds": end_to_end_seconds,
            "end_to_end_milliseconds_per_image": 1000.0 * end_to_end_seconds / image_count,
            # Backward-compatible alias. This is detector inference only.
            "milliseconds_per_image": 1000.0 * inference_seconds / image_count,
            **summary,
            **confusion_summary,
        }
        metrics_rows.append(row)

        if config.experiment.save_predictions:
            _save_json(condition_dir / "predictions.json", [item.as_coco() for item in detections])
        _save_json(condition_dir / "metrics.json", row)
        _save_json(condition_dir / "class_ap.json", class_ap)
        np.savetxt(condition_dir / "confusion_matrix.csv", confusion.matrix, delimiter=",", fmt="%d")
        (condition_dir / "coco_eval.txt").write_text(report, encoding="utf-8")

    metrics_path = config.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    return metrics_path

