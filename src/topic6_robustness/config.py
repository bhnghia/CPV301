from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    images_dir: Path
    annotations: Path
    limit: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "yolo11n.pt"
    device: str | int | None = None
    image_size: int = 640
    confidence: float = 0.001
    nms_iou: float = 0.7
    batch_size: int = 1


@dataclass(frozen=True)
class ExperimentConfig:
    corruptions: tuple[str, ...] = (
        "gaussian_noise",
        "salt_pepper",
        "fog",
        "low_light",
    )
    severities: tuple[int, ...] = (1, 2, 3, 4, 5)
    repeats: int = 3
    repeats_by_corruption: Mapping[str, int] = field(default_factory=dict)
    include_clean: bool = True
    include_recovery: bool = True
    recovery_severities: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    recovery_methods: Mapping[str, str] = field(default_factory=dict)
    save_predictions: bool = True
    confusion_confidence: float = 0.25
    confusion_iou: float = 0.5


@dataclass(frozen=True)
class ProjectConfig:
    project_name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    experiment: ExperimentConfig
    output_dir: Path
    source_path: Path = field(repr=False)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    source_path = Path(path).resolve()
    workspace = source_path.parent.parent
    with source_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    dataset_raw = raw["dataset"]
    model_raw = raw.get("model", {})
    experiment_raw = raw.get("experiment", {})

    dataset = DatasetConfig(
        images_dir=_resolve(workspace, dataset_raw["images_dir"]),
        annotations=_resolve(workspace, dataset_raw["annotations"]),
        limit=dataset_raw.get("limit"),
    )
    model = ModelConfig(
        weights=str(model_raw.get("weights", "yolo11n.pt")),
        device=model_raw.get("device"),
        image_size=int(model_raw.get("image_size", 640)),
        confidence=float(model_raw.get("confidence", 0.001)),
        nms_iou=float(model_raw.get("nms_iou", 0.7)),
        batch_size=int(model_raw.get("batch_size", 1)),
    )
    experiment = ExperimentConfig(
        corruptions=tuple(experiment_raw.get("corruptions", ExperimentConfig.corruptions)),
        severities=tuple(int(v) for v in experiment_raw.get("severities", (1, 2, 3, 4, 5))),
        repeats=int(experiment_raw.get("repeats", 3)),
        repeats_by_corruption={
            str(name): int(count)
            for name, count in experiment_raw.get("repeats_by_corruption", {}).items()
        },
        include_clean=bool(experiment_raw.get("include_clean", True)),
        include_recovery=bool(experiment_raw.get("include_recovery", True)),
        recovery_severities={
            str(name): tuple(int(level) for level in levels)
            for name, levels in experiment_raw.get("recovery_severities", {}).items()
        },
        recovery_methods={
            str(name): str(method)
            for name, method in experiment_raw.get("recovery_methods", {}).items()
        },
        save_predictions=bool(experiment_raw.get("save_predictions", True)),
        confusion_confidence=float(experiment_raw.get("confusion_confidence", 0.25)),
        confusion_iou=float(experiment_raw.get("confusion_iou", 0.5)),
    )

    if not experiment.severities or any(level not in range(1, 6) for level in experiment.severities):
        raise ValueError("Severity must contain integers from 1 to 5.")
    if experiment.repeats < 1:
        raise ValueError("Experiment repeats must be at least 1.")
    if model.batch_size < 1:
        raise ValueError("Model batch_size must be at least 1.")
    invalid_repeat_names = set(experiment.repeats_by_corruption) - set(experiment.corruptions)
    if invalid_repeat_names:
        raise ValueError(f"Unknown corruption(s) in repeats_by_corruption: {sorted(invalid_repeat_names)}")
    if any(count < 1 for count in experiment.repeats_by_corruption.values()):
        raise ValueError("All repeats_by_corruption values must be at least 1.")
    invalid_recovery_names = set(experiment.recovery_severities) - set(experiment.corruptions)
    if invalid_recovery_names:
        raise ValueError(f"Unknown corruption(s) in recovery_severities: {sorted(invalid_recovery_names)}")
    for corruption, levels in experiment.recovery_severities.items():
        if not levels or any(level not in range(1, 6) for level in levels):
            raise ValueError(f"Recovery severities for {corruption} must contain integers from 1 to 5.")
    invalid_recovery_method_names = set(experiment.recovery_methods) - set(experiment.corruptions)
    if invalid_recovery_method_names:
        raise ValueError(f"Unknown corruption(s) in recovery_methods: {sorted(invalid_recovery_method_names)}")

    return ProjectConfig(
        project_name=str(raw.get("project_name", source_path.stem)),
        seed=int(raw.get("seed", 2026)),
        dataset=dataset,
        model=model,
        experiment=experiment,
        output_dir=_resolve(workspace, raw.get("output_dir", f"runs/{source_path.stem}")),
        source_path=source_path,
    )

