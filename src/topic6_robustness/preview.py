from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .coco_data import CocoDataset
from .config import ProjectConfig
from .corruptions import apply_corruption
from .recovery import recover


def create_preview(
    config: ProjectConfig,
    severity: int = 3,
    image_id: int | None = None,
    output: str | Path | None = None,
) -> Path:
    if severity not in range(1, 6):
        raise ValueError("Severity must be between 1 and 5.")
    dataset = CocoDataset(config.dataset.images_dir, config.dataset.annotations, config.dataset.limit)
    selected_id = image_id if image_id is not None else dataset.image_ids[0]
    if selected_id not in dataset.image_ids:
        raise ValueError(f"Image id {selected_id} is not included in the configured dataset subset.")
    clean = dataset.load_rgb(selected_id)

    corruptions = config.experiment.corruptions
    figure, axes = plt.subplots(len(corruptions), 3, figsize=(13, 3.6 * len(corruptions)))
    if len(corruptions) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, corruption in enumerate(corruptions):
        sequence = np.random.SeedSequence([config.seed, selected_id, severity, 0])
        corrupted = apply_corruption(clean, corruption, severity, np.random.default_rng(sequence))
        recovered = recover(corrupted, corruption, severity)
        for axis, image, title in zip(
            axes[row],
            (clean, corrupted, recovered),
            ("Clean", f"{corruption} severity {severity}", "Recovered"),
            strict=True,
        ):
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")

    figure.suptitle(f"COCO image {selected_id}: corruption and recovery preview", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    destination = Path(output) if output else config.output_dir / f"preview_image_{selected_id}_s{severity}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination.resolve()

