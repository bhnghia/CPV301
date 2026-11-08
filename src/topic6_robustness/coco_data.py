from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# COCO's official download page publishes these large archives over HTTP.
# Using its HTTPS alias currently yields a hostname-mismatch certificate on
# some managed Windows environments, so do not disable TLS verification.
COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    path: Path
    width: int
    height: int


class DownloadProgress(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing archive: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=destination.name) as progress:
        urllib.request.urlretrieve(url, temporary, reporthook=progress.update_to)
    temporary.replace(destination)


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        handle.extractall(destination)


def download_coco_val2017(root: str | Path, keep_archives: bool = False) -> tuple[Path, Path]:
    root = Path(root).resolve()
    archives = root / "archives"
    val_archive = archives / "val2017.zip"
    annotations_archive = archives / "annotations_trainval2017.zip"

    _download(COCO_VAL_URL, val_archive)
    _download(COCO_ANNOTATIONS_URL, annotations_archive)
    _safe_extract(val_archive, root)
    _safe_extract(annotations_archive, root)

    if not keep_archives:
        shutil.rmtree(archives)

    images_dir = root / "val2017"
    annotations = root / "annotations" / "instances_val2017.json"
    if not images_dir.is_dir() or not annotations.is_file():
        raise RuntimeError("COCO extraction did not produce the expected files.")
    return images_dir, annotations


class CocoDataset:
    def __init__(self, images_dir: str | Path, annotations: str | Path, limit: int | None = None):
        from pycocotools.coco import COCO

        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {self.annotations_path}")

        self.coco = COCO(str(self.annotations_path))
        image_ids = sorted(self.coco.getImgIds())
        self.image_ids = image_ids[:limit] if limit else image_ids
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_name_to_id = {item["name"]: int(item["id"]) for item in categories}
        self.category_ids = sorted(self.category_name_to_id.values())
        self.category_id_to_name = {value: key for key, value in self.category_name_to_id.items()}

    def __len__(self) -> int:
        return len(self.image_ids)

    def record(self, image_id: int) -> ImageRecord:
        item = self.coco.loadImgs([image_id])[0]
        return ImageRecord(
            image_id=int(image_id),
            file_name=str(item["file_name"]),
            path=self.images_dir / item["file_name"],
            width=int(item["width"]),
            height=int(item["height"]),
        )

    def load_rgb(self, image_id: int) -> np.ndarray:
        record = self.record(image_id)
        bgr = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Unable to read image: {record.path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def ground_truth(self, image_id: int) -> list[dict]:
        annotation_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=False)
        return self.coco.loadAnns(annotation_ids)

    def write_manifest(self, path: str | Path) -> None:
        manifest = {
            "images_dir": str(self.images_dir.resolve()),
            "annotations": str(self.annotations_path.resolve()),
            "image_count": len(self),
            "image_ids": self.image_ids,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

