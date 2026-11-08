from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


GAUSSIAN_SIGMA = (8.0, 16.0, 24.0, 36.0, 50.0)
SALT_PEPPER_PROBABILITY = (0.005, 0.01, 0.03, 0.06, 0.10)
FOG_STRENGTH = (0.10, 0.20, 0.32, 0.45, 0.58)
LOW_LIGHT_FACTOR = (0.75, 0.60, 0.45, 0.30, 0.18)


def _validate_image(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape HxWx3.")
    if image.dtype != np.uint8:
        raise ValueError("Expected uint8 image data.")
    return image


def _severity_index(severity: int) -> int:
    if severity not in range(1, 6):
        raise ValueError("Severity must be between 1 and 5.")
    return severity - 1


def gaussian_noise(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    image = _validate_image(image)
    sigma = GAUSSIAN_SIGMA[_severity_index(severity)]
    noise = rng.normal(0.0, sigma, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def salt_pepper(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    image = _validate_image(image)
    probability = SALT_PEPPER_PROBABILITY[_severity_index(severity)]
    mask = rng.random(image.shape[:2])
    output = image.copy()
    output[mask < probability / 2.0] = 0
    output[(mask >= probability / 2.0) & (mask < probability)] = 255
    return output


def low_light(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    del rng
    image = _validate_image(image)
    factor = LOW_LIGHT_FACTOR[_severity_index(severity)]
    linear = image.astype(np.float32) / 255.0
    darkened = np.power(linear * factor, 1.08)
    return np.clip(darkened * 255.0, 0, 255).astype(np.uint8)


def fog(image: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    image = _validate_image(image)
    strength = FOG_STRENGTH[_severity_index(severity)]
    height, width = image.shape[:2]

    grid_h = max(2, height // 48)
    grid_w = max(2, width // 48)
    coarse = rng.random((grid_h, grid_w), dtype=np.float32)
    haze = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    blur_sigma = max(height, width) / 18.0
    haze = cv2.GaussianBlur(haze, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    haze = (haze - haze.min()) / max(float(np.ptp(haze)), 1e-6)

    alpha = strength * (0.72 + 0.28 * haze)
    output = image.astype(np.float32) * (1.0 - alpha[..., None]) + 255.0 * alpha[..., None]
    return np.clip(output, 0, 255).astype(np.uint8)


CORRUPTIONS: dict[str, Callable[[np.ndarray, int, np.random.Generator], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "salt_pepper": salt_pepper,
    "fog": fog,
    "low_light": low_light,
}


def apply_corruption(
    image: np.ndarray,
    name: str,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    try:
        transform = CORRUPTIONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown corruption '{name}'. Available: {sorted(CORRUPTIONS)}") from exc
    return transform(image, severity, rng)

