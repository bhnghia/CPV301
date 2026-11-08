from __future__ import annotations

import cv2
import numpy as np


def _clahe_rgb(image: np.ndarray, clip_limit: float) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def _dehaze_dark_channel(image: np.ndarray, severity: int) -> np.ndarray:
    source = image.astype(np.float32) / 255.0
    window = 9 if severity <= 2 else 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window, window))
    dark = cv2.erode(np.min(source, axis=2), kernel)

    flat_dark = dark.reshape(-1)
    flat_source = source.reshape(-1, 3)
    count = max(1, int(flat_dark.size * 0.001))
    brightest_indices = np.argpartition(flat_dark, -count)[-count:]
    atmosphere = flat_source[brightest_indices].mean(axis=0)
    atmosphere = np.maximum(atmosphere, 0.45)

    normalized = source / atmosphere[None, None, :]
    normalized_dark = cv2.erode(np.min(normalized, axis=2), kernel)
    transmission = 1.0 - 0.88 * normalized_dark
    transmission = cv2.GaussianBlur(transmission, (0, 0), sigmaX=3.0)
    transmission = np.clip(transmission, 0.22, 1.0)

    recovered = (source - atmosphere[None, None, :]) / transmission[..., None]
    recovered += atmosphere[None, None, :]
    return np.clip(recovered * 255.0, 0, 255).astype(np.uint8)


def recover(
    image: np.ndarray,
    corruption: str,
    severity: int,
    method: str | None = None,
) -> np.ndarray:
    """Apply a corruption-aware, deterministic preprocessing baseline."""
    if corruption == "gaussian_noise":
        selected = method or "fast_nlmeans"
        if selected == "bilateral":
            diameter = 5 + 2 * severity
            sigma = 18 + 8 * severity
            return cv2.bilateralFilter(image, diameter, sigma, sigma)
        if selected != "fast_nlmeans":
            raise ValueError(f"Unsupported Gaussian recovery method '{selected}'.")
        strength = 5 + 2 * severity
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    if corruption == "salt_pepper":
        kernel = 3 if severity <= 3 else 5
        return cv2.medianBlur(image, kernel)
    if corruption == "low_light":
        gamma = (0.82, 0.72, 0.62, 0.52, 0.44)[severity - 1]
        lookup = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)]).astype(np.uint8)
        brightened = cv2.LUT(image, lookup)
        return _clahe_rgb(brightened, clip_limit=1.5 + 0.25 * severity)
    if corruption == "fog":
        dehazed = _dehaze_dark_channel(image, severity)
        return _clahe_rgb(dehazed, clip_limit=1.3)
    raise ValueError(f"No recovery strategy for corruption '{corruption}'.")

