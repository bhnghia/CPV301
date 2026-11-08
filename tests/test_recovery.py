import numpy as np
import pytest

from topic6_robustness.recovery import recover


@pytest.mark.parametrize("name", ["gaussian_noise", "salt_pepper", "fog", "low_light"])
def test_recovery_preserves_shape_and_dtype(name: str) -> None:
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    output = recover(image, name, severity=3)
    assert output.shape == image.shape
    assert output.dtype == np.uint8


def test_bilateral_gaussian_recovery_preserves_shape_and_dtype() -> None:
    image = np.random.default_rng(8).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    output = recover(image, "gaussian_noise", severity=3, method="bilateral")
    assert output.shape == image.shape
    assert output.dtype == np.uint8

