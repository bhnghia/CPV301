import numpy as np
import pytest

from topic6_robustness.corruptions import apply_corruption


@pytest.fixture
def image() -> np.ndarray:
    values = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8)
    return values.reshape(64, 64, 3)


@pytest.mark.parametrize("name", ["gaussian_noise", "salt_pepper", "fog", "low_light"])
def test_corruptions_preserve_shape_and_dtype(image: np.ndarray, name: str) -> None:
    output = apply_corruption(image, name, 3, np.random.default_rng(7))
    assert output.shape == image.shape
    assert output.dtype == np.uint8


@pytest.mark.parametrize("name", ["gaussian_noise", "salt_pepper", "fog", "low_light"])
def test_corruptions_are_deterministic_for_same_seed(image: np.ndarray, name: str) -> None:
    left = apply_corruption(image, name, 3, np.random.default_rng(99))
    right = apply_corruption(image, name, 3, np.random.default_rng(99))
    np.testing.assert_array_equal(left, right)


def test_low_light_gets_darker_with_severity(image: np.ndarray) -> None:
    means = [
        apply_corruption(image, "low_light", level, np.random.default_rng(1)).mean()
        for level in range(1, 6)
    ]
    assert means == sorted(means, reverse=True)


def test_invalid_severity_is_rejected(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Severity"):
        apply_corruption(image, "fog", 0, np.random.default_rng(1))

