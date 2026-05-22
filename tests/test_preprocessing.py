"""Tests for the image preprocessing module."""

import numpy as np
import pytest

from src.preprocessing.image_processor import ImageProcessor, PreprocessingResult


@pytest.fixture
def processor():
    return ImageProcessor()


@pytest.fixture
def clean_white_image():
    """300x400 white RGB image — simulates a blank scanned page."""
    return np.full((300, 400, 3), 255, dtype=np.uint8)


@pytest.fixture
def noisy_gray_image():
    """Grayscale image with random noise — simulates a low-quality scan."""
    rng = np.random.default_rng(42)
    base = np.full((200, 300), 240, dtype=np.uint8)
    noise = rng.integers(0, 30, size=base.shape, dtype=np.uint8)
    return np.clip(base.astype(int) - noise, 0, 255).astype(np.uint8)


class TestImageProcessor:

    def test_process_numpy_array_succeeds(self, processor, clean_white_image):
        result = processor.process(clean_white_image)
        assert result.success is True
        assert result.processed_image is not None
        assert result.gray_image is not None
        assert result.binary_image is not None

    def test_process_returns_original_shape(self, processor, clean_white_image):
        result = processor.process(clean_white_image)
        assert result.original_shape == clean_white_image.shape

    def test_process_missing_file_returns_failure(self, processor):
        result = processor.process("/nonexistent/path/image.jpg")
        assert result.success is False
        assert result.error is not None

    def test_quality_score_range(self, processor, clean_white_image):
        result = processor.process(clean_white_image)
        assert 0.0 <= result.quality_score <= 1.0

    def test_skew_angle_is_float(self, processor, clean_white_image):
        result = processor.process(clean_white_image)
        assert isinstance(result.skew_angle, float)

    def test_gray_image_is_2d(self, processor, clean_white_image):
        result = processor.process(clean_white_image)
        assert result.gray_image.ndim == 2

    def test_binary_image_has_only_two_values(self, processor, clean_white_image):
        result = processor.process(clean_white_image)
        unique_vals = set(result.binary_image.flatten().tolist())
        assert unique_vals.issubset({0, 255})

    def test_crop_region_extracts_correct_slice(self, processor, clean_white_image):
        # Draw a black rectangle at a known position
        img = clean_white_image.copy()
        img[50:100, 80:200] = 0
        crop = ImageProcessor.crop_region(img, (80, 50, 200, 100))
        assert crop.shape == (50, 120, 3)
        assert crop.mean() < 10   # should be mostly black

    def test_process_grayscale_input(self, processor, noisy_gray_image):
        result = processor.process(noisy_gray_image)
        assert result.success is True
