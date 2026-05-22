"""Tests for barcode reader (no pyzbar required for unit tests — uses mocking)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ocr.barcode_reader import BarcodeReader, BarcodeResult
from config import BarcodeConfig


@pytest.fixture
def reader():
    cfg = BarcodeConfig(min_area=100, scan_attempts=2, rotation_step=5)
    with patch.object(BarcodeReader, "_check_pyzbar"):
        return BarcodeReader(cfg=cfg)


@pytest.fixture
def barcode_image():
    """Synthetic high-frequency horizontal stripe pattern — looks like a barcode."""
    img = np.zeros((60, 200), dtype=np.uint8)
    for col in range(0, 200, 4):
        img[:, col:col+2] = 255
    return img


class TestBarcodeReader:

    def test_no_result_on_empty_image(self, reader):
        result = reader.decode(np.array([]))
        assert result.detected is False
        assert result.value is None

    def test_no_result_on_none(self, reader):
        result = reader.decode(None)
        assert result.detected is False

    def test_successful_decode(self, reader, barcode_image):
        mock_decoded = MagicMock()
        mock_decoded.data = b"8801234567890"
        mock_decoded.type = "CODE128"
        mock_decoded.rect = MagicMock(left=10, top=5, width=180, height=50)

        with patch("pyzbar.pyzbar.decode", return_value=[mock_decoded]):
            result = reader.decode(barcode_image)

        assert result.detected is True
        assert result.value == "8801234567890"
        assert result.barcode_format == "CODE128"
        assert result.confidence > 0.0

    def test_failed_decode_returns_no_result(self, reader, barcode_image):
        with patch("pyzbar.pyzbar.decode", return_value=[]):
            result = reader.decode(barcode_image)

        assert result.detected is False
        assert result.value is None

    def test_confidence_decreases_with_attempts(self, reader, barcode_image):
        mock_decoded = MagicMock()
        mock_decoded.data = b"12345678"
        mock_decoded.type = "EAN8"
        mock_decoded.rect = MagicMock(left=0, top=0, width=100, height=40)

        # Fail first decode, succeed on second
        with patch("pyzbar.pyzbar.decode", side_effect=[[], [mock_decoded]]):
            result = reader.decode(barcode_image)

        assert result.detected is True
        assert result.attempts == 2
        assert result.confidence < 0.99   # degraded by retry

    def test_rgb_image_converted_to_gray(self, reader):
        rgb = np.full((60, 200, 3), 128, dtype=np.uint8)
        with patch("pyzbar.pyzbar.decode", return_value=[]):
            result = reader.decode(rgb)
        assert result.detected is False   # no barcode — just testing no crash
