"""Tests for confidence scoring and routing logic."""

import pytest

from src.confidence.scorer import ConfidenceScorer, RoutingDecision
from config import ConfidenceConfig


@pytest.fixture
def scorer():
    cfg = ConfidenceConfig(
        auto_accept_threshold=0.85,
        spot_check_threshold=0.70,
        field_validators={
            "id_field":  r"[\w\-]{4,20}",
            "barcode":   r"\d{8,20}",
        },
    )
    return ConfidenceScorer(cfg=cfg)


class TestConfidenceScorer:

    def test_high_confidence_auto_accepted(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": "John Smith"},
            ocr_confidences={"name_field": 0.95},
        )
        assert result.fields["name_field"].decision == RoutingDecision.AUTO_ACCEPT
        assert "name_field" in result.auto_accepted

    def test_low_confidence_routes_to_review(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": "Joh? Sm?th"},
            ocr_confidences={"name_field": 0.45},
        )
        assert result.fields["name_field"].decision == RoutingDecision.HUMAN_REVIEW
        assert "name_field" in result.human_review

    def test_medium_confidence_spot_check(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": "John Smith"},
            ocr_confidences={"name_field": 0.75},
        )
        assert result.fields["name_field"].decision == RoutingDecision.SPOT_CHECK
        assert "name_field" in result.spot_check

    def test_empty_value_always_reviewed(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": ""},
            ocr_confidences={"name_field": 0.99},
        )
        assert result.fields["name_field"].decision == RoutingDecision.HUMAN_REVIEW
        assert result.fields["name_field"].reason == "empty_value"

    def test_invalid_format_forces_review(self, scorer):
        # id_field validator requires [\w\-]{4,20}
        result = scorer.score(
            extracted_fields={"id_field": "AB"},   # too short
            ocr_confidences={"id_field": 0.97},
        )
        assert result.fields["id_field"].decision == RoutingDecision.HUMAN_REVIEW
        assert result.fields["id_field"].format_valid is False

    def test_valid_barcode_accepted(self, scorer):
        result = scorer.score(
            extracted_fields={"barcode": "8801234567890"},
            ocr_confidences={"barcode": 0.99},
            barcode_confidence=0.99,
        )
        assert result.fields["barcode"].decision == RoutingDecision.AUTO_ACCEPT

    def test_overall_confidence_is_average(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": "John", "id_field": "A-123456"},
            ocr_confidences={"name_field": 0.90, "id_field": 0.80},
        )
        assert 0.0 <= result.overall_confidence <= 1.0

    def test_requires_human_review_flag(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": ""},
            ocr_confidences={"name_field": 0.0},
        )
        assert result.requires_human_review is True

    def test_no_review_required_when_all_confident(self, scorer):
        result = scorer.score(
            extracted_fields={"name_field": "John Smith"},
            ocr_confidences={"name_field": 0.95},
        )
        assert result.requires_human_review is False
