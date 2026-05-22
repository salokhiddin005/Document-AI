"""
Confidence scoring and routing engine.

For each extracted field this module produces:
  - A final confidence score (combining OCR confidence + format validation)
  - A routing decision:  AUTO_ACCEPT | SPOT_CHECK | HUMAN_REVIEW
  - A human-readable reason when routing to review

Routing thresholds are read from config so they can be tuned per deployment
without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger

from config import settings, ConfidenceConfig


# ─────────────────────────────────────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────────────────────────────────────

class RoutingDecision(str, Enum):
    AUTO_ACCEPT  = "AUTO_ACCEPT"   # confident + valid → goes straight to DB
    SPOT_CHECK   = "SPOT_CHECK"    # medium confidence → human checks a sample
    HUMAN_REVIEW = "HUMAN_REVIEW"  # low confidence or invalid → must be reviewed


@dataclass
class FieldScore:
    field_name: str
    raw_value: str
    ocr_confidence: float       # from the OCR model (0–1)
    format_valid: bool          # passed regex / length validation
    final_confidence: float     # composite score (0–1)
    decision: RoutingDecision
    reason: str                 # human-readable explanation


@dataclass
class ScoringResult:
    fields: Dict[str, FieldScore]
    overall_confidence: float
    requires_human_review: bool
    auto_accepted: List[str]
    spot_check: List[str]
    human_review: List[str]


# ─────────────────────────────────────────────────────────────────────────────
#  Scorer
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceScorer:

    def __init__(self, cfg: Optional[ConfidenceConfig] = None):
        self.cfg = cfg or settings.confidence

    # ── Public ────────────────────────────────────────────────────────────────

    def score(
        self,
        extracted_fields: Dict[str, str],       # {field_name: raw_text}
        ocr_confidences: Dict[str, float],       # {field_name: 0–1 score}
        barcode_confidence: Optional[float] = None,
    ) -> ScoringResult:
        """
        Score all fields and return routing decisions.

        *extracted_fields*  — the OCR output values
        *ocr_confidences*   — the OCR model's own confidence per field
        *barcode_confidence*— if present, overrides the OCR confidence for
                              the 'barcode' field (barcode decoders are
                              almost always right or completely wrong)
        """
        field_scores: Dict[str, FieldScore] = {}

        for field_name, raw_value in extracted_fields.items():
            ocr_conf = ocr_confidences.get(field_name, 0.0)

            # Barcode confidence comes from the decoder, not OCR
            if field_name == "barcode" and barcode_confidence is not None:
                ocr_conf = barcode_confidence

            fs = self._score_field(field_name, raw_value, ocr_conf)
            field_scores[field_name] = fs

        auto_accepted  = [n for n, s in field_scores.items() if s.decision == RoutingDecision.AUTO_ACCEPT]
        spot_check     = [n for n, s in field_scores.items() if s.decision == RoutingDecision.SPOT_CHECK]
        human_review   = [n for n, s in field_scores.items() if s.decision == RoutingDecision.HUMAN_REVIEW]

        all_confs = [s.final_confidence for s in field_scores.values() if s.raw_value]
        overall   = round(sum(all_confs) / len(all_confs), 3) if all_confs else 0.0

        logger.info(
            f"Scoring: auto={len(auto_accepted)} spot={len(spot_check)} "
            f"review={len(human_review)} overall={overall:.2f}"
        )

        return ScoringResult(
            fields=field_scores,
            overall_confidence=overall,
            requires_human_review=len(human_review) > 0,
            auto_accepted=auto_accepted,
            spot_check=spot_check,
            human_review=human_review,
        )

    # ── Field-level scoring ───────────────────────────────────────────────────

    def _score_field(
        self, field_name: str, raw_value: str, ocr_confidence: float
    ) -> FieldScore:

        # Empty value → always needs review
        if not raw_value or not raw_value.strip():
            return FieldScore(
                field_name=field_name,
                raw_value=raw_value,
                ocr_confidence=0.0,
                format_valid=False,
                final_confidence=0.0,
                decision=RoutingDecision.HUMAN_REVIEW,
                reason="empty_value",
            )

        # Format validation
        format_valid, format_reason = self._validate_format(field_name, raw_value)

        # Composite confidence:
        # - OCR confidence carries 80% of the weight
        # - Format validity carries 20% (binary bonus)
        format_bonus = 0.05 if format_valid else -0.10
        final_conf = round(
            min(max(ocr_confidence + format_bonus, 0.0), 1.0), 3
        )

        # Always send to human review if format is wrong, regardless of OCR confidence
        if not format_valid:
            decision = RoutingDecision.HUMAN_REVIEW
            reason   = f"format_invalid: {format_reason}"
        else:
            decision, reason = self._route(final_conf)

        return FieldScore(
            field_name=field_name,
            raw_value=raw_value,
            ocr_confidence=round(ocr_confidence, 3),
            format_valid=format_valid,
            final_confidence=final_conf,
            decision=decision,
            reason=reason,
        )

    def _validate_format(self, field_name: str, value: str) -> tuple[bool, str]:
        """
        Validate *value* against the configured regex for *field_name*.
        Returns (is_valid, reason_string).
        """
        validator = self.cfg.get_validator(field_name)
        if validator is None:
            return True, "no_validator"  # no rule defined → assume valid

        if validator.search(value):
            return True, "pattern_matched"

        return False, f"expected_pattern={validator.pattern}"

    def _route(self, confidence: float) -> tuple[RoutingDecision, str]:
        if confidence >= self.cfg.auto_accept_threshold:
            return RoutingDecision.AUTO_ACCEPT, "high_confidence"
        if confidence >= self.cfg.spot_check_threshold:
            return RoutingDecision.SPOT_CHECK, "medium_confidence"
        return RoutingDecision.HUMAN_REVIEW, "low_confidence"
