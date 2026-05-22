"""
Central configuration — loads configs/settings.yaml and exposes typed dataclasses.
Every module imports from here; never hard-code thresholds in business logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


ROOT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
#  Typed config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    target_dpi: int = 300
    max_skew_angle: float = 45.0
    denoise_strength: int = 10
    contrast_clip_limit: float = 2.0
    contrast_tile_size: int = 8
    min_quality_score: float = 0.30


@dataclass
class OCRConfig:
    language: List[str] = field(default_factory=lambda: ["en", "korean"])
    device: str = "cpu"                   # "cpu" or "gpu"
    use_textline_orientation: bool = True # auto-detect rotated/upside-down text
    text_det_thresh: float = 0.3
    text_det_box_thresh: float = 0.5
    text_recognition_batch_size: int = 6
    enable_mkldnn: bool = False           # disable to avoid PIR inference errors on Windows


@dataclass
class BarcodeConfig:
    min_area: int = 1000
    scan_attempts: int = 3
    rotation_step: int = 5


@dataclass
class ConfidenceConfig:
    auto_accept_threshold: float = 0.85
    spot_check_threshold: float = 0.70
    field_validators: Dict[str, str] = field(default_factory=dict)

    def get_validator(self, field_name: str) -> Optional[re.Pattern]:
        pattern = self.field_validators.get(field_name)
        return re.compile(pattern) if pattern else None


@dataclass
class ReviewConfig:
    db_path: Path = ROOT / "data" / "review_queue.db"
    crops_dir: Path = ROOT / "data" / "feedback" / "crops"


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 20


@dataclass
class AppConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    barcode: BarcodeConfig = field(default_factory=BarcodeConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    api: APIConfig = field(default_factory=APIConfig)


# ─────────────────────────────────────────────────────────────────────────────
#  Loader
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(yaml_path: Optional[Path] = None) -> AppConfig:
    yaml_path = yaml_path or ROOT / "configs" / "settings.yaml"

    raw: dict = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    p = raw.get("preprocessing", {})
    o = raw.get("ocr", {})
    b = raw.get("barcode", {})
    c = raw.get("confidence", {})
    r = raw.get("review", {})
    a = raw.get("api", {})

    review_cfg = ReviewConfig(
        db_path=ROOT / r.get("db_path", "data/review_queue.db"),
        crops_dir=ROOT / r.get("crops_dir", "data/feedback/crops"),
    )

    return AppConfig(
        preprocessing=PreprocessingConfig(**{k: p[k] for k in PreprocessingConfig.__dataclass_fields__ if k in p}),
        ocr=OCRConfig(**{k: o[k] for k in OCRConfig.__dataclass_fields__ if k in o}),
        barcode=BarcodeConfig(**{k: b[k] for k in BarcodeConfig.__dataclass_fields__ if k in b}),
        confidence=ConfidenceConfig(
            auto_accept_threshold=c.get("auto_accept_threshold", 0.85),
            spot_check_threshold=c.get("spot_check_threshold", 0.70),
            field_validators=c.get("field_validators", {}),
        ),
        review=review_cfg,
        api=APIConfig(**{k: a[k] for k in APIConfig.__dataclass_fields__ if k in a}),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Singleton — import this everywhere
# ─────────────────────────────────────────────────────────────────────────────
settings = load_config()
