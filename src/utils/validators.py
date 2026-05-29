"""Input validation utilities for SynthID Remover."""
import os
from pathlib import Path
from typing import Tuple, Optional, List


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


def validate_image_path(path: str) -> Path:
    """Validate image file path."""
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"Image not found: {path}")
    if not p.is_file():
        raise ValidationError(f"Not a file: {path}")

    valid_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")
    if p.suffix.lower() not in valid_ext:
        raise ValidationError(f"Unsupported format: {p.suffix}. Use: {valid_ext}")

    # Check file size (max 50MB)
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > 50:
        raise ValidationError(f"Image too large: {size_mb:.1f}MB (max 50MB)")

    return p


def validate_prompt(prompt: str, max_length: int = 1000) -> str:
    """Validate and clean prompt text."""
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError("Prompt cannot be empty")
    if len(prompt) > max_length:
        raise ValidationError(f"Prompt too long: {len(prompt)} chars (max {max_length})")
    return prompt


def validate_denoise(value: float) -> float:
    """Validate denoise strength."""
    if not 0.0 <= value <= 1.0:
        raise ValidationError(f"Denoise must be 0.0-1.0, got {value}")
    return round(value, 3)


def validate_steps(value: int) -> int:
    """Validate step count."""
    if not 1 <= value <= 100:
        raise ValidationError(f"Steps must be 1-100, got {value}")
    return value


def validate_seed(value: int) -> int:
    """Validate seed value."""
    if value < -1 or value > 2**32:
        raise ValidationError(f"Seed must be -1 or 0-{2**32}, got {value}")
    return value


def validate_quality(value: int) -> int:
    """Validate output quality."""
    if not 1 <= value <= 100:
        raise ValidationError(f"Quality must be 1-100, got {value}")
    return value


def validate_model_paths(model_paths: dict) -> Tuple[List[str], List[str]]:
    """Validate all model paths. Returns (present, missing)."""
    present = []
    missing = []
    for name, path in model_paths.items():
        if path.exists():
            present.append(name)
        else:
            missing.append(name)
    return present, missing


def validate_output_dir(path: str) -> Path:
    """Validate and create output directory."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if not os.access(p, os.W_OK):
        raise ValidationError(f"Cannot write to output directory: {path}")
    return p
