"""Persistent settings manager for SynthID Remover GUI."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


SETTINGS_FILE = Path.home() / ".synthid-remover" / "settings.json"


@dataclass
class UserSettings:
    """User-configurable application settings."""
    # Window
    window_width: int = 1400
    window_height: int = 900
    window_x: Optional[int] = None
    window_y: Optional[int] = None

    # Prompts
    last_positive_prompt: str = "high quality photo, detailed, sharp focus"
    last_negative_prompt: str = "watermark, synthid, text, logo, blurry, low quality, artifact"

    # Processing
    default_denoise_auto: bool = False
    default_denoise_manual: float = 0.20
    default_face_enhancement: bool = True
    default_face_denoise_scale: float = 0.5
    default_steps: int = 9
    default_output_format: str = "png"
    default_output_quality: int = 95

    # Paths
    last_input_dir: str = ""
    last_output_dir: str = ""

    # Advanced
    theme: str = "dark"
    auto_check_updates: bool = True
    show_comparison_on_complete: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserSettings":
        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class SettingsManager:
    """Manages loading and saving of user settings."""

    def __init__(self, settings_file: Optional[Path] = None):
        self.settings_file = settings_file or SETTINGS_FILE
        self._settings = UserSettings()
        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> UserSettings:
        """Load settings from file."""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                self._settings = UserSettings.from_dict(data)
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load settings: {e}. Using defaults.")
        return self._settings

    def save(self):
        """Save current settings to file."""
        try:
            with open(self.settings_file, "w") as f:
                json.dump(self._settings.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

    @property
    def settings(self) -> UserSettings:
        return self._settings

    def update(self, **kwargs):
        """Update settings and save."""
        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                setattr(self._settings, key, value)
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value."""
        return getattr(self._settings, key, default)
