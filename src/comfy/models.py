"""Model location checks and downloads for ComfyUI."""
from pathlib import Path
from typing import Dict, List, Tuple

from config import COMFYUI_DIR, MODEL_SPECS


class ComfyModelManager:
    def __init__(self, comfy_dir: Path = COMFYUI_DIR):
        self.comfy_dir = Path(comfy_dir)

    def model_path(self, name: str) -> Path:
        spec = MODEL_SPECS[name]
        return self.comfy_dir / "models" / spec["subfolder"] / spec["filename"]

    @property
    def model_paths(self) -> Dict[str, Path]:
        return {name: self.model_path(name) for name in MODEL_SPECS}

    @property
    def missing_models(self) -> List[str]:
        return [name for name, path in self.model_paths.items() if not path.exists()]

    def validate(self) -> Tuple[List[str], List[str]]:
        present = []
        missing = []
        for name, path in self.model_paths.items():
            if path.exists():
                present.append(name)
            else:
                missing.append(name)
        return present, missing

    def ensure_dirs(self) -> None:
        for spec in MODEL_SPECS.values():
            (self.comfy_dir / "models" / spec["subfolder"]).mkdir(parents=True, exist_ok=True)

    def download_missing(self, force: bool = False) -> Dict[str, Path]:
        from huggingface_hub import hf_hub_download

        self.ensure_dirs()
        downloaded = {}
        for name, spec in MODEL_SPECS.items():
            dest = self.model_path(name)
            if dest.exists() and not force:
                downloaded[name] = dest
                continue

            tmp = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=spec["filename"],
                local_dir=str(dest.parent),
                local_dir_use_symlinks=False,
            )
            tmp_path = Path(tmp)
            if tmp_path != dest:
                if dest.exists():
                    dest.unlink()
                tmp_path.replace(dest)
            downloaded[name] = dest
        return downloaded
