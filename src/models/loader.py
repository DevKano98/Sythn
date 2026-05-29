"""Model loading and management for SynthID Remover"""
import os
import torch
from pathlib import Path
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of all pipeline models."""

    def __init__(self, model_paths: Dict[str, Path], device: str = "cpu"):
        self.model_paths = model_paths
        self.device = device
        self.loaded_models: Dict[str, Any] = {}
        self._check_models_exist()

    def _check_models_exist(self):
        """Verify all model files are present."""
        missing = []
        for name, path in self.model_paths.items():
            if not path.exists():
                missing.append(f"  {name}: {path}")
        if missing:
            logger.warning("Missing model files (download required):\n" + "\n".join(missing))

    def load_gguf_model(self, name: str, n_ctx: int = 2048,
                        n_gpu_layers: int = -1, embedding: bool = False):
        """Load a GGUF model using llama-cpp-python."""
        cache_key = f"{name}:embedding" if embedding else name
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        path = self.model_paths.get(name)
        if not path or not path.exists():
            raise FileNotFoundError(f"Model {name} not found at {path}")

        try:
            from llama_cpp import Llama
            model = Llama(
                model_path=str(path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers if self.device == "cuda" else 0,
                embedding=embedding,
                verbose=False,
            )
            self.loaded_models[cache_key] = model
            logger.info(f"Loaded GGUF model: {name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            raise

    def load_safetensors(self, name: str) -> Dict[str, torch.Tensor]:
        """Load a safetensors file."""
        if name in self.loaded_models:
            return self.loaded_models[name]

        path = self.model_paths.get(name)
        if not path or not path.exists():
            raise FileNotFoundError(f"Model {name} not found at {path}")

        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(path), device=self.device)
            self.loaded_models[name] = state_dict
            logger.info(f"Loaded safetensors: {name}")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            raise

    def load_yolo_face(self, name: str = "face_detection"):
        """Load YOLOv8 face detection model."""
        if name in self.loaded_models:
            return self.loaded_models[name]

        path = self.model_paths.get(name)
        if not path or not path.exists():
            raise FileNotFoundError(f"Model {name} not found at {path}")

        try:
            from ultralytics import YOLO
            model = YOLO(str(path))
            if self.device == "cuda":
                model.to("cuda")
            self.loaded_models[name] = model
            logger.info(f"Loaded YOLO model: {name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            raise

    def load_vae(self, name: str):
        """Load VAE from safetensors. Returns state dict."""
        return self.load_safetensors(name)

    def load_controlnet(self, name: str = "controlnet"):
        """Load ControlNet from safetensors."""
        return self.load_safetensors(name)

    def get_model(self, name: str):
        """Get a loaded model by name."""
        return self.loaded_models.get(name)

    def unload_model(self, name: str):
        """Unload a specific model to free memory."""
        if name in self.loaded_models:
            del self.loaded_models[name]
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info(f"Unloaded model: {name}")

    def unload_all(self):
        """Unload all models."""
        self.loaded_models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("All models unloaded")

    @property
    def available_models(self):
        """List of successfully loaded models."""
        return list(self.loaded_models.keys())

    @property
    def missing_models(self):
        """List of model files that don't exist."""
        return [name for name, path in self.model_paths.items() if not path.exists()]
