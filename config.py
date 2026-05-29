"""Application configuration for the ComfyUI-backed SynthID Remover."""
import os
from pathlib import Path

APP_NAME = "SynthID Remover"
APP_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"
ASSETS_DIR = BASE_DIR / "assets"

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", BASE_DIR / "ComfyUI")).expanduser()
COMFY_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
COMFY_BASE_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"

for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_SPECS = {
    "diffusion_model": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "z_image_turbo_bf16.safetensors",
        "subfolder": "diffusion_models",
    },
    "text_encoder": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "qwen_3_4b.safetensors",
        "subfolder": "text_encoders",
    },
    "vae": {
        "repo_id": "Comfy-Org/z_image_turbo",
        "filename": "ae.safetensors",
        "subfolder": "vae",
    },
    "controlnet": {
        "repo_id": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        "filename": "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors",
        "subfolder": "controlnet",
    },
    "face_detector": {
        "repo_id": "deepghs/yolo-face",
        "filename": "yolov8n-face.pt",
        "subfolder": "ultralytics",
    },
}

DEFAULTS = {
    "positive_prompt": "high quality photo, detailed, sharp focus",
    "negative_prompt": "watermark, synthid, text, logo, blurry, low quality, artifact",
    "denoise_auto": False,
    "denoise_manual": 0.20,
    "face_enhancement": True,
    "face_denoise_scale": 0.50,
    "steps": 9,
    "cfg_scale": 1.0,
    "seed": -1,
    "output_format": "png",
    "output_quality": 95,
    "canny_low": 50,
    "canny_high": 150,
}


def comfy_model_path(name: str, comfy_dir: Path = COMFYUI_DIR) -> Path:
    spec = MODEL_SPECS[name]
    return comfy_dir / "models" / spec["subfolder"] / spec["filename"]


MODEL_PATHS = {name: comfy_model_path(name) for name in MODEL_SPECS}
