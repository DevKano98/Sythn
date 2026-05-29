"""ComfyUI workflow API JSON generation."""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from config import DEFAULTS, MODEL_SPECS


@dataclass
class WorkflowSettings:
    positive_prompt: str = DEFAULTS["positive_prompt"]
    negative_prompt: str = DEFAULTS["negative_prompt"]
    denoise: float = DEFAULTS["denoise_manual"]
    face_enabled: bool = DEFAULTS["face_enhancement"]
    face_denoise: float = DEFAULTS["denoise_manual"] * DEFAULTS["face_denoise_scale"]
    steps: int = DEFAULTS["steps"]
    cfg_scale: float = DEFAULTS["cfg_scale"]
    seed: int = DEFAULTS["seed"]
    canny_low: int = DEFAULTS["canny_low"]
    canny_high: int = DEFAULTS["canny_high"]


BASE_REQUIRED_NODES = {
    "LoadImage",
    "VAELoader",
    "VAEEncode",
    "CLIPLoader",
    "CLIPTextEncode",
    "UNETLoader",
    "ControlNetLoader",
    "ControlNetApplyAdvanced",
    "KSampler",
    "VAEDecode",
    "SaveImage",
}

CANNY_NODE_CANDIDATES = ["CannyEdgePreprocessor", "Canny"]

FACE_REQUIRED_NODES = {
    "UltralyticsDetectorProvider",
    "BboxDetectorForEach",
    "InpaintCrop",
    "InpaintStitch",
}


def _available_nodes(object_info: Optional[Dict[str, Any]]) -> Set[str]:
    return set(object_info or {})


def validate_required_nodes(object_info: Optional[Dict[str, Any]], face_enabled: bool) -> None:
    if object_info is None:
        return
    available = _available_nodes(object_info)
    missing = sorted(BASE_REQUIRED_NODES - available)
    if not any(node in available for node in CANNY_NODE_CANDIDATES):
        missing.append("CannyEdgePreprocessor or Canny")
    if face_enabled:
        missing.extend(sorted(FACE_REQUIRED_NODES - available))
    if missing:
        raise RuntimeError(
            "ComfyUI is missing required nodes: "
            + ", ".join(missing)
            + ". Install the matching Z-Image, ControlNet preprocessor, and face/inpaint custom nodes."
        )


def build_workflow(uploaded_image_name: str, settings: WorkflowSettings,
                   object_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    validate_required_nodes(object_info, settings.face_enabled)
    available = _available_nodes(object_info)
    canny_class = "CannyEdgePreprocessor" if "CannyEdgePreprocessor" in available else "Canny"
    seed = settings.seed if settings.seed >= 0 else 0

    workflow: Dict[str, Any] = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": uploaded_image_name},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODEL_SPECS["vae"]["filename"]},
        },
        "3": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["1", 0], "vae": ["2", 0]},
        },
        "4": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MODEL_SPECS["text_encoder"]["filename"],
                "type": "stable_diffusion",
            },
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 0], "text": settings.positive_prompt},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 0], "text": settings.negative_prompt},
        },
        "7": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": MODEL_SPECS["diffusion_model"]["filename"],
                "weight_dtype": "default",
            },
        },
        "8": {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": MODEL_SPECS["controlnet"]["filename"]},
        },
        "9": {
            "class_type": canny_class,
            "inputs": {
                "image": ["1", 0],
                "low_threshold": settings.canny_low,
                "high_threshold": settings.canny_high,
            },
        },
        "10": {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "control_net": ["8", 0],
                "image": ["9", 0],
                "strength": 1.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["7", 0],
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["3", 0],
                "seed": seed,
                "steps": settings.steps,
                "cfg": settings.cfg_scale,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": settings.denoise,
            },
        },
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["2", 0]},
        },
        "13": {
            "class_type": "SaveImage",
            "inputs": {"images": ["12", 0], "filename_prefix": "synthid_base"},
        },
    }

    if settings.face_enabled:
        workflow.update({
            "20": {
                "class_type": "UltralyticsDetectorProvider",
                "inputs": {"model_name": MODEL_SPECS["face_detector"]["filename"]},
            },
            "21": {
                "class_type": "BboxDetectorForEach",
                "inputs": {"bbox_detector": ["20", 0], "image": ["12", 0], "threshold": 0.50},
            },
            "22": {
                "class_type": "InpaintCrop",
                "inputs": {"image": ["12", 0], "segs": ["21", 0], "padding": 32},
            },
            "23": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["22", 0], "vae": ["2", 0]},
            },
            "24": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["7", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["23", 0],
                    "seed": seed + 1,
                    "steps": settings.steps,
                    "cfg": settings.cfg_scale,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": settings.face_denoise,
                },
            },
            "25": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["24", 0], "vae": ["2", 0]},
            },
            "26": {
                "class_type": "InpaintStitch",
                "inputs": {"inpainted_image": ["25", 0], "crop_data": ["22", 1]},
            },
            "27": {
                "class_type": "SaveImage",
                "inputs": {"images": ["26", 0], "filename_prefix": "synthid_final"},
            },
        })

    return workflow


def find_output_image(history_item: Dict[str, Any], prefer_final: bool = True) -> Dict[str, str]:
    outputs = history_item.get("outputs", {})
    node_order = ["27", "13"] if prefer_final else ["13", "27"]
    for node_id in node_order:
        for image in outputs.get(node_id, {}).get("images", []):
            return image
    for output in outputs.values():
        for image in output.get("images", []):
            return image
    raise RuntimeError("ComfyUI completed but did not return an output image")
