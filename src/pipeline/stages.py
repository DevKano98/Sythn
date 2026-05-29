"""Individual pipeline stages for SynthID Remover"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import logging

from ..utils.image import (
    canny_edge_detect, numpy_to_tensor, tensor_to_numpy,
    create_mask_from_bbox, feather_mask, composite_images,
    calculate_adaptive_denoise, resize_image
)
from ..models.inference import VAEInference, DiffusionSampler, FaceEnhancer, GGUFInference

logger = logging.getLogger(__name__)


class Stage1_ImageInput:
    """Stage 1: Load and normalize input image."""

    def __init__(self, max_size: int = 2048):
        self.max_size = max_size

    def process(self, image_path: str, max_size: Optional[int] = None) -> Dict[str, Any]:
        from ..utils.image import load_image
        img = load_image(image_path)
        original_size = img.shape[:2]
        img = resize_image(img, max_size or self.max_size)

        return {
            "image": img,
            "original_size": original_size,
            "current_size": img.shape[:2],
            "path": image_path
        }


class Stage2_AdaptiveDenoise:
    """Stage 2: Calculate denoise strength based on resolution."""

    def process(self, image: np.ndarray, auto: bool = True, 
                manual_value: float = 0.12, scale: float = 1.0) -> Dict[str, Any]:
        if auto:
            denoise = calculate_adaptive_denoise(image, scale=scale)
        else:
            denoise = manual_value

        info = f"Resolution: {image.shape[1]}x{image.shape[0]} → Denoise: {denoise}"
        logger.info(info)

        return {
            "denoise": denoise,
            "info": info,
            "width": image.shape[1],
            "height": image.shape[0]
        }


class Stage3_EdgeExtraction:
    """Stage 3: Canny edge detection for ControlNet."""

    def __init__(self, low: int = 50, high: int = 150):
        self.low = low
        self.high = high

    def process(self, image: np.ndarray, low: Optional[int] = None,
                high: Optional[int] = None) -> Dict[str, Any]:
        edges = canny_edge_detect(image, low or self.low, high or self.high)
        return {
            "edge_map": edges,
            "edge_tensor": numpy_to_tensor(edges)
        }


class Stage4_ControlNet:
    """Stage 4: ControlNet conditioning."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def process(self, edge_tensor: torch.Tensor, 
              prompt_embedding: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        # Load controlnet if available
        try:
            controlnet_state = self.model_manager.load_controlnet("controlnet")
            # ControlNet processing would happen here
            # For now, return edge conditioning
            conditioning = {
                "controlnet_state": controlnet_state,
                "edge_tensor": edge_tensor,
                "prompt_embedding": prompt_embedding
            }
        except FileNotFoundError:
            logger.warning("ControlNet not found, using edge map directly")
            conditioning = {"edge_tensor": edge_tensor}

        return {"conditioning": conditioning}


class Stage5_PromptEncoding:
    """Stage 5: Encode text prompts using Qwen VL."""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    def process(self, positive: str, negative: str) -> Dict[str, Any]:
        try:
            model = self.model_manager.load_gguf_model("prompt_encoder", embedding=True)
            logger.info(f"Encoding prompts: pos={len(positive)} chars, neg={len(negative)} chars")
            positive_embedding = self._encode_with_model(model, positive)
            negative_embedding = self._encode_with_model(model, negative)

            return {
                "positive_embedding": positive_embedding,
                "negative_embedding": negative_embedding,
                "model": model
            }
        except Exception as e:
            logger.warning(f"Prompt encoder failed: {e}, using text directly")
            return {
                "positive_embedding": positive,
                "negative_embedding": negative,
                "model": None
            }

    @staticmethod
    def _encode_with_model(model, prompt: str) -> torch.Tensor:
        if hasattr(model, "create_embedding"):
            try:
                output = model.create_embedding(prompt)
                data = output.get("data", [])
                if data and "embedding" in data[0]:
                    return torch.tensor(data[0]["embedding"], dtype=torch.float32).unsqueeze(0)
            except Exception as exc:
                logger.warning("Prompt embedding API failed: %s", exc)

        arr = GGUFInference._stable_text_embedding(prompt)
        return torch.from_numpy(arr)


class Stage6_LatentEncoding:
    """Stage 6: VAE encode image to latent space."""

    def __init__(self, model_manager, device: str = "cpu"):
        self.model_manager = model_manager
        self.device = device
        self.vae: Optional[VAEInference] = None

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            vae_state = self.model_manager.load_vae("vae")
            image_tensor = numpy_to_tensor(image, self.device)
            self.vae = VAEInference(vae_state, self.device)
            latent = self.vae.encode(image_tensor)

            return {
                "latent": latent,
                "image_tensor": image_tensor,
                "vae_state": vae_state,
                "vae": self.vae
            }
        except Exception as e:
            logger.warning(f"VAE encode failed: {e}, using direct tensor")
            image_tensor = numpy_to_tensor(image, self.device)
            self.vae = VAEInference({}, self.device)
            return {
                "latent": self.vae.encode(image_tensor),
                "image_tensor": image_tensor,
                "vae": self.vae
            }

    def decode(self, latent: torch.Tensor) -> np.ndarray:
        if self.vae is None:
            self.vae = VAEInference({}, self.device)
        decoded = self.vae.decode(latent)
        return tensor_to_numpy(decoded)


class Stage7_MainReconstruction:
    """Stage 7: Main diffusion reconstruction."""

    def __init__(self, model_manager, device: str = "cpu"):
        self.model_manager = model_manager
        self.device = device

    def process(self, latent: torch.Tensor, conditioning: Dict,
                prompt_data: Dict[str, Any], vae: VAEInference,
                denoise: float, steps: int = 4, seed: int = -1,
                cfg_scale: float = 7.0) -> Dict[str, Any]:
        try:
            model = self.model_manager.load_gguf_model("reconstruction")

            if seed < 0:
                seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)

            logger.info(f"Reconstructing with denoise={denoise}, steps={steps}, seed={seed}")

            controlnet = None
            if "controlnet_state" in conditioning:
                from ..models.inference import ControlNetInference
                controlnet = ControlNetInference(conditioning["controlnet_state"], self.device)
            sampler = DiffusionSampler(model, vae, controlnet, self.device)
            positive = prompt_data.get("positive_embedding")
            negative = prompt_data.get("negative_embedding")
            if not isinstance(positive, torch.Tensor):
                positive = torch.zeros((1, 1), device=latent.device)
            else:
                positive = positive.to(latent.device)
            if not isinstance(negative, torch.Tensor):
                negative = torch.zeros_like(positive)
            else:
                negative = negative.to(latent.device)

            edge = conditioning.get("edge_tensor")
            if isinstance(edge, torch.Tensor):
                edge = edge.to(latent.device)
                if edge.shape[-2:] != latent.shape[-2:]:
                    edge = torch.nn.functional.interpolate(edge, size=latent.shape[-2:], mode="bilinear")

            reconstructed = sampler.sample(
                latent,
                positive,
                negative,
                edge_map=edge,
                num_steps=steps,
                denoise_strength=denoise,
                cfg_scale=cfg_scale,
                seed=seed,
            )

            return {
                "reconstructed_latent": reconstructed,
                "seed": seed,
                "model": model
            }
        except Exception as e:
            logger.error(f"Reconstruction failed: {e}")
            raise


class Stage8_FaceDetection:
    """Stage 8: Detect faces using YOLOv8."""

    def __init__(self, model_manager, conf: float = 0.5):
        self.model_manager = model_manager
        self.conf = conf

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        try:
            model = self.model_manager.load_yolo_face("face_detection")

            # YOLOv8 inference
            results = model(image, conf=self.conf, verbose=False)

            faces = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box[:4])
                        # Add padding
                        pad = int(min(x2-x1, y2-y1) * 0.2)
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(image.shape[1], x2 + pad)
                        y2 = min(image.shape[0], y2 + pad)
                        faces.append((x1, y1, x2, y2))

            logger.info(f"Detected {len(faces)} face(s)")
            return {
                "faces": faces,
                "count": len(faces),
                "model": model
            }
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {"faces": [], "count": 0}


class Stage9_FaceSegmentation:
    """Stage 9: Generate face masks using MediaPipe or SAM."""

    def process(self, image: np.ndarray, faces: List[Tuple[int, ...]]) -> Dict[str, Any]:
        masks = []
        for bbox in faces:
            # Create mask from bbox with feathering
            mask = create_mask_from_bbox(image.shape, bbox)
            mask = feather_mask(mask, radius=20)
            masks.append(mask)

        return {
            "masks": masks,
            "face_regions": [image[y1:y2, x1:x2] for x1, y1, x2, y2 in faces]
        }


class Stage10_FaceReconstruction:
    """Stage 10: Face-specific enhancement."""

    def __init__(self, model_manager, device: str = "cpu"):
        self.model_manager = model_manager
        self.device = device

    def process(self, face_regions: List[np.ndarray],
                denoise: float = 0.08, steps: int = 4) -> Dict[str, Any]:
        enhanced = []

        for i, face in enumerate(face_regions):
            try:
                # Load face reconstruction model if available
                model = self.model_manager.load_gguf_model("face_reconstruction")
                vae_state = self.model_manager.load_vae("face_vae")
                vae = VAEInference(vae_state, self.device)
                enhancer = FaceEnhancer(model, vae, self.device)
                face_tensor = numpy_to_tensor(face, self.device)
                face_latent = vae.encode(face_tensor)
                enhanced_latent = enhancer.enhance(face_latent, denoise=denoise, steps=steps)
                enhanced_face = vae.decode(enhanced_latent)
                enhanced.append(tensor_to_numpy(enhanced_face))

                logger.info(f"Enhanced face {i+1}/{len(face_regions)}")
            except Exception as e:
                logger.warning(f"Face enhancement failed for face {i+1}: {e}")
                enhanced.append(face)  # Return original if enhancement fails

        return {"enhanced_faces": enhanced}


class Stage11_Compositing:
    """Stage 11: Merge enhanced faces back into image."""

    def process(self, base_image: np.ndarray, enhanced_faces: List[np.ndarray],
                faces: List[Tuple[int, ...]], masks: List[np.ndarray]) -> Dict[str, Any]:
        result = base_image.copy()

        for i, (face, bbox, mask) in enumerate(zip(enhanced_faces, faces, masks)):
            result = composite_images(result, face, mask, bbox)
            logger.info(f"Composited face {i+1}")

        return {"final_image": result}


class Stage12_Export:
    """Stage 12: Save final output."""

    def __init__(self, output_dir: str, format: str = "png", quality: int = 95):
        self.output_dir = output_dir
        self.format = format
        self.quality = quality

    def process(self, image: np.ndarray, original_path: str,
                suffix: str = "_synthid_removed",
                format: Optional[str] = None,
                quality: Optional[int] = None) -> Dict[str, Any]:
        from pathlib import Path
        from ..utils.image import save_image

        orig_name = Path(original_path).stem
        output_format = (format or self.format).lower()
        output_quality = quality if quality is not None else self.quality
        ext = f".{output_format}"
        output_path = Path(self.output_dir) / f"{orig_name}{suffix}{ext}"

        save_image(image, str(output_path), output_quality)
        logger.info(f"Saved: {output_path}")

        return {
            "output_path": str(output_path),
            "format": output_format,
            "quality": output_quality
        }
