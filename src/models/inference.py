"""Model inference wrappers for SynthID Remover

Provides high-level interfaces for running inference on:
- Qwen Image reconstruction (GGUF)
- Qwen VL prompt encoding (GGUF)
- Face reconstruction (GGUF)
- VAE encode/decode (safetensors)
- ControlNet (safetensors)
"""
import torch
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import hashlib

logger = logging.getLogger(__name__)


class GGUFInference:
    """Wrapper for GGUF model inference using llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 2048, n_gpu_layers: int = -1, device: str = "cpu"):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if device == "cuda" else 0
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from llama_cpp import Llama
                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    embedding=True,
                    verbose=False,
                )
                logger.info(f"Loaded GGUF model: {Path(self.model_path).name}")
            except Exception as e:
                logger.error(f"Failed to load GGUF model: {e}")
                raise
        return self._model

    def encode_prompt(self, prompt: str, max_tokens: int = 512) -> np.ndarray:
        """Encode text prompt to embedding vector."""
        model = self._load()
        if hasattr(model, "create_embedding"):
            output = model.create_embedding(prompt)
            data = output.get("data", [])
            if data and "embedding" in data[0]:
                return np.asarray(data[0]["embedding"], dtype=np.float32)[None, :]

        if hasattr(model, "embed"):
            tokens = model.tokenize(prompt.encode("utf-8"))[:max_tokens]
            embedding = model.embed(tokens)
            return np.asarray(embedding, dtype=np.float32)

        return self._stable_text_embedding(prompt)

    @staticmethod
    def _stable_text_embedding(prompt: str, size: int = 768) -> np.ndarray:
        """Deterministic fallback when the GGUF runtime cannot expose embeddings."""
        digest = hashlib.blake2b(prompt.encode("utf-8"), digest_size=64).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        values = np.resize(values, size)
        values = (values / 127.5) - 1.0
        return values[None, :]

    def generate_image_tokens(self, prompt: str, image_tokens: int = 256) -> List[int]:
        """Generate image token sequence from prompt."""
        model = self._load()
        output = model(prompt, max_tokens=image_tokens, temperature=0.7)
        tokens = output.get("choices", [{}])[0].get("text", "")
        return [ord(c) for c in tokens]  # Simplified


class VAEInference:
    """VAE encode/decode for latent space conversion."""

    def __init__(self, state_dict: Dict[str, torch.Tensor], device: str = "cpu"):
        self.state_dict = state_dict
        self.device = device
        self._build_vae()

    def _build_vae(self):
        """Build VAE from state dict."""
        self.vae = None
        if not self.state_dict:
            return
        try:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_config({
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": [
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D",
                ],
                "in_channels": 3,
                "latent_channels": 4,
                "layers_per_block": 2,
                "norm_num_groups": 32,
                "out_channels": 3,
                "sample_size": 1024,
                "scaling_factor": 0.18215,
                "up_block_types": [
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                ],
            })
            missing, unexpected = self.vae.load_state_dict(self.state_dict, strict=False)
            if len(unexpected) > len(self.state_dict) * 0.25:
                logger.warning("VAE state dict does not match AutoencoderKL layout; using tensor fallback")
                self.vae = None
            else:
                if missing:
                    logger.warning("VAE loaded with %d missing keys", len(missing))
                self.vae.to(self.device).eval()
        except Exception as exc:
            logger.warning("Could not build diffusers VAE from state dict: %s", exc)

    def encode(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image [B,3,H,W] to latent [B,C,h,w]."""
        if self.vae is None:
            return torch.nn.functional.avg_pool2d(image_tensor, 8)
        with torch.no_grad():
            latent = self.vae.encode(image_tensor).latent_dist.sample()
            return latent * getattr(self.vae.config, "scaling_factor", 1.0)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent [B,C,h,w] to image [B,3,H,W]."""
        if self.vae is None:
            return torch.nn.functional.interpolate(latent, scale_factor=8, mode="bilinear")
        with torch.no_grad():
            scale = getattr(self.vae.config, "scaling_factor", 1.0)
            return self.vae.decode(latent / scale).sample


class ControlNetInference:
    """ControlNet for structural guidance."""

    def __init__(self, state_dict: Dict[str, torch.Tensor], device: str = "cpu"):
        self.state_dict = state_dict
        self.device = device

    def apply(self, latent: torch.Tensor, edge_conditioning: torch.Tensor,
              timestep: int, prompt_embedding: torch.Tensor) -> torch.Tensor:
        """Apply ControlNet conditioning to latent."""
        if edge_conditioning.shape[-2:] != latent.shape[-2:]:
            edge_conditioning = torch.nn.functional.interpolate(
                edge_conditioning, size=latent.shape[-2:], mode="bilinear"
            )
        if edge_conditioning.shape[1] != latent.shape[1]:
            edge_conditioning = edge_conditioning.mean(dim=1, keepdim=True)
            edge_conditioning = edge_conditioning.repeat(1, latent.shape[1], 1, 1)
        edge_weight = 0.8
        return latent + edge_conditioning * edge_weight * 0.1


class DiffusionSampler:
    """Diffusion sampling loop for image reconstruction."""

    def __init__(self, model, vae: VAEInference, controlnet: Optional[ControlNetInference] = None,
                 device: str = "cpu"):
        self.model = model
        self.vae = vae
        self.controlnet = controlnet
        self.device = device

    def sample(self, latent: torch.Tensor, prompt_embedding: torch.Tensor,
               negative_embedding: torch.Tensor, edge_map: Optional[torch.Tensor] = None,
               num_steps: int = 4, denoise_strength: float = 0.12,
               cfg_scale: float = 7.0, seed: int = -1) -> torch.Tensor:
        """Run diffusion sampling.

        Args:
            latent: Initial latent (encoded from input image)
            prompt_embedding: Positive prompt embedding
            negative_embedding: Negative prompt embedding
            edge_map: ControlNet edge conditioning
            num_steps: Number of denoising steps (4 for Lightning)
            denoise_strength: How much noise to add/remove (0.0-1.0)
            cfg_scale: Classifier-free guidance scale
            seed: Random seed (-1 for random)

        Returns:
            Reconstructed latent
        """
        if seed >= 0:
            torch.manual_seed(seed)

        generator = torch.Generator(device=latent.device)
        if seed < 0:
            seed = torch.seed() % (2**32)
        generator.manual_seed(int(seed))

        noise = torch.randn(latent.shape, generator=generator, device=latent.device, dtype=latent.dtype)
        noisy_latent = latent + noise * denoise_strength

        for step in range(num_steps):
            if self.controlnet and edge_map is not None:
                noisy_latent = self.controlnet.apply(
                    noisy_latent, edge_map, step, prompt_embedding
                )

            pred_noise = self._predict_noise(noisy_latent, step, prompt_embedding, negative_embedding)
            step_size = denoise_strength / max(num_steps, 1)
            noisy_latent = noisy_latent - pred_noise * step_size

        return noisy_latent

    def _predict_noise(self, latent: torch.Tensor, timestep: int,
                       prompt_embedding: torch.Tensor,
                       negative_embedding: torch.Tensor) -> torch.Tensor:
        """Call an architecture-specific model adapter when one is available."""
        if hasattr(self.model, "predict_noise"):
            return self.model.predict_noise(
                latent=latent,
                timestep=timestep,
                prompt_embedding=prompt_embedding,
                negative_embedding=negative_embedding,
            )

        if callable(self.model) and not self.model.__class__.__module__.startswith("llama_cpp"):
            predicted = self.model(latent, timestep, prompt_embedding, negative_embedding)
            if isinstance(predicted, torch.Tensor):
                return predicted

        logger.warning_once = getattr(logger, "warning_once", set())
        key = "missing_diffusion_adapter"
        if key not in logger.warning_once:
            logger.warning(
                "Loaded reconstruction model has no latent diffusion adapter; preserving latent after noise schedule"
            )
            logger.warning_once.add(key)
        return torch.zeros_like(latent)


class FaceEnhancer:
    """Face-specific enhancement using dedicated face model."""

    def __init__(self, model, vae: VAEInference, device: str = "cpu"):
        self.model = model
        self.vae = vae
        self.device = device

    def enhance(self, face_latent: torch.Tensor, prompt: str = "detailed face, high quality",
                denoise: float = 0.08, steps: int = 4) -> torch.Tensor:
        """Enhance a face region in latent space.

        Uses lower denoise than main reconstruction to preserve identity.
        """
        if hasattr(self.model, "enhance_face"):
            return self.model.enhance_face(face_latent, prompt=prompt, denoise=denoise, steps=steps)

        sampler = DiffusionSampler(self.model, self.vae, device=self.device)
        prompt_embedding = torch.zeros((1, 1), device=face_latent.device)
        return sampler.sample(
            face_latent,
            prompt_embedding,
            prompt_embedding,
            num_steps=steps,
            denoise_strength=denoise,
        )
