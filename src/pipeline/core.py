"""Core pipeline orchestrator for SynthID Remover"""
import os
import time
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..models.loader import ModelManager
from .stages import (
    Stage1_ImageInput, Stage2_AdaptiveDenoise, Stage3_EdgeExtraction,
    Stage4_ControlNet, Stage5_PromptEncoding, Stage6_LatentEncoding,
    Stage7_MainReconstruction, Stage8_FaceDetection, Stage9_FaceSegmentation,
    Stage10_FaceReconstruction, Stage11_Compositing, Stage12_Export
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline run."""
    image_path: str
    positive_prompt: str = "high quality photo, detailed, sharp focus"
    negative_prompt: str = "watermark, synthid, text, logo, blurry, low quality, artifact"
    denoise_auto: bool = True
    denoise_manual: float = 0.12
    denoise_scale: float = 1.0
    face_enhancement: bool = True
    face_denoise_scale: float = 0.7
    steps: int = 4
    cfg_scale: float = 7.0
    canny_low: int = 50
    canny_high: int = 150
    seed: int = -1
    output_format: str = "png"
    output_quality: int = 95
    output_suffix: str = "_synthid_removed"
    max_image_size: int = 2048


@dataclass 
class PipelineResult:
    """Result of a pipeline run."""
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    stages_completed: int = 0
    total_stages: int = 12
    metadata: Dict[str, Any] = field(default_factory=dict)


class SynthIDPipeline:
    """End-to-end SynthID removal pipeline."""

    def __init__(self, model_manager: ModelManager, output_dir: str, device: str = "cpu"):
        self.model_manager = model_manager
        self.output_dir = output_dir
        self.device = device
        self._setup_stages()
        self._cancelled = False

    def _setup_stages(self):
        """Initialize all pipeline stages."""
        self.stage1 = Stage1_ImageInput()
        self.stage2 = Stage2_AdaptiveDenoise()
        self.stage3 = Stage3_EdgeExtraction()
        self.stage4 = Stage4_ControlNet(self.model_manager)
        self.stage5 = Stage5_PromptEncoding(self.model_manager)
        self.stage6 = Stage6_LatentEncoding(self.model_manager, self.device)
        self.stage7 = Stage7_MainReconstruction(self.model_manager, self.device)
        self.stage8 = Stage8_FaceDetection(self.model_manager)
        self.stage9 = Stage9_FaceSegmentation()
        self.stage10 = Stage10_FaceReconstruction(self.model_manager, self.device)
        self.stage11 = Stage11_Compositing()
        self.stage12 = Stage12_Export(self.output_dir)

    def cancel(self):
        """Signal pipeline to cancel on next stage check."""
        self._cancelled = True

    def _check_cancelled(self) -> bool:
        return self._cancelled

    def run(self, config: PipelineConfig, 
            progress_callback: Optional[Callable[[int, int, str], None]] = None) -> PipelineResult:
        """
        Execute full pipeline.

        Args:
            config: Pipeline configuration
            progress_callback: Optional callback(current_stage, total_stages, status_message)

        Returns:
            PipelineResult with output info or error
        """
        start_time = time.time()
        metadata = {}

        def update_progress(stage_num: int, message: str):
            if progress_callback:
                progress_callback(stage_num, 12, message)
            logger.info(f"Stage {stage_num}/12: {message}")

        try:
            # Stage 1: Image Input
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(1, "Loading image...")
            s1 = self.stage1.process(config.image_path, config.max_image_size)
            image = s1["image"]
            metadata["input_size"] = s1["current_size"]

            # Stage 2: Adaptive Denoise
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(2, "Calculating adaptive denoise...")
            s2 = self.stage2.process(image, config.denoise_auto, 
                                     config.denoise_manual, config.denoise_scale)
            denoise = s2["denoise"]
            metadata["denoise"] = denoise
            metadata["resolution_info"] = s2["info"]

            # Stage 3: Edge Extraction
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(3, "Extracting edges (Canny)...")
            s3 = self.stage3.process(image, config.canny_low, config.canny_high)
            edge_tensor = s3["edge_tensor"]

            # Stage 4: ControlNet Conditioning
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(4, "Preparing ControlNet conditioning...")
            s4 = self.stage4.process(edge_tensor)
            conditioning = s4["conditioning"]

            # Stage 5: Prompt Encoding
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(5, "Encoding prompts...")
            s5 = self.stage5.process(config.positive_prompt, config.negative_prompt)
            prompt_data = s5

            # Stage 6: Latent Encoding
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(6, "Encoding to latent space...")
            s6 = self.stage6.process(image)
            latent = s6["latent"]
            vae = s6["vae"]

            # Stage 7: Main Reconstruction
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(7, f"Main reconstruction (denoise={denoise}, steps={config.steps})...")
            s7 = self.stage7.process(
                latent, conditioning, prompt_data, vae, denoise,
                config.steps, config.seed, config.cfg_scale
            )
            reconstructed_latent = s7["reconstructed_latent"]
            metadata["seed"] = s7["seed"]

            base_image = self.stage6.decode(reconstructed_latent)

            # Stage 8: Face Detection
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(8, "Detecting faces...")
            s8 = self.stage8.process(image)  # Detect on original for better accuracy
            faces = s8["faces"]
            face_count = s8["count"]
            metadata["face_count"] = face_count

            if face_count > 0 and config.face_enhancement:
                # Stage 9: Face Segmentation
                if self._check_cancelled():
                    return PipelineResult(False, error_message="Cancelled by user")
                update_progress(9, "Segmenting faces...")
                s9 = self.stage9.process(image, faces)
                masks = s9["masks"]
                face_regions = s9["face_regions"]

                # Stage 10: Face Reconstruction
                if self._check_cancelled():
                    return PipelineResult(False, error_message="Cancelled by user")
                update_progress(10, "Enhancing faces...")
                face_denoise = denoise * config.face_denoise_scale
                s10 = self.stage10.process(face_regions, face_denoise, config.steps)
                enhanced_faces = s10["enhanced_faces"]

                # Stage 11: Compositing
                if self._check_cancelled():
                    return PipelineResult(False, error_message="Cancelled by user")
                update_progress(11, "Compositing faces...")
                s11 = self.stage11.process(base_image, enhanced_faces, faces, masks)
                final_image = s11["final_image"]
            else:
                if face_count == 0:
                    update_progress(8, "No faces detected, skipping face enhancement")
                final_image = base_image

            # Stage 12: Export
            if self._check_cancelled():
                return PipelineResult(False, error_message="Cancelled by user")
            update_progress(12, "Saving output...")
            s12 = self.stage12.process(
                final_image,
                config.image_path,
                config.output_suffix,
                config.output_format,
                config.output_quality,
            )
            output_path = s12["output_path"]

            elapsed = time.time() - start_time

            return PipelineResult(
                success=True,
                output_path=output_path,
                processing_time=elapsed,
                stages_completed=12,
                total_stages=12,
                metadata=metadata
            )

        except Exception as e:
            logger.exception("Pipeline failed")
            elapsed = time.time() - start_time
            return PipelineResult(
                success=False,
                error_message=str(e),
                processing_time=elapsed,
                stages_completed=0,
                metadata=metadata
            )

    def run_batch(self, configs: list, 
                  progress_callback: Optional[Callable] = None) -> list:
        """Process multiple images sequentially."""
        results = []
        for i, config in enumerate(configs):
            def wrapped_progress(stage, total, msg):
                if progress_callback:
                    progress_callback(i, len(configs), stage, total, msg)

            result = self.run(config, wrapped_progress)
            results.append(result)
        return results
