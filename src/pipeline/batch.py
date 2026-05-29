"""Batch processing utilities for SynthID Remover."""
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from ..pipeline.core import SynthIDPipeline, PipelineConfig, PipelineResult
from ..utils.helpers import get_image_files, ensure_dir
from ..utils.validators import validate_image_path, validate_output_dir

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of a batch processing job."""
    total: int
    successful: int
    failed: int
    results: List[PipelineResult]
    total_time: float
    errors: List[str]


class BatchProcessor:
    """Process multiple images in batch mode."""

    def __init__(self, pipeline: SynthIDPipeline, output_dir: str):
        self.pipeline = pipeline
        self.output_dir = ensure_dir(output_dir)

    def process_folder(self, input_dir: str, 
                       config_template: Optional[PipelineConfig] = None,
                       progress_callback: Optional[Callable] = None,
                       file_filter: Optional[str] = None) -> BatchResult:
        """Process all images in a folder.

        Args:
            input_dir: Folder containing images
            config_template: Base config to use for all images (image_path will be overridden)
            progress_callback: Called with (current, total, filename, result)
            file_filter: Optional glob pattern to filter files

        Returns:
            BatchResult with summary
        """
        start = time.time()

        if file_filter:
            files = list(Path(input_dir).glob(file_filter))
            files = [str(f) for f in files if f.is_file()]
        else:
            files = get_image_files(input_dir)

        if not files:
            logger.warning(f"No images found in {input_dir}")
            return BatchResult(0, 0, 0, [], 0.0, ["No images found"])

        results = []
        errors = []
        successful = 0

        for i, file_path in enumerate(files):
            try:
                validate_image_path(file_path)

                if config_template:
                    config = PipelineConfig(
                        image_path=file_path,
                        positive_prompt=config_template.positive_prompt,
                        negative_prompt=config_template.negative_prompt,
                        denoise_auto=config_template.denoise_auto,
                        denoise_manual=config_template.denoise_manual,
                        denoise_scale=config_template.denoise_scale,
                        face_enhancement=config_template.face_enhancement,
                        face_denoise_scale=config_template.face_denoise_scale,
                        steps=config_template.steps,
                        cfg_scale=config_template.cfg_scale,
                        canny_low=config_template.canny_low,
                        canny_high=config_template.canny_high,
                        seed=config_template.seed,
                        output_format=config_template.output_format,
                        output_quality=config_template.output_quality,
                        output_suffix=config_template.output_suffix,
                        max_image_size=config_template.max_image_size,
                    )
                else:
                    config = PipelineConfig(image_path=file_path)

                def stage_progress(stage, total, msg):
                    if progress_callback:
                        progress_callback(i, len(files), stage, total, 
                                        Path(file_path).name, msg)

                result = self.pipeline.run(config, stage_progress)
                results.append(result)

                if result.success:
                    successful += 1
                    logger.info(f"OK {file_path}")
                else:
                    errors.append(f"{file_path}: {result.error_message}")
                    logger.error(f"FAILED {file_path}: {result.error_message}")

                if progress_callback:
                    progress_callback(i, len(files), 12, 12, 
                                    Path(file_path).name, 
                                    "Done" if result.success else "Failed")

            except Exception as e:
                logger.exception(f"Failed to process {file_path}")
                errors.append(f"{file_path}: {str(e)}")
                results.append(PipelineResult(False, error_message=str(e)))

        total_time = time.time() - start

        return BatchResult(
            total=len(files),
            successful=successful,
            failed=len(files) - successful,
            results=results,
            total_time=total_time,
            errors=errors
        )

    def process_list(self, file_paths: List[str],
                     config_template: Optional[PipelineConfig] = None,
                     progress_callback: Optional[Callable] = None) -> BatchResult:
        """Process a specific list of image files."""
        start = time.time()
        results = []
        errors = []
        successful = 0

        for i, file_path in enumerate(file_paths):
            try:
                validate_image_path(file_path)
                if config_template:
                    config = PipelineConfig(
                        image_path=file_path,
                        positive_prompt=config_template.positive_prompt,
                        negative_prompt=config_template.negative_prompt,
                        denoise_auto=config_template.denoise_auto,
                        denoise_manual=config_template.denoise_manual,
                        denoise_scale=config_template.denoise_scale,
                        face_enhancement=config_template.face_enhancement,
                        face_denoise_scale=config_template.face_denoise_scale,
                        steps=config_template.steps,
                        cfg_scale=config_template.cfg_scale,
                        canny_low=config_template.canny_low,
                        canny_high=config_template.canny_high,
                        seed=config_template.seed,
                        output_format=config_template.output_format,
                        output_quality=config_template.output_quality,
                        output_suffix=config_template.output_suffix,
                        max_image_size=config_template.max_image_size,
                    )
                else:
                    config = PipelineConfig(image_path=file_path)

                def stage_progress(stage, total, msg):
                    if progress_callback:
                        progress_callback(i, len(file_paths), stage, total,
                                          Path(file_path).name, msg)

                result = self.pipeline.run(config, stage_progress)
                results.append(result)
                if result.success:
                    successful += 1
                else:
                    errors.append(f"{file_path}: {result.error_message}")
            except Exception as e:
                logger.exception(f"Failed to process {file_path}")
                errors.append(f"{file_path}: {str(e)}")
                results.append(PipelineResult(False, error_message=str(e)))

        total_time = time.time() - start
        return BatchResult(
            total=len(file_paths),
            successful=successful,
            failed=len(file_paths) - successful,
            results=results,
            total_time=total_time,
            errors=errors,
        )
