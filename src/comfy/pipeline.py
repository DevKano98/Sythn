"""End-to-end ComfyUI execution pipeline."""
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image

from config import COMFY_BASE_URL, COMFYUI_DIR, OUTPUT_DIR
from src.utils.image import calculate_adaptive_denoise, load_image
from .client import ComfyClient
from .manager import ComfyManager
from .models import ComfyModelManager
from .workflow import WorkflowSettings, build_workflow, find_output_image


@dataclass
class PipelineConfig:
    image_path: str
    positive_prompt: str = "high quality photo, detailed, sharp focus"
    negative_prompt: str = "watermark, synthid, text, logo, blurry, low quality, artifact"
    denoise_auto: bool = False
    denoise_manual: float = 0.20
    face_enhancement: bool = True
    face_denoise_scale: float = 0.50
    steps: int = 9
    cfg_scale: float = 1.0
    canny_low: int = 50
    canny_high: int = 150
    seed: int = -1
    output_format: str = "png"
    output_quality: int = 95
    output_suffix: str = "_synthid_removed"


@dataclass
class PipelineResult:
    success: bool
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    stages_completed: int = 0
    total_stages: int = 8
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComfyPipeline:
    def __init__(self, output_dir: str = str(OUTPUT_DIR),
                 comfy_dir: Path = COMFYUI_DIR,
                 base_url: str = COMFY_BASE_URL,
                 manage_server: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manager = ComfyManager(comfy_dir=comfy_dir)
        self.client = ComfyClient(base_url)
        self.models = ComfyModelManager(comfy_dir=comfy_dir)
        self.manage_server = manage_server
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def stop(self):
        self.manager.stop()

    def run(self, config: PipelineConfig,
            progress_callback: Optional[Callable[[int, int, str], None]] = None) -> PipelineResult:
        start = time.time()
        total = 8

        def progress(stage: int, message: str) -> None:
            if progress_callback:
                progress_callback(stage, total, message)

        try:
            progress(1, "Checking ComfyUI and models")
            missing = self.models.missing_models
            if missing:
                raise RuntimeError(
                    "Missing ComfyUI models: " + ", ".join(missing)
                    + ". Run python download_models.py."
                )

            if self.manage_server:
                self.client = self.manager.start(wait=True)
            else:
                self.client.wait_until_ready()

            progress(2, "Reading ComfyUI node registry")
            object_info = self.client.get_object_info()

            progress(3, "Uploading image")
            uploaded_name = self.client.upload_image(config.image_path)

            progress(4, "Building workflow")
            denoise = config.denoise_manual
            if config.denoise_auto:
                denoise = calculate_adaptive_denoise(load_image(config.image_path))
            settings = WorkflowSettings(
                positive_prompt=config.positive_prompt.strip(),
                negative_prompt=config.negative_prompt.strip(),
                denoise=denoise,
                face_enabled=config.face_enhancement,
                face_denoise=denoise * config.face_denoise_scale,
                steps=config.steps,
                cfg_scale=config.cfg_scale,
                seed=config.seed,
                canny_low=config.canny_low,
                canny_high=config.canny_high,
            )
            workflow = build_workflow(uploaded_name, settings, object_info)

            progress(5, "Queueing workflow")
            prompt_id = self.client.queue_prompt(workflow, client_id=str(uuid.uuid4()))
            if self._cancelled:
                return PipelineResult(False, error_message="Cancelled by user")

            progress(6, "Waiting for ComfyUI result")
            history_item = self.client.wait_for_result(
                prompt_id,
                progress=lambda msg: progress(6, msg),
            )

            progress(7, "Downloading output")
            image_info = find_output_image(history_item, prefer_final=config.face_enhancement)
            tmp_path = self.output_dir / f"{Path(config.image_path).stem}{config.output_suffix}.png"
            self.client.download_view(
                image_info["filename"],
                image_info.get("subfolder", ""),
                image_info.get("type", "output"),
                str(tmp_path),
            )

            progress(8, "Saving requested output format")
            output_path = self._encode_output(tmp_path, config)
            if output_path != tmp_path:
                tmp_path.unlink(missing_ok=True)

            return PipelineResult(
                success=True,
                output_path=str(output_path),
                processing_time=time.time() - start,
                stages_completed=total,
                total_stages=total,
                metadata={"prompt_id": prompt_id, "denoise": denoise},
            )
        except Exception as exc:
            return PipelineResult(
                success=False,
                error_message=str(exc),
                processing_time=time.time() - start,
            )

    def _encode_output(self, source: Path, config: PipelineConfig) -> Path:
        fmt = config.output_format.lower()
        if fmt == "jpg":
            fmt = "jpeg"
        target = self.output_dir / f"{Path(config.image_path).stem}{config.output_suffix}.{config.output_format.lower()}"
        if source.suffix.lower().lstrip(".") == config.output_format.lower():
            if source != target:
                shutil.move(str(source), str(target))
            return target

        image = Image.open(source)
        if fmt == "jpeg":
            image = image.convert("RGB")
            image.save(target, format="JPEG", quality=config.output_quality)
        elif fmt == "webp":
            image.save(target, format="WEBP", quality=config.output_quality)
        else:
            image.save(target, format="PNG")
        return target
