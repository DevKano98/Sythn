"""Pipeline stages and orchestration for SynthID Remover."""
from .core import SynthIDPipeline, PipelineConfig, PipelineResult
from .stages import (
    Stage1_ImageInput, Stage2_AdaptiveDenoise, Stage3_EdgeExtraction,
    Stage4_ControlNet, Stage5_PromptEncoding, Stage6_LatentEncoding,
    Stage7_MainReconstruction, Stage8_FaceDetection, Stage9_FaceSegmentation,
    Stage10_FaceReconstruction, Stage11_Compositing, Stage12_Export
)
from .batch import BatchProcessor, BatchResult

__all__ = [
    "SynthIDPipeline",
    "PipelineConfig",
    "PipelineResult",
    "Stage1_ImageInput",
    "Stage2_AdaptiveDenoise",
    "Stage3_EdgeExtraction",
    "Stage4_ControlNet",
    "Stage5_PromptEncoding",
    "Stage6_LatentEncoding",
    "Stage7_MainReconstruction",
    "Stage8_FaceDetection",
    "Stage9_FaceSegmentation",
    "Stage10_FaceReconstruction",
    "Stage11_Compositing",
    "Stage12_Export",
    "BatchProcessor",
    "BatchResult",
]
