import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import APP_NAME, APP_VERSION, DEFAULTS, MODEL_SPECS
from src.comfy.models import ComfyModelManager
from src.comfy.pipeline import PipelineConfig
from src.comfy.workflow import WorkflowSettings, build_workflow
from src.utils.image import calculate_adaptive_denoise, canny_edge_detect, composite_images


class TestConfig(unittest.TestCase):
    def test_app_metadata(self):
        self.assertEqual(APP_NAME, "SynthID Remover")
        self.assertEqual(APP_VERSION, "2.0.0")

    def test_real_model_specs(self):
        self.assertEqual(MODEL_SPECS["diffusion_model"]["filename"], "z_image_turbo_bf16.safetensors")
        self.assertEqual(MODEL_SPECS["text_encoder"]["filename"], "qwen_3_4b.safetensors")
        self.assertEqual(MODEL_SPECS["vae"]["filename"], "ae.safetensors")


class TestImageUtils(unittest.TestCase):
    def test_adaptive_denoise(self):
        self.assertAlmostEqual(calculate_adaptive_denoise(np.zeros((512, 512, 3), dtype=np.uint8)), 0.08, delta=0.02)
        self.assertAlmostEqual(calculate_adaptive_denoise(np.zeros((1024, 1024, 3), dtype=np.uint8)), 0.12, delta=0.02)

    def test_canny_shape(self):
        edges = canny_edge_detect(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
        self.assertEqual(edges.shape, (128, 128, 3))

    def test_composite(self):
        bg = np.ones((100, 100, 3), dtype=np.uint8) * 100
        fg = np.ones((50, 50, 3), dtype=np.uint8) * 200
        mask = np.ones((50, 50), dtype=np.uint8) * 128
        result = composite_images(bg, fg, mask, (10, 10, 60, 60))
        self.assertEqual(result.shape, (100, 100, 3))


class TestComfyWorkflow(unittest.TestCase):
    def test_pipeline_config_defaults(self):
        config = PipelineConfig(image_path="image.png")
        self.assertEqual(config.steps, 9)
        self.assertEqual(config.cfg_scale, 1.0)
        self.assertTrue(config.face_enhancement)

    def test_workflow_contains_required_core_nodes(self):
        workflow = build_workflow("input.png", WorkflowSettings(face_enabled=False), object_info=None)
        classes = {node["class_type"] for node in workflow.values()}
        self.assertIn("LoadImage", classes)
        self.assertIn("ControlNetLoader", classes)
        self.assertIn("KSampler", classes)
        self.assertIn("SaveImage", classes)

    def test_model_manager_paths_target_comfy_folders(self):
        manager = ComfyModelManager()
        self.assertIn("diffusion_models", manager.model_path("diffusion_model").parts)
        self.assertIn("text_encoders", manager.model_path("text_encoder").parts)
        self.assertIn("controlnet", manager.model_path("controlnet").parts)


if __name__ == "__main__":
    unittest.main()
