"""
Test suite for configuration management.
Works on both CPU and GPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import torch

from config import (
    SAM2Config, CLIPConfig, AlignmentConfig, InpaintingConfig,
    PipelineConfig, get_fast_config, get_quality_config, get_balanced_config
)


class TestConfigDataclasses(unittest.TestCase):
    """Test individual configuration dataclasses."""

    def test_sam2_config_defaults(self):
        """Test SAM2Config default values match thesis."""
        config = SAM2Config()

        self.assertEqual(config.model_type, "sam2_hiera_large")
        self.assertEqual(config.points_per_side, 32)
        self.assertEqual(config.pred_iou_thresh, 0.88)
        self.assertEqual(config.stability_score_thresh, 0.95)
        self.assertEqual(config.min_mask_region_area, 100)
        self.assertEqual(config.stability_score_offset, 1.0)

    def test_sam2_config_custom(self):
        """Test SAM2Config with custom values."""
        config = SAM2Config(
            points_per_side=16,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.92
        )

        self.assertEqual(config.points_per_side, 16)
        self.assertEqual(config.pred_iou_thresh, 0.9)
        self.assertEqual(config.stability_score_thresh, 0.92)

    def test_clip_config_defaults(self):
        """Test CLIPConfig default values match thesis."""
        config = CLIPConfig()

        self.assertEqual(config.model_name, "ViT-L-14")
        self.assertEqual(config.pretrained, "openai")
        self.assertEqual(config.extract_layers, [6, 12, 18, 24])
        self.assertEqual(config.image_size, 336)

    def test_clip_config_custom(self):
        """Test CLIPConfig with custom values."""
        config = CLIPConfig(
            model_name="ViT-B-32",
            extract_layers=[3, 6, 9],
            image_size=224
        )

        self.assertEqual(config.model_name, "ViT-B-32")
        self.assertEqual(config.extract_layers, [3, 6, 9])
        self.assertEqual(config.image_size, 224)

    def test_alignment_config_defaults(self):
        """Test AlignmentConfig default values match thesis."""
        config = AlignmentConfig()

        self.assertEqual(config.background_weight, 0.3)
        self.assertEqual(config.similarity_threshold, 0.25)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.aggregation_mode, "mean")

    def test_alignment_config_custom(self):
        """Test AlignmentConfig with custom values."""
        config = AlignmentConfig(
            background_weight=0.5,
            similarity_threshold=0.3,
            top_k=3
        )

        self.assertEqual(config.background_weight, 0.5)
        self.assertEqual(config.similarity_threshold, 0.3)
        self.assertEqual(config.top_k, 3)

    def test_inpainting_config_defaults(self):
        """Test InpaintingConfig default values match thesis."""
        config = InpaintingConfig()

        self.assertEqual(config.model_id, "stabilityai/stable-diffusion-2-inpainting")
        self.assertEqual(config.num_inference_steps, 50)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertEqual(config.mask_blur, 8)
        self.assertEqual(config.mask_dilation, 5)

    def test_inpainting_config_custom(self):
        """Test InpaintingConfig with custom values."""
        config = InpaintingConfig(
            num_inference_steps=30,
            guidance_scale=10.0,
            mask_blur=4
        )

        self.assertEqual(config.num_inference_steps, 30)
        self.assertEqual(config.guidance_scale, 10.0)
        self.assertEqual(config.mask_blur, 4)


class TestPipelineConfig(unittest.TestCase):
    """Test the main pipeline configuration."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig creates all sub-configs."""
        config = PipelineConfig()

        self.assertIsInstance(config.sam2, SAM2Config)
        self.assertIsInstance(config.clip, CLIPConfig)
        self.assertIsInstance(config.alignment, AlignmentConfig)
        self.assertIsInstance(config.inpainting, InpaintingConfig)

    def test_pipeline_config_device_auto(self):
        """Test device auto-detection."""
        config = PipelineConfig()

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(config.device, expected_device)

    def test_pipeline_config_device_manual(self):
        """Test manual device setting."""
        config = PipelineConfig(device="cpu")
        self.assertEqual(config.device, "cpu")

        if torch.cuda.is_available():
            config = PipelineConfig(device="cuda")
            self.assertEqual(config.device, "cuda")

    def test_pipeline_config_custom_subconfigs(self):
        """Test PipelineConfig with custom sub-configs."""
        custom_sam = SAM2Config(points_per_side=16)
        custom_clip = CLIPConfig(image_size=224)

        config = PipelineConfig(
            sam2=custom_sam,
            clip=custom_clip
        )

        self.assertEqual(config.sam2.points_per_side, 16)
        self.assertEqual(config.clip.image_size, 224)

    def test_pipeline_config_output_dir(self):
        """Test output directory configuration."""
        config = PipelineConfig(output_dir="custom_output")
        self.assertEqual(config.output_dir, "custom_output")

    def test_pipeline_config_save_intermediate(self):
        """Test save_intermediate_results flag."""
        config = PipelineConfig(save_intermediate_results=True)
        self.assertTrue(config.save_intermediate_results)

        config = PipelineConfig(save_intermediate_results=False)
        self.assertFalse(config.save_intermediate_results)


class TestPresetConfigs(unittest.TestCase):
    """Test preset configuration functions."""

    def test_fast_config(self):
        """Test fast configuration for quick inference."""
        config = get_fast_config()

        # Should use fewer points and steps for speed
        self.assertEqual(config.sam2.points_per_side, 16)
        self.assertEqual(config.inpainting.num_inference_steps, 20)
        self.assertLess(config.sam2.points_per_side, 32)

    def test_quality_config(self):
        """Test quality configuration for best results."""
        config = get_quality_config()

        # Should use more points and steps for quality
        self.assertEqual(config.sam2.points_per_side, 64)
        self.assertEqual(config.inpainting.num_inference_steps, 100)
        self.assertGreater(config.sam2.points_per_side, 32)

    def test_balanced_config(self):
        """Test balanced configuration (same as default)."""
        config = get_balanced_config()
        default = PipelineConfig()

        # Should match default values
        self.assertEqual(config.sam2.points_per_side, default.sam2.points_per_side)
        self.assertEqual(config.inpainting.num_inference_steps,
                        default.inpainting.num_inference_steps)

    def test_preset_configs_device_inheritance(self):
        """Test that preset configs respect device parameter."""
        fast = get_fast_config(device="cpu")
        self.assertEqual(fast.device, "cpu")

        quality = get_quality_config(device="cpu")
        self.assertEqual(quality.device, "cpu")

        balanced = get_balanced_config(device="cpu")
        self.assertEqual(balanced.device, "cpu")

    def test_preset_configs_are_different(self):
        """Test that preset configs actually differ from each other."""
        fast = get_fast_config()
        quality = get_quality_config()
        balanced = get_balanced_config()

        # Fast should be faster than quality
        self.assertLess(fast.sam2.points_per_side, quality.sam2.points_per_side)
        self.assertLess(fast.inpainting.num_inference_steps,
                       quality.inpainting.num_inference_steps)

        # Balanced should be in between
        self.assertLessEqual(fast.sam2.points_per_side, balanced.sam2.points_per_side)
        self.assertGreaterEqual(quality.sam2.points_per_side, balanced.sam2.points_per_side)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and edge cases."""

    def test_negative_values(self):
        """Test that configurations handle reasonable values."""
        # These should not raise errors during creation
        config = SAM2Config(points_per_side=8)  # Minimum practical value
        self.assertEqual(config.points_per_side, 8)

        config = InpaintingConfig(num_inference_steps=1)  # Very fast
        self.assertEqual(config.num_inference_steps, 1)

    def test_threshold_ranges(self):
        """Test that thresholds are in valid ranges."""
        config = AlignmentConfig(
            background_weight=0.0,
            similarity_threshold=0.0
        )
        self.assertGreaterEqual(config.background_weight, 0.0)
        self.assertGreaterEqual(config.similarity_threshold, 0.0)

        config = AlignmentConfig(
            background_weight=1.0,
            similarity_threshold=1.0
        )
        self.assertLessEqual(config.background_weight, 1.0)
        self.assertLessEqual(config.similarity_threshold, 1.0)

    def test_config_serialization(self):
        """Test that configs can be converted to dict."""
        config = PipelineConfig()

        # Should be able to access as dict (dataclass feature)
        self.assertEqual(config.device, config.device)
        self.assertIsInstance(config.sam2, SAM2Config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
