"""
Test suite for the main pipeline.
Works on both CPU and GPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch
from PIL import Image

from pipeline import OpenVocabSegmentationPipeline
from config import PipelineConfig


class TestPipeline(unittest.TestCase):
    """Test the main segmentation pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {self.device}")

        # Create config for testing
        self.config = PipelineConfig()
        self.config.device = self.device
        self.config.sam2.points_per_side = 16  # Faster for testing

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.sam_generator)
        self.assertIsNotNone(pipeline.mask_aligner)
        self.assertIsNotNone(pipeline.inpainter)

    def test_segmentation_only(self):
        """Test segmentation without editing."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        # Create test image
        test_image = Image.new('RGB', (512, 512), color='red')

        # Segment
        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="red area",
            edit_operation=None,
            top_k=1
        )

        # Check result structure
        self.assertIn('original_image', result)
        self.assertIn('masks', result)
        self.assertIn('visualization', result)
        self.assertIn('timing', result)

        # Check masks
        self.assertGreater(len(result['masks']), 0)
        for mask_data in result['masks']:
            self.assertIn('mask', mask_data)
            self.assertIn('score', mask_data)

        # Check timing
        self.assertIn('sam2_generation', result['timing'])
        self.assertIn('clip_alignment', result['timing'])
        self.assertIn('total', result['timing'])

    def test_remove_operation(self):
        """Test removing objects from image."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='blue')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="blue area",
            edit_operation="remove",
            top_k=1
        )

        # Should have edited image
        self.assertIn('edited_image', result)
        self.assertIsInstance(result['edited_image'], Image.Image)

        # Should have inpainting timing
        self.assertIn('inpainting', result['timing'])

    def test_replace_operation(self):
        """Test replacing objects in image."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='green')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="green area",
            edit_operation="replace",
            edit_prompt="a red area",
            top_k=1
        )

        # Should have edited image
        self.assertIn('edited_image', result)
        self.assertIsInstance(result['edited_image'], Image.Image)

    def test_style_operation(self):
        """Test style transfer operation."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='yellow')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="yellow area",
            edit_operation="style",
            edit_prompt="oil painting style",
            top_k=1
        )

        # Should have edited image
        self.assertIn('edited_image', result)
        self.assertIsInstance(result['edited_image'], Image.Image)

    def test_multiple_masks(self):
        """Test returning multiple top-k masks."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='purple')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="purple area",
            edit_operation=None,
            top_k=3
        )

        # Should return up to 3 masks
        self.assertLessEqual(len(result['masks']), 3)
        self.assertGreater(len(result['masks']), 0)

    def test_invalid_operation(self):
        """Test handling of invalid edit operation."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='orange')

        with self.assertRaises(ValueError):
            pipeline.segment_and_edit(
                test_image,
                text_prompt="orange area",
                edit_operation="invalid_op"
            )

    def test_pipeline_with_numpy_input(self):
        """Test pipeline accepts numpy array input."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        # Create numpy array
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="test object",
            edit_operation=None
        )

        self.assertIn('masks', result)
        self.assertGreater(len(result['masks']), 0)

    def test_timing_measurements(self):
        """Test that timing is properly measured for all stages."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='cyan')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="cyan area",
            edit_operation="remove"
        )

        # Check all timing components
        timing = result['timing']
        self.assertIn('sam2_generation', timing)
        self.assertIn('clip_alignment', timing)
        self.assertIn('inpainting', timing)
        self.assertIn('total', timing)

        # All times should be positive
        for key, value in timing.items():
            self.assertGreater(value, 0, f"{key} should have positive time")

    def test_empty_edit_prompt_for_remove(self):
        """Test that remove operation works without edit_prompt."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        test_image = Image.new('RGB', (512, 512), color='magenta')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="magenta area",
            edit_operation="remove",
            edit_prompt=None  # Should work without this
        )

        self.assertIn('edited_image', result)

    def test_pipeline_preserves_image_size(self):
        """Test that output image size matches input."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        # Test with non-square image
        test_image = Image.new('RGB', (640, 480), color='brown')

        result = pipeline.segment_and_edit(
            test_image,
            text_prompt="brown area",
            edit_operation="replace",
            edit_prompt="blue background"
        )

        edited = result['edited_image']
        self.assertEqual(edited.size, test_image.size)


class TestPipelineBenchmark(unittest.TestCase):
    """Test the benchmark functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {self.device}")

        self.config = PipelineConfig()
        self.config.device = self.device
        self.config.sam2.points_per_side = 16

    def test_benchmark_method(self):
        """Test the benchmark method exists and works."""
        pipeline = OpenVocabSegmentationPipeline(config=self.config)

        # Check method exists
        self.assertTrue(hasattr(pipeline, 'benchmark'))

        # Create simple test data
        test_data = [
            {
                'image': Image.new('RGB', (256, 256), color='red'),
                'prompt': 'red area',
                'ground_truth': np.ones((256, 256), dtype=bool)
            }
        ]

        # Run benchmark
        results = pipeline.benchmark(test_data, num_samples=1)

        # Check results structure
        self.assertIn('metrics', results)
        self.assertIn('samples', results)


if __name__ == '__main__':
    unittest.main(verbosity=2)
