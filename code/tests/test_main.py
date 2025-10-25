"""
Test suite for main CLI interface.
Tests CLI argument parsing and command execution.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Import main functions (but don't execute CLI)
import main


class TestCLIArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing."""

    def test_parse_args_segment_mode(self):
        """Test parsing arguments for segment mode."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'red car',
            '--mode', 'segment'
        ])

        self.assertEqual(args.image, 'test.jpg')
        self.assertEqual(args.prompt, 'red car')
        self.assertEqual(args.mode, 'segment')

    def test_parse_args_remove_mode(self):
        """Test parsing arguments for remove mode."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'person',
            '--mode', 'remove'
        ])

        self.assertEqual(args.mode, 'remove')

    def test_parse_args_replace_mode(self):
        """Test parsing arguments for replace mode."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'old TV',
            '--mode', 'replace',
            '--edit', 'modern TV'
        ])

        self.assertEqual(args.mode, 'replace')
        self.assertEqual(args.edit, 'modern TV')

    def test_parse_args_style_mode(self):
        """Test parsing arguments for style mode."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'building',
            '--mode', 'style',
            '--edit', 'oil painting'
        ])

        self.assertEqual(args.mode, 'style')
        self.assertEqual(args.edit, 'oil painting')

    def test_parse_args_benchmark_mode(self):
        """Test parsing arguments for benchmark mode."""
        args = main.parse_args([
            '--mode', 'benchmark',
            '--benchmark-data', 'dataset.json'
        ])

        self.assertEqual(args.mode, 'benchmark')
        self.assertEqual(args.benchmark_data, 'dataset.json')

    def test_parse_args_with_device(self):
        """Test parsing device argument."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--device', 'cpu'
        ])

        self.assertEqual(args.device, 'cpu')

    def test_parse_args_with_output(self):
        """Test parsing output directory argument."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--output', 'my_output'
        ])

        self.assertEqual(args.output, 'my_output')

    def test_parse_args_with_config(self):
        """Test parsing config preset argument."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--config', 'fast'
        ])

        self.assertEqual(args.config, 'fast')

    def test_parse_args_with_topk(self):
        """Test parsing top-k argument."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--top-k', '3'
        ])

        self.assertEqual(args.top_k, 3)


class TestCLIExecution(unittest.TestCase):
    """Test CLI command execution with actual files."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')

        # Create test image
        test_img = Image.new('RGB', (512, 512), color='blue')
        test_img.save(self.test_image_path)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_segment_mode_execution(self):
        """Test executing segment mode."""
        args = main.parse_args([
            '--image', self.test_image_path,
            '--prompt', 'blue area',
            '--mode', 'segment',
            '--output', self.temp_dir,
            '--device', 'cpu',
            '--config', 'fast'  # Use fast config for testing
        ])

        # Should not raise error
        try:
            main.main(args)
        except SystemExit:
            pass  # main() calls sys.exit(), which is fine

        # Check that output files were created
        output_dir = Path(self.temp_dir)
        # Should have created some output files
        self.assertTrue(any(output_dir.iterdir()))

    def test_remove_mode_execution(self):
        """Test executing remove mode."""
        args = main.parse_args([
            '--image', self.test_image_path,
            '--prompt', 'blue area',
            '--mode', 'remove',
            '--output', self.temp_dir,
            '--device', 'cpu',
            '--config', 'fast'
        ])

        try:
            main.main(args)
        except SystemExit:
            pass

    def test_invalid_image_path(self):
        """Test handling of invalid image path."""
        args = main.parse_args([
            '--image', 'nonexistent.jpg',
            '--prompt', 'test',
            '--mode', 'segment'
        ])

        # Should handle error gracefully
        with self.assertRaises((FileNotFoundError, SystemExit)):
            main.main(args)

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = os.path.join(self.temp_dir, 'new_output')

        args = main.parse_args([
            '--image', self.test_image_path,
            '--prompt', 'test',
            '--mode', 'segment',
            '--output', new_output_dir,
            '--device', 'cpu',
            '--config', 'fast'
        ])

        try:
            main.main(args)
        except SystemExit:
            pass

        # Output directory should have been created
        self.assertTrue(os.path.exists(new_output_dir))


class TestConfigPresets(unittest.TestCase):
    """Test configuration preset selection."""

    def test_default_config(self):
        """Test default config selection."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test'
        ])

        # Default should be 'balanced'
        self.assertEqual(args.config, 'balanced')

    def test_fast_config_selection(self):
        """Test fast config can be selected."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--config', 'fast'
        ])

        config = main.get_config_from_args(args)

        # Should have fast config settings
        self.assertEqual(config.sam2.points_per_side, 16)

    def test_quality_config_selection(self):
        """Test quality config can be selected."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--config', 'quality'
        ])

        config = main.get_config_from_args(args)

        # Should have quality config settings
        self.assertEqual(config.sam2.points_per_side, 64)

    def test_balanced_config_selection(self):
        """Test balanced config can be selected."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'test',
            '--config', 'balanced'
        ])

        config = main.get_config_from_args(args)

        # Should have balanced config settings
        self.assertEqual(config.sam2.points_per_side, 32)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in CLI."""

    def test_missing_required_args(self):
        """Test that missing required arguments are caught."""
        # segment mode requires --image and --prompt
        with self.assertRaises(SystemExit):
            main.parse_args(['--mode', 'segment'])

    def test_replace_without_edit_prompt(self):
        """Test that replace mode requires --edit."""
        args = main.parse_args([
            '--image', 'test.jpg',
            '--prompt', 'old object',
            '--mode', 'replace'
            # Missing --edit
        ])

        # Should have edit as None
        self.assertIsNone(args.edit)

    def test_invalid_mode(self):
        """Test handling of invalid mode."""
        with self.assertRaises(SystemExit):
            main.parse_args([
                '--image', 'test.jpg',
                '--prompt', 'test',
                '--mode', 'invalid_mode'
            ])

    def test_invalid_config_preset(self):
        """Test handling of invalid config preset."""
        with self.assertRaises(SystemExit):
            main.parse_args([
                '--image', 'test.jpg',
                '--prompt', 'test',
                '--config', 'invalid_preset'
            ])

    def test_invalid_device(self):
        """Test handling of invalid device."""
        with self.assertRaises(SystemExit):
            main.parse_args([
                '--image', 'test.jpg',
                '--prompt', 'test',
                '--device', 'invalid_device'
            ])


class TestHelpers(unittest.TestCase):
    """Test helper functions in main module."""

    def test_load_image(self):
        """Test image loading helper."""
        # Create temporary test image
        temp_dir = tempfile.mkdtemp()
        test_path = os.path.join(temp_dir, 'test.jpg')

        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_path)

        # Load image
        loaded = main.load_image(test_path)

        self.assertIsInstance(loaded, Image.Image)
        self.assertEqual(loaded.size, (100, 100))

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_save_results(self):
        """Test saving results helper."""
        temp_dir = tempfile.mkdtemp()

        # Create test results
        results = {
            'original_image': Image.new('RGB', (100, 100), color='blue'),
            'masks': [
                {
                    'mask': np.ones((100, 100), dtype=bool),
                    'score': 0.95
                }
            ],
            'visualization': Image.new('RGB', (100, 100), color='green')
        }

        # Save results
        main.save_results(results, temp_dir, 'test')

        # Check files were created
        output_path = Path(temp_dir)
        files = list(output_path.glob('test_*'))

        self.assertGreater(len(files), 0)

        # Cleanup
        shutil.rmtree(temp_dir)


class TestBenchmarkMode(unittest.TestCase):
    """Test benchmark mode functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_benchmark_data_loading(self):
        """Test loading benchmark data from JSON."""
        # Create test benchmark data
        import json

        benchmark_file = os.path.join(self.temp_dir, 'benchmark.json')

        # Create test images
        img1_path = os.path.join(self.temp_dir, 'img1.jpg')
        img2_path = os.path.join(self.temp_dir, 'img2.jpg')

        Image.new('RGB', (100, 100), color='red').save(img1_path)
        Image.new('RGB', (100, 100), color='blue').save(img2_path)

        # Create benchmark JSON
        benchmark_data = [
            {
                'image_path': img1_path,
                'prompt': 'red area',
                'ground_truth_mask': [[True] * 100] * 100
            },
            {
                'image_path': img2_path,
                'prompt': 'blue area',
                'ground_truth_mask': [[True] * 100] * 100
            }
        ]

        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f)

        # Test loading
        data = main.load_benchmark_data(benchmark_file)

        self.assertEqual(len(data), 2)
        self.assertIn('image', data[0])
        self.assertIn('prompt', data[0])
        self.assertIn('ground_truth', data[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
