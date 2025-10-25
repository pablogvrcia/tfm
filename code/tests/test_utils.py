"""
Test suite for utility functions and evaluation metrics.
Works on both CPU and GPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from PIL import Image

from utils import (
    compute_iou, compute_precision_recall, compute_f1,
    compute_boundary_f1, compute_mean_iou,
    create_mask_overlay, create_side_by_side,
    resize_image_pil, resize_mask
)


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics from Chapter 4.2."""

    def test_compute_iou_perfect_match(self):
        """Test IoU with identical masks."""
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)

        iou = compute_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 1.0, places=5)

    def test_compute_iou_no_overlap(self):
        """Test IoU with no overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:50, :] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:100, :] = True

        iou = compute_iou(mask1, mask2)
        self.assertAlmostEqual(iou, 0.0, places=5)

    def test_compute_iou_partial_overlap(self):
        """Test IoU with partial overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:60, :] = True  # Top 60%

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[40:100, :] = True  # Bottom 60%

        # Overlap is 20%, union is 100% of image
        iou = compute_iou(mask1, mask2)

        # Intersection: 20 rows, Union: 100 rows
        expected_iou = 20.0 / 100.0
        self.assertAlmostEqual(iou, expected_iou, places=2)

    def test_compute_iou_empty_masks(self):
        """Test IoU with empty masks."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask2 = np.zeros((100, 100), dtype=bool)

        iou = compute_iou(mask1, mask2)
        # Should return 0.0 for empty masks (convention)
        self.assertEqual(iou, 0.0)

    def test_precision_recall_perfect(self):
        """Test precision and recall with perfect match."""
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)

        precision, recall = compute_precision_recall(mask1, mask2)
        self.assertAlmostEqual(precision, 1.0, places=5)
        self.assertAlmostEqual(recall, 1.0, places=5)

    def test_precision_recall_no_match(self):
        """Test precision and recall with no overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:50, :] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:100, :] = True

        precision, recall = compute_precision_recall(mask1, mask2)
        self.assertAlmostEqual(precision, 0.0, places=5)
        self.assertAlmostEqual(recall, 0.0, places=5)

    def test_precision_recall_values(self):
        """Test precision and recall with known values."""
        # Ground truth: center 50x50 square
        gt = np.zeros((100, 100), dtype=bool)
        gt[25:75, 25:75] = True  # 2500 pixels

        # Prediction: slightly larger 60x60 square
        pred = np.zeros((100, 100), dtype=bool)
        pred[20:80, 20:80] = True  # 3600 pixels

        # Intersection: 50x50 = 2500
        # Precision: 2500 / 3600 = 0.694
        # Recall: 2500 / 2500 = 1.0

        precision, recall = compute_precision_recall(pred, gt)
        self.assertAlmostEqual(precision, 2500 / 3600, places=2)
        self.assertAlmostEqual(recall, 1.0, places=2)

    def test_compute_f1_perfect(self):
        """Test F1 score with perfect match."""
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)

        f1 = compute_f1(mask1, mask2)
        self.assertAlmostEqual(f1, 1.0, places=5)

    def test_compute_f1_no_match(self):
        """Test F1 score with no overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:50, :] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:100, :] = True

        f1 = compute_f1(mask1, mask2)
        self.assertAlmostEqual(f1, 0.0, places=5)

    def test_compute_f1_formula(self):
        """Test F1 score formula: F1 = 2 * (P * R) / (P + R)."""
        # Create masks with known precision and recall
        gt = np.zeros((100, 100), dtype=bool)
        gt[25:75, 25:75] = True

        pred = np.zeros((100, 100), dtype=bool)
        pred[20:80, 20:80] = True

        precision, recall = compute_precision_recall(pred, gt)
        f1 = compute_f1(pred, gt)

        expected_f1 = 2 * (precision * recall) / (precision + recall)
        self.assertAlmostEqual(f1, expected_f1, places=5)

    def test_boundary_f1(self):
        """Test boundary F1 score computation."""
        # Create masks with clear boundaries
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[20:80, 20:80] = True

        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[25:75, 25:75] = True

        # Should be able to compute boundary F1
        bf1 = compute_boundary_f1(mask1, mask2, threshold=2.0)

        # Boundary F1 should be between 0 and 1
        self.assertGreaterEqual(bf1, 0.0)
        self.assertLessEqual(bf1, 1.0)

    def test_mean_iou_single_class(self):
        """Test mean IoU with single class."""
        pred_masks = [np.ones((100, 100), dtype=bool)]
        gt_masks = [np.ones((100, 100), dtype=bool)]

        mean_iou, per_class = compute_mean_iou(pred_masks, gt_masks)
        self.assertAlmostEqual(mean_iou, 1.0, places=5)
        self.assertEqual(len(per_class), 1)

    def test_mean_iou_multiple_classes(self):
        """Test mean IoU with multiple classes."""
        # Class 1: perfect match
        pred1 = np.zeros((100, 100), dtype=bool)
        pred1[0:50, :] = True
        gt1 = pred1.copy()

        # Class 2: partial match
        pred2 = np.zeros((100, 100), dtype=bool)
        pred2[50:100, :] = True
        gt2 = np.zeros((100, 100), dtype=bool)
        gt2[60:100, :] = True

        pred_masks = [pred1, pred2]
        gt_masks = [gt1, gt2]

        mean_iou, per_class = compute_mean_iou(pred_masks, gt_masks)

        # Class 1 should have IoU = 1.0
        self.assertAlmostEqual(per_class[0], 1.0, places=2)

        # Mean should be average of both
        self.assertEqual(len(per_class), 2)
        self.assertAlmostEqual(mean_iou, np.mean(per_class), places=5)


class TestVisualization(unittest.TestCase):
    """Test visualization utility functions."""

    def test_create_mask_overlay(self):
        """Test creating mask overlay on image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True

        overlay = create_mask_overlay(image, mask, color=(255, 0, 0), alpha=0.5)

        # Check shape
        self.assertEqual(overlay.shape, image.shape)

        # Check dtype
        self.assertEqual(overlay.dtype, np.uint8)

        # Check that overlay is different from original
        self.assertFalse(np.array_equal(overlay, image))

    def test_create_mask_overlay_multiple_masks(self):
        """Test creating overlay with multiple masks."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        masks = [
            np.zeros((100, 100), dtype=bool),
            np.zeros((100, 100), dtype=bool)
        ]
        masks[0][10:40, 10:40] = True
        masks[1][60:90, 60:90] = True

        overlay = create_mask_overlay(
            image,
            masks,
            color=[(255, 0, 0), (0, 255, 0)],
            alpha=0.5
        )

        self.assertEqual(overlay.shape, image.shape)

    def test_create_side_by_side(self):
        """Test creating side-by-side comparison."""
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='blue')

        result = create_side_by_side([img1, img2])

        # Should be twice as wide
        self.assertEqual(result.size, (200, 100))

    def test_create_side_by_side_multiple(self):
        """Test side-by-side with multiple images."""
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='green'),
            Image.new('RGB', (100, 100), color='blue')
        ]

        result = create_side_by_side(images)

        # Should be three times as wide
        self.assertEqual(result.size, (300, 100))

    def test_create_side_by_side_with_labels(self):
        """Test side-by-side with labels."""
        images = [
            Image.new('RGB', (100, 100), color='red'),
            Image.new('RGB', (100, 100), color='blue')
        ]
        labels = ["Original", "Edited"]

        result = create_side_by_side(images, labels=labels)

        # Should still work with labels
        self.assertIsInstance(result, Image.Image)


class TestImageProcessing(unittest.TestCase):
    """Test image processing utilities."""

    def test_resize_image_pil(self):
        """Test resizing PIL image."""
        img = Image.new('RGB', (100, 100), color='red')

        resized = resize_image_pil(img, target_size=(50, 50))

        self.assertEqual(resized.size, (50, 50))

    def test_resize_image_pil_maintain_aspect(self):
        """Test resizing with aspect ratio maintained."""
        img = Image.new('RGB', (200, 100))

        # Resize to fit in 100x100, maintaining aspect
        resized = resize_image_pil(img, target_size=(100, 100), maintain_aspect=True)

        # Should be 100x50 (maintains 2:1 ratio)
        self.assertEqual(resized.size[0], 100)
        self.assertLessEqual(resized.size[1], 100)

    def test_resize_mask(self):
        """Test resizing binary mask."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True

        resized = resize_mask(mask, target_size=(50, 50))

        self.assertEqual(resized.shape, (50, 50))
        self.assertEqual(resized.dtype, bool)

        # Should still have some True values
        self.assertTrue(np.any(resized))

    def test_resize_mask_upscale(self):
        """Test upscaling mask."""
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:40, 10:40] = True

        resized = resize_mask(mask, target_size=(100, 100))

        self.assertEqual(resized.shape, (100, 100))
        self.assertTrue(np.any(resized))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_metrics_with_uint8_masks(self):
        """Test that metrics work with uint8 masks (0/255)."""
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[0:50, :] = 255

        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[0:50, :] = 255

        # Should handle uint8 masks
        iou = compute_iou(mask1, mask2)
        self.assertGreater(iou, 0.9)  # Should be close to 1.0

    def test_metrics_with_float_masks(self):
        """Test that metrics work with float masks (0.0/1.0)."""
        mask1 = np.zeros((100, 100), dtype=np.float32)
        mask1[0:50, :] = 1.0

        mask2 = np.zeros((100, 100), dtype=np.float32)
        mask2[0:50, :] = 1.0

        # Should handle float masks
        iou = compute_iou(mask1, mask2)
        self.assertGreater(iou, 0.9)

    def test_overlay_with_empty_mask(self):
        """Test overlay with empty mask."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)

        # Should not raise error
        overlay = create_mask_overlay(image, mask)
        self.assertEqual(overlay.shape, image.shape)

    def test_overlay_with_full_mask(self):
        """Test overlay with full mask."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=bool)

        # Should not raise error
        overlay = create_mask_overlay(image, mask)
        self.assertEqual(overlay.shape, image.shape)


if __name__ == '__main__':
    unittest.main(verbosity=2)
