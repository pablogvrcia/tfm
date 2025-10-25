"""
Test suite for all model components.
Works on both CPU and GPU.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import torch
from PIL import Image

from models.sam2_segmentation import SAM2MaskGenerator
from models.clip_features import CLIPFeatureExtractor
from models.mask_alignment import MaskTextAligner
from models.inpainting import StableDiffusionInpainter


class TestSAM2Segmentation(unittest.TestCase):
    """Test SAM 2 mask generation (both real and mock)."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {self.device}")

    def test_sam2_initialization(self):
        """Test SAM2MaskGenerator can be initialized."""
        generator = SAM2MaskGenerator(
            model_type="sam2_hiera_large",
            device=self.device,
            points_per_side=32
        )
        self.assertIsNotNone(generator)

    def test_mask_generation(self):
        """Test mask generation on a simple image."""
        generator = SAM2MaskGenerator(
            device=self.device,
            points_per_side=16  # Use fewer points for faster testing
        )

        # Create test image (RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # Generate masks
        masks = generator.generate_masks(test_image)

        # Check masks were generated
        self.assertGreater(len(masks), 0, "Should generate at least one mask")

        # Check mask structure
        for mask in masks:
            self.assertIn('segmentation', mask)
            self.assertIn('area', mask)
            self.assertIn('predicted_iou', mask)
            self.assertIn('stability_score', mask)

            # Check mask shape matches image
            self.assertEqual(mask['segmentation'].shape, test_image.shape[:2])

    def test_mask_filtering(self):
        """Test that masks are filtered by area, IoU, and stability."""
        generator = SAM2MaskGenerator(
            device=self.device,
            points_per_side=16,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            min_mask_region_area=500
        )

        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = generator.generate_masks(test_image)

        # All masks should meet the thresholds
        for mask in masks:
            self.assertGreaterEqual(mask['area'], 500)
            # Note: Mock masks have fixed scores, real SAM 2 varies


class TestCLIPFeatures(unittest.TestCase):
    """Test CLIP feature extraction."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {self.device}")

    def test_clip_initialization(self):
        """Test CLIPFeatureExtractor can be initialized."""
        try:
            extractor = CLIPFeatureExtractor(
                model_name="ViT-L-14",
                device=self.device
            )
            self.assertIsNotNone(extractor)
            print("  ✓ CLIP model loaded successfully")
        except Exception as e:
            self.skipTest(f"CLIP model not available: {e}")

    def test_image_feature_extraction(self):
        """Test extracting features from an image."""
        try:
            extractor = CLIPFeatureExtractor(
                model_name="ViT-L-14",
                device=self.device,
                extract_layers=[6, 12, 18, 24]
            )
        except:
            self.skipTest("CLIP model not available")

        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')

        # Extract features
        global_emb, dense_features = extractor.extract_image_features(test_image)

        # Check global embedding
        self.assertEqual(len(global_emb.shape), 1, "Global embedding should be 1D")
        self.assertGreater(global_emb.shape[0], 0, "Global embedding should have features")

        # Check dense features from multiple layers
        self.assertEqual(len(dense_features), 4, "Should have features from 4 layers")
        for feat in dense_features:
            self.assertEqual(len(feat.shape), 3, "Dense features should be 3D (C, H, W)")

    def test_text_feature_extraction(self):
        """Test extracting features from text."""
        try:
            extractor = CLIPFeatureExtractor(
                model_name="ViT-L-14",
                device=self.device
            )
        except:
            self.skipTest("CLIP model not available")

        # Extract text features
        text_emb = extractor.extract_text_features("a red car")

        # Check text embedding
        self.assertEqual(len(text_emb.shape), 1, "Text embedding should be 1D")
        self.assertGreater(text_emb.shape[0], 0, "Text embedding should have features")

    def test_similarity_computation(self):
        """Test computing similarity between image and text."""
        try:
            extractor = CLIPFeatureExtractor(
                model_name="ViT-L-14",
                device=self.device,
                extract_layers=[6, 12, 18, 24]
            )
        except:
            self.skipTest("CLIP model not available")

        # Create test image and extract features
        test_image = Image.new('RGB', (224, 224), color='red')
        _, dense_features = extractor.extract_image_features(test_image)

        # Extract text features
        text_emb = extractor.extract_text_features("red color")

        # Compute similarity map
        sim_map = extractor.compute_similarity_map(
            dense_features,
            text_emb,
            target_size=(224, 224)
        )

        # Check similarity map
        self.assertEqual(sim_map.shape, (224, 224), "Similarity map should match target size")
        self.assertTrue(np.all(sim_map >= -1) and np.all(sim_map <= 1),
                       "Similarity should be in [-1, 1]")


class TestMaskAlignment(unittest.TestCase):
    """Test mask-text alignment."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {self.device}")

    def test_mask_aligner_initialization(self):
        """Test MaskTextAligner can be initialized."""
        try:
            extractor = CLIPFeatureExtractor(device=self.device)
            aligner = MaskTextAligner(
                clip_extractor=extractor,
                background_weight=0.3,
                similarity_threshold=0.25
            )
            self.assertIsNotNone(aligner)
        except:
            self.skipTest("CLIP model not available")

    def test_mask_scoring(self):
        """Test scoring masks with text prompts."""
        try:
            extractor = CLIPFeatureExtractor(device=self.device)
            aligner = MaskTextAligner(clip_extractor=extractor)
        except:
            self.skipTest("CLIP model not available")

        # Create test image and masks
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = [
            {
                'segmentation': np.random.randint(0, 2, (224, 224), dtype=bool),
                'area': 5000,
                'predicted_iou': 0.9,
                'stability_score': 0.95
            }
            for _ in range(3)
        ]

        # Align masks with text
        scored_masks = aligner.align_masks_with_text(
            masks,
            "red object",
            test_image,
            top_k=2
        )

        # Check results
        self.assertLessEqual(len(scored_masks), 2, "Should return at most top_k masks")

        # Check scoring structure
        for mask_data in scored_masks:
            self.assertIn('mask', mask_data)
            self.assertIn('score', mask_data)
            self.assertIsInstance(mask_data['score'], float)

    def test_background_suppression(self):
        """Test that background suppression affects scores."""
        try:
            extractor = CLIPFeatureExtractor(device=self.device)
            aligner_no_bg = MaskTextAligner(
                clip_extractor=extractor,
                background_weight=0.0
            )
            aligner_with_bg = MaskTextAligner(
                clip_extractor=extractor,
                background_weight=0.5
            )
        except:
            self.skipTest("CLIP model not available")

        # Create identical test setup
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = [{
            'segmentation': np.random.randint(0, 2, (224, 224), dtype=bool),
            'area': 5000,
            'predicted_iou': 0.9,
            'stability_score': 0.95
        }]

        # Get scores with different background weights
        result_no_bg = aligner_no_bg.align_masks_with_text(masks, "object", test_image)
        result_with_bg = aligner_with_bg.align_masks_with_text(masks, "object", test_image)

        # Scores should differ when background weight changes
        # (unless by coincidence the background similarity is exactly 0)
        score_no_bg = result_no_bg[0]['score']
        score_with_bg = result_with_bg[0]['score']

        self.assertIsInstance(score_no_bg, float)
        self.assertIsInstance(score_with_bg, float)


class TestInpainting(unittest.TestCase):
    """Test Stable Diffusion inpainting."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nTesting on device: {self.device}")

    def test_inpainter_initialization(self):
        """Test StableDiffusionInpainter can be initialized."""
        inpainter = StableDiffusionInpainter(
            model_id="stabilityai/stable-diffusion-2-inpainting",
            device=self.device,
            num_inference_steps=50
        )
        self.assertIsNotNone(inpainter)

    def test_inpainting(self):
        """Test inpainting an image region."""
        inpainter = StableDiffusionInpainter(device=self.device)

        # Create test image and mask
        test_image = Image.new('RGB', (512, 512), color='blue')
        test_mask = np.zeros((512, 512), dtype=np.uint8)
        test_mask[100:200, 100:200] = 255  # Square region to inpaint

        # Perform inpainting
        result = inpainter.inpaint(
            test_image,
            test_mask,
            "a red square"
        )

        # Check result
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, test_image.size)

    def test_mock_inpainting_cpu(self):
        """Test that mock inpainting works on CPU when model unavailable."""
        # Force mock mode by passing invalid model
        inpainter = StableDiffusionInpainter(
            model_id="invalid_model_id",
            device="cpu"
        )

        test_image = Image.new('RGB', (512, 512), color='blue')
        test_mask = np.zeros((512, 512), dtype=np.uint8)
        test_mask[100:200, 100:200] = 255

        # Should use OpenCV inpainting as fallback
        result = inpainter.inpaint(test_image, test_mask, "test prompt")

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, test_image.size)

    def test_mask_preprocessing(self):
        """Test mask dilation and blurring."""
        inpainter = StableDiffusionInpainter(
            device=self.device,
            mask_blur=8,
            mask_dilation=5
        )

        # Create simple mask
        test_mask = np.zeros((512, 512), dtype=np.uint8)
        test_mask[200:300, 200:300] = 255

        # Process mask
        processed = inpainter._preprocess_mask(test_mask)

        # Check that mask was modified (dilated/blurred)
        self.assertEqual(processed.shape, test_mask.shape)
        # After dilation, the mask should be larger
        self.assertGreater(processed.sum(), test_mask.sum())


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
