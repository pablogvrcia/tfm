#!/usr/bin/env python3
"""
Simplified comprehensive test suite for the Open-Vocabulary Segmentation Pipeline.
Tests actual implementation - works on both CPU and GPU.

Usage:
    python test_all.py              # Run all tests
    python test_all.py --verbose    # Verbose output
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
from PIL import Image

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{BOLD} {text}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_test(name, passed, details=""):
    """Print test result."""
    if passed:
        print(f"  {GREEN}✓{RESET} {name}")
        if details:
            print(f"    {details}")
    else:
        print(f"  {RED}✗{RESET} {name}")
        if details:
            print(f"    {RED}{details}{RESET}")


def test_imports():
    """Test that all modules can be imported."""
    print_header("Test 1: Module Imports")

    tests = []

    try:
        from config import PipelineConfig, SAM2Config, CLIPConfig
        tests.append(("config module", True, ""))
    except Exception as e:
        tests.append(("config module", False, str(e)))

    try:
        from models.sam2_segmentation import SAM2MaskGenerator
        tests.append(("SAM2 module", True, ""))
    except Exception as e:
        tests.append(("SAM2 module", False, str(e)))

    try:
        from models.clip_features import CLIPFeatureExtractor
        tests.append(("CLIP module", True, ""))
    except Exception as e:
        tests.append(("CLIP module", False, str(e)))

    try:
        from models.mask_alignment import MaskTextAligner
        tests.append(("Mask alignment module", True, ""))
    except Exception as e:
        tests.append(("Mask alignment module", False, str(e)))

    try:
        from models.inpainting import StableDiffusionInpainter
        tests.append(("Inpainting module", True, ""))
    except Exception as e:
        tests.append(("Inpainting module", False, str(e)))

    try:
        from pipeline import OpenVocabSegmentationPipeline
        tests.append(("Pipeline module", True, ""))
    except Exception as e:
        tests.append(("Pipeline module", False, str(e)))

    try:
        import utils
        tests.append(("Utils module", True, ""))
    except Exception as e:
        tests.append(("Utils module", False, str(e)))

    for name, passed, details in tests:
        print_test(name, passed, details)

    return all(passed for _, passed, _ in tests)


def test_config():
    """Test configuration system."""
    print_header("Test 2: Configuration System")

    from config import PipelineConfig, SAM2Config, CLIPConfig, get_fast_config, get_quality_config

    tests = []

    try:
        config = PipelineConfig()
        assert hasattr(config, 'sam2')
        assert hasattr(config, 'clip')
        assert hasattr(config, 'alignment')
        assert hasattr(config, 'inpainting')
        tests.append(("PipelineConfig creation", True, "All sub-configs present"))
    except Exception as e:
        tests.append(("PipelineConfig creation", False, str(e)))

    try:
        sam2 = SAM2Config()
        assert sam2.points_per_side == 32
        assert sam2.pred_iou_thresh == 0.88
        tests.append(("SAM2Config defaults", True, f"points_per_side={sam2.points_per_side}"))
    except Exception as e:
        tests.append(("SAM2Config defaults", False, str(e)))

    try:
        clip = CLIPConfig()
        assert clip.model_name == "ViT-L-14"
        assert clip.extract_layers == [6, 12, 18, 24]
        tests.append(("CLIPConfig defaults", True, f"model={clip.model_name}"))
    except Exception as e:
        tests.append(("CLIPConfig defaults", False, str(e)))

    try:
        fast = get_fast_config()
        assert fast.sam2.points_per_side == 16
        tests.append(("Fast preset config", True, f"points={fast.sam2.points_per_side}"))
    except Exception as e:
        tests.append(("Fast preset config", False, str(e)))

    try:
        quality = get_quality_config()
        assert quality.sam2.points_per_side == 64
        tests.append(("Quality preset config", True, f"points={quality.sam2.points_per_side}"))
    except Exception as e:
        tests.append(("Quality preset config", False, str(e)))

    for name, passed, details in tests:
        print_test(name, passed, details)

    return all(passed for _, passed, _ in tests)


def test_sam2_masks():
    """Test SAM 2 mask generation."""
    print_header("Test 3: SAM 2 Mask Generation")

    from models.sam2_segmentation import SAM2MaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tests = []

    try:
        generator = SAM2MaskGenerator(device=device, points_per_side=8)
        tests.append(("SAM2 initialization", True, f"device={device}"))
    except Exception as e:
        tests.append(("SAM2 initialization", False, str(e)))
        return False

    try:
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        masks = generator.generate_masks(test_image)
        assert len(masks) > 0
        tests.append(("Mask generation", True, f"{len(masks)} masks generated"))
    except Exception as e:
        tests.append(("Mask generation", False, str(e)))

    try:
        # Check mask structure (MaskCandidate dataclass)
        if len(masks) > 0:
            mask = masks[0]
            assert hasattr(mask, 'mask') or hasattr(mask, 'segmentation')
            assert hasattr(mask, 'area')
            assert hasattr(mask, 'predicted_iou')
            tests.append(("Mask structure", True, f"area={mask.area}"))
    except Exception as e:
        tests.append(("Mask structure", False, str(e)))

    for name, passed, details in tests:
        print_test(name, passed, details)

    return all(passed for _, passed, _ in tests)


def test_clip_features():
    """Test CLIP feature extraction."""
    print_header("Test 4: CLIP Feature Extraction")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tests = []

    try:
        from models.clip_features import CLIPFeatureExtractor
        extractor = CLIPFeatureExtractor(device=device)
        tests.append(("CLIP initialization", True, f"device={device}"))
    except Exception as e:
        tests.append(("CLIP initialization", False, str(e)))
        # Print all tests as skipped
        for name, _, _ in tests:
            print_test(name, False, "Skipped due to initialization failure")
        return False

    try:
        test_image = Image.new('RGB', (224, 224), color='red')
        global_emb, dense_features = extractor.extract_image_features(test_image)
        assert global_emb is not None
        assert len(dense_features) > 0
        tests.append(("Image feature extraction", True, f"{len(dense_features)} layers"))
    except Exception as e:
        tests.append(("Image feature extraction", False, str(e)))

    try:
        text_emb = extractor.extract_text_features("a red car")
        assert text_emb is not None
        assert len(text_emb.shape) > 0
        tests.append(("Text feature extraction", True, f"shape={text_emb.shape}"))
    except Exception as e:
        tests.append(("Text feature extraction", False, str(e)))

    for name, passed, details in tests:
        print_test(name, passed, details)

    return all(passed for _, passed, _ in tests)


def test_utils():
    """Test utility functions."""
    print_header("Test 5: Utility Functions")

    import utils

    tests = []

    try:
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)
        iou = utils.compute_iou(mask1, mask2)
        assert 0.99 < iou <= 1.0
        tests.append(("IoU computation", True, f"IoU={iou:.3f}"))
    except Exception as e:
        tests.append(("IoU computation", False, str(e)))

    try:
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[0:50, :] = True
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[0:50, :] = True
        precision, recall = utils.compute_precision_recall(mask1, mask2)
        assert precision == 1.0 and recall == 1.0
        tests.append(("Precision/Recall", True, f"P={precision:.2f}, R={recall:.2f}"))
    except Exception as e:
        tests.append(("Precision/Recall", False, str(e)))

    try:
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)
        f1 = utils.compute_f1(mask1, mask2)
        assert 0.99 < f1 <= 1.0
        tests.append(("F1 score", True, f"F1={f1:.3f}"))
    except Exception as e:
        tests.append(("F1 score", False, str(e)))

    try:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        overlay = utils.create_mask_overlay(image, mask)
        assert overlay.shape == image.shape
        tests.append(("Mask overlay", True, f"shape={overlay.shape}"))
    except Exception as e:
        tests.append(("Mask overlay", False, str(e)))

    for name, passed, details in tests:
        print_test(name, passed, details)

    return all(passed for _, passed, _ in tests)


def test_pipeline_integration():
    """Test full pipeline integration."""
    print_header("Test 6: Pipeline Integration")

    from pipeline import OpenVocabSegmentationPipeline
    from config import PipelineConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tests = []

    try:
        pipeline = OpenVocabSegmentationPipeline(
            device=device,
            verbose=False  # Suppress output during tests
        )
        tests.append(("Pipeline initialization", True, f"device={device}"))
    except Exception as e:
        tests.append(("Pipeline initialization", False, str(e)))
        for name, _, _ in tests:
            print_test(name, False, "Skipped")
        return False

    try:
        test_image = Image.new('RGB', (256, 256), color='blue')
        result = pipeline.segment(
            test_image,
            text_prompt="blue area",
            top_k=1
        )
        assert result.segmentation_masks is not None
        assert result.timing is not None
        tests.append(("Segmentation pipeline", True, f"{len(result.segmentation_masks)} masks"))
    except Exception as e:
        tests.append(("Segmentation pipeline", False, str(e)))

    try:
        if result and result.timing:
            timing = result.timing
            assert 'sam2_generation' in timing
            assert 'clip_alignment' in timing
            total_time = sum(timing.values())
            tests.append(("Timing measurements", True, f"total={total_time:.2f}s"))
    except Exception as e:
        tests.append(("Timing measurements", False, str(e)))

    for name, passed, details in tests:
        print_test(name, passed, details)

    return all(passed for _, passed, _ in tests)


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description='Run comprehensive tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    print_header("Open-Vocabulary Segmentation Pipeline - Test Suite")

    # Detect environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run all tests
    start_time = time.time()

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("SAM 2 Masks", test_sam2_masks()))
    results.append(("CLIP Features", test_clip_features()))
    results.append(("Utilities", test_utils()))
    results.append(("Pipeline", test_pipeline_integration()))

    elapsed_time = time.time() - start_time

    # Print summary
    print_header("Test Summary")

    total = len(results)
    passed = sum(1 for _, result in results if result)
    failed = total - passed

    for name, result in results:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {name}: {status}")

    print(f"\n{BOLD}Total: {total} | Passed: {passed} | Failed: {failed}{RESET}")
    print(f"Time: {elapsed_time:.2f}s")

    if failed == 0:
        print(f"\n{GREEN}{BOLD}✓ All tests passed!{RESET}")
        return 0
    else:
        print(f"\n{RED}{BOLD}✗ {failed} test(s) failed{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
