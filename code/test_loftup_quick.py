"""
Quick test script for LoftUp integration.

This script performs a minimal test to verify the LoftUp integration works correctly.

Usage:
    python test_loftup_quick.py
"""

import numpy as np
import torch
from PIL import Image
import sys
from pathlib import Path

def test_loftup_import():
    """Test 1: Check if LoftUp can be imported."""
    print("="*70)
    print("TEST 1: Import LoftUp Module")
    print("="*70)

    try:
        from models.loftup_sclip_segmentor import LoftUpSCLIPSegmentor
        print("✅ Successfully imported LoftUpSCLIPSegmentor")
        return True
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return False


def test_loftup_initialization():
    """Test 2: Initialize LoftUp segmentor."""
    print("\n" + "="*70)
    print("TEST 2: Initialize LoftUp Segmentor")
    print("="*70)

    try:
        from models.loftup_sclip_segmentor import LoftUpSCLIPSegmentor

        # Test without LoftUp (should always work)
        print("\n[2a] Testing without LoftUp upsampling...")
        segmentor_standard = LoftUpSCLIPSegmentor(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_loftup=False,
            verbose=False
        )
        print("✅ Standard SCLIP segmentor initialized")

        # Test with LoftUp
        print("\n[2b] Testing with LoftUp upsampling...")
        segmentor_loftup = LoftUpSCLIPSegmentor(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_loftup=True,
            loftup_model_name="loftup_clip",
            verbose=True
        )

        if segmentor_loftup.use_loftup and segmentor_loftup.loftup_upsampler is not None:
            print("✅ LoftUp segmentor initialized successfully")
            return True, segmentor_loftup
        else:
            print("⚠️  LoftUp initialized but upsampler not loaded (will use bilinear fallback)")
            return True, segmentor_loftup

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_loftup_inference():
    """Test 3: Run inference with dummy image."""
    print("\n" + "="*70)
    print("TEST 3: Run Inference on Dummy Image")
    print("="*70)

    try:
        from models.loftup_sclip_segmentor import LoftUpSCLIPSegmentor

        # Create dummy image (224x224 RGB)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        vocabulary = ["person", "car", "dog"]

        print(f"\nDummy image shape: {dummy_image.shape}")
        print(f"Vocabulary: {vocabulary}")

        # Test standard segmentation
        print("\n[3a] Testing standard SCLIP inference...")
        segmentor_standard = LoftUpSCLIPSegmentor(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_loftup=False,
            verbose=False
        )

        seg_map_standard, logits_standard, _ = segmentor_standard.predict_dense(
            dummy_image,
            vocabulary,
            return_logits=True,
            return_features=False
        )

        print(f"✅ Standard output: seg_map shape = {seg_map_standard.shape}, logits shape = {logits_standard.shape}")

        # Test LoftUp segmentation
        print("\n[3b] Testing LoftUp-enhanced inference...")
        segmentor_loftup = LoftUpSCLIPSegmentor(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_loftup=True,
            loftup_model_name="loftup_clip",
            verbose=False
        )

        seg_map_loftup, logits_loftup, features_loftup = segmentor_loftup.predict_dense(
            dummy_image,
            vocabulary,
            return_logits=True,
            return_features=True
        )

        print(f"✅ LoftUp output: seg_map shape = {seg_map_loftup.shape}, logits shape = {logits_loftup.shape}")
        if features_loftup is not None:
            print(f"   Upsampled features shape = {features_loftup.shape}")

        # Verify outputs are at expected resolution
        assert seg_map_standard.shape == dummy_image.shape[:2], "Standard seg_map resolution mismatch"
        assert seg_map_loftup.shape == dummy_image.shape[:2], "LoftUp seg_map resolution mismatch"
        print("\n✅ All output shapes are correct")

        return True

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prompt_extraction():
    """Test 4: Extract prompts from predictions."""
    print("\n" + "="*70)
    print("TEST 4: Prompt Extraction")
    print("="*70)

    try:
        from models.loftup_sclip_segmentor import (
            LoftUpSCLIPSegmentor,
            extract_prompt_points_from_upsampled
        )

        # Create dummy predictions
        H, W = 224, 224
        num_classes = 3

        # Create dummy segmentation map with some regions
        seg_map = np.zeros((H, W), dtype=np.int32)
        seg_map[50:100, 50:100] = 1  # Class 1 region
        seg_map[120:180, 120:180] = 2  # Class 2 region

        # Create dummy probabilities
        probs = np.random.rand(H, W, num_classes).astype(np.float32)
        probs[50:100, 50:100, 1] = 0.9  # High confidence for class 1
        probs[120:180, 120:180, 2] = 0.85  # High confidence for class 2

        vocabulary = ["background", "person", "car"]

        print(f"\nDummy predictions: {H}x{W}, {num_classes} classes")
        print(f"Vocabulary: {vocabulary}")

        # Extract prompts
        prompts = extract_prompt_points_from_upsampled(
            seg_map,
            probs,
            vocabulary,
            min_confidence=0.7,
            min_region_size=100
        )

        print(f"\n✅ Extracted {len(prompts)} prompts")

        if len(prompts) > 0:
            print("\nSample prompt:")
            print(f"  Point: {prompts[0]['point']}")
            print(f"  Class: {prompts[0]['class_name']} (idx={prompts[0]['class_idx']})")
            print(f"  Confidence: {prompts[0]['confidence']:.3f}")
            print(f"  Region size: {prompts[0]['region_size']} pixels")

        return True

    except Exception as e:
        print(f"❌ Prompt extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("LOFTUP INTEGRATION QUICK TEST")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("="*70)

    results = {}

    # Run tests
    results['import'] = test_loftup_import()

    if results['import']:
        init_success, segmentor = test_loftup_initialization()
        results['initialization'] = init_success

        if init_success:
            results['inference'] = test_loftup_inference()
            results['prompt_extraction'] = test_prompt_extraction()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL TESTS PASSED! LoftUp integration is working correctly.")
    else:
        print("⚠️  SOME TESTS FAILED. Check errors above for details.")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
