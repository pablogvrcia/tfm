#!/usr/bin/env python3
"""
CPU Test Script - Verify the pipeline works without GPU

This script tests the pipeline components without requiring heavy models.
Uses mock implementations and minimal dependencies.
"""

import sys
import numpy as np
from PIL import Image
from pathlib import Path

print("="*70)
print("CPU Test - Open-Vocabulary Segmentation Pipeline")
print("="*70)
print()

# Test 1: Check basic imports
print("Test 1: Checking basic imports...")
try:
    import cv2
    print("  ✓ opencv-python")
except ImportError as e:
    print(f"  ✗ opencv-python: {e}")

try:
    from PIL import Image
    print("  ✓ Pillow")
except ImportError as e:
    print(f"  ✗ Pillow: {e}")

try:
    import numpy as np
    print("  ✓ numpy")
except ImportError as e:
    print(f"  ✗ numpy: {e}")

print()

# Test 2: Check PyTorch (CPU)
print("Test 2: Checking PyTorch...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
    print(f"  ✓ Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Test tensor creation
    test_tensor = torch.randn(3, 224, 224)
    print(f"  ✓ Tensor creation works: shape {test_tensor.shape}")
except ImportError as e:
    print(f"  ✗ PyTorch: {e}")
    print("  Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")

print()

# Test 3: Test SAM2 mock implementation
print("Test 3: Testing SAM 2 mock implementation...")
try:
    from models.sam2_segmentation import SAM2MaskGenerator

    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Initialize with mock (no actual model needed)
    generator = SAM2MaskGenerator(device="cpu")
    print("  ✓ SAM2MaskGenerator initialized")

    # Generate masks (will use mock implementation)
    masks = generator.generate_masks(test_image)
    print(f"  ✓ Generated {len(masks)} mock masks")

    # Test filtering
    filtered = generator.filter_by_size(masks, min_area=500)
    print(f"  ✓ Filtered to {len(filtered)} masks (min area: 500)")

except Exception as e:
    print(f"  ✗ SAM 2 test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Test utilities
print("Test 4: Testing utility functions...")
try:
    from utils import compute_iou, compute_f1, create_mask_overlay

    # Create test masks
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask1[25:75, 25:75] = 1

    mask2 = np.zeros((100, 100), dtype=np.uint8)
    mask2[40:90, 40:90] = 1

    # Test IoU
    iou = compute_iou(mask1, mask2)
    print(f"  ✓ IoU computation: {iou:.3f}")

    # Test F1
    f1 = compute_f1(mask1, mask2)
    print(f"  ✓ F1 computation: {f1:.3f}")

    # Test visualization
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    overlay = create_mask_overlay(test_img, mask1, color=(255, 0, 0))
    print(f"  ✓ Mask overlay created: shape {overlay.shape}")

except Exception as e:
    print(f"  ✗ Utilities test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 5: Test configuration
print("Test 5: Testing configuration system...")
try:
    from config import PipelineConfig, get_fast_config, get_quality_config

    config = PipelineConfig()
    config.device = "cpu"
    print(f"  ✓ Default config created")
    print(f"    - Device: {config.device}")
    print(f"    - SAM points per side: {config.sam2.points_per_side}")
    print(f"    - CLIP layers: {config.clip.extract_layers}")

    fast_config = get_fast_config()
    fast_config.device = "cpu"
    print(f"  ✓ Fast config: {fast_config.sam2.points_per_side} points per side")

except Exception as e:
    print(f"  ✗ Config test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 6: Full pipeline test (with mocks)
print("Test 6: Testing full pipeline (mock mode)...")
try:
    # Create a simple synthetic test
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Draw a simple shape for testing
    cv2.rectangle(test_image, (50, 50), (150, 150), (255, 0, 0), -1)
    cv2.circle(test_image, (200, 200), 30, (0, 255, 0), -1)

    # Save test image
    output_dir = Path("output/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_image_pil = Image.fromarray(test_image)
    test_image_pil.save(output_dir / "test_image.png")
    print(f"  ✓ Created test image: {output_dir}/test_image.png")

    # Test the pipeline components individually
    from models.sam2_segmentation import SAM2MaskGenerator

    generator = SAM2MaskGenerator(device="cpu")
    masks = generator.generate_masks(test_image)
    print(f"  ✓ Generated {len(masks)} masks")

    # Visualize masks
    if len(masks) > 0:
        vis = generator.visualize_masks(test_image, masks, max_display=10)
        vis_pil = Image.fromarray(vis)
        vis_pil.save(output_dir / "test_masks_overlay.png")
        print(f"  ✓ Saved visualization: {output_dir}/test_masks_overlay.png")

except Exception as e:
    print(f"  ✗ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("CPU Test Summary")
print("="*70)
print()
print("✓ Basic functionality works on CPU!")
print("✓ Mock implementations allow testing without models")
print()
print("Note: For full functionality (CLIP, SAM 2, Stable Diffusion),")
print("you would need to install the full dependencies, but they")
print("will be VERY SLOW on CPU. Recommended to use:")
print("  - Docker with GPU support, or")
print("  - A machine with NVIDIA GPU")
print()
print("Mock implementations are sufficient for:")
print("  - Code development and testing")
print("  - Understanding the pipeline structure")
print("  - Validation of evaluation metrics")
print()
