#!/usr/bin/env python3
"""
Quick test script to verify the installation is working correctly.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 70)
    print("Testing Package Imports")
    print("=" * 70)
    print()

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("open_clip", "OpenCLIP"),
        ("diffusers", "Diffusers (Stable Diffusion)"),
        ("transformers", "Transformers"),
        ("skimage", "scikit-image"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
    ]

    failed = []

    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"✓ {display_name:30s} OK")
        except ImportError as e:
            print(f"✗ {display_name:30s} FAILED: {e}")
            failed.append(display_name)

    # Test SAM 2 (optional)
    try:
        import sam2
        print(f"✓ {'SAM 2':30s} OK")
    except ImportError:
        print(f"⚠ {'SAM 2':30s} Not installed (will use mock)")

    print()

    if failed:
        print(f"ERROR: {len(failed)} package(s) failed to import:")
        for pkg in failed:
            print(f"  - {pkg}")
        return False

    return True


def test_cuda():
    """Test CUDA availability and GPU."""
    print("=" * 70)
    print("Testing CUDA Setup")
    print("=" * 70)
    print()

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"GPU memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"Compute capability: {props.major}.{props.minor}")

        # Test a simple operation
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print("✓ CUDA operations working")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ CUDA operation failed: {e}")
            return False

        print()
        return True
    else:
        print("✗ CUDA not available")
        print("Please check your CUDA installation and GPU drivers")
        print()
        return False


def test_models():
    """Test that models can be loaded."""
    print("=" * 70)
    print("Testing Model Loading")
    print("=" * 70)
    print()

    import torch

    # Test CLIP
    try:
        import open_clip
        print("Testing CLIP...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai',
            device='cpu'  # Use CPU for test to save memory
        )
        print("✓ CLIP model loaded successfully")
        del model
    except Exception as e:
        print(f"✗ CLIP loading failed: {e}")
        return False

    # Test SAM 2
    import os
    checkpoint_exists = os.path.exists("checkpoints/sam2_hiera_large.pt")

    if checkpoint_exists:
        print("✓ SAM 2 checkpoint found")
        try:
            import sam2
            print("✓ SAM 2 package installed")
        except ImportError:
            print("⚠ SAM 2 package not installed (checkpoint exists)")
    else:
        print("⚠ SAM 2 checkpoint not found")
        print("  Download with: python scripts/download_sam2_checkpoints.py")

    # Test Stable Diffusion (just check if it can be imported)
    try:
        from diffusers import StableDiffusionInpaintPipeline
        print("✓ Stable Diffusion pipeline available")
    except Exception as e:
        print(f"✗ Stable Diffusion import failed: {e}")
        return False

    print()
    return True


def test_pipeline_imports():
    """Test that our pipeline modules can be imported."""
    print("=" * 70)
    print("Testing Pipeline Modules")
    print("=" * 70)
    print()

    modules = [
        "models.sam2_segmentation",
        "models.clip_features",
        "models.mask_alignment",
        "models.inpainting",
        "pipeline",
        "config",
        "utils",
    ]

    for module in modules:
        try:
            __import__(module)
            print(f"✓ {module:30s} OK")
        except Exception as e:
            print(f"✗ {module:30s} FAILED: {e}")
            return False

    print()
    return True


def main():
    """Run all tests."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Open-Vocabulary Segmentation - Installation Test" + " " * 9 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("CUDA Setup", test_cuda()))
    results.append(("Model Loading", test_models()))
    results.append(("Pipeline Modules", test_pipeline_imports()))

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False

    print()

    if all_passed:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "🎉 All tests passed! 🎉" + " " * 25 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
        print("Your installation is working correctly!")
        print()
        print("Next steps:")
        print("  1. Download SAM 2 checkpoint (if not done):")
        print("     python scripts/download_sam2_checkpoints.py --model sam2_hiera_large")
        print()
        print("  2. Run a test segmentation:")
        print("     python main.py --image photo.jpg --prompt 'person' --mode segment")
        print()
        print("See README.md and SETUP.md for more information.")
        print()
        return 0
    else:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 21 + "❌ Some tests failed" + " " * 26 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
        print("Please check the errors above and:")
        print("  1. Make sure all dependencies are installed:")
        print("     pip install -r requirements.txt")
        print()
        print("  2. Check CUDA installation:")
        print("     nvidia-smi")
        print()
        print("  3. See SETUP.md for troubleshooting")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
