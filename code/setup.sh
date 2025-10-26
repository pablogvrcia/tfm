#!/bin/bash

# Setup script for Open-Vocabulary Semantic Segmentation Pipeline
# For use with NVIDIA GTX 1060 6GB or better

set -e  # Exit on error

echo "======================================================================="
echo "Open-Vocabulary Semantic Segmentation Pipeline - Setup"
echo "Master's Thesis Implementation"
echo "======================================================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

echo "✓ NVIDIA drivers detected"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check Python version - use python3.12 if available
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD=python3.12
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD=python3.11
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
else
    PYTHON_CMD=python3
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Using Python: $PYTHON_CMD (version $PYTHON_VERSION)"

if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "ERROR: Python 3.10 or higher required"
    echo "Available Python versions:"
    ls /usr/bin/python* 2>/dev/null | grep -E "python3\.[0-9]+" || true
    exit 1
fi

echo "✓ Python version OK"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q

echo ""
echo "======================================================================="
echo "Installing PyTorch with CUDA 11.8 support..."
echo "This may take a few minutes..."
echo "======================================================================="
echo ""

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "======================================================================="
echo "Installing other dependencies..."
echo "This may take 10-15 minutes..."
echo "======================================================================="
echo ""

pip install -r requirements.txt

echo ""
echo "======================================================================="
echo "Verifying CUDA setup..."
echo "======================================================================="
echo ""

python << 'EOF'
import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("✓ CUDA is working correctly!")
else:
    print("ERROR: CUDA is not available. Check your installation.")
    sys.exit(1)
EOF

echo ""
echo "======================================================================="
echo "Checking for SAM 2 checkpoints..."
echo "======================================================================="
echo ""

if [ ! -d "checkpoints" ] || [ -z "$(ls -A checkpoints/*.pt 2>/dev/null)" ]; then
    echo "No SAM 2 checkpoints found."
    echo ""
    read -p "Download SAM 2 checkpoint? (Y/n): " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        echo "Downloading sam2_hiera_large (~224MB)..."
        python scripts/download_sam2_checkpoints.py --model sam2_hiera_large
    else
        echo "Skipping checkpoint download."
        echo "You can download later with:"
        echo "  python scripts/download_sam2_checkpoints.py --model sam2_hiera_large"
    fi
else
    echo "✓ SAM 2 checkpoints found:"
    ls -lh checkpoints/*.pt
fi

echo ""
echo "======================================================================="
echo "Running quick test..."
echo "======================================================================="
echo ""

python << 'EOF'
try:
    print("Testing imports...")
    import torch
    import torchvision
    import numpy as np
    import cv2
    from PIL import Image
    import open_clip
    from diffusers import StableDiffusionInpaintPipeline

    print("✓ All core imports successful")

    # Test SAM 2
    try:
        import sam2
        print("✓ SAM 2 imported successfully")
    except ImportError:
        print("⚠ SAM 2 not found (will use mock implementation)")

    print("")
    print("All tests passed!")

except Exception as e:
    print(f"ERROR: {e}")
    import sys
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "Setup complete! 🎉"
    echo "======================================================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2. Run a test:"
    echo "   python main.py --image photo.jpg --prompt \"person\" --mode segment"
    echo ""
    echo "3. Check SETUP.md for more usage examples"
    echo ""
    echo "For help:"
    echo "   python main.py --help"
    echo ""
else
    echo ""
    echo "Setup encountered errors. Please check the output above."
    exit 1
fi
