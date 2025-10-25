#!/bin/bash
# Build and test CPU-only Docker image

set -e

echo "======================================================================="
echo "Building CPU-only Docker Image"
echo "======================================================================="
echo ""
echo "This will create a lighter image without GPU support."
echo "Suitable for testing and development on machines without NVIDIA GPUs."
echo ""

# Build the image
echo "Building image..."
docker build -f Dockerfile.cpu -t openvocab-segmentation:cpu .

echo ""
echo "✓ Image built successfully!"
echo ""
echo "Image size:"
docker images openvocab-segmentation:cpu --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo ""
echo "======================================================================="
echo "Testing the image..."
echo "======================================================================="
echo ""

# Run basic test
docker run --rm openvocab-segmentation:cpu python3 -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CPU threads:', torch.get_num_threads())
print('✓ Basic test passed!')
"

echo ""
echo "======================================================================="
echo "Running comprehensive CPU test..."
echo "======================================================================="
echo ""

# Run the full CPU test script
docker run --rm \
    -v "$(pwd)/output:/app/output" \
    openvocab-segmentation:cpu \
    python3 test_cpu.py

echo ""
echo "======================================================================="
echo "Build Complete!"
echo "======================================================================="
echo ""
echo "Usage examples:"
echo ""
echo "  # Interactive shell"
echo "  docker run --rm -it openvocab-segmentation:cpu /bin/bash"
echo ""
echo "  # Run test script"
echo "  docker run --rm -v \$(pwd)/output:/app/output openvocab-segmentation:cpu python3 test_cpu.py"
echo ""
echo "  # Process an image (with mocks)"
echo "  docker run --rm -it \\"
echo "    -v \$(pwd)/input:/app/input:ro \\"
echo "    -v \$(pwd)/output:/app/output:rw \\"
echo "    openvocab-segmentation:cpu \\"
echo "    python3 main.py --image /app/input/photo.jpg --prompt \"car\" --device cpu"
echo ""
echo "Note: CPU mode uses mock implementations for SAM 2 and Stable Diffusion"
echo "      For full functionality, use the GPU version on a machine with NVIDIA GPU"
echo ""
