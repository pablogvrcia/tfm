# CPU-Only Usage Guide

This guide explains how to run the pipeline on machines **without GPU support**.

## Quick Start (Docker - Recommended)

### 1. Build the CPU-only Docker image:

```bash
./docker-build-cpu.sh
```

This builds a lighter image (~3GB vs ~12GB for GPU version) that works on any machine.

### 2. Run the test:

```bash
docker run --rm \
    -v $(pwd)/output:/app/output \
    openvocab-segmentation:cpu \
    python3 test_cpu.py
```

### 3. Interactive shell:

```bash
docker run --rm -it openvocab-segmentation:cpu /bin/bash

# Inside container:
python3 test_cpu.py
python3 main.py --help
```

## Important Notes for CPU Mode

### ⚠️ Limitations

1. **Mock Implementations**: Without installing heavy models, the pipeline uses mock implementations:
   - SAM 2 → Uses scikit-image superpixels (~200 segments)
   - Stable Diffusion → Uses OpenCV inpainting (basic)
   - CLIP → Would need to be installed (can work on CPU but slow)

2. **Performance**:
   - Full models on CPU are **10-100x slower** than GPU
   - SAM 2: ~5-10 minutes per image on CPU vs 2-4 seconds on GPU
   - Stable Diffusion: ~30-60 minutes per image on CPU vs 5-10 seconds on GPU

3. **Recommended Use Cases**:
   - ✅ Code testing and development
   - ✅ Understanding pipeline structure
   - ✅ Validation of metrics (IoU, F1, etc.)
   - ✅ Testing with mock implementations
   - ❌ Production use
   - ❌ Real segmentation tasks (use GPU or cloud)

## What Works on CPU

### ✅ Fully Functional (Fast)

- Configuration system
- Utility functions (IoU, F1, visualization)
- Mock SAM 2 implementation (superpixels)
- Mock inpainting (OpenCV)
- Pipeline structure and flow
- All evaluation metrics
- CLI interface

### ⚠️ Partially Functional (Slow)

If you install full dependencies:

```bash
# Inside Docker container or local install
pip install open-clip-torch  # CLIP works on CPU but slow
# SAM 2 and Stable Diffusion are too slow on CPU
```

- CLIP: Works but 10x slower (~1-2 seconds per image)
- Basic inference: Possible but impractical

### ❌ Not Recommended

- SAM 2 on CPU: Too slow (5-10 min per image)
- Stable Diffusion on CPU: Extremely slow (30-60 min per image)

## Running Tests

### Test 1: Basic CPU Test

```bash
docker run --rm openvocab-segmentation:cpu python3 test_cpu.py
```

This tests:
- Import system
- PyTorch CPU functionality
- Mock SAM 2 implementation
- Utility functions
- Configuration system
- Basic pipeline flow

Expected output:
```
Test 1: Checking basic imports...
  ✓ opencv-python
  ✓ Pillow
  ✓ numpy

Test 2: Checking PyTorch...
  ✓ PyTorch version: 2.x.x
  ✓ CUDA available: False
  ✓ Device: cpu

...

✓ Basic functionality works on CPU!
```

### Test 2: Mock Segmentation

```bash
# Create a test image
docker run --rm \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:cpu \
    python3 test_cpu.py
```

Outputs:
- `output/test/test_image.png` - Synthetic test image
- `output/test/test_masks_overlay.png` - Masks visualization

## Local Installation (Without Docker)

If you prefer not to use Docker:

```bash
# Install CPU-only dependencies
pip install -r requirements-cpu.txt

# Run test
python3 test_cpu.py
```

## For Production Use

**Don't use CPU mode for production!** Instead:

### Option 1: Use a GPU Machine

Rent a cloud GPU instance:
- **Google Colab** (Free GPU): https://colab.research.google.com/
- **AWS EC2 g4dn.xlarge** ($0.50/hour): NVIDIA T4 GPU
- **Paperspace** ($0.45/hour): P4000 GPU
- **Lambda Labs** ($0.50/hour): RTX A6000

### Option 2: Docker on GPU Machine

```bash
# On GPU machine
git clone [repo]
cd code/

# Build GPU version
docker build -t openvocab-segmentation:latest .

# Run
./docker-run.sh segment input/photo.jpg "red car"
```

### Option 3: Google Colab

Upload the code to Colab and run:

```python
# In Colab notebook
!git clone [repo]
%cd code/
!pip install -r requirements.txt
!python main.py --image sample.jpg --prompt "car" --device cuda
```

## Comparison: CPU vs GPU

| Task | CPU Time | GPU Time (RTX 3090) | Speedup |
|------|----------|---------------------|---------|
| SAM 2 Masks | 5-10 min | 2-4 sec | 75-150x |
| CLIP Features | 1-2 sec | 0.1 sec | 10-20x |
| SD Inpainting | 30-60 min | 5-10 sec | 180-360x |
| **Total Pipeline** | **40-70 min** | **10-20 sec** | **120-200x** |

## FAQ

**Q: Can I use the CPU version for my thesis experiments?**
A: No. The mock implementations don't provide accurate segmentation. Use GPU or cloud.

**Q: Why are the results different from GPU?**
A: CPU mode uses simplified algorithms (superpixels instead of SAM 2, basic inpainting instead of Stable Diffusion).

**Q: Can I install the full models and run on CPU?**
A: Technically yes, but it's impractically slow. Not recommended.

**Q: What's the minimum RAM needed?**
A: 8GB for mock mode, 16GB+ if you install full models on CPU.

**Q: Can I debug my code with CPU mode?**
A: Yes! CPU mode is perfect for development, testing, and debugging.

## Recommended Workflow

1. **Development** (CPU mode):
   ```bash
   # Test code structure, fix bugs
   python3 test_cpu.py
   ```

2. **Testing** (CPU mode):
   ```bash
   # Validate metrics, check outputs
   python3 -c "from utils import compute_iou; print(compute_iou(...))"
   ```

3. **Production** (GPU mode):
   ```bash
   # On GPU machine or cloud
   python3 main.py --image photo.jpg --prompt "car" --device cuda
   ```

## Getting Help

If you encounter issues:

1. Check this guide
2. Review [DOCKER.md](DOCKER.md)
3. Check the build log: `cat build.log`
4. Open an issue on GitHub

## Next Steps

- For GPU setup: See [README.md](README.md)
- For Docker details: See [DOCKER.md](DOCKER.md)
- For quick start: See [QUICK_START.md](QUICK_START.md)
