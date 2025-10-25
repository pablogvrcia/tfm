# SAM 2 Setup Guide

## Quick Start

SAM 2 (Segment Anything Model 2) requires downloading model checkpoints (~224 MB for the large model).

### Option 1: Download Checkpoints (Recommended for GPU)

```bash
# Download the default large model (~224 MB)
python3 download_sam2_checkpoints.py

# Download a smaller/faster model
python3 download_sam2_checkpoints.py --model sam2_hiera_tiny  # ~38 MB

# Download all models
python3 download_sam2_checkpoints.py --all  # ~388 MB total

# List available models
python3 download_sam2_checkpoints.py --list
```

### Option 2: Use Mock Implementation (CPU Testing)

For CPU testing and development, the mock implementation works without downloading:

```python
from models.sam2_segmentation import SAM2MaskGenerator

# Will automatically use mock if checkpoints not available
generator = SAM2MaskGenerator(device='cpu')
```

The mock uses scikit-image superpixels (~200 segments, very fast).

## Available Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| sam2_hiera_tiny | ~38 MB | Fastest | Good | Quick testing |
| sam2_hiera_small | ~46 MB | Fast | Better | Development |
| sam2_hiera_base_plus | ~80 MB | Medium | Great | Balanced |
| sam2_hiera_large | ~224 MB | Slower | Best | Production (default) |

## Using Downloaded Checkpoints

Once downloaded, SAM 2 will automatically use the checkpoints:

```python
from models.sam2_segmentation import SAM2MaskGenerator
import numpy as np

# Initialize with checkpoint (auto-detected from checkpoints/ folder)
generator = SAM2MaskGenerator(
    model_type="sam2_hiera_large",
    device="cuda",  # or "cpu" but will be slow
    points_per_side=32
)

# Generate masks
image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
masks = generator.generate_masks(image)

print(f"Generated {len(masks)} masks")
```

## Docker Usage

### Build with Checkpoint Download

```bash
# Build and download checkpoints during build
docker build -f Dockerfile.cpu -t openvocab-segmentation:cpu --build-arg DOWNLOAD_SAM2=true .
```

### Download in Running Container

```bash
# Start container
docker run -it -v $(pwd):/workspace openvocab-segmentation:cpu bash

# Download inside container
cd /workspace
python3 download_sam2_checkpoints.py
```

## Checkpoint Locations

Checkpoints are saved to:
- **Local**: `./checkpoints/`
- **Docker**: `/app/checkpoints/`

Example:
```
checkpoints/
├── sam2_hiera_large.pt          # Model weights
├── sam2_hiera_large_info.txt    # Model info
└── ...
```

## Manual Download

If the script doesn't work, download manually:

```bash
# Create directory
mkdir -p checkpoints

# Download checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt \
     -O checkpoints/sam2_hiera_large.pt
```

## Verification

Test that SAM 2 works:

```bash
python3 -c "
from models.sam2_segmentation import SAM2MaskGenerator
import numpy as np

gen = SAM2MaskGenerator(device='cpu')
img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
masks = gen.generate_masks(img)
print(f'Generated {len(masks)} masks')
"
```

Expected output:
- **With checkpoints**: "Generated 100-300 masks" (actual SAM 2)
- **Without checkpoints**: "Warning: SAM 2 initialization failed... Generated 1-10 masks" (mock)

## Troubleshooting

### "MissingConfigException" Error
**Cause**: SAM 2 library installed but checkpoints not downloaded
**Fix**: Run `python3 download_sam2_checkpoints.py`

### Slow Performance on CPU
**Normal**: SAM 2 is very slow on CPU (5-10 min per image)
**Recommendation**: Use GPU or the mock implementation for CPU

### Out of Memory
**Cause**: SAM 2 large model requires ~8GB GPU RAM
**Fix**: Use smaller model:
```python
generator = SAM2MaskGenerator(model_type="sam2_hiera_small", device="cuda")
```

## Performance Comparison

| Setup | Masks Generated | Time per Image | Quality |
|-------|----------------|----------------|---------|
| Mock (CPU) | 1-10 | <1s | Low (superpixels) |
| SAM 2 Tiny (GPU) | 100-200 | 2-3s | Good |
| SAM 2 Large (GPU) | 150-300 | 4-6s | Excellent |
| SAM 2 Large (CPU) | 150-300 | 5-10 min | Excellent |

## For Thesis Experiments

**Recommended setup**:
1. **Development/Testing**: Use mock on CPU
2. **Experiments**: Use SAM 2 Large on GPU
3. **Benchmarks**: Download checkpoints first, use GPU

**Commands**:
```bash
# Download checkpoints once
python3 download_sam2_checkpoints.py

# Run experiments on GPU
python3 main.py --image dataset/img.jpg --prompt "car" --device cuda

# Or use Docker with GPU
docker run --gpus all -v $(pwd):/app openvocab-segmentation:latest \
    python3 main.py --image /app/dataset/img.jpg --prompt "car"
```

## References

- SAM 2 Paper: https://arxiv.org/abs/2408.00714
- SAM 2 GitHub: https://github.com/facebookresearch/segment-anything-2
- Model Checkpoints: https://github.com/facebookresearch/segment-anything-2#model-checkpoints
