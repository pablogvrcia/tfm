# Setup Guide - Open-Vocabulary Semantic Segmentation Pipeline

This guide will help you set up and run the complete pipeline on your GTX 1060 6GB GPU.

## Prerequisites

- NVIDIA GTX 1060 6GB or better
- CUDA 11.8 or 12.x installed
- Python 3.10 or 3.11
- At least 20GB free disk space (for models and checkpoints)

## Step 1: Create Virtual Environment

```bash
cd /home/pablo/tfm/code

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8 (compatible with GTX 1060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

**Note:** This will take 10-15 minutes and download ~8GB of packages.

## Step 3: Verify CUDA is Working

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

You should see:
```
CUDA available: True
Device: NVIDIA GeForce GTX 1060 6GB
```

## Step 4: Download SAM 2 Checkpoints

The pipeline needs SAM 2 model checkpoints. Download the one you want to use:

```bash
# For best quality (recommended, ~224MB)
python scripts/download_sam2_checkpoints.py --model sam2_hiera_large

# Or for faster inference (~38MB)
python scripts/download_sam2_checkpoints.py --model sam2_hiera_tiny

# Or download all models
python scripts/download_sam2_checkpoints.py --all
```

Checkpoints will be saved to `checkpoints/` directory.

## Step 5: Test the Pipeline

Run a simple test to make sure everything works:

```bash
# Test with segmentation only
python main.py --image photo.jpg --prompt "person" --mode segment --visualize

# Check the output
ls -lh output/
```

You should see generated files in the `output/` directory.

## Usage Examples

### 1. Segment Objects

```bash
python main.py \
  --image photo.jpg \
  --prompt "red car" \
  --mode segment \
  --top-k 5 \
  --visualize
```

### 2. Remove Objects

```bash
python main.py \
  --image photo.jpg \
  --prompt "person in background" \
  --mode remove \
  --visualize
```

### 3. Replace Objects

```bash
python main.py \
  --image photo.jpg \
  --prompt "old laptop" \
  --mode replace \
  --edit "modern MacBook Pro" \
  --visualize
```

### 4. Style Transfer

```bash
python main.py \
  --image photo.jpg \
  --prompt "building" \
  --mode style \
  --edit "watercolor painting style" \
  --visualize
```

### 5. Benchmark Performance

```bash
python main.py \
  --image photo.jpg \
  --mode benchmark
```

## Configuration Presets

The pipeline has three quality presets:

- **fast**: Faster but lower quality (good for testing)
- **balanced**: Good balance of speed and quality (default)
- **quality**: Best quality but slower

Usage:
```bash
python main.py --image photo.jpg --prompt "dog" --config quality
```

## Expected Performance (GTX 1060 6GB)

With the GTX 1060 6GB, expect:

- **SAM 2 mask generation**: 3-6 seconds per image
- **CLIP feature extraction**: 0.2-0.5 seconds
- **Mask alignment**: 0.1-0.2 seconds
- **Stable Diffusion inpainting**: 8-15 seconds per edit
- **Total pipeline**: 15-30 seconds for full segmentation + editing

## Memory Management

The GTX 1060 has 6GB VRAM. To avoid out-of-memory errors:

1. Use `sam2_hiera_tiny` or `sam2_hiera_small` for SAM 2
2. Close other GPU-heavy applications
3. Process one image at a time
4. Use smaller image sizes (resize to max 1024px if needed)

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Use smaller SAM 2 model
python scripts/download_sam2_checkpoints.py --model sam2_hiera_tiny

# In pipeline.py, change model to tiny
# Or use environment variable
export SAM_MODEL=sam2_hiera_tiny
```

### Issue: SAM 2 checkpoint not found

**Solution:**
```bash
# Download the checkpoint
python scripts/download_sam2_checkpoints.py --model sam2_hiera_large

# Verify it's downloaded
ls -lh checkpoints/
```

### Issue: Slow inference

**Solution:**
- Use `--config fast` preset
- Use smaller SAM 2 model (tiny or small)
- Resize input images to 512-768px

### Issue: ImportError for SAM 2

**Solution:**
```bash
# Reinstall SAM 2
pip uninstall sam2 -y
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## Directory Structure

```
code/
├── checkpoints/          # SAM 2 model checkpoints
├── models/              # Model modules (CLIP, SAM2, etc.)
├── scripts/             # Utility scripts
├── output/              # Generated outputs
├── main.py              # Main entry point
├── pipeline.py          # Complete pipeline
├── requirements.txt     # Python dependencies
└── SETUP.md            # This file
```

## Next Steps

1. Try the examples above with your own images
2. Experiment with different prompts
3. Adjust configuration for your use case
4. Check the thesis document for methodology details

## Support

For issues specific to:
- **SAM 2**: https://github.com/facebookresearch/segment-anything-2
- **CLIP**: https://github.com/mlfoundations/open_clip
- **Stable Diffusion**: https://github.com/huggingface/diffusers

For this implementation, check the README.md and code comments.
