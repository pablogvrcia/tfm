# Quick Start Guide

## Installation (5 minutes)

```bash
cd code/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

## Basic Usage (1 minute)

### Segment an object:
```bash
python main.py --image photo.jpg --prompt "red car" --mode segment
```

### Remove an object:
```bash
python main.py --image photo.jpg --prompt "person" --mode remove
```

### Replace an object:
```bash
python main.py --image photo.jpg --prompt "old TV" \
    --mode replace --edit "modern flat screen TV"
```

## Python API

```python
from pipeline import OpenVocabSegmentationPipeline

# Initialize
pipeline = OpenVocabSegmentationPipeline()

# Segment
result = pipeline.segment("image.jpg", "red car")

# Edit
result = pipeline.segment_and_edit("image.jpg", "person", "remove")
```

## Key Files

- **[main.py](main.py)** - CLI entry point
- **[pipeline.py](pipeline.py)** - Main pipeline (400 lines)
- **[models/sam2_segmentation.py](models/sam2_segmentation.py)** - SAM 2 (270 lines)
- **[models/clip_features.py](models/clip_features.py)** - CLIP (280 lines)
- **[models/mask_alignment.py](models/mask_alignment.py)** - Alignment (300 lines)
- **[models/inpainting.py](models/inpainting.py)** - Stable Diffusion (280 lines)
- **[config.py](config.py)** - Configuration (170 lines)
- **[utils.py](utils.py)** - Utilities (420 lines)

## Output Structure

```
output/
├── original.png          # Original input image
├── segmentation.png      # Masks overlaid on image
├── similarity_map.png    # CLIP similarity heatmap
├── comparison_grid.png   # Multi-view comparison
├── edited.png           # Final edited result
└── comparison.png       # Before/after comparison
```

## Configuration Presets

**Fast** (30s/image): Fewer masks, faster inference
```bash
python main.py --image photo.jpg --prompt "car" --config fast
```

**Balanced** (10-20s/image): Default, good tradeoff
```bash
python main.py --image photo.jpg --prompt "car"
```

**Quality** (60s/image): More masks, better results
```bash
python main.py --image photo.jpg --prompt "car" --config quality
```

## Troubleshooting

**CUDA out of memory**: Use `--device cpu` or reduce image size

**No masks found**: Lower similarity threshold in config

**Poor inpainting**: Increase `num_inference_steps` in config

## Next Steps

- See [README.md](README.md) for detailed documentation
- Check [examples/basic_usage.py](examples/basic_usage.py) for code examples
- Read the thesis PDF for methodology details
