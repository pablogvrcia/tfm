# Open-Vocabulary Semantic Segmentation with Generative AI

Implementation of the Master's Thesis: **"Open-Vocabulary Semantic Segmentation for Generative AI"** by Pablo García García, Universidad de Zaragoza.

This system combines SAM 2, CLIP, and Stable Diffusion to enable flexible, language-driven image segmentation and editing without requiring training on specific object categories.

## Features

- **Zero-Shot Segmentation**: Segment arbitrary objects using natural language descriptions
- **Multi-Scale CLIP Features**: Extract dense vision-language features from multiple transformer layers (6, 12, 18, 24)
- **High-Quality Masks**: Automatic mask generation with SAM 2 (32×32 point grid, IoU threshold 0.88)
- **Generative Editing**: Remove, replace, or restyle objects using Stable Diffusion
- **Comprehensive Evaluation**: Metrics including mIoU, Precision/Recall, F1, FID, CLIP Score

## Architecture

The pipeline consists of 4 main stages:

1. **SAM 2 Mask Generation** (Chapter 3.2.2)
   - Automatic mask generation with 1024 point prompts (32×32 grid)
   - Hierarchical coverage across multiple scales
   - IoU filtering (>0.88) and stability scoring (>0.95)

2. **CLIP Feature Extraction** (Chapter 3.2.1)
   - ViT-L/14 model at 336×336 resolution
   - Multi-scale features from layers 6, 12, 18, 24
   - Prompt ensembling for robustness

3. **Mask-Text Alignment** (Chapter 3.2.3)
   - Cosine similarity scoring: S_i = (1/|M_i|) Σ sim(f_p, e_t)
   - Background suppression with α=0.3
   - Spatial weighting for boundary noise reduction

4. **Stable Diffusion Inpainting** (Chapter 3.2.4)
   - SD v2 inpainting with 50 steps
   - Guidance scale 7.5 for prompt adherence
   - Mask blur (8px) and dilation (5px) for seamless blending

## Installation

### Option 1: Docker (Recommended)

**Easiest way to get started!** No dependency management needed.

```bash
# Build Docker image
./scripts/docker-run.sh build

# Run segmentation
./scripts/docker-run.sh segment input/photo.jpg "red car"
```

See [docs/DOCKER.md](docs/DOCKER.md) for complete Docker documentation.

### Option 2: Local Installation

**Prerequisites:**
- Python 3.8+
- CUDA-capable GPU (recommended: NVIDIA RTX 3090 or better)
- 16GB+ RAM
- 8GB+ VRAM

**Setup:**

```bash
# Clone repository
cd code/

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SAM 2
pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Download model checkpoints (automatic on first run)
# - SAM 2: sam2_hiera_large (~900MB)
# - CLIP: ViT-L/14 (~900MB)
# - Stable Diffusion: SD v2-inpainting (~5GB)
```

### Option 3: Using Makefile

```bash
make install        # Install dependencies
make docker-build   # Build Docker image
make docker-run     # Run Docker container
```

## Usage

### Basic Segmentation

```bash
# Segment objects based on text prompt
python main.py --image examples/living_room.jpg --prompt "red sofa" --mode segment

# Segment with custom configuration
python main.py --image examples/street.jpg --prompt "blue car" \
    --mode segment --config quality --top-k 3
```

### Object Removal

```bash
# Remove object from image
python main.py --image examples/room.jpg --prompt "lamp on table" --mode remove
```

### Object Replacement

```bash
# Replace object with new content
python main.py --image examples/office.jpg \
    --prompt "old computer monitor" \
    --mode replace \
    --edit "modern ultrawide curved monitor"
```

### Style Transfer

```bash
# Apply style to specific region
python main.py --image examples/furniture.jpg \
    --prompt "wooden chair" \
    --mode style \
    --edit "modern metal chair"
```

### Benchmarking

```bash
# Run performance benchmark
python main.py --image examples/test.jpg --mode benchmark
```

## Advanced Usage

### Python API

```python
from pipeline import OpenVocabSegmentationPipeline
from PIL import Image

# Initialize pipeline
pipeline = OpenVocabSegmentationPipeline(
    device="cuda",
    verbose=True
)

# Segmentation only
result = pipeline.segment(
    "image.jpg",
    text_prompt="red car",
    top_k=5
)

# Segmentation + Editing
result = pipeline.segment_and_edit(
    "image.jpg",
    text_prompt="person wearing hat",
    edit_operation="remove"
)

# Access results
print(f"Found {len(result.segmentation_masks)} masks")
print(f"Top score: {result.segmentation_masks[0].final_score:.3f}")
result.edited_image.save("output.png")

# Visualize
visualizations = pipeline.visualize_results(result)
```

### Configuration Presets

```python
from config import get_fast_config, get_quality_config, get_balanced_config

# Fast mode (30s per image)
config = get_fast_config()

# Quality mode (60s per image)
config = get_quality_config()

# Balanced mode (default, 10-20s per image)
config = get_balanced_config()
```

### Custom Configuration

```python
from config import PipelineConfig

config = PipelineConfig()

# SAM 2 settings
config.sam2.points_per_side = 48  # More masks
config.sam2.pred_iou_thresh = 0.90  # Higher quality

# CLIP settings
config.clip.extract_layers = [12, 24]  # Fewer layers (faster)
config.clip.image_size = 336

# Alignment settings
config.alignment.background_weight = 0.4  # Stronger suppression
config.alignment.similarity_threshold = 0.30  # Higher threshold

# Inpainting settings
config.inpainting.num_inference_steps = 75  # More steps (better quality)
config.inpainting.guidance_scale = 9.0  # Stronger prompt adherence
```

## Performance

Benchmarks on NVIDIA RTX 3090:

| Stage | Time | Notes |
|-------|------|-------|
| SAM 2 Generation | 2-4s | ~200 masks, 32×32 grid |
| CLIP Alignment | 0.1s | 4-layer features, 200 masks |
| Inpainting | 5-10s | 50 steps, 512×512 |
| **Total** | **10-20s** | End-to-end pipeline |

## Evaluation Metrics

Segmentation performance on standard benchmarks:

| Dataset | mIoU (%) | Notes |
|---------|----------|-------|
| PASCAL VOC | 58.4 | 20 classes |
| COCO-Stuff | 35.9 | 171 classes |
| ADE20K | 33.2 | 150 classes |

Zero-shot performance on COCO-Open:
- **Novel classes**: 32.4% mIoU
- **Base classes**: 45.2% mIoU

## Project Structure

```
code/
├── models/
│   ├── sam2_segmentation.py   # SAM 2 mask generation
│   ├── clip_features.py       # CLIP feature extraction
│   ├── mask_alignment.py      # Mask-text alignment
│   └── inpainting.py          # Stable Diffusion inpainting
├── pipeline.py                # Main pipeline integration
├── config.py                  # Configuration presets
├── utils.py                   # Utility functions
├── main.py                    # CLI entry point
├── examples/                  # Example scripts
└── requirements.txt           # Python dependencies
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Basic segmentation
- Object removal and replacement
- Batch processing
- Custom configurations
- Evaluation on benchmark datasets

## Limitations

Known limitations (see Chapter 5.3 for details):

1. **Small objects**: Objects < 32×32 pixels often missed
2. **Heavy occlusions**: Incomplete masks for heavily occluded objects
3. **Domain shift**: Degraded performance on artistic/sketch images
4. **Inpainting artifacts**: Complex textures (text, patterns) may have artifacts
5. **Ambiguous prompts**: Vague queries like "thing on table" fail

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{garcia2025openvocab,
  title={Open-Vocabulary Semantic Segmentation for Generative AI},
  author={García García, Pablo},
  school={Universidad de Zaragoza},
  year={2025}
}
```

## References

Key papers implemented:
- SAM 2: Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024
- MaskCLIP: Zhou et al., "Extract Free Dense Labels from CLIP", ECCV 2022
- CLIPSeg: Lüddecke & Ecker, "Image Segmentation Using Text and Image Prompts", CVPR 2022
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
- Stable Diffusion: Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022

## License

This code is provided for academic and research purposes. See LICENSE for details.

## Acknowledgments

- Meta AI for SAM 2
- OpenAI for CLIP
- Stability AI for Stable Diffusion
- Universidad de Zaragoza

## Contact

Pablo García García
Universidad de Zaragoza
Email: [your-email]

For issues and questions, please open an issue on GitHub.
