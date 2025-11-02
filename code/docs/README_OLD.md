# Open-Vocabulary Semantic Segmentation for Generative AI

Master's Thesis Implementation by Pablo García García

This repository implements an open-vocabulary semantic segmentation pipeline that integrates SAM 2, CLIP, and Stable Diffusion for flexible, language-driven image understanding and manipulation.

## 🎯 What is This?

This system allows you to:
- **Segment any object** using natural language descriptions (even objects never seen during training)
- **Remove objects** from images realistically
- **Replace objects** with AI-generated alternatives
- **Apply style transfer** to specific regions

**Example:** You can say "the red car in the background" and the system will find it, segment it, and remove/replace it - all without any manual selection!

## 🏗️ Architecture

The pipeline combines four state-of-the-art models:

1. **SAM 2** (Meta): Generates high-quality segmentation masks
2. **CLIP** (OpenAI): Aligns visual regions with text descriptions
3. **Mask Alignment**: Scores masks based on semantic similarity
4. **Stable Diffusion v2**: Performs realistic inpainting and generation

```
Input Image + Text Prompt
    ↓
SAM 2: Generate 100-300 mask candidates
    ↓
CLIP: Extract dense vision-language features
    ↓
Alignment: Score masks against text prompt
    ↓
Select top-K masks
    ↓
Stable Diffusion: Edit selected regions
    ↓
Output Image
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd /home/pablo/tfm/code
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Download SAM 2 checkpoints
- Verify everything works

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Download SAM 2 checkpoints
python scripts/download_sam2_checkpoints.py --model sam2_hiera_large
```

See [SETUP.md](SETUP.md) for detailed instructions.

## 📦 Requirements

- **GPU**: NVIDIA GTX 1060 6GB or better
- **CUDA**: 11.8 or 12.x
- **Python**: 3.10+
- **Disk Space**: ~20GB (for models and checkpoints)
- **RAM**: 16GB recommended

## 🎨 Usage Examples

### 1. Segment Objects

Find and highlight objects matching a text description:

```bash
python main.py \
  --image photo.jpg \
  --prompt "person wearing red jacket" \
  --mode segment \
  --top-k 5 \
  --visualize
```

**Output**: `output/segmentation.png` with top 5 matching masks highlighted

### 2. Remove Objects

Remove objects from the scene:

```bash
python main.py \
  --image photo.jpg \
  --prompt "traffic cone" \
  --mode remove \
  --visualize
```

**Output**: `output/edited.png` with the object removed and background filled naturally

### 3. Replace Objects

Replace one object with another:

```bash
python main.py \
  --image photo.jpg \
  --prompt "old television" \
  --mode replace \
  --edit "modern 4K smart TV" \
  --visualize
```

**Output**: `output/edited.png` with the old TV replaced by a modern one

### 4. Style Transfer

Apply artistic styles to specific regions:

```bash
python main.py \
  --image photo.jpg \
  --prompt "building facade" \
  --mode style \
  --edit "impressionist painting style, like Monet" \
  --visualize
```

**Output**: `output/edited.png` with the building rendered in impressionist style

### 5. Benchmark Performance

Test the pipeline's performance:

```bash
python main.py --image photo.jpg --mode benchmark
```

## ⚙️ Configuration

Three quality presets are available:

| Preset | Speed | Quality | Best For |
|--------|-------|---------|----------|
| `fast` | Fastest | Good | Testing, iteration |
| `balanced` | Medium | Very Good | General use (default) |
| `quality` | Slowest | Best | Final results, thesis figures |

Usage:
```bash
python main.py --image photo.jpg --prompt "dog" --config quality
```

## 📊 Performance (GTX 1060 6GB)

Expected processing times:

- **SAM 2 mask generation**: 3-6 seconds
- **CLIP alignment**: 0.2-0.5 seconds
- **Stable Diffusion inpainting**: 8-15 seconds
- **Total pipeline**: 15-30 seconds (segmentation + editing)

## 🏭 Project Structure

```
code/
├── models/                    # Model implementations
│   ├── sam2_segmentation.py  # SAM 2 mask generation
│   ├── clip_features.py      # CLIP feature extraction
│   ├── mask_alignment.py     # Mask-text alignment
│   └── inpainting.py         # Stable Diffusion inpainting
├── scripts/                   # Utility scripts
│   └── download_sam2_checkpoints.py
├── checkpoints/               # SAM 2 model weights
├── output/                    # Generated results
├── main.py                    # CLI entry point
├── pipeline.py                # Complete pipeline
├── config.py                  # Configuration presets
├── utils.py                   # Helper functions
├── requirements.txt           # Python dependencies
├── setup.sh                   # Automated setup script
├── SETUP.md                   # Detailed setup guide
└── README.md                  # This file
```

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Use smaller SAM 2 model
python scripts/download_sam2_checkpoints.py --model sam2_hiera_tiny

# Or reduce image size
python main.py --image large.jpg --prompt "car" --mode segment
# Pipeline automatically resizes if needed
```

### SAM 2 Checkpoint Not Found

```bash
# Download the checkpoint
python scripts/download_sam2_checkpoints.py --model sam2_hiera_large

# Verify
ls -lh checkpoints/
```

### Slow Inference

- Use `--config fast`
- Use smaller SAM 2 model (`sam2_hiera_tiny` or `sam2_hiera_small`)
- Close other GPU applications

See [SETUP.md](SETUP.md) for more troubleshooting tips.

## 📚 Thesis Context

This implementation supports the Master's thesis:

**"Open-Vocabulary Semantic Segmentation for Generative AI"**

Key contributions:
1. Integration of SAM 2, CLIP, and Stable Diffusion into unified pipeline
2. Multi-scale CLIP feature extraction strategy (+4.2% mIoU improvement)
3. Zero-shot segmentation of arbitrary objects via natural language
4. Practical system achieving 15-30s end-to-end latency

See the thesis document (in `/home/pablo/tfm/overleaf`) for:
- Detailed methodology (Chapter 3)
- Experimental results (Chapter 4)
- Ablation studies and analysis

## 📖 Citation

```bibtex
@mastersthesis{garcia2025openvocab,
  title={Open-Vocabulary Semantic Segmentation for Generative AI},
  author={García García, Pablo},
  year={2025},
  school={Universidad de Zaragoza}
}
```

## 🔗 References

- **SAM 2**: [Segment Anything in Images and Videos](https://github.com/facebookresearch/segment-anything-2)
- **CLIP**: [Learning Transferable Visual Models](https://github.com/openai/CLIP)
- **Stable Diffusion**: [High-Resolution Image Synthesis](https://github.com/Stability-AI/stablediffusion)

## 📝 License

This code is for academic and research purposes as part of a Master's thesis.

## 🤝 Acknowledgments

- Thesis supervisors: Alejandro Pérez Yus, María Santos Villafranca
- Universidad de Zaragoza, Escuela de Ingeniería y Arquitectura
- Meta AI (SAM 2), OpenAI (CLIP), Stability AI (Stable Diffusion)

---

For detailed setup instructions, see [SETUP.md](SETUP.md)

For usage examples and troubleshooting, run `python main.py --help`
