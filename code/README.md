# Open-Vocabulary Semantic Segmentation for Generative AI

Master's Thesis Implementation - Universidad de Zaragoza

This repository implements **two complementary approaches** for open-vocabulary semantic segmentation with generative image editing capabilities.

## 🚀 Quick Start

### Default: Dense SCLIP + SAM2 Refinement (Approach 2)

```bash
# Replace car with Rayo McQueen
python main.py --image photo.jpg --prompt "car" --mode replace \
  --edit "Rayo McQueen from Cars movie" --visualize

# Segment sky (stuff class - SCLIP's strength)
python main.py --image landscape.jpg --prompt "sky" --mode segment \
  --vocabulary sky clouds ocean mountains --visualize
```

### Fast: Proposal-Based SAM2+CLIP (Approach 1)

```bash
# Use --use-proposals for faster, discrete object segmentation
python main.py --image photo.jpg --prompt "person" --mode remove \
  --use-proposals --visualize
```

## 📊 Two Complementary Approaches

This implementation provides **both methodologies** from the thesis:

### Approach 1: Proposal-Based (SAM2+CLIP)

**Chapter Reference:** Chapter 2, Section 2.1

**How it works:**
1. SAM2 generates mask proposals
2. CLIP scores masks with multi-scale voting (224px, 336px, 512px)
3. Adaptive selection chooses best masks
4. Stable Diffusion performs inpainting

**Best for:**
- ✅ Discrete objects (cars, people, furniture)
- ✅ Speed-critical applications (2-4s per image)
- ✅ Interactive editing scenarios
- ✅ Multi-instance detection

**Results:** **69.3% mIoU** on PASCAL VOC 2012

**Usage:**
```bash
python main.py --image photo.jpg --prompt "car" --use-proposals
```

### Approach 2: Dense SCLIP + SAM2 Refinement (Default)

**Chapter Reference:** Chapter 2, Section 2.2

**How it works:**
1. **SCLIP dense prediction** with Cross-layer Self-Attention (CSA)
2. **SAM2 refinement** via majority voting (our novel contribution)
3. Stable Diffusion performs inpainting

**Best for:**
- ✅ Stuff classes (sky, grass, water, road)
- ✅ Semantic scene understanding
- ✅ Datasets with many classes (COCO-Stuff: 171)
- ✅ Fine-grained semantic consistency

**Results:**
- **49.52% mIoU** on COCO-Stuff (best training-free, +83% over ITACLIP)
- **48.09% mIoU** on PASCAL VOC

**Usage:**
```bash
# Default - no flag needed
python main.py --image photo.jpg --prompt "sky" --vocabulary sky clouds ocean
```

## 📖 Command-Line Reference

### Required Arguments

```bash
--image PATH          # Input image path
--prompt TEXT         # Target object/region description
```

### Operation Modes

```bash
--mode segment        # Segmentation only (default)
--mode remove         # Object removal
--mode replace        # Object replacement (requires --edit)
--mode style          # Style transfer (requires --edit)
```

### Method Selection

```bash
--use-proposals       # Use proposal-based (SAM2+CLIP) instead of dense (SCLIP+SAM2)
```

### Dense Approach Options

```bash
--vocabulary CLASS1 CLASS2 ...    # Additional classes for better context
                                   # Example: --vocabulary sky ocean road building
```

### Proposal Approach Options

```bash
--top-k N             # Number of masks to return (default: 5)
--adaptive            # Adaptive selection (auto-determines mask count)
```

### General Options

```bash
--edit TEXT           # Edit description (for replace/style modes)
--output DIR          # Output directory (default: output/)
--device cuda/cpu     # Computation device (default: cuda)
--visualize           # Save visualization images
--no-save             # Don't save outputs
```

## 📝 Usage Examples

### Example 1: Car Replacement (Proposal-Based - Fast)

```bash
python main.py --image street.jpg --prompt "car" --mode replace \
  --edit "futuristic flying car" --use-proposals --visualize
```

**Why proposals?** Cars are discrete objects - proposal-based is faster and more accurate.

### Example 2: Sky Editing (Dense - Better Quality)

```bash
python main.py --image landscape.jpg --prompt "sky" --mode style \
  --edit "dramatic sunset with orange clouds" \
  --vocabulary sky clouds ocean mountains --visualize
```

**Why dense?** Sky is a "stuff" class - dense prediction excels here.

### Example 3: Road Replacement (Dense with Context)

```bash
python main.py --image street.jpg --prompt "road" --mode replace \
  --edit "snowy ski slope" \
  --vocabulary road asphalt ocean sky mountain vegetation --visualize
```

**Vocabulary helps!** More context classes improve SCLIP's understanding.

### Example 4: Person Removal (Proposal-Based)

```bash
python main.py --image photo.jpg --prompt "person" --mode remove \
  --use-proposals --visualize
```

**Fast removal:** Proposal-based works in ~4 seconds.

## 🎯 When to Use Which Approach?

### Use Dense SCLIP + SAM2 (Default) When:

- ✅ **YES for:** Stuff classes (sky, grass, water, road, floor)
- ✅ **YES for:** Semantic scene understanding
- ✅ **YES for:** Datasets with many classes
- ❌ **NOT for:** Small discrete objects (<32×32 pixels)
- ⚠️ **Trade-off:** Slower (~30s per image)

**Examples:** sky, ocean, grass, road, floor, wall, ceiling, snow, sand

### Use Proposal-Based SAM2+CLIP When:

- ✅ **YES for:** Discrete objects (car, person, chair, dog)
- ✅ **YES for:** Speed-critical applications (2-4s per image)
- ✅ **YES for:** Interactive editing
- ✅ **YES for:** Multi-instance scenarios
- ❌ **NOT for:** Stuff classes (will miss large amorphous regions)

**Examples:** person, car, chair, dog, laptop, bottle, cup, phone

## 📂 Output Files

### Dense Approach (Default)

```
output/
├── original.png              # Input image
├── sclip_prediction.png      # Dense SCLIP segmentation (red overlay)
├── sam2_refined_mask.png     # SAM2-refined mask (green overlay)
├── edited.png                # Final edited image (if editing mode)
└── comparison.png            # Side-by-side comparison (if editing mode)
```

### Proposal Approach (--use-proposals)

```
output/
├── original.png              # Input image
├── segmentation.png          # Top scored masks visualization
├── similarity_map.png        # CLIP similarity heatmap
├── edited.png                # Final edited image (if editing mode)
└── comparison.png            # Side-by-side comparison (if editing mode)
```

## 🔬 Performance Metrics

### Dense SCLIP + SAM2 (Our Extension)

| Dataset | Method | mIoU | Improvement |
|---------|--------|------|-------------|
| COCO-Stuff | SCLIP (CSA only) | 35.41% | Baseline |
| COCO-Stuff | **SCLIP + SAM2** | **49.52%** | **+39.9%** |
| PASCAL VOC | SCLIP (CSA only) | 38.50% | Baseline |
| PASCAL VOC | **SCLIP + SAM2** | **48.09%** | **+24.9%** |

### Comparison to State-of-the-Art

| Method | COCO-Stuff | PASCAL VOC | Approach | Speed |
|--------|------------|------------|----------|-------|
| MaskCLIP (ViT-B/16) | 12.5% | 21.7% | Dense | - |
| ITACLIP | 27.0% | 67.9% | Dense + I+T+A | - |
| **Our SCLIP+SAM2** | **49.52%** | 48.09% | Dense + Refinement | 30s |
| **Our SAM2+CLIP** | - | **69.3%** | Proposal-based | 4s |

**Key Achievement:** +83% improvement over ITACLIP on COCO-Stuff (27.0% → 49.52%)

## 📖 Citation

```bibtex
@mastersthesis{garcia2025ovs,
  title={Open-Vocabulary Semantic Segmentation for Generative AI},
  author={García García, Pablo},
  school={Universidad de Zaragoza},
  year={2025}
}
```

## 🙏 Acknowledgments

- **SCLIP** - CSA attention mechanism (Wang et al., ECCV 2024)
- **SAM 2** - Segment Anything (Ravi et al., 2024)
- **CLIP** - Vision-language model (Radford et al., ICML 2021)
- **Stable Diffusion** - Image inpainting (Rombach et al., CVPR 2022)

---

**Universidad de Zaragoza** • **2024-2025**
