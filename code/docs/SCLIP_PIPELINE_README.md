# SCLIP-based Segmentation and Editing Pipeline

This directory contains the implementation of **Approach 2** from the thesis: **Extended SCLIP with Novel SAM2 Refinement**.

## Quick Start

### Replace car with Rayo McQueen

```bash
python main_sclip.py \
  --image photo.jpg \
  --prompt "car" \
  --mode replace \
  --edit "Rayo McQueen from Cars movie, red racing car with lightning bolt" \
  --visualize
```

### Segment only (no editing)

```bash
python main_sclip.py \
  --image photo.jpg \
  --prompt "car" \
  --mode segment \
  --visualize
```

### Remove object

```bash
python main_sclip.py \
  --image photo.jpg \
  --prompt "person" \
  --mode remove \
  --visualize
```

## How It Works

This pipeline implements the methodology described in **Chapter 2, Section 2.2** of the thesis:

### Stage 1: SCLIP Dense Prediction (Foundation)

Uses SCLIP's Cross-layer Self-Attention (CSA) to extract dense semantic features:

```
CSA(Q, K, V) = softmax((QQ^T + KK^T) / √d) V
```

- Modifies CLIP's attention mechanism for better spatial consistency
- Extracts features from layers 6, 12, 18, 24
- Performs sliding window inference (224px crops, 112px stride)
- Generates pixel-wise class predictions

**Result:** Dense segmentation map (H × W) with class labels

### Stage 2: SAM2 Mask Refinement (Our Novel Contribution)

Refines SCLIP predictions using SAM2's high-quality boundaries via majority voting:

1. Generate SAM2 mask proposals
2. For each SAM2 mask, compute overlap with SCLIP prediction
3. Keep masks where ≥60% of pixels match SCLIP class
4. Combine refined masks with OR operation

**Result:** High-quality binary mask with precise boundaries

### Stage 3: Stable Diffusion Inpainting (Optional)

For editing operations (remove/replace/style):

1. Use refined mask as inpainting region
2. Apply Stable Diffusion v2 inpainting
3. Generate realistic edits based on text prompts

## Command Line Options

### Required Arguments

- `--image PATH`: Input image path
- `--prompt TEXT`: Target object description
- `--mode {segment,remove,replace,style}`: Operation mode

### Optional Arguments

- `--edit TEXT`: Edit description (required for replace/style)
- `--vocabulary CLASS1 CLASS2 ...`: Additional class names
- `--output DIR`: Output directory (default: `output_sclip/`)
- `--device {cuda,cpu}`: Computation device
- `--use-sam-refinement`: Enable SAM2 refinement (default: True)
- `--visualize`: Save visualization images

## Output Files

### Segmentation Mode

- `original.png`: Input image
- `sclip_prediction.png`: Dense SCLIP segmentation (red overlay)
- `sam2_refined_mask.png`: SAM2-refined mask (green overlay)

### Editing Modes (remove/replace/style)

- `original.png`: Input image
- `sclip_prediction.png`: Dense SCLIP segmentation
- `sam2_refined_mask.png`: SAM2-refined mask
- `edited.png`: Final edited image
- `comparison.png`: Side-by-side comparison (Original | Mask | Edited)

## Performance

### COCO-Stuff 164K (Annotation-Free)

| Method | mIoU | Setting |
|--------|------|---------|
| SCLIP (CSA only) | 35.41% | Dense prediction |
| **SCLIP + SAM2 (ours)** | **49.52%** | Dense + refinement |

### PASCAL VOC 2012 (Annotation-Free)

| Method | mIoU | Setting |
|--------|------|---------|
| SCLIP (CSA only) | 38.50% | Dense prediction |
| **SCLIP + SAM2 (ours)** | **48.09%** | Dense + refinement |

### Improvements

- **+39.9% relative** over SCLIP alone on COCO-Stuff
- **+83% improvement** over ITACLIP (27.0% → 49.52%) on COCO-Stuff
- **+24.9% relative** over SCLIP alone on PASCAL VOC

## Comparison: Proposal-Based vs. Dense Prediction

This repository implements **both approaches** from the thesis:

### Approach 1: Proposal-Based (SAM2+CLIP)

**File:** `main.py`

**Best for:**
- Discrete objects (cars, people, chairs)
- Speed-critical applications (2-4s per image)
- Interactive editing scenarios
- Multi-instance detection

**Results:** 69.3% mIoU on PASCAL VOC

### Approach 2: Dense Prediction (SCLIP+SAM2)

**File:** `main_sclip.py` (this approach)

**Best for:**
- Stuff classes (sky, grass, water, road)
- Semantic scene understanding
- Datasets with many classes (COCO-Stuff: 171 classes)
- Fine-grained semantic consistency

**Results:** 49.52% mIoU on COCO-Stuff (best training-free result)

## Technical Details

### Text Feature Caching

The pipeline caches CLIP text embeddings to avoid recomputation:

- **First image:** 37.55s (encodes text + processes image)
- **Subsequent images:** 26.57s (reuses cached text)
- **Speedup:** 41% (1.41×)

### Sliding Window Inference

SCLIP uses overlapping windows for better coverage:

- **Crop size:** 224px × 224px
- **Stride:** 112px (50% overlap)
- **Upsampling:** Image resized to 2048px max dimension
- **Aggregation:** Average overlapping predictions

### SAM2 Configuration

- **Model:** `sam2_hiera_tiny` (fast variant)
- **Points per side:** 32 (32×32 grid)
- **IoU threshold:** 0.88
- **Stability threshold:** 0.95
- **Typical masks:** 100-300 per image

## Examples

### Example 1: Car Replacement

```bash
python main_sclip.py \
  --image street.jpg \
  --prompt "car" \
  --mode replace \
  --edit "futuristic flying car from Blade Runner" \
  --visualize
```

### Example 2: Sky Style Transfer

```bash
python main_sclip.py \
  --image landscape.jpg \
  --prompt "sky" \
  --mode style \
  --edit "dramatic sunset with orange and purple clouds" \
  --visualize
```

### Example 3: Person Removal

```bash
python main_sclip.py \
  --image photo.jpg \
  --prompt "person" \
  --mode remove \
  --visualize
```

### Example 4: Multi-Class Vocabulary

```bash
python main_sclip.py \
  --image street.jpg \
  --prompt "car" \
  --vocabulary car road building tree sky person \
  --mode segment \
  --visualize
```

This helps SCLIP understand the scene context better by providing related classes.

## Limitations

1. **Small objects:** Objects <32×32 pixels may be missed
2. **Occlusion:** Heavily occluded objects may be incomplete
3. **Speed:** ~27-30s per image (6.75× slower than proposal-based)
4. **Domain shift:** Performance degrades on artistic/sketch images

## Citation

If you use this implementation, please cite:

```bibtex
@mastersthesis{garcia2025sclip,
  title={Open-Vocabulary Semantic Segmentation for Generative AI},
  author={García García, Pablo},
  school={Universidad de Zaragoza},
  year={2025},
  note={Extended SCLIP with novel SAM2 refinement layer}
}

@inproceedings{sclip2024,
  title={Self-Attention Dense Vision-Language Inference with Improved Cross-Layer Feature Aggregation},
  author={Wang, Zhaoqing and Lu, Yu and Li, Qiang and Tao, Xunqiang and Guo, Yandong and Gong, Mingming and Liu, Tongliang},
  booktitle={ECCV},
  pages={1--18},
  year={2024}
}
```

## References

- **SCLIP paper:** Wang et al., ECCV 2024
- **SAM2 paper:** Ravi et al., arXiv 2024
- **Stable Diffusion:** Rombach et al., CVPR 2022
- **CLIP:** Radford et al., ICML 2021

## Troubleshooting

### "No pixels found for class X"

**Solutions:**
1. Add related classes: `--vocabulary car vehicle automobile`
2. Try different prompt variations
3. Check if object is actually in the image

### Out of memory

**Solutions:**
1. Use smaller model: Modify `main_sclip.py` to use `ViT-B/16` instead of `ViT-L/14`
2. Reduce slide crop: `slide_crop=112` instead of 224
3. Use CPU: `--device cpu`

### Slow inference

**Expected:** Dense prediction is slower than proposal-based (~30s vs 4s)

**Optimizations:**
1. Text caching (already enabled, 41% speedup)
2. Use faster SAM variant: `sam2_hiera_tiny`
3. Disable SAM refinement: Remove `--use-sam-refinement` flag
4. Use proposal-based approach instead: `python main.py`

---

**Generated:** November 1, 2024
**Approach:** Dense Prediction (SCLIP) + Novel SAM2 Refinement
**Thesis Chapter:** Chapter 2, Section 2.2
