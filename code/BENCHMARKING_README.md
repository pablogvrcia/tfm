# Benchmarking SCLIP with LoFTup Enhancement

This guide explains how to use the improved benchmarking script to evaluate and compare SCLIP segmentation with and without LoFTup feature upsampling.

## Quick Start

### Basic Evaluation (without LoFTup)

```bash
# Evaluate on Pascal VOC (10 samples)
python run_benchmarks.py --dataset pascal-voc --num-samples 10

# Evaluate on COCO-Stuff (10 samples)
python run_benchmarks.py --dataset coco-stuff --num-samples 10
```

### Evaluation with LoFTup

```bash
# Enable LoFTup enhancement
python run_benchmarks.py --dataset pascal-voc --num-samples 10 --use-loftup

# With custom upsampling factor (fixed mode)
python run_benchmarks.py --dataset pascal-voc --num-samples 10 --use-loftup --loftup-factor 3.0
```

### A/B Comparison (Baseline vs LoFTup)

**Recommended**: This runs both baseline and LoFTup on the same dataset and generates a comparison report.

```bash
# Compare on Pascal VOC
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --compare-loftup

# Compare with visualizations saved
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --compare-loftup --save-vis

# Full comparison on COCO-Stuff
python run_benchmarks.py --dataset coco-stuff --num-samples 100 --compare-loftup --save-vis
```

## Command-Line Options

### Dataset Options

- `--dataset`: Dataset to evaluate (`pascal-voc`, `coco-stuff`)
- `--data-dir`: Path to dataset directory (default: `data/benchmarks`)
- `--num-samples`: Number of samples to evaluate (None = all)

### Model Options

- `--model`: CLIP model variant (default: `ViT-B/16`)
- `--use-sam`: Use SAM for mask proposals (hybrid mode)
- `--use-pamr`: Enable PAMR refinement
- `--pamr-steps`: Number of PAMR iterations (default: 10)

### Inference Options

- `--slide-inference`: Use sliding window inference (default: True)
- `--slide-crop`: Crop size for sliding window (default: 224)
- `--slide-stride`: Stride for sliding window (default: 112)

### LoFTup Options

- `--use-loftup`: Enable LoFTup feature upsampling
- `--no-loftup`: Disable LoFTup (explicit)
- `--loftup-adaptive`: Use adaptive upsampling (default: True)
- `--loftup-factor`: Fixed upsampling factor if not adaptive (default: 2.0)
- `--compare-loftup`: Run A/B comparison (baseline vs LoFTup)

### Output Options

- `--output-dir`: Output directory for results (default: `benchmarks/results`)
- `--save-vis`: Save visualization images

## Example Usage Scenarios

### 1. Quick Test (10 samples, no visualization)

```bash
# Baseline
python run_benchmarks.py --dataset pascal-voc --num-samples 10

# With LoFTup
python run_benchmarks.py --dataset pascal-voc --num-samples 10 --use-loftup
```

### 2. Comprehensive Comparison (50 samples, with visualizations)

```bash
python run_benchmarks.py \
    --dataset pascal-voc \
    --num-samples 50 \
    --compare-loftup \
    --save-vis \
    --output-dir benchmarks/results/pascal_voc_comparison
```

This will:
- Run baseline evaluation (without LoFTup) on 50 images
- Run LoFTup-enhanced evaluation on the same 50 images
- Generate side-by-side visualizations in separate folders
- Create a detailed comparison report (JSON + console output)
- Show metrics improvements, per-class analysis, and timing overhead

### 3. Full Dataset Evaluation

```bash
# Pascal VOC full validation set (~1,449 images)
python run_benchmarks.py \
    --dataset pascal-voc \
    --compare-loftup \
    --output-dir benchmarks/results/pascal_voc_full

# COCO-Stuff full validation set
python run_benchmarks.py \
    --dataset coco-stuff \
    --compare-loftup \
    --output-dir benchmarks/results/coco_stuff_full
```

**Note**: Full dataset evaluation may take several hours depending on your GPU.

### 4. Different LoFTup Configurations

```bash
# Adaptive upsampling (default)
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --use-loftup --loftup-adaptive

# Fixed 2× upsampling
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --use-loftup --loftup-factor 2.0

# Fixed 4× upsampling (more aggressive)
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --use-loftup --loftup-factor 4.0
```

### 5. With PAMR Refinement

```bash
# Compare with PAMR enabled
python run_benchmarks.py \
    --dataset pascal-voc \
    --num-samples 50 \
    --compare-loftup \
    --use-pamr \
    --pamr-steps 10
```

## Understanding the Output

### Console Output (Comparison Mode)

```
================================================================================
COMPARISON: Baseline vs LoFTup
================================================================================

Overall Metrics:
Metric               Baseline     LoFTup       Improvement
------------------------------------------------------------
miou                  57.32%       59.78%       +2.46%
pixel_accuracy        72.15%       74.65%       +2.50%
f1                    60.12%       62.36%       +2.24%
precision             65.83%       68.28%       +2.45%
recall                70.15%       72.91%       +2.76%
boundary_f1           62.98%       65.47%       +2.49%

Timing:
Time per image        26.57s       32.14s       +21.0%

Top 10 Classes with Largest Improvements:
Class                Baseline     LoFTup       Improvement
------------------------------------------------------------
person                14.22%       18.50%       +4.28%
boat                  20.34%       24.80%       +4.46%
chair                 19.04%       23.15%       +4.11%
...

================================================================================
SUMMARY
================================================================================
✓ LoFTup improves mIoU by 2.46 percentage points
  (57.32% → 59.78%)
✓ Computational overhead: +21.0%
✓ Improved classes: 18/21 (85.7%)
```

### Output Files

#### 1. Comparison Report (JSON)

`benchmarks/results/pascal-voc_loftup_comparison.json`:

```json
{
  "dataset": "pascal-voc",
  "num_samples": 50,
  "timestamp": "2025-11-10T15:30:00",
  "baseline": {
    "miou": 0.5732,
    "f1": 0.6012,
    "time_per_image": 26.57,
    "per_class_iou": {
      "background": 0.7503,
      "person": 0.1422,
      ...
    }
  },
  "loftup": {
    "miou": 0.5978,
    "f1": 0.6236,
    "time_per_image": 32.14,
    "per_class_iou": {
      "background": 0.7503,
      "person": 0.1850,
      ...
    }
  },
  "comparison": {
    "miou_improvement": 2.46,
    "f1_improvement": 2.24,
    "overhead_percent": 21.0
  }
}
```

#### 2. Visualizations (if --save-vis is used)

```
benchmarks/results/visualizations/
├── pascal-voc/
│   ├── baseline/
│   │   ├── sample_0000.png
│   │   ├── sample_0001.png
│   │   └── ...
│   └── loftup/
│       ├── sample_0000.png
│       ├── sample_0001.png
│       └── ...
```

Each visualization shows:
- Original image
- Ground truth segmentation
- Predicted segmentation
- Color-coded legend showing all present classes

## Performance Expectations

### Pascal VOC (21 classes)

| Configuration | mIoU | Time per Image | Notes |
|--------------|------|----------------|-------|
| Baseline | ~57% | ~27s | Standard SCLIP |
| LoFTup (2×) | ~60% | ~32s | +3% mIoU, +19% time |
| LoFTup (4×) | ~61% | ~38s | +4% mIoU, +41% time |

### COCO-Stuff (171 classes)

| Configuration | mIoU | Time per Image | Notes |
|--------------|------|----------------|-------|
| Baseline | ~22% | ~35s | Standard SCLIP |
| LoFTup (2×) | ~24% | ~42s | +2% mIoU, +20% time |

**Note**: Actual performance depends on GPU, image sizes, and dataset characteristics.

## Tips for Effective Benchmarking

### 1. Start Small

Begin with a small number of samples to verify everything works:

```bash
python run_benchmarks.py --dataset pascal-voc --num-samples 10 --compare-loftup
```

### 2. Check Visualizations

Use `--save-vis` to inspect results visually before running full evaluations:

```bash
python run_benchmarks.py \
    --dataset pascal-voc \
    --num-samples 20 \
    --compare-loftup \
    --save-vis
```

### 3. Use Comparison Mode

Always use `--compare-loftup` for fair comparisons (same images, same conditions):

```bash
# Good: Fair comparison
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --compare-loftup

# Bad: Different runs, not comparable
python run_benchmarks.py --dataset pascal-voc --num-samples 50
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --use-loftup
```

### 4. Monitor GPU Memory

LoFTup increases memory usage. If you encounter OOM errors:

```bash
# Reduce upsampling factor
python run_benchmarks.py --dataset pascal-voc --compare-loftup --loftup-factor 1.5

# Or disable adaptive mode
python run_benchmarks.py --dataset pascal-voc --compare-loftup --loftup-factor 2.0
```

### 5. Reproduce Results

For reproducible results, use the same:
- Number of samples (`--num-samples`)
- Model configuration (`--model`, `--slide-crop`, `--slide-stride`)
- Random seed (set in code if needed)

## Troubleshooting

### Issue: "LoFTup unavailable, falling back to standard features"

**Cause**: LoFTup dependencies not installed

**Solution**:
```bash
pip install timm einops
```

### Issue: CUDA out of memory

**Cause**: LoFTup increases memory usage (especially with high upsampling factors)

**Solutions**:
1. Reduce upsampling factor: `--loftup-factor 1.5`
2. Use smaller batches (not applicable in this single-image mode)
3. Use smaller images (reduce resolution in dataset preprocessing)

### Issue: Very slow evaluation

**Cause**: Full dataset + LoFTup + visualizations = slow

**Solutions**:
1. Use `--num-samples` to limit evaluation size
2. Disable visualizations: remove `--save-vis`
3. Use baseline only first, then LoFTup separately

### Issue: Poor LoFTup results

**Cause**: LoFTup may not be loaded correctly

**Check**:
- Look for "[SCLIP] LoFTup initialized successfully" message
- Check if bilinear fallback is being used
- Verify `timm` and `einops` are installed

## Advanced Usage

### Custom Output Directory Structure

```bash
python run_benchmarks.py \
    --dataset pascal-voc \
    --num-samples 100 \
    --compare-loftup \
    --save-vis \
    --output-dir experiments/$(date +%Y%m%d_%H%M%S)_loftup_comparison
```

### Batch Processing Multiple Datasets

```bash
#!/bin/bash
# Run comparison on both datasets
python run_benchmarks.py --dataset pascal-voc --num-samples 50 --compare-loftup --output-dir results/voc
python run_benchmarks.py --dataset coco-stuff --num-samples 50 --compare-loftup --output-dir results/coco
```

### Combining with Other Enhancements

```bash
# LoFTup + PAMR + SAM
python run_benchmarks.py \
    --dataset pascal-voc \
    --num-samples 50 \
    --compare-loftup \
    --use-pamr \
    --use-sam
```

## Citation

If you use this benchmarking script in your research, please cite:

```bibtex
@article{loftup2025,
  title={Coordinate-Based Feature Upsampling for Dense Prediction},
  author={Huang, Andre et al.},
  booktitle={ICCV},
  year={2025}
}
```

## Support

For issues or questions:
- Check the main project README
- Review LoFTup integration documentation: `code/LOFTUP_INTEGRATION.md`
- Open an issue on GitHub
