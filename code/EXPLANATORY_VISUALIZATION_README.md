# CLIP-Guided SAM Explanatory Visualization

Comprehensive educational visualization tool that demonstrates every step of the CLIP-guided SAM pipeline with interactive HTML output.

## Overview

This tool generates **18 detailed visualizations** showing the complete process from raw image to final segmentation, plus an **interactive HTML viewer** for easy exploration.

## Features

### ✨ Complete Pipeline Visualization (18 Steps)

1. **Raw Image** - Original input
2. **Text Prompt Processing** - Vocabulary and multi-descriptor expansion
3. **SCLIP Sliding Window Patches** - 224×224 grid overlay
4. **Visual-Text Similarity** - Cosine similarity computation
5. **Per-Class Confidence Maps** - Individual heatmaps per class
6. **SCLIP Dense Prediction** - Argmax colored segmentation
7. **SCLIP Prediction Confidences** - Multi-view (heatmap + uncertain regions + weighted overlay)
8. **Confidence Thresholding** - Regions passing min_confidence
9. **Connected Components** - Labeled regions per class
10. **Min Region Size Filter** - Before/after filtering
11. **Positive Point Prompts** - Centroid or multi-point visualization
12. **Negative Point Prompts** - Background point extraction
13. **SAM Prompting** - All prompts overlaid (green=positive, red=negative)
14. **SAM Multi-Mask Candidates** - 3 masks per prompt with scores
15. **Best Mask Selection** - Highest scoring mask per prompt
16. **Overlap Resolution** - IoU-based merging
17. **Final Segmentation** - Clean output
18. **Comparison Grid** - Side-by-side: Raw | SCLIP | Confidences | Final

### 🌐 Interactive HTML Visualization

- **Single-page interface** with all 18 steps
- **Click navigation** between steps
- **Keyboard shortcuts** (← → arrows)
- **Responsive design** for different screen sizes
- **Statistics dashboard** with per-class metrics
- **Professional styling** ready for presentations

### 📊 Optional Per-Class Detailed Views

For each detected class:
- Confidence map
- Connected components
- Filtered regions
- Prompt points (positive + negative)
- SAM mask candidates
- Final selected mask
- Statistics (histogram, size distribution)

## Installation

No additional dependencies beyond the main project requirements:
```bash
# Already installed from main project
pip install torch torchvision
pip install open-clip-torch
pip install matplotlib scipy pillow opencv-python
pip install sam2  # For SAM 2 functionality
```

## Usage

### Basic Usage

```bash
python clip_guided_sam_explanatory.py \
    --image examples/football_frame.png \
    --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background \
    --output explanatory_results/football \
    --min-confidence 0.3 \
    --create-html
```

### Advanced Usage with Negative Prompts

```bash
python clip_guided_sam_explanatory.py \
    --image examples/football_frame.png \
    --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background \
    --output explanatory_results/football_advanced \
    --min-confidence 0.3 \
    --min-region-size 100 \
    --points-per-cluster 3 \
    --negative-points-per-cluster 2 \
    --negative-confidence-threshold 0.8 \
    --create-html \
    --per-class-details
```

### Cityscapes Example

```bash
python clip_guided_sam_explanatory.py \
    --image examples/cityscapes_sample.png \
    --vocabulary road sidewalk building wall fence pole person car \
    --output explanatory_results/cityscapes \
    --min-confidence 0.3 \
    --min-region-size 200 \
    --points-per-cluster 2 \
    --negative-points-per-cluster 1 \
    --create-html
```

### Quick Start with Examples

```bash
# Run all examples
chmod +x run_explanatory_example.sh
./run_explanatory_example.sh
```

## Parameters

### Required Parameters

- `--image PATH` - Path to input image
- `--vocabulary CLASSES...` - List of class names (space-separated)
- `--output DIR` - Output directory for results

### SCLIP Parameters

- `--min-confidence FLOAT` - Minimum CLIP confidence threshold (default: 0.3)
  - Lower values → more regions, but less reliable
  - Higher values → fewer regions, but more confident
  - Recommended: 0.3 for general use, 0.5 for high-quality data

- `--min-region-size INT` - Minimum region size in pixels (default: 100)
  - Filters out tiny regions that are likely noise
  - Larger values → fewer, larger regions
  - Recommended: 100-200 for high-res images, 50-100 for low-res

### SAM Prompt Parameters

- `--points-per-cluster INT` - Number of positive points per cluster (default: 1)
  - 1 = centroid only (fastest)
  - 3-5 = multiple spatially-diverse points (better coverage)
  - 10+ = dense sampling (slower but thorough)

- `--negative-points-per-cluster INT` - Number of negative points per cluster (default: 0)
  - 0 = disabled (no background points)
  - 1-2 = light background guidance
  - 3-5 = strong background disambiguation
  - Helps SAM distinguish foreground from background

- `--negative-confidence-threshold FLOAT` - Min confidence for negative regions (default: 0.8)
  - Only high-confidence competing classes serve as negative examples
  - Higher = more selective negative points

- `--iou-threshold FLOAT` - IoU threshold for overlap merging (default: 0.8)
  - Higher = less aggressive merging (keeps more separate masks)
  - Lower = more aggressive merging (fewer overlaps)

### Visualization Options

- `--create-html` - Generate interactive HTML (default: True)
- `--per-class-details` - Generate per-class detailed views (optional)
- `--device {cuda,cpu}` - Computation device (default: auto-detect)

## Output Structure

```
output_dir/
├── index.html                    # Interactive visualization
├── steps/                        # Individual step images
│   ├── 01_raw_image.png
│   ├── 02_text_prompts.png
│   ├── 03_sclip_patches.png
│   ├── 04_similarity.png
│   ├── 05_confidence_maps.png
│   ├── 06_dense_prediction.png
│   ├── 07_prediction_confidences.png
│   ├── 08_thresholding.png
│   ├── 09_connected_components.png
│   ├── 10_size_filter.png
│   ├── 11_positive_prompts.png
│   ├── 12_negative_prompts.png
│   ├── 13_sam_prompting.png
│   ├── 14_sam_candidates.png
│   ├── 15_mask_selection.png
│   ├── 16_overlap_resolution.png
│   ├── 17_final_segmentation.png
│   └── 18_comparison_grid.png
├── per_class/ (if --per-class-details)
│   ├── class_0_Lionel_Messi.png
│   ├── class_1_Luis_Suarez.png
│   └── ...
└── data/                         # Intermediate data
    ├── confidences.npy           # CLIP confidence maps
    ├── prompts.json              # Extracted prompt points
    └── statistics.json           # Pipeline statistics
```

## Interactive HTML Features

### Navigation
- **Step buttons** at top - Click to jump to any step
- **Previous/Next buttons** at bottom
- **Keyboard shortcuts**:
  - `←` Previous step
  - `→` Next step

### Display
- **Full-resolution images** for all steps
- **Smooth animations** between steps
- **Professional styling** suitable for presentations
- **Mobile-responsive** layout

### Information
- **Step descriptions** explaining each visualization
- **Statistics panel** (when applicable)
- **Parameter display** showing configuration used

## Use Cases

### 1. Educational/Tutorial
Perfect for:
- Teaching how CLIP-guided SAM works
- Thesis/dissertation presentations
- Research paper supplementary materials
- Blog posts explaining the methodology

### 2. Debugging
Helps identify:
- Why certain classes are missed (check confidence maps)
- Boundary issues (check thresholding step)
- Prompt quality (check prompt extraction steps)
- SAM mask quality (check candidates step)

### 3. Parameter Tuning
Visualize effects of:
- Different confidence thresholds
- Region size filters
- Positive/negative point counts
- IoU thresholds

### 4. Results Comparison
Compare:
- SCLIP vs SAM-refined outputs
- Different parameter configurations
- Various prompt strategies

## Tips for Best Results

### For High-Quality Visualizations

1. **Use diverse vocabulary**: Mix stuff (grass, sky) and things (person, car)
2. **Tune confidence threshold**: Start at 0.3, increase if too noisy
3. **Enable negative prompts**: Helps with class disambiguation
4. **Use multiple points per cluster**: Better for large objects

### For Fast Iteration

1. **Start with 1 point per cluster**
2. **Use higher confidence threshold** (0.5-0.7)
3. **Disable per-class details**
4. **Use smaller images** for testing

### For Paper/Presentation

1. **Enable all features** (negative prompts, multi-point)
2. **Use --per-class-details** for key classes
3. **Generate multiple variants** with different parameters
4. **Export Step 18 (comparison grid)** for figures

## Troubleshooting

### No prompts extracted
- **Problem**: No high-confidence regions found
- **Solution**: Lower --min-confidence (try 0.2)

### Too many prompts
- **Problem**: Visualization cluttered with points
- **Solution**: Increase --min-confidence or --min-region-size

### Poor SAM masks
- **Problem**: Masks don't match objects well
- **Solution**:
  - Increase --points-per-cluster (try 3-5)
  - Add negative prompts (--negative-points-per-cluster 2)
  - Check if CLIP confidences are good (Step 7)

### Slow generation
- **Problem**: Takes too long to generate
- **Solution**:
  - Use CPU if GPU memory is limited
  - Reduce image resolution
  - Disable --per-class-details
  - Use fewer classes in vocabulary

## Technical Details

### Step 7: Confidence Visualization

Shows three views:
1. **Heatmap**: Max probability at each pixel (0.0-1.0)
2. **Weighted Overlay**: Opacity proportional to confidence
3. **Uncertain Regions**: Pixels below threshold marked in red

This is crucial for understanding:
- Where CLIP is confident
- Boundary uncertainties
- Which regions need refinement

### Negative Points

Negative points are extracted from:
- High-confidence regions of OTHER classes
- At least 10 pixels away from positive region
- Selected with spatial diversity (farthest-point sampling)

Benefits:
- Helps SAM distinguish similar-looking classes
- Improves boundary precision
- Reduces false positives

### Color Palette

Uses golden ratio spacing for visually distinct colors:
- Optimal for human perception
- Works well for colorblind viewers
- Consistent across all visualizations

## Citation

If you use this visualization tool in your research, please cite:

```bibtex
@misc{clipguidedsam_explanatory,
  title={CLIP-Guided SAM Explanatory Visualization},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/yourrepo}}
}
```

## License

Same license as the main CLIP-Guided SAM project.

## Acknowledgments

- SCLIP paper for the dense prediction approach
- SAM 2 for the segmentation foundation model
- CLIP for the visual-semantic alignment

---

**Questions or Issues?**

Open an issue on GitHub or contact the maintainers.

**Want to Contribute?**

Pull requests welcome! Areas for improvement:
- Additional visualization types
- Animation generation
- Per-region statistics
- Video support
