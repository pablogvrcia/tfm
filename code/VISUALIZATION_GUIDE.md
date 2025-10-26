# Thesis Visualization Guide

## 🎨 Available Visualization Scripts

You now have **2 visualization scripts** ready for generating thesis figures:

### 1. `quick_viz.py` - Fast Single Figure (Recommended for testing)

**What it creates:**
- One comprehensive 6-panel figure showing:
  - Original image
  - SAM 2 mask proposals
  - CLIP similarity heatmap
  - Top 3 aligned masks

**Usage:**
```bash
cd /home/pablo/tfm/code
source venv/bin/activate
python quick_viz.py
```

**Output:** `figures/segmentation_<prompt>.png` (200 DPI, ~900KB)

**Customization:** Edit the script to change:
```python
image_path = "photo.jpg"      # Your image
text_prompt = "person"         # What to segment
```

**Time:** ~15-20 seconds

### 2. `create_visualizations.py` - Full Thesis Figures (For final thesis)

**What it creates:**
- Figure 1: SAM 2 mask generation analysis (3 panels)
- Figure 2: CLIP similarity heatmaps (3 panels)
- Figure 3: Top-K mask alignment (6-9 panels)
- Figure 4: Multi-prompt comparison (3-4 panels)

**Usage:**
```bash
cd /home/pablo/tfm/code
source venv/bin/activate
python create_visualizations.py
```

**Output:** `thesis_figures/*.png` (300 DPI, publication quality)

**Customization:** Edit `main()` function:
```python
IMAGE_PATH = "photo.jpg"
MAIN_PROMPT = "person"
```

**Time:** ~45-60 seconds

## 📊 What Each Figure Shows

### Figure 1: SAM 2 Mask Generation
- **(a)** Original image
- **(b)** Top 50 mask proposals (color-coded)
- **(c)** Mask area distribution histogram

**Purpose:** Demonstrates SAM 2's comprehensive mask generation capability (§3.2.2)

### Figure 2: CLIP Similarity
- **(a)** Original image
- **(b)** Dense similarity heatmap
- **(c)** Similarity overlay on image

**Purpose:** Shows dense vision-language feature extraction (§3.2.1)

### Figure 3: Mask-Text Alignment
- **Grid of 6-9 panels:** Top-ranked masks with scores
- Each panel shows: mask overlay, bounding box, score, area, IoU

**Purpose:** Demonstrates mask-text alignment scoring (§3.2.3)

### Figure 4: Multi-Prompt Comparison
- **Multiple columns:** Different prompts on same image
- Shows open-vocabulary capability

**Purpose:** Zero-shot segmentation of diverse objects

## 🚀 Quick Start Guide

### Step 1: Prepare Your Image

Make sure you have a test image:
```bash
ls photo.jpg  # Should exist
```

Or use your own:
```bash
cp /path/to/your/image.jpg myimage.jpg
```

### Step 2: Run Quick Visualization

```bash
cd /home/pablo/tfm/code
source venv/bin/activate

# Edit prompt if needed
nano quick_viz.py  # Change line: text_prompt = "YOUR_OBJECT"

# Run
python quick_viz.py

# View result
xdg-open figures/segmentation_*.png  # Or use image viewer
```

### Step 3: Check Results

The figure shows:
- If **masks are found**: Red highlighted regions with scores
- If **no matches**: Try different prompts or images

### Step 4: Generate Full Thesis Figures

```bash
# Edit to match your image content
nano create_visualizations.py  # Change MAIN_PROMPT

# Generate all figures
python create_visualizations.py

# Results in thesis_figures/
ls -lh thesis_figures/
```

## 💡 Tips for Good Results

### Choosing Prompts

✅ **Good prompts** (specific, visual):
- "person wearing red jacket"
- "blue car"
- "wooden chair"
- "laptop computer"
- "green plant"

❌ **Poor prompts** (too abstract):
- "happiness"
- "justice"
- "the concept of"

### If No Matches Found

1. **Try different prompts** that actually exist in the image
2. **Lower the threshold** in `models/mask_alignment.py`:
   ```python
   similarity_threshold: float = 0.15  # Was 0.25
   ```
3. **Use different images** with clear objects
4. **Check similarity scores** even if below threshold

### Optimizing for Your GTX 1060

The scripts are already optimized:
- ✅ Uses `sam2_hiera_tiny` (1-2 GB VRAM)
- ✅ No Stable Diffusion loaded (~3.5 GB saved)
- ✅ Total memory: ~2 GB / 6 GB
- ✅ Safe for your hardware

## 📝 For Your Thesis

### Chapter 3: Methodology

Use these figures to illustrate:

**Figure 3.X:** "SAM 2 generates comprehensive mask proposals at multiple scales"
→ Use `fig1_sam2_masks.png`

**Figure 3.Y:** "Dense CLIP features enable pixel-wise text-image alignment"
→ Use `fig2_clip_similarity.png`

**Figure 3.Z:** "Top-K masks ranked by semantic similarity to text prompt"
→ Use `fig3_alignment.png`

### Chapter 4: Results

**Qualitative Results:**
- Show segmentation working on diverse objects
- Use `fig4_comparison.png` for multiple prompts
- Demonstrate zero-shot capability

**Quantitative Results:**
- Timing information (printed by scripts)
- Memory usage (shown in output)
- Number of masks generated

### Chapter 5: Discussion

**Figures showing:**
- Successful segmentations
- Failure cases (no matches)
- Comparison with baselines

## 🎓 Example Workflow

### For a Single Result Figure:

```bash
# 1. Prepare
cd /home/pablo/tfm/code
source venv/bin/activate

# 2. Quick test
python quick_viz.py

# 3. Check output
ls -lh figures/

# 4. Copy to thesis
cp figures/segmentation_person.png ~/tfm/overleaf/figures/
```

### For Complete Thesis Figures:

```bash
# 1. Configure
nano create_visualizations.py
# Set: IMAGE_PATH and MAIN_PROMPT

# 2. Generate all
python create_visualizations.py

# 3. Review
ls -lh thesis_figures/

# 4. Copy to thesis
cp thesis_figures/* ~/tfm/overleaf/figures/
```

## 📊 Figure Specifications

### Quick Viz
- **Format:** PNG
- **DPI:** 200 (good for screen, preview)
- **Size:** ~900 KB
- **Dimensions:** 15" × 10"
- **Use for:** Testing, iteration, quick results

### Thesis Figures
- **Format:** PNG
- **DPI:** 300 (publication quality)
- **Size:** ~2-3 MB each
- **Dimensions:** Varies by figure
- **Use for:** Final thesis submission

## 🔧 Customization

### Change Color Scheme

Edit visualization scripts:
```python
# For masks
colored_mask[mask > 0] = [255, 0, 0]  # Red
# Change to: [0, 0, 255] for Blue

# For colormaps
cmap='hot'  # Options: 'viridis', 'plasma', 'jet', 'hot'
```

### Adjust Figure Size

```python
fig = plt.figure(figsize=(15, 10))  # Width x Height in inches
```

### Add More Panels

Modify GridSpec:
```python
gs = GridSpec(3, 3, ...)  # 3 rows × 3 columns = 9 panels
```

## 📈 Next Steps

1. **Generate figures for your dataset**
   - Run on COCO validation images
   - Create figures for different object categories
   - Show diverse examples

2. **Create comparison figures**
   - Your method vs baselines
   - Different model sizes (tiny vs large)
   - Different prompts

3. **Make ablation study figures**
   - With/without background suppression
   - Different similarity thresholds
   - Different number of masks

## ✅ Checklist for Thesis

- [ ] Generate Figure 1 (SAM 2 masks) for representative image
- [ ] Generate Figure 2 (CLIP similarity) for main examples
- [ ] Generate Figure 3 (Alignment) showing top results
- [ ] Generate Figure 4 (Comparison) with multiple prompts
- [ ] Create figures for at least 3-5 diverse images
- [ ] Include both success and failure cases
- [ ] Add figure captions explaining methodology
- [ ] Reference figures in thesis text (§3.2.X, §4.X)

## 🎉 You're Ready!

Everything is set up to generate publication-quality figures for your thesis:

✅ **Visualization scripts** - Ready to run
✅ **Example figure** - Already generated (`figures/segmentation_person.png`)
✅ **GPU optimized** - Works on your GTX 1060
✅ **Publication quality** - 300 DPI output

**Just run the scripts and include the figures in your thesis!** 📊✨
