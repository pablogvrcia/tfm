# Fix: Why "tire" query returns wrong masks

## Problem Analysis

Looking at your output, querying "tire" returns:
- ✓ #1: Actual tire (0.184 score)
- ✗ #2: Front grille (0.176 score)
- ✗ #3: License plate (0.165 score)
- ✗ #4: Headlight (0.168 score)
- ✗ #5: Road/lane marking (0.151 score)

**Only 1/5 masks is correct!** The similarity map shows CLIP activating on the wrong regions.

## Root Causes

### 1. CLIP's Semantic Confusion
CLIP ViT-L/14 has trouble distinguishing "tire" from other car parts because:
- Training data bias: "tire" appears in images with entire cars
- Circular shape confusion: Grille, headlights have circular elements
- Co-occurrence: License plates often appear near tires in training images

### 2. Low Spatial Resolution
- CLIP processes at 336×336 with 14×14 patches
- Each patch = 24×24 pixels
- Small objects like tires become 2-3 patches → very coarse representation

### 3. Mask Cropping Issues
Current method (line 230 `mask_alignment.py`):
```python
def _extract_masked_region(self, image, mask):
    # Crops to bounding box
    # Sets background to mean color
    # Passes cropped region to CLIP
```

**Problem:** Small masks (tire) get dominated by background context when cropped.

## Solutions

### Solution 1: Improve Prompt Engineering (Quick Fix)

**Change the text prompt to be more specific:**

```python
# Bad prompt (ambiguous)
"tire"

# Better prompts (more specific)
"car tire"
"wheel tire"
"rubber tire"
"black tire"
```

**Test this:**
```bash
python main.py --image car.jpg --prompt "car tire" --mode segment
python main.py --image car.jpg --prompt "black rubber tire" --mode segment
```

### Solution 2: Add Negative Prompts (Medium Fix)

Modify `mask_alignment.py` to use contrastive scoring:

```python
def align_masks_with_text(self, masks, text_prompt, image, ...):
    # Current: Only positive similarity
    sim_score = similarity(mask, "tire")

    # Better: Contrastive with negatives
    positive_prompts = ["tire", "car wheel", "rubber tire"]
    negative_prompts = ["grille", "license plate", "headlight", "road", "lane marking"]

    pos_score = max([similarity(mask, p) for p in positive_prompts])
    neg_score = max([similarity(mask, p) for p in negative_prompts])

    final_score = pos_score - 0.5 * neg_score
```

### Solution 3: Use Object-Centric CLIP Embeddings (Best Fix)

The current method extracts the masked region with background. Better approach:

```python
def _extract_masked_region_improved(self, image, mask):
    """Extract only the object pixels, no background."""
    # Get bounding box
    y_indices, x_indices = np.where(mask > 0)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Crop to bbox
    cropped_img = image[y_min:y_max+1, x_min:x_max+1].copy()
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

    # NEW: Set background to black (not mean color)
    # This makes CLIP focus on the object foreground
    cropped_img[cropped_mask == 0] = [0, 0, 0]

    # NEW: Ensure minimum size (CLIP needs ~224px)
    h, w = cropped_img.shape[:2]
    if h < 224 or w < 224:
        scale = max(224 / h, 224 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        cropped_img = cv2.resize(cropped_img, (new_w, new_h))

    return cropped_img
```

### Solution 4: Use Dense CLIP Features Directly (Alternative)

Instead of cropping masks, use the original dense similarity map:

```python
def _compute_mask_score_dense(self, mask, similarity_map):
    """
    Use dense CLIP features without cropping.
    More reliable for small objects.
    """
    # Get top 50% of pixels within mask (focus on most relevant)
    mask_scores = similarity_map[mask > 0]

    # Use 90th percentile instead of mean (reduces noise)
    if len(mask_scores) > 10:
        score = np.percentile(mask_scores, 90)
    else:
        score = mask_scores.mean() if len(mask_scores) > 0 else 0

    return score
```

### Solution 5: Add Semantic Class Filtering (Most Robust)

Use CLIP to first classify what category each mask belongs to:

```python
def _classify_mask_category(self, mask_img):
    """Classify mask into semantic categories."""
    categories = [
        "tire", "wheel",           # Target
        "grille", "front of car",  # Confusers
        "license plate", "number plate",
        "headlight", "car light",
        "road", "street", "lane"
    ]

    # Get CLIP similarity for all categories
    scores = {}
    for cat in categories:
        scores[cat] = self.clip_similarity(mask_img, cat)

    # Return top category
    return max(scores, key=scores.get)

def align_masks_with_text(self, masks, text_prompt, ...):
    # First, classify each mask
    for mask in masks:
        mask.category = self._classify_mask_category(mask_img)

    # Filter: only keep masks matching query semantic category
    if "tire" in text_prompt or "wheel" in text_prompt:
        masks = [m for m in masks if m.category in ["tire", "wheel"]]

    # Then score the filtered masks
    ...
```

## Recommended Implementation

I'll implement **Solution 3 + Solution 4** as they're complementary and don't require prompt changes:

1. **Improve mask cropping** (focus on object, ensure min size)
2. **Use percentile scoring** instead of mean (more robust to noise)
3. **Add category filtering** as optional feature

This should dramatically improve "tire" detection!

## Expected Results After Fix

```
Query: "tire"

Before:
  #1: Tire (0.184)         ✓
  #2: Grille (0.176)       ✗
  #3: License (0.165)      ✗
  #4: Headlight (0.168)    ✗
  #5: Road (0.151)         ✗

After:
  #1: Front-left tire (0.42)   ✓
  #2: Front-right tire (0.39)  ✓
  #3: Rear tire (partial) (0.28) ✓ (if visible)
  (Other car parts score < 0.15)
```

## Why This Happens in Research

This is a **well-known limitation** of CLIP-based segmentation:
- CLIP was trained on image-text pairs, not pixel-level labels
- It learns object co-occurrence (tires appear with cars → confusion)
- Small objects get poor representations at 336px resolution

**Citation opportunity for thesis:**
> "We observe that CLIP's image-level training leads to semantic confusion for small part-level queries, where visually similar components (e.g., grilles, headlights) receive high similarity scores for 'tire' queries. This motivates our contrastive scoring approach..."

## Next Steps

1. Test Solution 1 (better prompts) immediately
2. Implement Solution 3 + 4 in code
3. Compare before/after results
4. Document in thesis Chapter 4 (Experiments - Failure Cases)
