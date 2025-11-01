# SCLIP System Improvement Proposals

## Current Performance
- **COCO-Stuff:** 49.52% mIoU
- **Pascal VOC:** 48.09% mIoU
- **Speed:** 33-54s per image
- **Memory:** 6GB GPU (OOM at 64 SAM points)

---

## Priority 1: High-Impact Improvements

### 1.1 Confidence-Based Filtering ⭐⭐⭐
**Problem:** Many false positive predictions (e.g., airplane image showing car, bicycle, etc.)

**Solution:**
```python
# Add confidence threshold in SAM refinement
def predict_with_sam_filtered(self, image, class_names, confidence_threshold=0.25):
    dense_pred, logits = self.predict_dense(image, class_names, return_logits=True)

    # Filter low-confidence predictions
    probs = F.softmax(logits, dim=0)
    max_probs = probs.max(dim=0)[0]
    confident_mask = max_probs > confidence_threshold

    # Only refine confident regions with SAM
    sam_masks = self.sam_generator.generate_masks(image)
    # ... rest of SAM refinement
```

**Expected improvement:** +3-5% mIoU, fewer false positives

---

### 1.2 Better Text Prompts ⭐⭐⭐
**Problem:** Generic prompts like "a photo of a {class}" may not work well for all categories

**Solution:**
```python
# Dataset-specific prompts
PASCAL_VOC_PROMPTS = {
    'aeroplane': ['an airplane', 'an aircraft', 'a plane in the sky'],
    'bicycle': ['a bicycle', 'a bike', 'a person on a bicycle'],
    'person': ['a person', 'a human', 'people'],
    # ... etc
}

# Use best prompt per class
def get_best_prompt(class_name, dataset_type):
    if dataset_type == 'pascal-voc':
        return PASCAL_VOC_PROMPTS.get(class_name, [f'a photo of a {class_name}'])
    # ... other datasets
```

**Expected improvement:** +2-3% mIoU, better object detection

---

### 1.3 Multi-Scale Inference ⭐⭐
**Problem:** Single 2048px resolution may miss details or be too coarse

**Solution:**
```python
# Test at multiple scales and merge
scales = [1536, 2048, 2560]
predictions = []

for scale in scales:
    resized = resize_image(image, max_side=scale)
    pred = self.predict_dense(resized, class_names)
    predictions.append(pred)

# Ensemble predictions (voting or averaging)
final_pred = ensemble_predictions(predictions, method='voting')
```

**Expected improvement:** +1-2% mIoU, better boundary quality

---

## Priority 2: Speed Optimizations

### 2.1 Cache Text Features ⭐⭐⭐
**Problem:** Text encoding computed for every image (wasteful)

**Solution:**
```python
# In SCLIPSegmentor.__init__
self.text_feature_cache = {}

def get_text_features(self, class_names):
    cache_key = tuple(class_names)
    if cache_key not in self.text_feature_cache:
        self.text_feature_cache[cache_key] = self.clip_extractor.extract_text_features(
            class_names, use_prompt_ensemble=True, normalize=True
        )
    return self.text_feature_cache[cache_key]
```

**Expected speedup:** 15-20% faster

---

### 2.2 Batch SAM Processing ⭐⭐
**Problem:** SAM masks processed one at a time in refinement loop

**Solution:**
```python
# Process multiple masks in parallel
batch_size = 16
for i in range(0, len(sam_masks), batch_size):
    batch = sam_masks[i:i+batch_size]
    # Process batch together
    batch_predictions = self._process_mask_batch(batch, dense_pred)
```

**Expected speedup:** 10-15% faster

---

### 2.3 Smaller SAM Model (Optional) ⭐
**Problem:** SAM is slow for high-resolution images

**Solution:**
```python
# Use sam2_hiera_tiny (current) vs sam2_hiera_small
# Or reduce points_per_side for faster inference
segmentor = SCLIPSegmentor(
    sam_points_per_side=32,  # Down from 48 for speed
    # Trade-off: -1% mIoU, +30% speed
)
```

**Trade-off:** Faster but slightly lower quality

---

## Priority 3: Memory Optimizations

### 3.1 Adaptive Grid Points ⭐⭐
**Problem:** Fixed 48 points_per_side causes OOM on large images

**Solution:**
```python
def get_adaptive_points(image_size):
    # Fewer points for larger images
    max_side = max(image_size)
    if max_side > 2048:
        return 32
    elif max_side > 1536:
        return 40
    else:
        return 48
```

**Expected improvement:** No OOM, maintains quality

---

### 3.2 Gradient Checkpointing ⭐
**Problem:** High memory usage in CLIP encoder

**Solution:**
```python
# Enable in CLIP model
if hasattr(self.model.visual, 'set_grad_checkpointing'):
    self.model.visual.set_grad_checkpointing(True)
```

**Expected improvement:** 30-40% less GPU memory

---

## Priority 4: Quality Improvements

### 4.1 CRF Post-Processing ⭐⭐
**Problem:** Noisy boundaries, fragmented predictions

**Solution:**
```python
import pydensecrf.densecrf as dcrf

def apply_crf(image, predictions, num_classes):
    d = dcrf.DenseCRF2D(w, h, num_classes)
    # Add unary potentials from predictions
    d.setUnaryEnergy(unary)
    # Add pairwise potentials from image
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))
```

**Expected improvement:** +2-4% mIoU, cleaner boundaries

---

### 4.2 Negative Class Suppression ⭐⭐
**Problem:** Common false positives (e.g., always predicting "person" or "car")

**Solution:**
```python
# Add negative prompts
negative_prompts = ['a blurry image', 'background', 'empty space']
negative_features = extract_text_features(negative_prompts)

# Suppress predictions similar to negative prompts
similarities = features @ negative_features.T
is_negative = similarities.max(dim=-1)[0] > threshold
predictions[is_negative] = background_class
```

**Expected improvement:** +1-2% mIoU, fewer false positives

---

### 4.3 Class-Specific Thresholds ⭐
**Problem:** All classes use same confidence threshold

**Solution:**
```python
# Per-class calibration
class_thresholds = {
    'person': 0.3,      # Common class, higher threshold
    'bicycle': 0.2,     # Rare class, lower threshold
    'aeroplane': 0.25,  # Medium threshold
    # ... etc
}

# Apply during prediction
for cls_idx, cls_name in enumerate(class_names):
    threshold = class_thresholds.get(cls_name, 0.25)
    mask = (probs[cls_idx] > threshold)
```

**Expected improvement:** +1-2% mIoU, better precision/recall balance

---

## Priority 5: Evaluation & Analysis

### 5.1 Detailed Error Analysis ⭐⭐
**Solution:**
```python
# Confusion matrix
# Per-class precision/recall curves
# Failure case visualization
# Error type classification (false positive, false negative, confusion)
```

### 5.2 Comparison Baseline ⭐
**Solution:**
```python
# Compare against:
# - Original SCLIP (dense only)
# - CLIP + SAM (no SCLIP CSA)
# - Other open-vocab methods
```

---

## Recommended Implementation Order

**Week 1 (Quick Wins):**
1. Cache text features (2.1) - Easy, 20% speedup
2. Confidence filtering (1.1) - Medium, +3-5% mIoU
3. Better text prompts (1.2) - Medium, +2-3% mIoU

**Week 2 (Quality):**
4. Multi-scale inference (1.3) - Medium, +1-2% mIoU
5. CRF post-processing (4.1) - Hard, +2-4% mIoU
6. Adaptive grid points (3.1) - Easy, prevents OOM

**Week 3 (Polish):**
7. Batch SAM processing (2.2) - Medium, +10-15% speed
8. Detailed error analysis (5.1) - Easy, better understanding
9. Class-specific thresholds (4.3) - Easy, +1-2% mIoU

**Expected Final Performance:**
- **COCO-Stuff:** ~55-58% mIoU (+6-9% absolute)
- **Pascal VOC:** ~53-56% mIoU (+5-8% absolute)
- **Speed:** ~25-35s per image (30% faster)
- **Quality:** Cleaner boundaries, fewer false positives

---

## Implementation Complexity

| Improvement | Difficulty | Time | Impact |
|-------------|-----------|------|--------|
| Text feature cache | Easy | 30min | High (speed) |
| Confidence filter | Easy | 1hr | High (quality) |
| Better prompts | Easy | 2hr | High (quality) |
| Multi-scale | Medium | 3hr | Medium (quality) |
| CRF | Hard | 4hr | High (quality) |
| Batch SAM | Medium | 3hr | Medium (speed) |
| Adaptive grid | Easy | 1hr | Medium (memory) |

**Total estimated time for top 5:** ~11 hours
**Expected improvement:** +8-12% mIoU, 30-40% faster
