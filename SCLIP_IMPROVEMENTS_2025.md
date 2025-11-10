# SCLIP Improvements: 2025 Research Summary

## Overview

This document summarizes cutting-edge techniques from 2025 research to improve SCLIP (CLIP-based dense prediction) output quality, addressing the root cause of poor segmentation performance.

**Current Status:**
- Baseline (point-only SAM): **70% mIoU** ✓ WORKS
- All SAM2 prompting improvements: **FAILED** (45.2% mIoU)
- **Root Cause:** SCLIP produces imprecise dense predictions → SAM2 cannot fix this

**Solution:** Improve SCLIP output quality using 2025 state-of-the-art techniques

---

## 1. Multi-Scale Ensemble

### What It Is

**Multi-scale ensemble** predicts segmentation at multiple image scales and combines the results to improve robustness and capture objects at different scales.

### How It Works

```
Original Image (1024x1024)
    ↓
┌───────────────────────────────────────┐
│  Scale 1: 0.5x  (512x512)            │
│  Scale 2: 0.75x (768x768)            │
│  Scale 3: 1.0x  (1024x1024)          │
│  Scale 4: 1.25x (1280x1280)          │
│  Scale 5: 1.5x  (1536x1536)          │
└───────────────────────────────────────┘
    ↓
Run SCLIP on each scale
    ↓
Resize all predictions to original size
    ↓
Combine via averaging or voting
    ↓
Final dense segmentation map
```

### Benefits

- **Small objects:** Captured better at larger scales (1.25x, 1.5x)
- **Large objects:** Captured better at smaller scales (0.5x, 0.75x)
- **Boundary refinement:** Different scales provide complementary boundary information
- **Robustness:** Reduces sensitivity to scale-specific artifacts

### Expected Improvement

**+3-8% mIoU** (based on ESC-Net CVPR 2025)

### Implementation Strategy

**Option A: Simple averaging**
```python
def multi_scale_ensemble(image, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
    H, W = image.shape[:2]
    predictions = []

    for scale in scales:
        # Resize image
        scaled_img = cv2.resize(image, None, fx=scale, fy=scale)

        # Run SCLIP
        seg_map, probs = sclip_predict(scaled_img)

        # Resize back to original
        seg_map_resized = cv2.resize(seg_map, (W, H), interpolation=cv2.INTER_NEAREST)
        probs_resized = cv2.resize(probs, (W, H), interpolation=cv2.INTER_LINEAR)

        predictions.append((seg_map_resized, probs_resized))

    # Average probabilities
    avg_probs = np.mean([p[1] for p in predictions], axis=0)
    final_seg_map = np.argmax(avg_probs, axis=-1)

    return final_seg_map, avg_probs
```

**Option B: Weighted averaging** (better performance)
```python
# Weight larger scales more for small objects
# Weight smaller scales more for large objects
weights = {0.5: 0.15, 0.75: 0.2, 1.0: 0.3, 1.25: 0.2, 1.5: 0.15}

weighted_probs = sum(w * pred[1] for w, pred in zip(weights.values(), predictions))
final_seg_map = np.argmax(weighted_probs, axis=-1)
```

---

## 2. Dense Feature Extraction from CLIP (2025 Papers)

### Paper 1: ExCEL (CVPR 2025) - Patch-Text Alignment

**arXiv:** [2503.20826](https://arxiv.org/abs/2503.20826)

#### Key Innovation

Shifts from **image-text alignment** to **patch-text alignment**, exploring CLIP's fine-grained dense prediction capabilities.

#### Technical Components

**A. Text Semantic Enrichment (TSE)**

Uses Large Language Models (LLMs) to generate detailed class descriptions:

```python
# Example: Instead of "person"
"person" → [
    "A human being with a head, torso, arms, and legs",
    "A person standing, sitting, or walking",
    "An individual wearing clothing",
    "A figure with human-like proportions and posture"
]
```

**Process:**
1. Feed class names to LLM (e.g., GPT-4)
2. Generate 3-5 detailed descriptions per class
3. Encode all descriptions with CLIP text encoder
4. Build dataset-wide knowledge base
5. Enrich text embeddings with implicit attribute information

**Implementation:**
```python
def text_semantic_enrichment(class_name, llm_model="gpt-4"):
    prompt = f"""Generate 5 detailed descriptions for the object class "{class_name}"
    that capture visual attributes, typical poses, common contexts, and distinctive features.
    Each description should be a single sentence."""

    descriptions = llm_model.generate(prompt)

    # Encode with CLIP
    text_features = []
    for desc in descriptions:
        features = clip_text_encoder(desc)
        text_features.append(features)

    # Average to create enriched embedding
    enriched_features = torch.mean(torch.stack(text_features), dim=0)
    return enriched_features
```

**B. Visual Calibration (VC)**

Two sub-components for refining frozen CLIP features:

**1. Static Visual Calibration (SVC):**
- Uses non-parametric methods (e.g., k-NN, spatial propagation)
- Propagates fine-grained knowledge from high-confidence regions
- No training required

```python
def static_visual_calibration(patch_features, confidence_map, k=5):
    # Find high-confidence patches
    high_conf_indices = confidence_map > 0.8

    # For each low-confidence patch, find k nearest high-confidence neighbors
    for i in low_conf_indices:
        neighbors = find_k_nearest(patch_features[i], patch_features[high_conf_indices], k=k)
        # Propagate features from neighbors
        patch_features[i] = weighted_average(neighbors, distance_based_weights)

    return patch_features
```

**2. Learnable Visual Calibration (LVC):**
- Lightweight learnable module (small parameter count)
- Dynamically adjusts frozen CLIP features
- Learns to shift features toward semantically diverse distributions

```python
class LearnableVisualCalibration(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.calibration_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, frozen_features):
        adjustment = self.calibration_net(frozen_features)
        calibrated_features = frozen_features + adjustment
        return calibrated_features
```

#### Performance

- **PASCAL VOC:** State-of-the-art results
- **MS COCO:** State-of-the-art results
- **Training costs:** Reduced compared to full fine-tuning
- **Advantage:** Retains CLIP's training-free benefits

---

### Paper 2: ResCLIP (CVPR 2025) - Residual Attention

**arXiv:** [2411.15851](https://arxiv.org/html/2411.15851)
**Code:** [github.com/yvhangyang/ResCLIP](https://github.com/yvhangyang/ResCLIP)

#### Key Innovation

Training-free approach that modifies CLIP's attention mechanism to capture **class-specific features** and **local consistency** for dense prediction.

#### Technical Components

**A. Residual Cross-correlation Self-attention (RCS)**

Aggregates attention from intermediate CLIP layers to recover spatial localization:

**Motivation:**
- Final CLIP layer: spatially-invariant (good for global features, bad for dense prediction)
- Intermediate layers: class-specific localization (but weaker global understanding)
- **Solution:** Blend both via residual connection

**Formula:**
```
𝒜c = 1/N ∑(i=s to e) 𝒜qk^i    // Average attention from layers s to e
𝒜rcs = (1−λrcs)·𝒜s + λrcs·𝒜c  // Residual blend with self-attention
```

**Implementation:**
```python
def residual_cross_correlation_self_attention(clip_model, layer_start=6, layer_end=11, lambda_rcs=0.5):
    # Extract attention maps from intermediate layers
    intermediate_attentions = []
    for layer_idx in range(layer_start, layer_end + 1):
        qk_attention = clip_model.visual.transformer.resblocks[layer_idx].attn.attention_map
        intermediate_attentions.append(qk_attention)

    # Average intermediate attentions
    avg_intermediate = torch.mean(torch.stack(intermediate_attentions), dim=0)

    # Get self-attention from final layer
    final_self_attention = clip_model.visual.transformer.resblocks[-1].attn.self_attention_map

    # Residual blend
    rcs_attention = (1 - lambda_rcs) * final_self_attention + lambda_rcs * avg_intermediate

    return rcs_attention
```

**B. Semantic Feedback Refinement (SFR)**

Uses segmentation masks to refine attention by emphasizing semantically consistent regions:

**Process:**
1. Identify patches belonging to same semantic class
2. Apply connectivity analysis (spatial locality)
3. Distance-based decay using Chebyshev distance
4. Combine refined scores with original attention

**Formula:**
```
D(p,q) = exp(−d(p,q) / max(d))  // Chebyshev distance decay
𝒜sfr = (1−λsfr)·𝒜rcs + λsfr·𝒜semantic  // Blend with semantic feedback
```

**Implementation:**
```python
def semantic_feedback_refinement(attention_map, segmentation_mask, lambda_sfr=0.3):
    H, W = segmentation_mask.shape
    num_patches = attention_map.shape[0]
    patch_size = int(np.sqrt(H * W / num_patches))

    # Compute semantic consistency for each patch pair
    semantic_scores = torch.zeros_like(attention_map)

    for i in range(num_patches):
        for j in range(num_patches):
            # Get patch locations
            pi, pj = patch_coords(i, patch_size), patch_coords(j, patch_size)

            # Check if same semantic class
            same_class = (segmentation_mask[pi] == segmentation_mask[pj])

            if same_class:
                # Chebyshev distance
                dist = max(abs(pi[0] - pj[0]), abs(pi[1] - pj[1]))
                max_dist = max(H, W)
                decay = np.exp(-dist / max_dist)
                semantic_scores[i, j] = decay

    # Blend with original attention
    refined_attention = (1 - lambda_sfr) * attention_map + lambda_sfr * semantic_scores

    return refined_attention
```

#### Performance

- **VOC20:** +2.2% to +13.1% mIoU (depending on baseline)
- **ADE20K:** Consistent improvements
- **ViT-L/14:** Mitigates 13.5% performance drop when scaling up
- **State-of-the-art:** 40.3% mIoU (ViT-B/16), 39.1% mIoU (ViT-L/14)

#### Key Advantage

**Training-free, plug-and-play:** No retraining required, works with existing CLIP checkpoints

---

## 3. CLIP Prompt Engineering

### Current Approach (Baseline)

```python
# Simple class names
prompts = ["person", "car", "dog", "cat", ...]
```

### Improved Approach 1: Template-based Prompting

**OpenAI's 80 Templates** (from official CLIP notebook):

Top 7 templates selected via forward selection:
1. `"itap of a {class}."`
2. `"a bad photo of the {class}."`
3. `"a origami {class}."`
4. `"a photo of the large {class}."`
5. `"a {class} in a video game."`
6. `"art of the {class}."`
7. `"a photo of the small {class}."`

**Expected Improvement:** +1.5% on ImageNet (ViT-B/32)

**Implementation:**
```python
def ensemble_text_prompts(class_name, templates=None):
    if templates is None:
        templates = [
            "a photo of a {}.",
            "a photo of the {}.",
            "itap of a {}.",
            "a bad photo of the {}.",
            "a origami {}.",
            "a photo of the large {}.",
            "a photo of the small {}."
        ]

    # Generate all prompts
    prompts = [template.format(class_name) for template in templates]

    # Encode with CLIP
    text_features = []
    for prompt in prompts:
        features = clip_text_encoder(prompt)
        text_features.append(features)

    # Average embeddings (before normalization)
    ensemble_features = torch.mean(torch.stack(text_features), dim=0)

    # L2 normalize
    ensemble_features = F.normalize(ensemble_features, dim=-1)

    return ensemble_features
```

### Improved Approach 2: Context-Aware Prompting

**Roboflow Study:** Literal, descriptive prompts outperform domain-specific language

**Examples:**

| Class | Bad Prompt | Good Prompt |
|-------|-----------|-------------|
| person | "person" | "a person standing" |
| person | "human" | "a person walking or sitting" |
| car | "car" | "a car on the road" |
| dog | "dog" | "a dog with fur and four legs" |
| chair | "chair" | "a chair with legs and a backrest" |

**Expected Improvement:** 50% → 83% accuracy (Roboflow study, small dataset)

**Implementation:**
```python
# Context-enhanced class descriptions
CONTEXT_PROMPTS = {
    "person": [
        "a person standing",
        "a person walking",
        "a person sitting",
        "a human figure with a head and body"
    ],
    "car": [
        "a car on the road",
        "a vehicle with four wheels",
        "an automobile parked or driving"
    ],
    "dog": [
        "a dog with fur",
        "a canine with four legs and a tail",
        "a pet dog standing or sitting"
    ],
    # ... add for all classes
}

def context_aware_prompts(class_name):
    prompts = CONTEXT_PROMPTS.get(class_name, [f"a {class_name}"])

    # Encode all context variants
    text_features = [clip_text_encoder(p) for p in prompts]

    # Average
    ensemble_features = torch.mean(torch.stack(text_features), dim=0)
    ensemble_features = F.normalize(ensemble_features, dim=-1)

    return ensemble_features
```

### Improved Approach 3: Class Name Disambiguation

**Problem:** Ambiguous class names reduce accuracy

**OpenAI's examples:**
- `"nail"` → `"metal nail"` (not fingernail)
- `"kite"` → `"kite (bird of prey)"` (not toy kite)
- `"mole"` → `"mole (animal)"` (not skin mole)

**Implementation:**
```python
CLASS_DISAMBIGUATIONS = {
    "person": "person (human being)",
    "bear": "bear (animal)",
    "bat": "bat (baseball equipment)",  # or "bat (flying mammal)"
    "nail": "metal nail",
    "crane": "crane (construction vehicle)",  # or "crane (bird)"
}

def disambiguate_class_name(class_name):
    return CLASS_DISAMBIGUATIONS.get(class_name, class_name)
```

---

## 4. Implementation Roadmap

### Phase 3A: Training-Free CLIP Improvements (RECOMMENDED - START HERE)

**Why start here:**
- No training required
- Plug-and-play
- Expected +5-15% mIoU improvement

**Steps:**

1. **Implement ResCLIP (Residual Attention)** ⭐ HIGHEST PRIORITY
   - Training-free, plug-and-play
   - Code available: https://github.com/yvhangyang/ResCLIP
   - Expected: +2-13% mIoU
   - Effort: Medium (requires modifying CLIP attention mechanism)

2. **Implement Multi-Scale Ensemble** ⭐ HIGH PRIORITY
   - Simple to implement (no model changes)
   - Expected: +3-8% mIoU
   - Effort: Low (just run SCLIP at multiple scales)

3. **Implement Prompt Engineering** ⭐ HIGH PRIORITY
   - Very simple (just change text prompts)
   - Expected: +1.5-5% mIoU
   - Effort: Very Low (just modify prompt strings)

**Expected Combined Improvement:** +7-26% mIoU

---

### Phase 3B: ExCEL (Requires LLM + Light Training)

**Why later:**
- Requires LLM API calls (GPT-4) for text enrichment
- Requires training Learnable Visual Calibration module
- More complex implementation

**Steps:**

1. **Implement Text Semantic Enrichment (TSE)**
   - Use GPT-4 API to generate detailed class descriptions
   - Build dataset-wide knowledge base
   - Effort: Medium (requires LLM integration)

2. **Implement Static Visual Calibration (SVC)**
   - Non-parametric feature propagation
   - Effort: Medium

3. **Implement Learnable Visual Calibration (LVC)**
   - Train lightweight calibration network
   - Effort: High (requires training pipeline)

**Expected Combined Improvement:** +10-20% mIoU (SOTA on VOC/COCO)

---

## 5. Recommended Implementation Order

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3A: Training-Free Improvements (START HERE)          │
├─────────────────────────────────────────────────────────────┤
│ 1. Prompt Engineering (30 mins)          → +1.5-5% mIoU    │
│ 2. Multi-Scale Ensemble (2-3 hours)      → +3-8% mIoU      │
│ 3. ResCLIP Integration (1-2 days)        → +2-13% mIoU     │
├─────────────────────────────────────────────────────────────┤
│ Expected Total: +7-26% mIoU                                 │
│ Effort: 2-3 days                                            │
│ Training Required: NO ✓                                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ PHASE 3B: ExCEL (Advanced, Later)                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Text Semantic Enrichment (2-3 days)   → +5-10% mIoU     │
│ 2. Visual Calibration (3-5 days)         → +5-10% mIoU     │
├─────────────────────────────────────────────────────────────┤
│ Expected Total: +10-20% mIoU                                │
│ Effort: 1-2 weeks                                           │
│ Training Required: YES (lightweight)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Quick Start: Prompt Engineering (30 Minutes)

**Easiest way to get immediate improvement:**

### Step 1: Update `code/clip_guided_segmentation.py`

Add ensemble prompt function:
```python
def get_ensemble_text_features(class_names, clip_model):
    """Generate ensemble text features using multiple prompt templates."""

    templates = [
        "a photo of a {}.",
        "a photo of the {}.",
        "itap of a {}.",
        "a bad photo of the {}.",
        "a origami {}.",
        "a photo of the large {}.",
        "a photo of the small {}."
    ]

    text_features_all = []

    for class_name in class_names:
        # Generate all prompt variants
        prompts = [template.format(class_name) for template in templates]

        # Tokenize
        text_tokens = clip.tokenize(prompts).to(clip_model.device)

        # Encode
        with torch.no_grad():
            class_text_features = clip_model.encode_text(text_tokens)

        # Average embeddings (before final normalization)
        ensemble_feature = class_text_features.mean(dim=0, keepdim=True)

        # L2 normalize
        ensemble_feature = F.normalize(ensemble_feature, dim=-1)

        text_features_all.append(ensemble_feature)

    return torch.cat(text_features_all, dim=0)
```

### Step 2: Update text encoding in SCLIP predictor

Find where text features are encoded (search for `encode_text`), replace with:
```python
# OLD:
# text_features = clip_model.encode_text(text_tokens)

# NEW:
text_features = get_ensemble_text_features(class_names, clip_model)
```

### Step 3: Test

```bash
python run_benchmarks.py \
    --dataset coco-stuff \
    --num-samples 10 \
    --use-clip-guided-sam \
    --use-all-phase1 \
    --use-all-phase2a
```

**Expected:** +1.5-5% mIoU improvement with just better prompts!

---

## 7. Summary

### Current Situation
- SAM2 improvements all failed (box, negative points, multi-point)
- Root cause: SCLIP output is imprecise
- Need to improve SCLIP dense predictions

### Best Path Forward

**Phase 3A (Training-Free):**
1. ✅ Prompt Engineering (30 mins) → +1.5-5% mIoU
2. ✅ Multi-Scale Ensemble (2-3 hours) → +3-8% mIoU
3. ✅ ResCLIP (1-2 days) → +2-13% mIoU

**Total Expected: +7-26% mIoU in 2-3 days, no training**

### References

**2025 Papers:**
- **ExCEL:** [arXiv:2503.20826](https://arxiv.org/abs/2503.20826) - CVPR 2025
- **ResCLIP:** [arXiv:2411.15851](https://arxiv.org/html/2411.15851) - CVPR 2025 - [Code](https://github.com/yvhangyang/ResCLIP)
- **ESC-Net:** CVPR 2025 (SAM+CLIP combination)

**Prompt Engineering:**
- OpenAI CLIP Notebook: [Prompt Engineering for ImageNet](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
- Roboflow Guide: [CLIP Prompt Engineering](https://blog.roboflow.com/openai-clip-prompt-engineering/)
