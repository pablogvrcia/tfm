# Quick Citation Guide for Your Thesis

## TL;DR

**Your work is COMPLEMENTARY to MasQCLIP, not competitive.**
- MasQCLIP: Training-based, benchmark-focused
- Yours: Zero-shot, application-focused

**Your main novelty:** Adaptive mask selection algorithm

## Must-Cite Papers

### Primary (Related to your work)

1. **MasQCLIP** (ICCV 2023) - Xu et al.
   - When: Discussing training-based open-vocabulary segmentation
   - Where: Chapter 2 (Related Work), Chapter 5 (Comparison)

2. **MaskCLIP** (ICML 2023) - Ding et al.
   - When: Background on CLIP for segmentation
   - Where: Chapter 2.3

3. **CLIPSeg** (CVPR 2022) - Lüddecke & Ecker
   - When: Multi-scale CLIP features
   - Where: Chapter 3.2.1

4. **SAM 2** (2024) - Ravi et al.
   - When: Mask generation methodology
   - Where: Chapter 3.2.2

5. **CLIP** (ICML 2021) - Radford et al.
   - When: Vision-language foundation
   - Where: Chapter 2.1

## Your Novel Contributions (What to Emphasize)

### 1. Adaptive Mask Selection ⭐⭐⭐
```
First work to automatically determine top-K based on semantic granularity
- "car" → 1 mask
- "tires" → 4 masks  
- "mountains" → N masks
```

### 2. Complete Zero-Shot Pipeline ⭐⭐
```
First to combine SAM 2 + CLIP + Stable Diffusion
No training required (vs. MasQCLIP's 140K iterations)
```

### 3. Confuser Scoring ⭐
```
Explicit negative scoring to distinguish similar objects
"tire" vs "grille", "license plate"
```

## Positioning Statements (Copy-Paste Ready)

### Abstract
```
We present a zero-shot pipeline for language-driven image segmentation 
and editing, featuring a novel adaptive mask selection algorithm. Unlike 
training-based methods [Xu et al., 2023], our approach requires no training 
while supporting arbitrary text prompts.
```

### Related Work
```
Recent works like MasQCLIP [Xu et al., 2023] achieve strong benchmark results 
through supervised training, modifying CLIP's attention mechanism with learnable 
mask tokens. However, these approaches require extensive compute (140K iterations) 
and are limited to predefined categories.

Our work takes a complementary approach, prioritizing practical applicability 
through zero-shot inference while introducing adaptive mask selection to address 
the fixed-K limitation of prior methods.
```

### Comparison Section
```
While MasQCLIP [Xu et al., 2023] may achieve higher mIoU on standard benchmarks 
through supervised training, our zero-shot approach offers:
1. No training infrastructure required
2. Arbitrary text prompt flexibility  
3. Integrated generative editing
4. Novel adaptive mask selection

We position our work as complementary: MasQCLIP optimizes for benchmark accuracy, 
while we optimize for practical deployment and flexibility.
```

## What NOT to Say

❌ "We outperform MasQCLIP"
✅ "We offer complementary advantages for practical deployment"

❌ "First open-vocabulary segmentation work"
✅ "First zero-shot pipeline combining SAM 2, CLIP, and Stable Diffusion"

❌ "Novel CLIP integration"
✅ "Novel adaptive mask selection strategy"

## Quick Comparison Table (For Chapter 5)

```markdown
| Aspect | MasQCLIP | Our Work |
|--------|----------|----------|
| Training | 140K iterations | Zero-shot |
| Text prompts | Predefined | Arbitrary |
| Mask selection | Fixed top-K | **Adaptive** ✓ |
| Editing | No | **Yes (SD)** ✓ |
| Deployment | Complex | **Simple** ✓ |
```

## BibTeX Entries

```bibtex
@inproceedings{xu2023masqclip,
    author    = {Xu, Xin and Xiong, Tianyi and Ding, Zheng and Tu, Zhuowen},
    title     = {MasQCLIP for Open-Vocabulary Universal Image Segmentation},
    booktitle = {ICCV},
    year      = {2023},
}

@inproceedings{ding2023maskclip,
    author    = {Ding, Zheng and Wang, Jieke and Tu, Zhuowen},
    title     = {Open-Vocabulary Universal Image Segmentation with MaskCLIP},
    booktitle = {ICML},
    year      = {2023},
}
```

## Key Message

**Frame your work as solving a different problem:**
- MasQCLIP: "How to train a model for universal segmentation?"
- Yours: "How to make open-vocabulary segmentation practical without training?"

Both are valuable, neither is "better" - just different objectives.
