# Video Object Segmentation: Recent Benchmarks & Baselines (2022-2025)

This document provides recent state-of-the-art results on video object segmentation benchmarks for comparison with the CLIP-guided SAM2 approach implemented in this thesis.

## DAVIS 2017 Validation Set

### Semi-Supervised VOS (Standard Track)

| Method | Year | Venue | J&F (%) | J (%) | F (%) | Notes |
|--------|------|-------|---------|-------|-------|-------|
| **SAM 2** | 2024 | arXiv | **90.7** | 88.0 | 93.4 | SOTA, +2.6% improvement |
| Cutie | 2024 | CVPR | 90.1 | 87.6 | 92.6 | Strong baseline |
| DEVA | 2023 | ICCV | 86.9 | 84.2 | 89.6 | Open-vocabulary capable |
| XMem++ | 2023 | arXiv | 87.0 | 84.5 | 89.5 | Memory-based |
| DeAOT | 2022 | NeurIPS | 86.2 | 83.6 | 88.8 | Attention-based |
| XMem | 2022 | ECCV | 86.2 | 83.3 | 89.0 | Memory network |
| STCN | 2021 | NeurIPS | 85.3 | 82.0 | 88.6 | Space-time correspondence |
| AOT | 2021 | NeurIPS | 84.9 | 82.3 | 87.5 | Associating Objects with Transformers |

**Baseline (older):**
- STM (2019): 81.8% J&F
- FEELVOS (2019): 71.5% J&F

### DAVIS 2017 Test-Dev

| Method | Year | J&F (%) | J (%) | F (%) |
|--------|------|---------|-------|-------|
| SAM 2 | 2024 | **82.5** | 79.8 | 85.2 |
| Cutie | 2024 | 81.5 | 78.9 | 84.1 |
| XMem | 2022 | 81.0 | 77.7 | 84.3 |

## YouTube-VOS 2019 Validation

| Method | Year | Overall J&F (%) | J-Seen (%) | J-Unseen (%) | F-Seen (%) | F-Unseen (%) |
|--------|------|----------------|------------|--------------|------------|--------------|
| **SAM 2** | 2024 | **89.0** | 88.7 | 84.7 | 91.6 | 88.5 |
| Cutie | 2024 | 87.4 | 87.1 | 83.2 | 90.1 | 86.3 |
| DEVA | 2023 | 86.3 | 86.0 | 81.9 | 89.4 | 85.5 |
| XMem | 2022 | 85.5 | 85.2 | 80.3 | 88.9 | 84.5 |

## MOSE Dataset (Complex Scenes - 2023)

MOSE focuses on complex scenarios with occlusions, disappearance/reappearance, and crowded scenes.

### MOSE Validation Set

| Method | Year | Venue | J&F (%) | J (%) | F (%) | Notes |
|--------|------|-------|---------|-------|-------|-------|
| **SAM 2** | 2024 | arXiv | **77.9** | 74.8 | 81.0 | +6.2% improvement |
| 1st Place CVPR 2024 | 2024 | CVPR Workshop | 84.45* | - | - | Test set, challenge winner |
| Cutie | 2024 | ICCV | 81.39* | - | - | Test set, 3rd place |
| 2nd Place CVPR 2024 | 2024 | CVPR Workshop | 83.45* | - | - | Test set |

*Note: Test set results (with fine-tuning), validation results are generally lower

### MOSEv2 (2025 - More Challenging)

| Method | Year | J&F (%) | Notes |
|--------|------|---------|-------|
| SAM 2 | 2024 | **50.9** | Significant drop from MOSEv1 (76.4%) |

Shows that even SOTA struggles with very complex scenarios.

## LVOS (Long Videos - 2023)

| Method | Year | J&F (%) | Notes |
|--------|------|---------|-------|
| SAM 2 | 2024 | **75.8** | Zero-shot (no fine-tuning) |
| 4th Place LSVOS 2024 | 2024 | - | Using SAM 2 |

## Open-Vocabulary Video Segmentation

### Methods with Open-Vocabulary Capabilities

| Method | Year | Venue | Approach | DAVIS J&F (%) | Notes |
|--------|------|-------|----------|---------------|-------|
| **DEVA** | 2023 | ICCV | Decoupled: SAM + Tracking | 86.9 | Works with SAM, open-vocab |
| OV2Seg | 2023 | CVPR | CLIP + Mask2Former | - | Image-based open-vocab |
| VISA | 2023 | ICCV | Vision-Language | - | Open-vocabulary video instance seg |
| Your approach | 2025 | Thesis | **CLIP + SAM2** | **TBD** | Open-vocab with video tracking |

## Computational Performance

| Method | Speed (fps) | Memory (GB) | Notes |
|--------|-------------|-------------|-------|
| SAM 2 | ~5-10 | 8-12 | With CPU offloading |
| Cutie | ~15-20 | 6-8 | Optimized |
| XMem | ~20-30 | 4-6 | Efficient memory |

## Key Insights for Your Thesis

### Expected Performance Range

Based on your CLIP-guided SAM2 approach:

**Conservative Estimate:**
- DAVIS 2017: 65-75% J&F (open-vocabulary, no fine-tuning)
- YouTube-VOS: 60-70% J&F

**Optimistic Estimate (with good vocabulary):**
- DAVIS 2017: 75-85% J&F
- YouTube-VOS: 70-80% J&F

**Why lower than SAM 2 SOTA (90.7%)?**
1. SAM 2 uses ground truth first frame mask (semi-supervised)
2. Your approach is fully automatic with CLIP guidance (harder task)
3. Open-vocabulary setting (no class-specific training)

### Your Contribution

Your work bridges two important areas:
1. **Open-vocabulary** (like DEVA, but with SAM2)
2. **Automatic** (no manual first frame annotation)

**Comparison points:**
- vs. SAM 2: Fully automatic (no manual first frame)
- vs. DEVA: More recent SAM2 tracker + CLIP prompting strategy
- vs. Traditional VOS: Open-vocabulary capability

## Recommended Baselines for Comparison

### Must Compare Against:
1. **SAM 2** (2024) - 90.7% J&F on DAVIS 2017
   - Note: Uses ground truth first frame mask
2. **DEVA** (2023) - 86.9% J&F on DAVIS 2017
   - Most similar: open-vocabulary + SAM
3. **Cutie** (2024) - 90.1% J&F on DAVIS 2017
   - Strong recent baseline

### Good to Compare:
4. **XMem** (2022) - 86.2% J&F - Well-established baseline
5. **Zero-shot SAM 2** - Your approach should beat this

## Evaluation Recommendations

### Metrics to Report

1. **J&F metrics** (standard VOS):
   - J (region similarity)
   - F (boundary accuracy)
   - J&F (primary metric)

2. **Class-level mIoU** (semantic quality):
   - Per-class IoU
   - Mean across classes
   - Novel vs. seen class performance

3. **Temporal Stability (T)**:
   - Consistency across frames
   - Less flickering = better

4. **Efficiency**:
   - FPS (frames per second)
   - Number of prompts vs. grid search
   - Memory usage

### Datasets Priority

1. **DAVIS 2017** (required) - 30 validation videos
   - Standard benchmark
   - Comparable to all papers

2. **YouTube-VOS** (recommended) - If time permits
   - Larger scale
   - More diverse

3. **MOSE** (optional) - For complex scenes
   - Shows robustness
   - Recent benchmark

## References

### Key Papers to Cite

1. **SAM 2** (2024): "Segment Anything in Images and Videos"
   - arXiv:2408.00714
   - Current SOTA: 90.7% J&F on DAVIS 2017

2. **Cutie** (2024): "Putting the Object Back into Video Object Segmentation"
   - arXiv:2310.12982
   - Strong baseline: 90.1% J&F

3. **DEVA** (2023): "Tracking Anything with Decoupled Video Segmentation"
   - ICCV 2023
   - Open-vocabulary capable: 86.9% J&F

4. **XMem** (2022): "XMem: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model"
   - ECCV 2022
   - Established baseline: 86.2% J&F

5. **DAVIS** (2016): "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation"
   - CVPR 2016
   - Original benchmark

6. **YouTube-VOS** (2018): "YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark"
   - ECCV 2018

7. **MOSE** (2023): "MOSE: A New Dataset for Video Object Segmentation in Complex Scenes"
   - ICCV 2023

### Open-Vocabulary Papers

8. **OV2Seg** (2023): "Open-Vocabulary Semantic Segmentation with Mask-adapted CLIP"
   - CVPR 2023

9. **VISA** (2023): "Towards Open-Vocabulary Video Instance Segmentation"
   - ICCV 2023

## Summary Table for Thesis

| Aspect | Your Approach | SAM 2 (SOTA) | DEVA | Improvement Needed |
|--------|---------------|--------------|------|-------------------|
| Task | Open-vocab, automatic | Semi-supervised | Open-vocab | - |
| DAVIS J&F | **TBD** | 90.7% | 86.9% | Beat DEVA |
| First Frame | CLIP-guided (auto) | Manual GT mask | SAM (auto) | Better prompting |
| Vocabulary | Open (CLIP) | N/A | Open (SAM) | Smart class selection |
| Speed | ~15-30s/video | ~5-10 fps | Similar | Optimize prompts |

**Target:** Beat DEVA (86.9%) by leveraging SAM2's improvements + better CLIP prompting strategy.
