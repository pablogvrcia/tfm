# Video Segmentation Benchmarks

This document describes the video segmentation benchmark implementation for evaluating CLIP-guided SAM2 on standard video object segmentation datasets.

## Overview

The video benchmark system evaluates open-vocabulary semantic segmentation on video datasets using:
- **CLIP** for initial object detection and classification
- **SAM2** for video object tracking and segmentation
- **Standard VOS metrics** (J, F, J&F, T) for evaluation

## Supported Datasets

### 1. DAVIS (Densely Annotated VIdeo Segmentation)

**DAVIS 2016**: Single object segmentation (50 sequences)
- Download: https://davischallenge.org/davis2016/code.html
- Resolution: 480p
- Objects: 1 per video

**DAVIS 2017**: Multi-object segmentation (90 train + 30 val sequences)
- Download: https://davischallenge.org/davis2017/code.html
- Resolution: 480p or 1080p
- Objects: Multiple per video

Directory structure:
```
data/video_benchmarks/
└── DAVIS-2017/
    └── DAVIS/
        ├── Annotations/
        │   └── 480p/
        │       ├── bear/
        │       ├── bike-packing/
        │       └── ...
        ├── JPEGImages/
        │   └── 480p/
        │       ├── bear/
        │       ├── bike-packing/
        │       └── ...
        └── ImageSets/
            └── 2017/
                ├── train.txt
                └── val.txt
```

### 2. YouTube-VOS

**YouTube-VOS 2019**: Large-scale multi-object segmentation
- Download: https://youtube-vos.org/dataset/vos/
- Videos: 4,453 videos with 94 object categories
- Split: train / valid / test

Directory structure:
```
data/video_benchmarks/
└── youtube-vos-2019/
    ├── train/
    │   ├── Annotations/
    │   └── JPEGImages/
    ├── valid/
    │   ├── Annotations/
    │   ├── JPEGImages/
    │   └── meta.json
    └── test/
```

## Installation

### 1. Install Required Dependencies

```bash
pip install opencv-python scipy
```

All other dependencies are already included in the main project.

### 2. Download Datasets

Use the provided download script:

```bash
# Download DAVIS 2017 (recommended for quick start)
cd benchmarks
./download_video_datasets.sh davis-2017

# Download DAVIS 2016
./download_video_datasets.sh davis-2016

# Download YouTube-VOS (requires manual registration)
./download_video_datasets.sh youtube-vos

# Download all datasets
./download_video_datasets.sh all

# Verify existing downloads
./download_video_datasets.sh verify
```

**Note**: YouTube-VOS requires manual registration at https://youtube-vos.org/. Download the files and place them in `data/video_benchmarks/youtube-vos-2019/`, then run the script to extract.

## Usage

### Basic Usage

Evaluate on DAVIS 2017 validation set:

```bash
python run_video_benchmarks.py --dataset davis-2017 --num-samples 5
```

Evaluate on YouTube-VOS:

```bash
python run_video_benchmarks.py --dataset youtube-vos --num-samples 10
```

### Advanced Options

**Custom confidence threshold:**
```bash
python run_video_benchmarks.py --dataset davis-2017 \
    --min-confidence 0.5 \
    --min-region-size 50
```

**Save visualizations:**
```bash
python run_video_benchmarks.py --dataset davis-2017 \
    --num-samples 3 \
    --save-vis
```

**Limit frames per video (for faster testing):**
```bash
python run_video_benchmarks.py --dataset davis-2017 \
    --max-frames 30 \
    --num-samples 5
```

**Use custom vocabulary:**
```bash
python run_video_benchmarks.py --dataset davis-2017 \
    --vocabulary person car dog cat bird
```

**Enable performance profiling:**
```bash
python run_video_benchmarks.py --dataset davis-2017 \
    --enable-profiling
```

## Evaluation Metrics

The system computes standard Video Object Segmentation (VOS) metrics:

### J - Region Similarity (IoU)
- Measures the intersection-over-union between predicted and ground truth masks
- Range: [0, 1], higher is better
- Standard metric for segmentation quality

### F - Boundary Accuracy
- Measures the quality of object boundaries
- Computes F-measure on boundary pixels
- Range: [0, 1], higher is better
- More sensitive to boundary localization than J

### J&F - Overall Score
- Mean of J and F metrics
- Primary ranking metric for VOS benchmarks
- Balances region and boundary quality

### T - Temporal Stability
- Measures consistency across consecutive frames
- IoU between predictions in adjacent frames
- Range: [0, 1], higher is better
- Important for video quality (reduces flickering)

## Output

Results are saved to `benchmarks/video_results/`:

```
benchmarks/video_results/
├── davis-2017_results.json          # Detailed metrics
└── visualizations/                   # Sample frame visualizations (if --save-vis)
    └── davis-2017/
        └── bear/
            ├── frame_0000.png
            ├── frame_0015.png
            └── ...
```

Results JSON format:
```json
{
  "aggregated_metrics": {
    "J_mean": 0.65,
    "J_std": 0.12,
    "F_mean": 0.68,
    "F_std": 0.10,
    "J&F_mean": 0.665,
    "J&F_std": 0.11,
    "T_mean": 0.85,
    "T_std": 0.08,
    "num_videos": 30
  },
  "per_video_metrics": [
    {
      "J": 0.72,
      "F": 0.75,
      "J&F": 0.735,
      "T": 0.90,
      "num_objects": 2,
      "num_frames": 75
    },
    ...
  ]
}
```

## Implementation Details

### Workflow

1. **First Frame Analysis**:
   - Extract first frame from video
   - Run CLIP dense prediction to identify objects
   - Extract high-confidence regions as prompts

2. **Video Tracking**:
   - Use SAM2 video predictor with extracted prompts
   - Track objects across all frames
   - Generate per-frame segmentation masks

3. **Evaluation**:
   - Compute J, F, T metrics per video
   - Aggregate results across dataset
   - Save detailed metrics and visualizations

### Key Components

- `video_datasets.py`: Dataset loaders for DAVIS and YouTube-VOS
- `benchmarks/video_metrics.py`: VOS evaluation metrics (J, F, T)
- `run_video_benchmarks.py`: Main benchmark runner
- `models/video_segmentation.py`: SAM2 video tracking (already existed)

## Performance Notes

### Memory Considerations

- SAM2 uses CPU offloading to save GPU memory
- Each video is processed independently
- Frames are loaded on-demand

### Speed Optimizations

- **First frame only**: Prompts extracted once from first frame
- **Frame limiting**: Use `--max-frames` to limit processing
- **Sample limiting**: Use `--num-samples` to test on subset

### Expected Performance

Based on the CLIP-guided approach from `clip_guided_segmentation.py`:

**DAVIS 2017** (typical video: 75 frames, 480p):
- Processing time: ~15-30 seconds per video
- Memory: ~4-6 GB GPU
- Expected J&F: ~60-70% (depends on vocabulary quality)

**YouTube-VOS** (typical video: 100-200 frames):
- Processing time: ~30-60 seconds per video
- Memory: ~4-8 GB GPU

## Comparison with Image Benchmarks

| Aspect | Image Benchmarks | Video Benchmarks |
|--------|-----------------|------------------|
| Input | Single images | Frame sequences |
| Datasets | PASCAL-VOC, COCO-Stuff | DAVIS, YouTube-VOS |
| Metrics | mIoU, F1, Boundary F1 | J, F, J&F, T |
| Classes | Fixed (21 or 171) | Open-vocabulary |
| Processing | Per-image | Per-video tracking |
| Time | ~2s per image | ~30s per video |

## Troubleshooting

### No prompts extracted
- Lower `--min-confidence` (try 0.3-0.5)
- Reduce `--min-region-size` (try 50-100)
- Adjust vocabulary to match video content

### Out of memory
- Reduce `--max-frames` to limit frames
- Use smaller SAM2 model checkpoint
- Process fewer videos at once

### Poor performance
- Check if vocabulary matches video objects
- Increase `--min-confidence` for higher quality prompts
- Try different CLIP model (ViT-L/14 for better quality)

## Future Improvements

- [ ] Support for MOSE dataset (2024-2025)
- [ ] Support for SA-V dataset
- [ ] Multi-frame prompt extraction (not just first frame)
- [ ] Adaptive vocabulary based on video content
- [ ] Online learning for better tracking
- [ ] Support for referring video object segmentation

## References

1. **DAVIS**: Perazzi et al. "A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation", CVPR 2016
2. **YouTube-VOS**: Xu et al. "YouTube-VOS: A Large-Scale Video Object Segmentation Benchmark", arXiv 2018
3. **SAM2**: Ravi et al. "Segment Anything in Images and Videos", arXiv 2024
4. **SCLIP**: Wang et al. "SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference", arXiv 2023

## Citation

If you use this video benchmark implementation, please cite the original thesis:

```bibtex
@mastersthesis{garcia2025open,
  title={Open-Vocabulary Semantic Segmentation for Generative AI},
  author={García García, Pablo},
  year={2025},
  school={Universidad de Zaragoza}
}
```
