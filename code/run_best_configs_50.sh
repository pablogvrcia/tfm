#!/bin/bash
# Run comprehensive hyperparameter sweep for COCO-Stuff 50 samples
# Based on analysis of 10-sample results

echo "======================================================================"
echo "COCO-STUFF: Comprehensive Hyperparameter Sweep (50 samples)"
echo "======================================================================"
echo ""

BASE_ARGS="--dataset coco-stuff --num-samples 50 --save-vis --use-fp16 --batch-prompts"

# Config 1: Current Best (from 10-sample analysis)
echo "----------------------------------------------------------------------"
echo "[1/12] Running: Current Best Configuration"
echo "  - CLIP-guided SAM with optimal hyperparameters"
echo "  - min_confidence=0.15, min_region_size=5, iou_threshold=0.1"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/best

echo ""

# Config 2: Top7 Templates (faster)
echo "----------------------------------------------------------------------"
echo "[2/12] Running: Top7 Templates"
echo "  - 3-4x faster, expected +2-3% mIoU"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy top7 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/top7

echo ""

# Config 3: Top3 Templates (ultra-fast)
echo "----------------------------------------------------------------------"
echo "[3/12] Running: Top3 Templates"
echo "  - 5x faster than baseline"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy top3 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/top3

echo ""

# Config 4: Adaptive Templates
echo "----------------------------------------------------------------------"
echo "[4/12] Running: Adaptive Templates"
echo "  - Per-class templates, expected +3-5% mIoU"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy adaptive \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/adaptive

echo ""

# Config 5: All Phases + CLIP-RC
echo "----------------------------------------------------------------------"
echo "[5/12] Running: All Phases + CLIP-RC"
echo "  - Best for human parsing"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --use-clip-rc \
    --output-dir benchmarks/results/coco-50/all-phases-clip-rc

echo ""

# Config 6: Lower min_confidence (0.1)
echo "----------------------------------------------------------------------"
echo "[6/12] Running: Lower min_confidence=0.1"
echo "  - More regions, potentially better coverage"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.1 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/min-conf-0.1

echo ""

# Config 7: Higher min_confidence (0.25)
echo "----------------------------------------------------------------------"
echo "[7/12] Running: Higher min_confidence=0.25"
echo "  - Fewer but higher quality regions"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.25 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/min-conf-0.25

echo ""

# Config 8: Smaller region size (3)
echo "----------------------------------------------------------------------"
echo "[8/12] Running: min_region_size=3"
echo "  - Detect even smaller objects"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 3 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/region-size-3

echo ""

# Config 9: Larger region size (10)
echo "----------------------------------------------------------------------"
echo "[9/12] Running: min_region_size=10"
echo "  - Focus on larger objects"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 10 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/region-size-10

echo ""

# Config 10: Higher logit scale (60.0)
echo "----------------------------------------------------------------------"
echo "[10/12] Running: logit_scale=60.0"
echo "  - Sharper predictions"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 60.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/logit-scale-60

echo ""

# Config 11: Higher IoU threshold (0.5)
echo "----------------------------------------------------------------------"
echo "[11/12] Running: iou_threshold=0.5"
echo "  - Less aggressive overlap merging"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.5 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --use-all-phase1 \
    --use-all-phase2a \
    --output-dir benchmarks/results/coco-50/iou-thresh-0.5

echo ""

# Config 12: No phases (baseline)
echo "----------------------------------------------------------------------"
echo "[12/12] Running: No Phases (Baseline)"
echo "  - Pure CLIP-guided SAM without improvements"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS \
    --use-clip-guided-sam \
    --min-confidence 0.15 \
    --min-region-size 5 \
    --iou-threshold 0.1 \
    --logit-scale 40.0 \
    --template-strategy imagenet80 \
    --slide-inference --slide-crop 224 --slide-stride 112 \
    --output-dir benchmarks/results/coco-50/no-phases

echo ""

echo "======================================================================"
echo "ALL 12 CONFIGURATIONS COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved in benchmarks/results/coco-50/"
echo ""
echo "To analyze and compare results, run:"
echo "  python3 compare_results.py --dataset coco-stuff --detailed --recommendations"
echo ""
echo "Configuration Summary:"
echo "  1. best                  - Baseline best from 10-sample analysis"
echo "  2. top7                  - Faster templates (3-4x speedup)"
echo "  3. top3                  - Ultra-fast templates (5x speedup)"
echo "  4. adaptive              - Adaptive per-class templates"
echo "  5. all-phases-clip-rc    - All improvements enabled"
echo "  6. min-conf-0.1          - Lower confidence threshold"
echo "  7. min-conf-0.25         - Higher confidence threshold"
echo "  8. region-size-3         - Smaller region detection"
echo "  9. region-size-10        - Larger region focus"
echo "  10. logit-scale-60       - Sharper predictions"
echo "  11. iou-thresh-0.5       - Less overlap merging"
echo "  12. no-phases            - Pure baseline (no improvements)"
echo ""
