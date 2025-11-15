#!/bin/bash
# Test Cityscapes dataset with best configuration
# Cityscapes has 19 classes (urban scenes)

echo "======================================================================"
echo "CITYSCAPES BENCHMARK - 50 samples"
echo "======================================================================"
echo ""

BASE_ARGS="--dataset cityscapes --num-samples 50 --save-vis --use-fp16 --batch-prompts"

# Best configuration adapted for Cityscapes (fewer classes, urban scenes)
BEST_CONFIG="--use-clip-guided-sam \
  --min-confidence 0.15 \
  --min-region-size 5 \
  --iou-threshold 0.1 \
  --logit-scale 40.0 \
  --template-strategy adaptive \
  --use-all-phase1 \
  --use-all-phase2a"

echo "----------------------------------------------------------------------"
echo "Running: Best configuration on Cityscapes"
echo "  - 19 classes (urban scenes)"
echo "  - Adaptive templates"
echo "  - All phase improvements"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS $BEST_CONFIG \
    --output-dir benchmarks/results/cityscapes-best

echo ""

echo "======================================================================"
echo "CITYSCAPES BENCHMARK COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved in: benchmarks/results/cityscapes-best/"
echo ""
echo "To compare with other datasets:"
echo "  python3 compare_results.py --detailed"
echo ""
