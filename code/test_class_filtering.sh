#!/bin/bash
# Test class filtering on COCO-Stuff 10 samples
# Compares performance with and without class filtering

echo "======================================================================"
echo "CLASS FILTERING TEST - COCO-Stuff 10 samples"
echo "======================================================================"
echo ""

BASE_ARGS="--dataset coco-stuff --num-samples 10 --save-vis"
BEST_CONFIG="--use-clip-guided-sam --min-confidence 0.15 --min-region-size 5 --iou-threshold 0.1 --logit-scale 40.0 --template-strategy adaptive --use-all-phase1 --use-all-phase2a"

# Test 1: WITHOUT class filtering (baseline)
echo "----------------------------------------------------------------------"
echo "[1/4] Baseline: No class filtering (30.65% mIoU expected)"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS $BEST_CONFIG \
    --output-dir benchmarks/results/class-filter-test/no-filtering

echo ""

# Test 2: WITH class filtering (balanced preset)
echo "----------------------------------------------------------------------"
echo "[2/4] Class filtering: Balanced preset"
echo "  Expected: +5-10% mIoU, 2-3x faster"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS $BEST_CONFIG \
    --use-class-filtering \
    --class-filter-preset balanced \
    --output-dir benchmarks/results/class-filter-test/balanced

echo ""

# Test 3: WITH class filtering (precise preset)
echo "----------------------------------------------------------------------"
echo "[3/4] Class filtering: Precise preset"
echo "  Expected: Maximum accuracy"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS $BEST_CONFIG \
    --use-class-filtering \
    --class-filter-preset precise \
    --output-dir benchmarks/results/class-filter-test/precise

echo ""

# Test 4: WITH class filtering (fast preset)
echo "----------------------------------------------------------------------"
echo "[4/4] Class filtering: Fast preset"
echo "  Expected: Maximum speed"
echo "----------------------------------------------------------------------"
python3 run_benchmarks.py $BASE_ARGS $BEST_CONFIG \
    --use-class-filtering \
    --class-filter-preset fast \
    --output-dir benchmarks/results/class-filter-test/fast

echo ""

echo "======================================================================"
echo "TEST COMPLETE!"
echo "======================================================================"
echo ""
echo "Compare results:"
echo "  python3 compare_results.py --results-dir benchmarks/results/class-filter-test --detailed"
echo ""
