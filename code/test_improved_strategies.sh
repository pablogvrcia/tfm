#!/bin/bash
# Test different improved prompt extraction strategies on COCO-Stuff
#
# Usage:
#   bash test_improved_strategies.sh [num_samples]
#
# Example:
#   bash test_improved_strategies.sh 20  # Test on 20 samples

DATASET=${1:-coco-stuff}
NUM_SAMPLES=${2:-10}
OUTPUT_BASE="benchmarks/results/improved_strategies_${DATASET}"

echo "========================================"
echo "Testing Improved Prompt Strategies"
echo "========================================"
echo "Dataset: $DATASET"
echo "Samples: $NUM_SAMPLES"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "========================================"
echo "BASELINE: Current method (connected components)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-clip-guided-sam \
  --min-confidence 0.7 \
  --min-region-size 100 \
  --output-dir "${OUTPUT_BASE}/baseline_current" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "STRATEGY 1: Adaptive Thresholds (stuff vs thing)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-clip-guided-sam \
  --improved-strategy adaptive_threshold \
  --min-confidence 0.3 \
  --min-region-size 100 \
  --output-dir "${OUTPUT_BASE}/strategy1_adaptive" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "STRATEGY 2: Confidence-Weighted Sampling"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-clip-guided-sam \
  --improved-strategy confidence_weighted \
  --min-confidence 0.2 \
  --min-region-size 100 \
  --output-dir "${OUTPUT_BASE}/strategy2_conf_weighted" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "STRATEGY 3: Density-Based K-means"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-clip-guided-sam \
  --improved-strategy density_based \
  --min-confidence 0.2 \
  --min-region-size 50 \
  --output-dir "${OUTPUT_BASE}/strategy3_density" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "STRATEGY 4: Full Prob Map Exploitation (BEST)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-clip-guided-sam \
  --improved-strategy prob_map \
  --min-confidence 0.2 \
  --min-region-size 100 \
  --output-dir "${OUTPUT_BASE}/strategy4_prob_map" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "All strategies tested!"
echo "========================================"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To analyze results, run:"
echo "  python analyze_strategies.py $OUTPUT_BASE"
