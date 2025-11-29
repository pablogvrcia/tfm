#!/bin/bash
# Benchmark Comparison Script
# Compares Dense SCLIP, Blind Grid, and CLIP-Guided SAM
#
# Usage:
#   bash benchmark_comparison.sh [dataset] [num_samples]
#
# Examples:
#   bash benchmark_comparison.sh coco-stuff 10
#   bash benchmark_comparison.sh pascal-voc 50

DATASET=${1:-coco-stuff}
NUM_SAMPLES=${2:-10}
OUTPUT_BASE="benchmarks/results/comparison_${DATASET}"

echo "========================================"
echo "Benchmark Comparison"
echo "========================================"
echo "Dataset: $DATASET"
echo "Samples: $NUM_SAMPLES"
echo "Output: $OUTPUT_BASE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "========================================"
echo "1/4: Dense SCLIP Baseline (No SAM)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --output-dir "${OUTPUT_BASE}/dense_sclip" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "2/4: Blind Grid 32×32 (1024 prompts)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-blind-grid \
  --grid-size 32 \
  --output-dir "${OUTPUT_BASE}/blind_grid_32x32" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "3/4: Blind Grid 64×64 (4096 prompts)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-blind-grid \
  --grid-size 64 \
  --output-dir "${OUTPUT_BASE}/blind_grid_64x64" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "4/4: CLIP-Guided SAM (Intelligent)"
echo "========================================"
python run_benchmarks.py \
  --dataset "$DATASET" \
  --num-samples "$NUM_SAMPLES" \
  --use-clip-guided-sam \
  --min-confidence 0.2 \
  --min-region-size 50 \
  --output-dir "${OUTPUT_BASE}/clip_guided" \
  --save-vis \
  --enable-profiling

echo ""
echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "To analyze results, run:"
echo "  python analyze_comparison.py $OUTPUT_BASE"
