#!/bin/bash
# Benchmark script to test each computational optimization in isolation

DATASET="pascal-voc"
NUM_SAMPLES=5
BASE_ARGS="--dataset $DATASET --num-samples $NUM_SAMPLES --use-clip-guided-sam --enable-profiling --descriptor-file configs/cls_voc21.txt"

echo "=================================="
echo "Computational Optimization Benchmarks"
echo "Dataset: $DATASET"
echo "Samples: $NUM_SAMPLES"
echo "=================================="
echo ""

# 1. Baseline (no optimizations)
echo "1. Running baseline (no optimizations)..."
python run_benchmarks.py $BASE_ARGS \
  --no-use-fp16 \
  --no-batch-prompts \
  --no-use-compile
echo ""

# 2. FP16 only
echo "2. Running with FP16 only..."
python run_benchmarks.py $BASE_ARGS \
  --use-fp16 \
  --no-batch-prompts \
  --no-use-compile
echo ""

# 3. Batch prompts only
echo "3. Running with batch prompts only..."
python run_benchmarks.py $BASE_ARGS \
  --no-use-fp16 \
  --batch-prompts \
  --no-use-compile
echo ""

# 4. torch.compile only
echo "4. Running with torch.compile only..."
python run_benchmarks.py $BASE_ARGS \
  --no-use-fp16 \
  --no-batch-prompts \
  --use-compile
echo ""

# 5. All optimizations
echo "5. Running with all optimizations..."
python run_benchmarks.py $BASE_ARGS \
  --use-fp16 \
  --batch-prompts \
  --use-compile
echo ""

echo "=================================="
echo "Benchmark completed!"
echo "Check benchmarks/results/ for detailed results"
echo "=================================="
