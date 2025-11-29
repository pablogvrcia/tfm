#!/bin/bash
# Ablation Studies for Master's Thesis
# Runs systematic experiments on PASCAL VOC to populate thesis tables

set -e  # Exit on error

# Configuration
DATASET="pascal-voc"
DATA_DIR="../data/VOCdevkit/VOC2012"
NUM_SAMPLES=1449  # Full PASCAL VOC validation set (use 100-200 for quick testing)
OUTPUT_BASE="ablation_results"
DEVICE="cuda"

# Create output directory
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${OUTPUT_BASE}/logs"

# Function to run experiment and save results
run_experiment() {
    local name="$1"
    local args="$2"
    local log_file="${OUTPUT_BASE}/logs/${name}.log"
    local result_file="${OUTPUT_BASE}/${name}_results.json"

    echo ""
    echo "========================================="
    echo "Running: ${name}"
    echo "Args: ${args}"
    echo "========================================="

    python run_benchmarks.py \
        --dataset ${DATASET} \
        --data-dir ${DATA_DIR} \
        --num-samples ${NUM_SAMPLES} \
        --output-dir ${OUTPUT_BASE}/${name} \
        --enable-profiling \
        ${args} \
        2>&1 | tee ${log_file}

    echo "✓ Completed: ${name}"
    echo ""
}

# ============================================================================
# ABLATION STUDY 1: DenseCRF Boundary Refinement
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 1: DenseCRF Boundary Refinement            ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

run_experiment "01_baseline_sclip" \
    ""

run_experiment "01_sclip_densecrf" \
    "--use-densecrf"

# ============================================================================
# ABLATION STUDY 2: Descriptor Files
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 2: Text Descriptor Files                   ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

run_experiment "02_baseline_sclip" \
    ""

# Bad descriptors example (create a file with poor/conflicting descriptors)
run_experiment "02_sclip_bad_descriptors" \
    "--descriptor-file configs/cls_voc21_bad.txt"

# Good descriptors (the ones you've been using)
run_experiment "02_sclip_good_descriptors" \
    "--descriptor-file configs/cls_voc21.txt"

# ============================================================================
# ABLATION STUDY 3: Prompt Template Strategies
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 3: Prompt Template Strategies              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Baseline: single class name (no templates)
run_experiment "03_baseline_no_templates" \
    ""

run_experiment "03_imagenet80" \
    "--template-strategy imagenet80"

run_experiment "03_top7" \
    "--template-strategy top7"

run_experiment "03_top3" \
    "--template-strategy top3"

run_experiment "03_adaptive" \
    "--template-strategy adaptive"

# ============================================================================
# ABLATION STUDY 4: Intelligent Prompting (Incremental)
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 4: Intelligent Prompting (Incremental)     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Step 1: Baseline SCLIP (dense prediction only)
run_experiment "04_01_baseline_sclip" \
    ""

# Step 2: SCLIP + SAM2 (intelligent prompting, no descriptors/templates)
run_experiment "04_02_sclip_sam2" \
    "--clip-guided-sam"

# Step 3: + Best template strategy
run_experiment "04_03_sclip_sam2_templates" \
    "--clip-guided-sam --template-strategy imagenet80"

# Step 4: + Best descriptors
run_experiment "04_04_sclip_sam2_templates_descriptors" \
    "--clip-guided-sam --template-strategy imagenet80 --descriptor-file configs/cls_voc21.txt"

# Step 5: + DenseCRF (full system)
run_experiment "04_05_full_system" \
    "--clip-guided-sam --template-strategy imagenet80 --descriptor-file configs/cls_voc21.txt --use-densecrf"

# ============================================================================
# ABLATION STUDY 5: Computational Optimizations
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 5: Computational Optimizations             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Baseline: FP32, no compile, sequential prompts
run_experiment "05_01_baseline_fp32" \
    "--clip-guided-sam --no-use-fp16 --no-use-compile --no-batch-prompts"

# + FP16
run_experiment "05_02_fp16" \
    "--clip-guided-sam --use-fp16 --no-use-compile --no-batch-prompts"

# + FP16 + torch.compile
run_experiment "05_03_fp16_compile" \
    "--clip-guided-sam --use-fp16 --use-compile --no-batch-prompts"

# + FP16 + torch.compile + batch prompting (full optimization)
run_experiment "05_04_all_optimizations" \
    "--clip-guided-sam --use-fp16 --use-compile --batch-prompts"

# ============================================================================
# ABLATION STUDY 6: Hyperparameter Sensitivity
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 6: Hyperparameter Sensitivity              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Confidence threshold variations
run_experiment "06_conf_0.5" \
    "--clip-guided-sam --min-confidence 0.5"

run_experiment "06_conf_0.7_default" \
    "--clip-guided-sam --min-confidence 0.7"

run_experiment "06_conf_0.9" \
    "--clip-guided-sam --min-confidence 0.9"

# Area threshold variations
run_experiment "06_area_50" \
    "--clip-guided-sam --min-area 50"

run_experiment "06_area_100_default" \
    "--clip-guided-sam --min-area 100"

run_experiment "06_area_200" \
    "--clip-guided-sam --min-area 200"

# ============================================================================
# ABLATION STUDY 7: Grid Sampling vs Intelligent Prompting
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     ABLATION STUDY 7: Grid Sampling Comparison                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Note: These require grid sampling implementation
# If not available, skip or implement simple grid sampling

# run_experiment "07_grid_16x16" \
#     "--use-sam --grid-sampling --grid-size 16"

# run_experiment "07_grid_32x32" \
#     "--use-sam --grid-sampling --grid-size 32"

# run_experiment "07_grid_64x64" \
#     "--use-sam --grid-sampling --grid-size 64"

run_experiment "07_intelligent_prompting" \
    "--clip-guided-sam --template-strategy imagenet80 --descriptor-file configs/cls_voc21.txt"

# ============================================================================
# Generate Summary Report
# ============================================================================
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     Generating Summary Report                                  ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

python generate_ablation_summary.py --results-dir ${OUTPUT_BASE}

echo ""
echo "========================================="
echo "✓ All ablation studies completed!"
echo "Results saved to: ${OUTPUT_BASE}/"
echo "========================================="
