#!/bin/bash
#
# Example: Generate CLIP-Guided SAM Explanatory Visualization
#
# This script demonstrates the complete pipeline from raw image to final segmentation
# with detailed visualizations of every intermediate step.
#

# Basic example with football image
echo "Running explanatory visualization for football example..."
python clip_guided_sam_explanatory.py \
    --image examples/football_frame.png \
    --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background \
    --output explanatory_results/football_basic \
    --min-confidence 0.3 \
    --min-region-size 100 \
    --points-per-cluster 1 \
    --create-html

echo ""
echo "Basic visualization complete!"
echo "Open: explanatory_results/football_basic/index.html"
echo ""

# Advanced example with negative prompts and multi-point
echo "Running advanced visualization with negative prompts..."
python clip_guided_sam_explanatory.py \
    --image examples/football_frame.png \
    --vocabulary "Lionel Messi" "Luis Suarez" "Neymar Jr" grass crowd background \
    --output explanatory_results/football_advanced \
    --min-confidence 0.3 \
    --min-region-size 100 \
    --points-per-cluster 3 \
    --negative-points-per-cluster 2 \
    --negative-confidence-threshold 0.8 \
    --create-html \
    --per-class-details

echo ""
echo "Advanced visualization complete!"
echo "Open: explanatory_results/football_advanced/index.html"
echo ""

# Example with Cityscapes image
echo "Running visualization for Cityscapes example..."
python clip_guided_sam_explanatory.py \
    --image examples/cityscapes_sample.png \
    --vocabulary road sidewalk building wall fence pole person car \
    --output explanatory_results/cityscapes \
    --min-confidence 0.3 \
    --min-region-size 200 \
    --points-per-cluster 2 \
    --negative-points-per-cluster 1 \
    --create-html

echo ""
echo "Cityscapes visualization complete!"
echo "Open: explanatory_results/cityscapes/index.html"
echo ""

echo "="
echo "All visualizations generated successfully!"
echo "Check the explanatory_results/ directory for outputs."
echo "="
