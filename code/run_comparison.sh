#!/bin/bash
# Script to generate comparison figures for thesis

echo "Generating comparison figures..."
echo ""

# Example 1: Car scene (2-panel)
echo "[1/3] Car scene (2-panel)..."
python compare_prompting_approaches.py \
    --image examples/car.jpg \
    --vocabulary car sky road building background \
    --output ../overleaf/Imagenes/comparison_blind_vs_intelligent_2panel.png \
    --grid-size 64 \
    --min-confidence 0.7

# Example 2: Car scene (3-panel with SCLIP visualization)
echo "[2/3] Car scene (3-panel with SCLIP)..."
python compare_prompting_approaches.py \
    --image examples/car.jpg \
    --vocabulary car sky road building background \
    --output ../overleaf/Imagenes/comparison_blind_vs_intelligent_3panel.png \
    --grid-size 64 \
    --min-confidence 0.7 \
    --show-sclip

# Example 3: Basketball scene
echo "[3/3] Basketball scene (3-panel)..."
python compare_prompting_approaches.py \
    --image examples/basketball.jpg \
    --vocabulary person basketball court background \
    --output ../overleaf/Imagenes/comparison_basketball.png \
    --grid-size 64 \
    --min-confidence 0.6 \
    --show-sclip

echo ""
echo "✓ Comparison figures generated!"
echo "  - comparison_blind_vs_intelligent_2panel.png (2-panel for thesis)"
echo "  - comparison_blind_vs_intelligent_3panel.png (3-panel with SCLIP)"
echo "  - comparison_basketball.png (alternative example)"
echo ""
echo "Files saved to: ../overleaf/Imagenes/"
