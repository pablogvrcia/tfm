#!/bin/bash
# Quick compilation script for Master Thesis
# Usage: ./compile.sh [docker|local]

set -e

MODE="${1:-docker}"

echo "========================================="
echo "Master Thesis Compilation Script"
echo "========================================="
echo ""

case "$MODE" in
    docker)
        echo "→ Using Docker compilation..."
        echo ""

        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo "✗ Error: Docker is not installed"
            echo "  Please install Docker or use: ./compile.sh local"
            exit 1
        fi

        # Build Docker image
        echo "→ Building Docker image..."
        docker build -t thesis-latex . > /dev/null 2>&1
        echo "✓ Docker image ready"
        echo ""

        # Compile thesis
        echo "→ Compiling thesis..."
        docker run --rm -v "$(pwd)/overleaf:/thesis" thesis-latex \
            bash -c "latexmk -pdf -output-directory=build -interaction=nonstopmode -file-line-error main.tex"

        # Copy output
        if [ -f "overleaf/build/main.pdf" ]; then
            cp overleaf/build/main.pdf ./Master_Thesis.pdf
            echo ""
            echo "========================================="
            echo "✓ SUCCESS! PDF generated successfully"
            echo "========================================="
            echo ""
            echo "Output: ./Master_Thesis.pdf"

            # Show file size
            size=$(du -h ./Master_Thesis.pdf | cut -f1)
            echo "Size: $size"
            echo ""
        else
            echo ""
            echo "✗ Error: PDF generation failed"
            echo "Check the output above for errors"
            exit 1
        fi
        ;;

    local)
        echo "→ Using local LaTeX installation..."
        echo ""

        # Check if latexmk is installed
        if ! command -v latexmk &> /dev/null; then
            echo "✗ Error: latexmk is not installed"
            echo "  Install with: sudo apt-get install latexmk texlive-full"
            echo "  Or use: ./compile.sh docker"
            exit 1
        fi

        # Create build directory
        mkdir -p overleaf/build

        # Compile thesis
        echo "→ Compiling thesis..."
        cd overleaf
        latexmk -pdf -output-directory=build -interaction=nonstopmode -file-line-error main.tex
        cd ..

        # Copy output
        if [ -f "overleaf/build/main.pdf" ]; then
            cp overleaf/build/main.pdf ./Master_Thesis.pdf
            echo ""
            echo "========================================="
            echo "✓ SUCCESS! PDF generated successfully"
            echo "========================================="
            echo ""
            echo "Output: ./Master_Thesis.pdf"

            # Show file size
            size=$(du -h ./Master_Thesis.pdf | cut -f1)
            echo "Size: $size"
            echo ""
        else
            echo ""
            echo "✗ Error: PDF generation failed"
            echo "Check the output above for errors"
            exit 1
        fi
        ;;

    *)
        echo "✗ Error: Unknown mode '$MODE'"
        echo ""
        echo "Usage: ./compile.sh [docker|local]"
        echo ""
        echo "  docker - Compile using Docker (recommended)"
        echo "  local  - Compile using local LaTeX installation"
        echo ""
        exit 1
        ;;
esac
