# Master Thesis: Open-Vocabulary Semantic Segmentation for Generative AI

**Author:** Pablo García García
**Supervisors:** Alejandro Pérez Yus, María Santos Villafranca
**Institution:** Universidad de Zaragoza, Escuela de Ingeniería y Arquitectura
**Year:** 2025

## Overview

This thesis addresses open-vocabulary semantic segmentation integrated with generative AI for image editing. The system combines SAM 2, CLIP/MaskCLIP/CLIPSeg, and Stable Diffusion to enable flexible object discovery and manipulation based on natural language prompts.

## Repository Structure

```
master-thesis/
├── overleaf/                    # LaTeX source files
│   ├── main.tex                 # Main thesis document
│   ├── Capitulos/              # Chapter files
│   │   ├── Introduccion.tex
│   │   ├── Capitulo1.tex       # Background and Related Work
│   │   ├── Capitulo2.tex       # Methodology
│   │   ├── Capitulo3.tex       # Experiments and Evaluation
│   │   └── Conclusion.tex
│   ├── Bibliografia_TFM.bib    # Bibliography
│   ├── Imagenes/               # Figures and images
│   └── Anexos/                 # Appendices
├── Dockerfile                   # Docker image for LaTeX compilation
├── Makefile                     # Build automation
├── compile.sh                   # Quick compilation script
└── README.md                    # This file
```

## Compilation Methods

### Method 1: Docker (Recommended)

**Advantages:**
- No need to install LaTeX locally (>5GB)
- Consistent compilation environment
- Works on any OS with Docker

**Prerequisites:**
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))

**Quick Start:**
```bash
# Simple one-command compilation
./compile.sh docker

# Or using Make
make docker-compile
```

**The first build will take 5-10 minutes** as it downloads and installs TeX Live. Subsequent compilations are fast (~30 seconds).

### Method 2: Local LaTeX Installation

**Advantages:**
- Faster compilation after initial setup
- No Docker overhead

**Prerequisites:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install texlive-full latexmk

# macOS (with Homebrew)
brew install --cask mactex
```

**Compilation:**
```bash
# Using the script
./compile.sh local

# Or using Make
make build
```

### Method 3: Overleaf

Upload the `overleaf/` directory contents to [Overleaf](https://www.overleaf.com/) and compile online.

## Available Make Targets

```bash
make help            # Show all available targets
make build           # Compile locally
make docker-build    # Build Docker image
make docker-compile  # Compile using Docker
make docker-shell    # Interactive Docker shell
make clean           # Remove build artifacts
make view            # Open PDF (Linux)
make quick           # Fast compile (single pass)
```

## Docker Commands

### Build the Docker image:
```bash
docker build -t thesis-latex .
```

### Compile the thesis:
```bash
docker run --rm -v $(pwd)/overleaf:/thesis thesis-latex \
    latexmk -pdf -output-directory=build -interaction=nonstopmode main.tex
```

### Interactive shell (for debugging):
```bash
make docker-shell
# or
docker run --rm -it -v $(pwd)/overleaf:/thesis thesis-latex bash
```

## Output

After successful compilation, the PDF will be available at:
- `./Master_Thesis.pdf` (root directory)
- `./overleaf/build/main.pdf` (build directory)

## Troubleshooting

### Docker compilation fails
```bash
# Clean and rebuild
make clean
docker rmi thesis-latex
make docker-compile
```

### Local compilation errors
```bash
# Clean build artifacts
make clean

# Ensure all LaTeX packages are installed
sudo apt-get install texlive-full texlive-lang-spanish
```

### Bibliography not updating
```bash
# Run full latexmk cycle
cd overleaf
latexmk -pdf -output-directory=build main.tex
```

### Permission errors (Linux)
```bash
# Fix ownership of build directory
sudo chown -R $USER:$USER overleaf/build/
```

## Development Workflow

### Quick iteration cycle:
```bash
# 1. Edit LaTeX files in overleaf/Capitulos/
# 2. Compile
./compile.sh docker

# 3. View PDF
xdg-open Master_Thesis.pdf  # Linux
open Master_Thesis.pdf      # macOS
```

### Working with Docker:
```bash
# Open interactive shell
make docker-shell

# Inside container, you can run commands directly:
latexmk -pdf main.tex
pdflatex main.tex
biber main
```

## Key Technologies

- **LaTeX Distribution:** TeX Live 2022
- **Compiler:** latexmk (automating pdflatex + biber)
- **Container:** Docker (Ubuntu 22.04 base)
- **Bibliography:** BibTeX/Biber

## Performance Notes

| Method | First Compilation | Subsequent |
|--------|------------------|------------|
| Docker | ~10 min (initial) + ~30s | ~30s |
| Local  | ~30s | ~15s |
| Overleaf | ~45s | ~30s |

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{garcia2025openvocabulary,
  author  = {Pablo García García},
  title   = {Open-Vocabulary Semantic Segmentation for Generative AI},
  school  = {Universidad de Zaragoza},
  year    = {2025},
  type    = {Master's Thesis}
}
```

## License

This thesis and its LaTeX source code are provided for academic purposes.

## Contact

- **Author:** Pablo García García
- **Institution:** Universidad de Zaragoza
- **Email:** [Your email here]

---

**Last Updated:** January 2025
