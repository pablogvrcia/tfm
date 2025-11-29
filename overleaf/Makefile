# Makefile for Master Thesis Compilation
# Author: Pablo García García

# Variables
MAIN = main
OUTDIR = build
OVERLEAF_DIR = .

# LaTeX compiler options
LATEXMK_FLAGS = -pdf -output-directory=$(OUTDIR) -interaction=nonstopmode -file-line-error

.PHONY: all clean build docker-build docker-compile docker-shell help

# Default target
all: build

# Help target
help:
	@echo "Master Thesis Compilation Targets:"
	@echo "  make build          - Compile thesis locally (requires LaTeX)"
	@echo "  make docker-build   - Build Docker image with LaTeX"
	@echo "  make docker-compile - Compile thesis using Docker"
	@echo "  make docker-shell   - Open interactive shell in Docker container"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make view           - Open compiled PDF (Linux)"

# Create build directory
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Build thesis locally
build: $(OUTDIR)
	cd $(OVERLEAF_DIR) && latexmk $(LATEXMK_FLAGS) $(MAIN).tex
	@echo "✓ PDF generated at $(OVERLEAF_DIR)/$(OUTDIR)/$(MAIN).pdf"
	@cp $(OVERLEAF_DIR)/$(OUTDIR)/$(MAIN).pdf ./Master_Thesis.pdf
	@echo "✓ PDF copied to ./Master_Thesis.pdf"

# Clean build artifacts
clean:
	rm -rf $(OVERLEAF_DIR)/$(OUTDIR)
	rm -f Master_Thesis.pdf
	@echo "✓ Build artifacts cleaned"

# Build Docker image
docker-build:
	docker build -t thesis-latex .
	@echo "✓ Docker image 'thesis-latex' built successfully"

# Compile thesis using Docker
docker-compile: docker-build
	docker run --rm -v $(PWD)/.:/thesis thesis-latex \
		bash -c "latexmk -pdf -output-directory=build -interaction=nonstopmode -f -file-line-error main.tex"
	@if [ -f $(OVERLEAF_DIR)/build/$(MAIN).pdf ]; then \
		cp $(OVERLEAF_DIR)/build/$(MAIN).pdf ./Master_Thesis.pdf; \
		echo "✓ PDF generated successfully: ./Master_Thesis.pdf"; \
	else \
		echo "✗ Error: PDF generation failed"; \
		exit 1; \
	fi

# Open interactive shell in Docker container
docker-shell:
	docker run --rm -it -v $(PWD)/.:/thesis thesis-latex bash

# View PDF (Linux)
view:
	@if [ -f ./Master_Thesis.pdf ]; then \
		xdg-open ./Master_Thesis.pdf 2>/dev/null || echo "Please open Master_Thesis.pdf manually"; \
	else \
		echo "Error: Master_Thesis.pdf not found. Run 'make build' first."; \
	fi

# Quick compile (fewer passes for faster iteration)
quick: $(OUTDIR)
	cd $(OVERLEAF_DIR) && pdflatex -output-directory=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex
	@echo "✓ Quick compile complete (bibliography may be outdated)"
	@cp $(OVERLEAF_DIR)/$(OUTDIR)/$(MAIN).pdf ./Master_Thesis.pdf
