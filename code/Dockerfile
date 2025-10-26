# Open-Vocabulary Semantic Segmentation Pipeline
# Multi-stage build for optimized image size

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install -r requirements.txt

# Install SAM 2 from GitHub
RUN pip3 install git+https://github.com/facebookresearch/segment-anything-2.git

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY models/ ./models/
COPY examples/ ./examples/
# COPY checkpoints/ ./checkpoints/
COPY pipeline.py .
COPY config.py .
COPY utils.py .
COPY main.py .

# Create directories for models and outputs
RUN mkdir -p /app/models_cache /app/output /app/input

# Set HuggingFace cache directory
ENV HF_HOME=/app/models_cache \
    TRANSFORMERS_CACHE=/app/models_cache \
    TORCH_HOME=/app/models_cache

# Expose port for potential web interface (future)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1

# Default command (interactive shell)
CMD ["/bin/bash"]
