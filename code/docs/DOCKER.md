# Docker Deployment Guide

This guide explains how to run the Open-Vocabulary Segmentation Pipeline using Docker.

## Prerequisites

- **Docker** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **NVIDIA GPU** with CUDA 12.1+ support (optional, CPU mode available)
- **nvidia-docker** for GPU support ([Install nvidia-docker](https://github.com/NVIDIA/nvidia-docker))
- **16GB+ RAM** recommended
- **8GB+ VRAM** for GPU mode

## Quick Start

### Option 1: Using Helper Script (Recommended)

```bash
# Build the image
./docker-run.sh build

# Run interactive shell
./docker-run.sh interactive

# Inside the container:
python main.py --image input/photo.jpg --prompt "red car" --mode segment
```

### Option 2: Direct Docker Commands

```bash
# Build image
docker build -t openvocab-segmentation:latest .

# Run segmentation
docker run --rm -it --gpus all \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:latest \
    python3 main.py --image /app/input/photo.jpg --prompt "red car" --mode segment
```

### Option 3: Using Docker Compose

```bash
# Start with GPU
docker-compose up -d segmentation

# Start CPU-only version
docker-compose --profile cpu up -d segmentation-cpu

# Execute commands
docker-compose exec segmentation python3 main.py --image /app/input/photo.jpg --prompt "car"

# Stop
docker-compose down
```

## Usage Examples

### 1. Build the Image

```bash
./docker-run.sh build
```

This will:
- Download base CUDA image (~3GB)
- Install Python dependencies (~5GB)
- Download SAM 2, CLIP, Stable Diffusion on first run (~7GB total)
- Total build time: 10-15 minutes (first time)

### 2. Interactive Mode

```bash
./docker-run.sh interactive

# Inside container:
root@container:/app# python3 main.py --help
root@container:/app# python3 main.py --image input/photo.jpg --prompt "person" --mode segment
```

### 3. Segment Objects

```bash
# Place images in ./input/ directory
cp photo.jpg input/

# Run segmentation
./docker-run.sh segment input/photo.jpg "red car"

# Output will be in ./output/
ls output/
# → original.png, segmentation.png, similarity_map.png, comparison_grid.png
```

### 4. Remove Objects

```bash
./docker-run.sh remove input/photo.jpg "person in background"

# Output: original.png, edited.png, comparison.png
```

### 5. Replace Objects

```bash
./docker-run.sh replace input/room.jpg "old TV" "modern 4K OLED TV"

# Output: original.png, edited.png, segmentation.png, comparison.png
```

### 6. Run Benchmark

```bash
./docker-run.sh benchmark input/test.jpg
```

## Directory Structure

```
code/
├── input/              # Place your images here (read-only in container)
├── output/             # Generated outputs (writable from container)
├── Dockerfile          # Image definition
├── docker-compose.yml  # Multi-container setup
└── docker-run.sh       # Helper script
```

## Volume Mounts

The Docker setup uses three volume mounts:

1. **Input** (`./input → /app/input`): Read-only, place your images here
2. **Output** (`./output → /app/output`): Writable, outputs saved here
3. **Model Cache** (named volume): Persists downloaded models across runs

## GPU Support

### Check GPU Availability

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this works, you have GPU support! If not:

```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### CPU-Only Mode

If you don't have a GPU, remove `--gpus all` from docker commands:

```bash
docker run --rm -it \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:latest \
    python3 main.py --image /app/input/photo.jpg --prompt "car" --device cpu
```

Or use the CPU profile in docker-compose:

```bash
docker-compose --profile cpu up -d segmentation-cpu
```

## Configuration

### Custom Configurations

Mount a custom config file:

```bash
docker run --rm -it --gpus all \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    -v $(pwd)/my_config.py:/app/custom_config.py:ro \
    openvocab-segmentation:latest \
    python3 -c "
from custom_config import MyConfig
from pipeline import OpenVocabSegmentationPipeline
pipeline = OpenVocabSegmentationPipeline()
result = pipeline.segment('input/photo.jpg', 'red car')
"
```

### Environment Variables

Control behavior with environment variables:

```bash
docker run --rm -it --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_HOME=/app/models_cache \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:latest \
    python3 main.py --image /app/input/photo.jpg --prompt "car"
```

## Performance Optimization

### 1. Use Pre-built Image (Skip Build)

```bash
# Pull from registry (if available)
docker pull yourusername/openvocab-segmentation:latest

# Tag it
docker tag yourusername/openvocab-segmentation:latest openvocab-segmentation:latest
```

### 2. Persistent Model Cache

Models are automatically cached in a named Docker volume. To inspect:

```bash
# List volumes
docker volume ls | grep model-cache

# Inspect volume
docker volume inspect openvocab-model-cache

# Clear cache (if needed)
docker volume rm openvocab-model-cache
```

### 3. Batch Processing

Process multiple images efficiently:

```bash
# Place all images in input/
ls input/
# → image1.jpg, image2.jpg, image3.jpg

# Run batch script
docker run --rm -it --gpus all \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:latest \
    bash -c "
    for img in /app/input/*.jpg; do
        python3 main.py --image \$img --prompt 'car' --mode segment --output /app/output/\$(basename \$img .jpg)
    done
    "
```

### 4. Memory Limits

Adjust memory based on your system:

```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 32G  # Increase if needed
```

## Troubleshooting

### Issue: CUDA out of memory

```bash
# Use CPU mode
./docker-run.sh segment input/photo.jpg "car" --device cpu

# Or reduce image size
docker run --rm -it --gpus all \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:latest \
    python3 -c "
from PIL import Image
img = Image.open('/app/input/photo.jpg')
img = img.resize((512, 512))
img.save('/app/input/photo_small.jpg')
"
```

### Issue: Models not downloading

```bash
# Check internet connection in container
docker run --rm openvocab-segmentation:latest ping -c 3 huggingface.co

# Manually download models
docker run --rm -it --gpus all \
    -v model-cache:/app/models_cache \
    openvocab-segmentation:latest \
    python3 -c "
from transformers import CLIPModel
CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
"
```

### Issue: Permission denied on output

```bash
# Fix output directory permissions
sudo chown -R $USER:$USER output/
chmod -R 755 output/
```

### Issue: Container exits immediately

```bash
# Check logs
docker logs openvocab-pipeline

# Run with debug
docker run --rm -it --gpus all \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    openvocab-segmentation:latest \
    bash -c "python3 main.py --image /app/input/photo.jpg --prompt 'car' --mode segment -v"
```

## Advanced Usage

### Multi-GPU Support

```bash
# Use specific GPU
docker run --rm -it --gpus '"device=0"' \
    openvocab-segmentation:latest \
    python3 main.py --image /app/input/photo.jpg --prompt "car"

# Use multiple GPUs
docker run --rm -it --gpus '"device=0,1"' \
    openvocab-segmentation:latest \
    python3 main.py --image /app/input/photo.jpg --prompt "car"
```

### Development Mode

Mount source code for live development:

```bash
docker run --rm -it --gpus all \
    -v $(pwd)/input:/app/input:ro \
    -v $(pwd)/output:/app/output:rw \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/pipeline.py:/app/pipeline.py:ro \
    openvocab-segmentation:latest \
    bash
```

### Export Container as Tarball

```bash
# Save image
docker save openvocab-segmentation:latest | gzip > openvocab-segmentation.tar.gz

# Transfer to another machine
scp openvocab-segmentation.tar.gz user@remote:/path/

# Load on remote machine
docker load < openvocab-segmentation.tar.gz
```

## Maintenance

### Update Dependencies

```bash
# Rebuild without cache
docker build --no-cache -t openvocab-segmentation:latest .
```

### Clean Up

```bash
# Remove containers
docker container prune

# Remove images
docker image prune

# Remove volumes (WARNING: deletes cached models)
docker volume prune
```

### Check Container Size

```bash
docker images openvocab-segmentation:latest
# REPOSITORY                  TAG       SIZE
# openvocab-segmentation     latest    ~12GB
```

## Production Deployment

For production use, consider:

1. **Orchestration**: Use Kubernetes or Docker Swarm
2. **Load Balancing**: Add nginx reverse proxy
3. **Monitoring**: Add Prometheus/Grafana
4. **Logging**: Configure centralized logging
5. **Secrets**: Use Docker secrets for API keys
6. **CI/CD**: Automate builds with GitHub Actions

See production deployment examples in `deployment/` directory (if available).

## Support

For issues:
1. Check this guide
2. Review [README.md](README.md)
3. Check container logs: `docker logs <container-name>`
4. Open an issue on GitHub

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [Docker Compose](https://docs.docker.com/compose/)
- [CUDA Containers](https://hub.docker.com/r/nvidia/cuda)
