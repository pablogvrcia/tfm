#!/bin/bash
# Helper script to run the Docker container with common options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Open-Vocabulary Semantic Segmentation Pipeline${NC}"
echo -e "${GREEN}Docker Runner Script${NC}\n"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if nvidia-docker is available
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    HAS_GPU=true
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
else
    HAS_GPU=false
    echo -e "${YELLOW}⚠ No GPU detected, will use CPU mode${NC}"
fi

# Parse command line arguments
MODE=${1:-"interactive"}
IMAGE=${2:-""}
PROMPT=${3:-""}

# Function to print usage
usage() {
    echo "Usage: $0 [mode] [image] [prompt] [edit]"
    echo ""
    echo "Modes:"
    echo "  build         - Build the Docker image"
    echo "  interactive   - Start interactive shell (default)"
    echo "  segment       - Run segmentation"
    echo "  remove        - Remove object"
    echo "  replace       - Replace object (requires edit prompt)"
    echo "  benchmark     - Run benchmark"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 interactive"
    echo "  $0 segment input/photo.jpg \"red car\""
    echo "  $0 remove input/photo.jpg \"person\""
    echo "  $0 replace input/photo.jpg \"old TV\" \"modern flat screen\""
    exit 1
}

# Function to build image
build_image() {
    echo -e "\n${GREEN}Building Docker image...${NC}"
    docker build -t openvocab-segmentation:latest .
    echo -e "${GREEN}✓ Image built successfully${NC}"
}

# Function to run container
run_container() {
    local cmd=$1
    local gpu_flag=""

    if [ "$HAS_GPU" = true ]; then
        gpu_flag="--gpus all"
    fi

    # Create input and output directories if they don't exist
    mkdir -p input output

    docker run --rm -it \
        $gpu_flag \
        -v "$(pwd)/input:/app/input:ro" \
        -v "$(pwd)/output:/app/output:rw" \
        -v openvocab-model-cache:/app/models_cache:rw \
        --name openvocab-pipeline-run \
        openvocab-segmentation:latest \
        $cmd
}

# Main logic
case "$MODE" in
    build)
        build_image
        ;;

    interactive|shell|bash)
        echo -e "\n${GREEN}Starting interactive shell...${NC}"
        echo -e "${YELLOW}Tip: Run 'python main.py --help' for usage${NC}\n"
        run_container "/bin/bash"
        ;;

    segment)
        if [ -z "$IMAGE" ] || [ -z "$PROMPT" ]; then
            echo -e "${RED}Error: Image and prompt required${NC}"
            usage
        fi
        echo -e "\n${GREEN}Running segmentation...${NC}"
        echo -e "Image: $IMAGE"
        echo -e "Prompt: $PROMPT\n"
        run_container "python3 main.py --image /app/$IMAGE --prompt \"$PROMPT\" --mode segment"
        ;;

    remove)
        if [ -z "$IMAGE" ] || [ -z "$PROMPT" ]; then
            echo -e "${RED}Error: Image and prompt required${NC}"
            usage
        fi
        echo -e "\n${GREEN}Running object removal...${NC}"
        echo -e "Image: $IMAGE"
        echo -e "Target: $PROMPT\n"
        run_container "python3 main.py --image /app/$IMAGE --prompt \"$PROMPT\" --mode remove"
        ;;

    replace)
        EDIT_PROMPT=${4:-""}
        if [ -z "$IMAGE" ] || [ -z "$PROMPT" ] || [ -z "$EDIT_PROMPT" ]; then
            echo -e "${RED}Error: Image, prompt, and edit prompt required${NC}"
            usage
        fi
        echo -e "\n${GREEN}Running object replacement...${NC}"
        echo -e "Image: $IMAGE"
        echo -e "Target: $PROMPT"
        echo -e "Replace with: $EDIT_PROMPT\n"
        run_container "python3 main.py --image /app/$IMAGE --prompt \"$PROMPT\" --mode replace --edit \"$EDIT_PROMPT\""
        ;;

    benchmark)
        if [ -z "$IMAGE" ]; then
            echo -e "${RED}Error: Image required for benchmark${NC}"
            usage
        fi
        echo -e "\n${GREEN}Running benchmark...${NC}"
        run_container "python3 main.py --image /app/$IMAGE --mode benchmark"
        ;;

    help|--help|-h)
        usage
        ;;

    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        usage
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"
echo -e "Output files are in: ${YELLOW}./output/${NC}\n"
