#!/bin/bash
# Download and prepare benchmark datasets for open-vocabulary segmentation evaluation
#
# Datasets:
# - COCO-Stuff 164K
# - PASCAL VOC 2012
# - ADE20K
# - COCO-Open split (48 base + 17 novel classes)
#
# Usage:
#   ./download_datasets.sh [dataset_name]
#
# Examples:
#   ./download_datasets.sh all          # Download all datasets
#   ./download_datasets.sh coco-stuff   # Download only COCO-Stuff
#   ./download_datasets.sh pascal-voc   # Download only PASCAL VOC
#   ./download_datasets.sh ade20k       # Download only ADE20K

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-./data/benchmarks}"
mkdir -p "$DATA_DIR"

echo "================================================"
echo "Benchmark Dataset Downloader"
echo "================================================"
echo "Data directory: $DATA_DIR"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#===============================================================================
# Helper functions
#===============================================================================

check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "$DATA_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')

    if [ "$available_gb" -lt "$required_gb" ]; then
        echo -e "${RED}Error: Not enough disk space. Required: ${required_gb}GB, Available: ${available_gb}GB${NC}"
        exit 1
    fi
}

download_file() {
    local url=$1
    local output=$2

    if [ -f "$output" ]; then
        echo -e "${YELLOW}File already exists: $output${NC}"
        return 0
    fi

    echo "Downloading: $url"
    wget --no-check-certificate -O "$output" "$url" || {
        echo -e "${RED}Download failed: $url${NC}"
        return 1
    }
}

extract_archive() {
    local archive=$1
    local dest_dir=$2

    echo "Extracting: $archive"

    case "$archive" in
        *.tar.gz|*.tgz)
            tar -xzf "$archive" -C "$dest_dir"
            ;;
        *.zip)
            unzip -q "$archive" -d "$dest_dir"
            ;;
        *)
            echo -e "${RED}Unknown archive format: $archive${NC}"
            return 1
            ;;
    esac
}

#===============================================================================
# COCO-Stuff 164K
#===============================================================================

download_coco_stuff() {
    echo ""
    echo "================================================"
    echo "Downloading COCO-Stuff 164K"
    echo "================================================"
    echo ""
    echo "Size: ~50GB (images + annotations)"
    echo "Classes: 171 (91 thing + 80 stuff)"
    echo "Images: 164,062 (train + val)"
    echo ""

    check_disk_space 50

    local coco_dir="$DATA_DIR/coco_stuff"
    mkdir -p "$coco_dir"

    # Download COCO 2017 images (required for COCO-Stuff)
    echo "[1/3] Downloading COCO 2017 train images..."
    download_file \
        "http://images.cocodataset.org/zips/train2017.zip" \
        "$coco_dir/train2017.zip"

    echo "[2/3] Downloading COCO 2017 val images..."
    download_file \
        "http://images.cocodataset.org/zips/val2017.zip" \
        "$coco_dir/val2017.zip"

    echo "[3/3] Downloading COCO-Stuff annotations..."
    download_file \
        "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip" \
        "$coco_dir/stuffthingmaps_trainval2017.zip"

    # Extract
    echo "Extracting archives..."
    extract_archive "$coco_dir/train2017.zip" "$coco_dir"
    extract_archive "$coco_dir/val2017.zip" "$coco_dir"
    extract_archive "$coco_dir/stuffthingmaps_trainval2017.zip" "$coco_dir"

    # Verify
    if [ -d "$coco_dir/train2017" ] && [ -d "$coco_dir/annotations" ]; then
        echo -e "${GREEN}✓ COCO-Stuff downloaded successfully!${NC}"
        echo "  Location: $coco_dir"
    else
        echo -e "${RED}✗ COCO-Stuff download incomplete${NC}"
        return 1
    fi
}

#===============================================================================
# PASCAL VOC 2012
#===============================================================================

download_pascal_voc() {
    echo ""
    echo "================================================"
    echo "Downloading PASCAL VOC 2012"
    echo "================================================"
    echo ""
    echo "Size: ~2GB"
    echo "Classes: 20 object categories + background"
    echo "Images: 2,913 (train + val)"
    echo ""

    check_disk_space 3

    local voc_dir="$DATA_DIR/pascal_voc"
    mkdir -p "$voc_dir"

    # Download VOC 2012
    echo "[1/1] Downloading PASCAL VOC 2012 trainval..."
    download_file \
        "http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar" \
        "$voc_dir/VOCtrainval_11-May-2012.tar"

    # Extract
    echo "Extracting archives..."
    tar -xf "$voc_dir/VOCtrainval_11-May-2012.tar" -C "$voc_dir"
    
    # Verify
    if [ -d "$voc_dir/VOCdevkit/VOC2012" ]; then
        echo -e "${GREEN}✓ PASCAL VOC downloaded successfully!${NC}"
        echo "  Location: $voc_dir/VOCdevkit/VOC2012"
    else
        echo -e "${RED}✗ PASCAL VOC download incomplete${NC}"
        return 1
    fi
}

#===============================================================================
# ADE20K
#===============================================================================

download_ade20k() {
    echo ""
    echo "================================================"
    echo "Downloading ADE20K"
    echo "================================================"
    echo ""
    echo "Size: ~3GB"
    echo "Classes: 150 categories (diverse scenes)"
    echo "Images: 25,574 (train + val)"
    echo ""

    check_disk_space 4

    local ade_dir="$DATA_DIR/ade20k"
    mkdir -p "$ade_dir"

    # Download ADE20K
    echo "Downloading ADE20K dataset..."
    download_file \
        "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" \
        "$ade_dir/ADEChallengeData2016.zip"

    # Extract
    echo "Extracting archive..."
    extract_archive "$ade_dir/ADEChallengeData2016.zip" "$ade_dir"

    # Verify
    if [ -d "$ade_dir/ADEChallengeData2016" ]; then
        echo -e "${GREEN}✓ ADE20K downloaded successfully!${NC}"
        echo "  Location: $ade_dir/ADEChallengeData2016"
    else
        echo -e "${RED}✗ ADE20K download incomplete${NC}"
        return 1
    fi
}

#===============================================================================
# COCO-Open Vocabulary Split
#===============================================================================

create_coco_open_split() {
    echo ""
    echo "================================================"
    echo "Creating COCO-Open Vocabulary Split"
    echo "================================================"
    echo ""
    echo "Base classes: 48 COCO categories"
    echo "Novel classes: 17 COCO categories"
    echo ""

    # This requires COCO-Stuff to be downloaded first
    local coco_dir="$DATA_DIR/coco_stuff"
    if [ ! -d "$coco_dir/train2017" ]; then
        echo -e "${YELLOW}COCO-Stuff not found. Downloading first...${NC}"
        download_coco_stuff
    fi

    # Create split metadata
    local split_dir="$DATA_DIR/coco_open"
    mkdir -p "$split_dir"

    # Run Python script to create split
    python3 - << 'PYTHON'
import json
import os

# COCO-Open split definition (from OpenSeg paper)
BASE_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple'
]

NOVEL_CLASSES = [
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse'
]

split_info = {
    'base_classes': BASE_CLASSES,
    'novel_classes': NOVEL_CLASSES,
    'num_base': len(BASE_CLASSES),
    'num_novel': len(NOVEL_CLASSES),
    'description': 'COCO-Open split for evaluating open-vocabulary generalization'
}

split_file = os.path.join(os.environ.get('DATA_DIR', './data/benchmarks'), 'coco_open', 'split.json')
os.makedirs(os.path.dirname(split_file), exist_ok=True)

with open(split_file, 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"✓ Created COCO-Open split: {split_file}")
print(f"  Base classes: {len(BASE_CLASSES)}")
print(f"  Novel classes: {len(NOVEL_CLASSES)}")
PYTHON

    if [ -f "$split_dir/split.json" ]; then
        echo -e "${GREEN}✓ COCO-Open split created successfully!${NC}"
        echo "  Location: $split_dir"
    else
        echo -e "${RED}✗ COCO-Open split creation failed${NC}"
        return 1
    fi
}

#===============================================================================
# Main
#===============================================================================

main() {
    local dataset="${1:-all}"

    case "$dataset" in
        all)
            download_coco_stuff
            download_pascal_voc
            download_ade20k
            create_coco_open_split
            ;;
        coco-stuff)
            download_coco_stuff
            ;;
        pascal-voc|voc)
            download_pascal_voc
            ;;
        ade20k)
            download_ade20k
            ;;
        coco-open)
            create_coco_open_split
            ;;
        *)
            echo "Usage: $0 [all|coco-stuff|pascal-voc|ade20k|coco-open]"
            echo ""
            echo "Examples:"
            echo "  $0 all          # Download all datasets (~55GB)"
            echo "  $0 coco-stuff   # Download only COCO-Stuff (~50GB)"
            echo "  $0 pascal-voc   # Download only PASCAL VOC (~2GB)"
            echo "  $0 ade20k       # Download only ADE20K (~3GB)"
            echo "  $0 coco-open    # Create COCO-Open split"
            exit 1
            ;;
    esac

    echo ""
    echo "================================================"
    echo "✓ Dataset preparation complete!"
    echo "================================================"
    echo ""
    echo "Data directory structure:"
    tree -L 2 "$DATA_DIR" 2>/dev/null || ls -lh "$DATA_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Run benchmarks: python run_benchmarks.py --dataset all"
    echo "  2. Evaluate on specific dataset: python run_benchmarks.py --dataset pascal-voc"
    echo "  3. See results: cat benchmarks/results/benchmark_results.json"
}

main "$@"
