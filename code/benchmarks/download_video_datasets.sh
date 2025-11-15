#!/bin/bash
# Download and prepare video benchmark datasets for video object segmentation evaluation
#
# Datasets:
# - DAVIS 2016 (Single object)
# - DAVIS 2017 (Multi-object)
# - YouTube-VOS 2019
#
# Usage:
#   ./download_video_datasets.sh [dataset_name]
#
# Examples:
#   ./download_video_datasets.sh all           # Download all datasets
#   ./download_video_datasets.sh davis-2016    # Download only DAVIS 2016
#   ./download_video_datasets.sh davis-2017    # Download only DAVIS 2017
#   ./download_video_datasets.sh youtube-vos   # Download only YouTube-VOS

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-./data/video_benchmarks}"
mkdir -p "$DATA_DIR"

echo "================================================"
echo "Video Benchmark Dataset Downloader"
echo "================================================"
echo "Data directory: $DATA_DIR"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    wget --no-check-certificate -O "$output" "$url" || curl -L -o "$output" "$url" || {
        echo -e "${RED}Download failed: $url${NC}"
        echo -e "${YELLOW}Please download manually from: $url${NC}"
        return 1
    }
}

extract_archive() {
    local archive=$1
    local dest_dir=$2

    echo "Extracting: $(basename $archive)"

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
# DAVIS 2016 (Single Object)
#===============================================================================

download_davis_2016() {
    echo ""
    echo "================================================"
    echo "Downloading DAVIS 2016 (Single Object)"
    echo "================================================"
    echo ""
    echo "Size: ~1GB (480p resolution)"
    echo "Sequences: 50 (30 train + 20 val)"
    echo "Objects: 1 per video"
    echo "Paper: https://arxiv.org/abs/1704.00675"
    echo ""

    check_disk_space 2

    local davis_dir="$DATA_DIR/DAVIS-2016"
    mkdir -p "$davis_dir"

    echo -e "${BLUE}Downloading DAVIS 2016...${NC}"
    download_file \
        "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" \
        "$davis_dir/DAVIS-2016-trainval-480p.zip"

    # Extract
    echo "Extracting archive..."
    extract_archive "$davis_dir/DAVIS-2016-trainval-480p.zip" "$davis_dir"

    # Verify
    if [ -d "$davis_dir/DAVIS" ]; then
        echo -e "${GREEN}✓ DAVIS 2016 downloaded successfully!${NC}"
        echo "  Location: $davis_dir/DAVIS"
        echo "  Annotations: $davis_dir/DAVIS/Annotations/480p"
        echo "  Images: $davis_dir/DAVIS/JPEGImages/480p"
    else
        echo -e "${RED}✗ DAVIS 2016 download incomplete${NC}"
        echo -e "${YELLOW}Manual download: https://davischallenge.org/davis2016/code.html${NC}"
        return 1
    fi
}

#===============================================================================
# DAVIS 2017 (Multi-Object)
#===============================================================================

download_davis_2017() {
    echo ""
    echo "================================================"
    echo "Downloading DAVIS 2017 (Multi-Object)"
    echo "================================================"
    echo ""
    echo "Size: ~1GB (480p) or ~7GB (1080p)"
    echo "Sequences: 120 (90 train + 30 val)"
    echo "Objects: Multiple per video"
    echo "Paper: https://arxiv.org/abs/1704.00675"
    echo ""

    check_disk_space 2

    local davis_dir="$DATA_DIR/DAVIS-2017"
    mkdir -p "$davis_dir"

    echo -e "${BLUE}Downloading DAVIS 2017 (480p)...${NC}"
    download_file \
        "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" \
        "$davis_dir/DAVIS-2017-trainval-480p.zip"

    # Extract
    echo "Extracting archive..."
    extract_archive "$davis_dir/DAVIS-2017-trainval-480p.zip" "$davis_dir"

    # Verify
    if [ -d "$davis_dir/DAVIS" ]; then
        echo -e "${GREEN}✓ DAVIS 2017 downloaded successfully!${NC}"
        echo "  Location: $davis_dir/DAVIS"
        echo "  Annotations: $davis_dir/DAVIS/Annotations/480p"
        echo "  Images: $davis_dir/DAVIS/JPEGImages/480p"
        echo "  ImageSets: $davis_dir/DAVIS/ImageSets/2017"

        # Show number of sequences
        local num_train=$(wc -l < "$davis_dir/DAVIS/ImageSets/2017/train.txt" 2>/dev/null || echo "N/A")
        local num_val=$(wc -l < "$davis_dir/DAVIS/ImageSets/2017/val.txt" 2>/dev/null || echo "N/A")
        echo ""
        echo "  Train sequences: $num_train"
        echo "  Val sequences: $num_val"
    else
        echo -e "${RED}✗ DAVIS 2017 download incomplete${NC}"
        echo -e "${YELLOW}Manual download: https://davischallenge.org/davis2017/code.html${NC}"
        return 1
    fi

    # Optional: Download 1080p version
    read -p "Do you also want to download DAVIS 2017 in 1080p (~7GB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading DAVIS 2017 (1080p)..."
        check_disk_space 8

        download_file \
            "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip" \
            "$davis_dir/DAVIS-2017-trainval-1080p.zip"

        echo "Extracting 1080p archive..."
        extract_archive "$davis_dir/DAVIS-2017-trainval-1080p.zip" "$davis_dir"

        echo -e "${GREEN}✓ DAVIS 2017 (1080p) downloaded successfully!${NC}"
    fi
}

#===============================================================================
# YouTube-VOS 2019
#===============================================================================

download_youtube_vos() {
    echo ""
    echo "================================================"
    echo "Downloading YouTube-VOS 2019"
    echo "================================================"
    echo ""
    echo "Size: ~30GB (train) + ~3GB (valid)"
    echo "Videos: 4,453 videos"
    echo "Objects: 94 categories"
    echo "Paper: https://arxiv.org/abs/1809.03327"
    echo ""
    echo -e "${YELLOW}NOTE: YouTube-VOS requires registration at https://youtube-vos.org/${NC}"
    echo -e "${YELLOW}You will need to manually download the dataset after registration.${NC}"
    echo ""

    check_disk_space 35

    local ytv_dir="$DATA_DIR/youtube-vos-2019"
    mkdir -p "$ytv_dir"

    echo -e "${BLUE}Instructions for downloading YouTube-VOS:${NC}"
    echo ""
    echo "1. Go to: https://youtube-vos.org/dataset/vos/"
    echo "2. Register for an account (required)"
    echo "3. Download the following files:"
    echo "   - train.zip (images + annotations)"
    echo "   - valid.zip (images + annotations + meta.json)"
    echo "4. Place them in: $ytv_dir"
    echo "5. Run this script again to extract"
    echo ""

    # Check if files exist
    if [ -f "$ytv_dir/train.zip" ] || [ -f "$ytv_dir/valid.zip" ]; then
        echo -e "${GREEN}Found YouTube-VOS archives, extracting...${NC}"

        if [ -f "$ytv_dir/train.zip" ]; then
            echo "Extracting train.zip..."
            extract_archive "$ytv_dir/train.zip" "$ytv_dir"
        fi

        if [ -f "$ytv_dir/valid.zip" ]; then
            echo "Extracting valid.zip..."
            extract_archive "$ytv_dir/valid.zip" "$ytv_dir"
        fi

        # Verify
        if [ -d "$ytv_dir/train" ] || [ -d "$ytv_dir/valid" ]; then
            echo -e "${GREEN}✓ YouTube-VOS extracted successfully!${NC}"
            echo "  Location: $ytv_dir"

            if [ -d "$ytv_dir/train" ]; then
                local num_train=$(ls -1 "$ytv_dir/train/JPEGImages" 2>/dev/null | wc -l)
                echo "  Train videos: $num_train"
            fi

            if [ -d "$ytv_dir/valid" ]; then
                local num_val=$(ls -1 "$ytv_dir/valid/JPEGImages" 2>/dev/null | wc -l)
                echo "  Valid videos: $num_val"
            fi
        else
            echo -e "${RED}✗ YouTube-VOS extraction incomplete${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}YouTube-VOS files not found in $ytv_dir${NC}"
        echo -e "${YELLOW}Please download manually from https://youtube-vos.org/${NC}"
        return 1
    fi
}

#===============================================================================
# DAVIS Test-Dev (Optional)
#===============================================================================

download_davis_testdev() {
    echo ""
    echo "================================================"
    echo "Downloading DAVIS 2017 Test-Dev"
    echo "================================================"
    echo ""
    echo "Size: ~500MB"
    echo "Sequences: 30 (test-dev set without GT)"
    echo "Note: For online evaluation only"
    echo ""

    check_disk_space 1

    local davis_dir="$DATA_DIR/DAVIS-2017"
    mkdir -p "$davis_dir"

    echo -e "${BLUE}Downloading DAVIS 2017 Test-Dev...${NC}"
    download_file \
        "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip" \
        "$davis_dir/DAVIS-2017-test-dev-480p.zip"

    # Extract
    echo "Extracting archive..."
    extract_archive "$davis_dir/DAVIS-2017-test-dev-480p.zip" "$davis_dir"

    if [ -d "$davis_dir/DAVIS/JPEGImages/480p" ]; then
        echo -e "${GREEN}✓ DAVIS 2017 Test-Dev downloaded successfully!${NC}"
    else
        echo -e "${RED}✗ DAVIS 2017 Test-Dev download incomplete${NC}"
        return 1
    fi
}

#===============================================================================
# Verification and Summary
#===============================================================================

verify_dataset() {
    local dataset_name=$1
    local dataset_path=$2

    echo ""
    echo "Verifying $dataset_name..."

    if [ -d "$dataset_path" ]; then
        echo -e "${GREEN}✓ $dataset_name found at: $dataset_path${NC}"

        # Show directory structure
        echo ""
        echo "Directory structure:"
        tree -L 2 "$dataset_path" 2>/dev/null || ls -lh "$dataset_path"
        return 0
    else
        echo -e "${RED}✗ $dataset_name not found${NC}"
        return 1
    fi
}

print_summary() {
    echo ""
    echo "================================================"
    echo "Download Summary"
    echo "================================================"
    echo ""

    # Check what's downloaded
    echo "Downloaded datasets:"
    [ -d "$DATA_DIR/DAVIS-2016/DAVIS" ] && echo -e "  ${GREEN}✓${NC} DAVIS 2016" || echo -e "  ${RED}✗${NC} DAVIS 2016"
    [ -d "$DATA_DIR/DAVIS-2017/DAVIS" ] && echo -e "  ${GREEN}✓${NC} DAVIS 2017" || echo -e "  ${RED}✗${NC} DAVIS 2017"
    [ -d "$DATA_DIR/youtube-vos-2019/valid" ] && echo -e "  ${GREEN}✓${NC} YouTube-VOS 2019" || echo -e "  ${RED}✗${NC} YouTube-VOS 2019"

    echo ""
    echo "Disk usage:"
    du -sh "$DATA_DIR" 2>/dev/null || echo "N/A"

    echo ""
    echo "Next steps:"
    echo "  1. Run benchmarks:"
    echo "     python run_video_benchmarks.py --dataset davis-2017 --num-samples 5"
    echo ""
    echo "  2. Evaluate on YouTube-VOS:"
    echo "     python run_video_benchmarks.py --dataset youtube-vos --num-samples 10"
    echo ""
    echo "  3. See results:"
    echo "     cat benchmarks/video_results/davis-2017_results.json"
}

#===============================================================================
# Main
#===============================================================================

main() {
    local dataset="${1:-all}"

    case "$dataset" in
        all)
            download_davis_2016
            download_davis_2017
            download_youtube_vos
            ;;
        davis-2016)
            download_davis_2016
            ;;
        davis-2017)
            download_davis_2017
            ;;
        youtube-vos|ytv)
            download_youtube_vos
            ;;
        test-dev)
            download_davis_testdev
            ;;
        verify)
            verify_dataset "DAVIS 2016" "$DATA_DIR/DAVIS-2016/DAVIS"
            verify_dataset "DAVIS 2017" "$DATA_DIR/DAVIS-2017/DAVIS"
            verify_dataset "YouTube-VOS" "$DATA_DIR/youtube-vos-2019"
            ;;
        *)
            echo "Usage: $0 [all|davis-2016|davis-2017|youtube-vos|test-dev|verify]"
            echo ""
            echo "Datasets:"
            echo "  davis-2016    - DAVIS 2016 Single Object (~1GB)"
            echo "  davis-2017    - DAVIS 2017 Multi-Object (~1GB, optional 1080p ~7GB)"
            echo "  youtube-vos   - YouTube-VOS 2019 (~33GB, requires manual download)"
            echo "  test-dev      - DAVIS 2017 Test-Dev (~500MB)"
            echo "  all           - Download all datasets"
            echo "  verify        - Verify existing downloads"
            echo ""
            echo "Examples:"
            echo "  $0 davis-2017              # Download DAVIS 2017 only"
            echo "  $0 all                     # Download all datasets"
            echo "  $0 verify                  # Verify existing downloads"
            echo ""
            echo "Environment variables:"
            echo "  DATA_DIR     - Data directory (default: ./data/video_benchmarks)"
            exit 1
            ;;
    esac

    print_summary

    echo ""
    echo "================================================"
    echo "✓ Video dataset preparation complete!"
    echo "================================================"
}

main "$@"
