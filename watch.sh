#!/bin/bash
# Watch for changes and auto-compile thesis
# Usage: ./watch.sh

echo "========================================="
echo "Master Thesis - Auto-compile on changes"
echo "========================================="
echo ""
echo "Watching for changes in overleaf/..."
echo "Press Ctrl+C to stop"
echo ""

# Check if inotifywait is installed
if ! command -v inotifywait &> /dev/null; then
    echo "Installing inotify-tools..."
    sudo apt-get update && sudo apt-get install -y inotify-tools
fi

# Initial compilation
echo "→ Initial compilation..."
./compile.sh docker

# Watch for changes
while true; do
    inotifywait -r -e modify,create,delete \
        --exclude '(\.aux|\.log|\.out|\.toc|\.bbl|\.blg|build/)' \
        overleaf/Capitulos/ overleaf/*.tex overleaf/*.bib 2>/dev/null

    echo ""
    echo "→ Change detected, recompiling..."
    ./compile.sh docker
    echo ""
    echo "→ Waiting for changes..."
done
