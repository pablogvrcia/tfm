# Adaptive Mask Selection - Quick Start

## What is it?

Automatically determines how many masks to select based on your query:
- **"car"** → Selects 1 complete vehicle
- **"tires"** → Selects all 4 tires
- **"mountains"** → Selects all mountains in scene

## Usage

### Command Line

```bash
# Before (fixed selection)
python main.py --image photo.jpg --prompt "cars" --top-k 5

# After (adaptive selection)
python main.py --image photo.jpg --prompt "cars" --adaptive
```

### Python

```python
from pipeline import OpenVocabSegmentationPipeline

pipeline = OpenVocabSegmentationPipeline()

result = pipeline.segment(
    "photo.jpg",
    "tires",
    use_adaptive_selection=True  # ← Add this
)

print(f"Found {len(result.segmentation_masks)} tires")
```

## Try the Demo

```bash
# Full comparison (requires image)
python demo_adaptive_selection.py --image street.jpg

# Just test query analysis
python demo_adaptive_selection.py --analysis-only
```

## How It Works

1. **Analyzes your query** ("car" vs "cars" vs "tires")
2. **Builds mask hierarchy** (complete objects vs parts)
3. **Applies smart strategy** (different logic per query type)

## Files

- `models/adaptive_selection.py` - Main code
- `ADAPTIVE_SELECTION.md` - Full documentation
- `demo_adaptive_selection.py` - Interactive demo
- `test_adaptive.py` - Unit tests

## Tests

```bash
python test_adaptive.py
# Result: ✅ 4/4 tests passing
```

## Performance

Adds ~50ms overhead (negligible compared to SAM 2 / CLIP)

## Backward Compatible

Yes! Old code works unchanged. New parameter is optional.

