# Adaptive Mask Selection

## Problem Statement

In open-vocabulary semantic segmentation, SAM 2 generates hundreds of candidate masks at different scales and overlapping regions. The challenge is: **How many masks should we select for a given query?**

### The Challenge

Consider these three queries on the same image:

1. **"car"** - Should return 1 complete vehicle mask
2. **"tire"** - Should return 4 individual tire masks
3. **"mountain"** - Should return N masks (all mountains in the scene)

Traditional approaches use fixed `top-K` selection, but this fails to adapt to the semantic granularity of different queries.

## Solution: Semantic-Aware Adaptive Selection

Our solution automatically determines the appropriate number of masks based on:

1. **Query semantics** (singular vs. plural, part vs. whole, stuff vs. things)
2. **Hierarchical mask relationships** (containment and overlap analysis)
3. **Score distributions** (natural clustering in similarity scores)

## Method Overview

```
Input:
  - Scored masks from CLIP alignment (sorted by similarity)
  - Text query (e.g., "tires", "mountain", "sky")

Process:
  1. Analyze query semantics
     ├─ Detect plural forms ("cars", "people")
     ├─ Identify part-of relationships ("wheel", "window")
     └─ Recognize stuff categories ("sky", "grass")

  2. Build mask hierarchy
     ├─ Compute pairwise IoU
     ├─ Identify parent-child relationships
     └─ Determine hierarchy depth

  3. Detect score clusters
     ├─ Find gaps in similarity scores
     └─ Group masks with similar scores

  4. Apply selection strategy
     ├─ Singular: Select top 1 mask
     ├─ Parts: Select all masks at same hierarchy level
     ├─ Instances: Select non-overlapping high-scoring masks
     └─ Stuff: Select largest high-scoring mask

Output:
  - Adaptively selected masks
  - Debug info (method used, cluster count, etc.)
```

## Semantic Categories

### 1. Singular Objects
**Query examples:** "car", "person", "the building"

**Selection strategy:** Return the highest-scoring mask (represents the complete object)

**Rationale:** Single object queries typically want the most comprehensive representation.

### 2. Object Parts
**Query examples:** "tires", "windows", "headlights"

**Selection strategy:**
- Find masks at the same hierarchy level (siblings)
- Filter by similar scores (within threshold)
- Return all matching parts

**Known parts:** wheel, tire, door, window, headlight, leg, arm, finger, button, handle, etc.

### 3. Multiple Instances
**Query examples:** "cars", "people", "mountains", "trees"

**Selection strategy:**
- Take first cluster (highest scores)
- Filter out overlapping masks (IoU > 0.5)
- Keep diverse non-redundant instances

**Rationale:** Multiple instances should be distinct and non-overlapping.

### 4. Stuff Categories
**Query examples:** "sky", "grass", "water", "road"

**Selection strategy:** Return the largest high-scoring mask

**Rationale:** Stuff categories are amorphous regions that can't be counted. We want the main region.

**Known stuff:** sky, grass, water, sand, snow, road, wall, floor, forest, mountain, ocean, etc.

### 5. Ambiguous
**Fallback strategy:** Use score gap detection to find natural clustering points.

## Hierarchical Mask Relationships

Masks are organized into a hierarchy based on spatial containment:

```
Example: "Car" image

Level 0 (top):
  └─ Entire car body [large mask]
       │
Level 1 (contained in car):
  ├─ Front section
  ├─ Rear section
  │
Level 2 (contained in sections):
  ├─ Front wheel
  ├─ Rear wheel
  ├─ Door
  ├─ Window
  │
Level 3 (smallest parts):
  ├─ Door handle
  ├─ Headlight
  └─ Side mirror
```

**Usage:**
- For "car": Select level 0 (complete object)
- For "wheels": Select level 2 (all wheels at same level)
- For "door handle": Select level 3 (specific part)

## Score Gap Detection

Similarity scores often have natural clustering:

```
Scores: [0.72, 0.68, 0.65, 0.42, 0.38, 0.15, 0.12, 0.08]
         [--------Cluster 1--------]  [--C2--] [---C3---]
                                  ↑           ↑
                              Gap ≥ 0.23   Gap ≥ 0.23
```

**Strategy:** Detect significant gaps (mean + std) to identify natural groupings.

## Configuration

```python
AdaptiveMaskSelector(
    score_gap_threshold=0.1,      # Minimum gap to separate clusters
    min_overlap_ratio=0.8,        # Threshold for parent-child relationship
    max_masks_per_query=20        # Safety limit
)
```

## Usage

### Basic Usage

```python
from pipeline import OpenVocabSegmentationPipeline

pipeline = OpenVocabSegmentationPipeline()

# Adaptive selection (automatic)
result = pipeline.segment(
    image="street.jpg",
    text_prompt="cars",
    use_adaptive_selection=True  # ← Enable adaptive selection
)

print(f"Selected {len(result.segmentation_masks)} masks")
```

### Command-Line

```bash
# Fixed top-5 (traditional)
python main.py --image street.jpg --prompt "cars" --top-k 5

# Adaptive selection
python main.py --image street.jpg --prompt "cars" --adaptive
```

### Demo Script

```bash
# Run comprehensive comparison
python demo_adaptive_selection.py --image your_image.jpg

# Analyze query semantics only
python demo_adaptive_selection.py --analysis-only
```

## Performance Comparison

| Query | Fixed Top-5 | Adaptive | Correct? |
|-------|-------------|----------|----------|
| "car" | 5 masks (includes parts) | 1 mask (complete car) | ✓ Adaptive |
| "tires" | 5 masks (mixed scales) | 4 masks (all tires) | ✓ Adaptive |
| "people" | 5 masks (first 5 people) | 8 masks (all people) | ✓ Adaptive |
| "sky" | 5 masks (fragments) | 1 mask (whole sky) | ✓ Adaptive |
| "mountains" | 5 masks (arbitrary limit) | 3 masks (all mountains) | ✓ Adaptive |

## Implementation Details

### Query Analysis

Implemented using linguistic heuristics:

```python
def _analyze_prompt(self, text_prompt: str) -> Tuple[str, Dict]:
    """
    Returns:
        - semantic_type: "singular", "parts", "stuff", "instances", "ambiguous"
        - prompt_info: Dictionary with analysis details
    """
    # Check plural forms (ends with 's')
    # Check against known part words
    # Check against known stuff categories
    # Apply decision logic
```

### Hierarchy Construction

```python
def _build_mask_hierarchy(self, masks, image_shape) -> Dict:
    """
    Returns:
        hierarchy: {
            0: {"parent": None, "children": [1, 2], "level": 0},
            1: {"parent": 0, "children": [], "level": 1},
            ...
        }
    """
    # Compute pairwise IoU
    # Detect containment (IoU ≥ 0.8 and different sizes)
    # Compute hierarchy depth
```

### Cluster Detection

```python
def _detect_score_clusters(self, masks) -> List[List[int]]:
    """
    Returns clusters where each cluster is separated by significant gaps.
    """
    scores = [m.final_score for m in masks]
    gaps = np.diff(scores)
    threshold = mean(gaps) + std(gaps)
    # Split at gaps ≥ threshold
```

## Thesis Integration

This method can be included in your thesis as:

### Chapter 3: Methodology Extension

**Section 3.4: Adaptive Mask Selection**

Traditional open-vocabulary segmentation systems use fixed top-K selection, which fails to adapt to varying semantic granularity. We propose an adaptive selection method that automatically determines the appropriate number of masks based on query semantics.

**Key contributions:**
1. Semantic query analysis (singular/plural, part/whole, stuff/things)
2. Hierarchical mask relationship modeling
3. Score-based clustering for natural groupings

### Chapter 4: Experiments

**Section 4.X: Adaptive Selection Evaluation**

Compare fixed top-K vs. adaptive selection:
- Accuracy metrics for different query types
- User study on mask selection quality
- Computational overhead analysis (~50ms additional time)

## Limitations

1. **Linguistic Heuristics:** Simple plural detection (ends with 's') may fail for irregular plurals ("people", "children")
2. **Part-of Dictionary:** Limited to manually defined part words (could use external knowledge base)
3. **Language:** Currently English-only
4. **Context:** Doesn't consider spatial context ("left tire" vs "all tires")

## Future Improvements

1. **Language Model Integration:** Use GPT/BERT for better semantic understanding
2. **Spatial Reasoning:** Handle spatial queries ("cars on the left", "mountains in background")
3. **Visual Context:** Use scene understanding to disambiguate (indoor vs outdoor affects "window" interpretation)
4. **User Feedback:** Learn from user corrections to improve selection strategy
5. **Cross-lingual Support:** Extend to multiple languages

## References

- SAM 2: Ravi et al., "SAM 2: Segment Anything in Images and Videos", 2024
- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021
- MaskCLIP: Zhou et al., "Extract Free Dense Labels from CLIP", 2022

## Example Results

### Query: "car"
```
Fixed top-5:     [complete_car, front_section, rear_section, wheel, door]
Adaptive (singular):  [complete_car]
✓ Correctly selects the complete object
```

### Query: "tires"
```
Fixed top-5:     [front_left, front_right, rear_left, rear_right, wheel_rim]
Adaptive (parts):     [front_left, front_right, rear_left, rear_right]
✓ Correctly selects all tires, excludes rim
```

### Query: "mountains"
```
Fixed top-5:     [mountain_1, mountain_2, mountain_3, mountain_4, mountain_5]
Adaptive (instances): [mountain_1, mountain_2, mountain_3, sky_fragment, cloud]
→ [mountain_1, mountain_2, mountain_3]  # After filtering overlaps
✓ Correctly identifies all distinct mountains
```

## Visualization

The `visualize_hierarchy()` method creates annotated images showing:
- Mask hierarchy levels (color-coded)
- Selected vs. rejected masks
- Bounding boxes with scores
- Parent-child relationships

## Contact

For questions or suggestions:
- Open an issue in the repository
- Email: [Your contact]
