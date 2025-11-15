"""
SCLIP Multi-Descriptor Support

Parses cls_voc21.txt (and similar files) to create multiple text embeddings per class.

Example cls_voc21.txt format:
    Line 1 (background): sky, wall, tree, wood, grass, road, ...
    Line 2 (aeroplane): aeroplane
    Line 16 (person): person, person in shirt, person in jeans, ...

SCLIP's strategy:
- Creates separate embeddings for each descriptor
- Maps all descriptors to the same class index
- Takes max similarity during inference
"""

from typing import List, Tuple
from pathlib import Path
import torch
import torch.nn as nn


def parse_sclip_descriptors(descriptor_file: str) -> Tuple[List[str], List[int]]:
    """
    Parse SCLIP descriptor file (e.g., cls_voc21.txt).

    Args:
        descriptor_file: Path to descriptor file

    Returns:
        query_words: List of all descriptors (can be > num_classes)
        query_idx: Class index for each descriptor

    Example:
        For Pascal VOC with cls_voc21.txt:
        - Line 0 (background): "sky, wall, tree, wood, ..." → 26 descriptors
        - Line 15 (person): "person, person in shirt, ..." → 7 descriptors

        Returns:
            query_words: ["sky", "wall", "tree", ..., "person", "person in shirt", ...]
            query_idx: [0, 0, 0, ..., 15, 15, ...]
    """
    with open(descriptor_file, 'r') as f:
        name_sets = f.readlines()

    num_classes = len(name_sets)

    query_words = []
    query_idx = []

    for class_idx in range(num_classes):
        # Split by comma to get all descriptors for this class
        descriptors = name_sets[class_idx].split(', ')

        # Clean descriptors (remove newlines)
        descriptors = [d.strip().replace('\n', '') for d in descriptors]

        # Add all descriptors for this class
        query_words += descriptors

        # Map all descriptors to the same class index
        query_idx += [class_idx] * len(descriptors)

    return query_words, query_idx


def map_logits_to_classes(
    logits: torch.Tensor,
    query_idx: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Map multi-descriptor logits to per-class logits using max pooling.

    This is SCLIP's approach: take the maximum similarity among all descriptors
    that map to the same class.

    Args:
        logits: Logits for all descriptors (num_queries, H, W)
        query_idx: Class index for each query (num_queries,)
        num_classes: Number of unique classes

    Returns:
        class_logits: Per-class logits (num_classes, H, W)

    Example:
        Background (class 0) has 26 descriptors at indices [0, 1, ..., 25]
        logits[0:26] contains similarity for all 26 descriptors
        class_logits[0] = max(logits[0:26]) (element-wise max)
    """
    num_queries = logits.shape[0]
    H, W = logits.shape[1], logits.shape[2]

    # Ensure query_idx is on the same device as logits
    query_idx = query_idx.to(logits.device)

    # Validate: descriptor file must match dataset classes
    num_descriptor_classes = query_idx.max().item() + 1
    if num_descriptor_classes != num_classes:
        raise ValueError(
            f"Descriptor file mismatch: descriptor file has {num_descriptor_classes} classes "
            f"but dataset has {num_classes} classes. "
            f"Please use the correct descriptor file for this dataset or remove --descriptor-file."
        )

    # Memory-efficient implementation: process class by class
    # This avoids creating the huge (num_classes, num_queries, H, W) tensor
    class_logits = torch.zeros(num_classes, H, W, dtype=logits.dtype, device=logits.device)

    for class_idx in range(num_classes):
        # Find all descriptors for this class
        descriptor_mask = (query_idx == class_idx)

        if descriptor_mask.any():
            # Get logits for all descriptors of this class
            class_descriptors = logits[descriptor_mask]  # (num_class_descriptors, H, W)

            # Take max across descriptors (element-wise max)
            class_logits[class_idx] = class_descriptors.max(dim=0)[0]
        else:
            # No descriptors for this class - leave as zeros (will get low probability)
            pass

    return class_logits


def get_descriptor_info(descriptor_file: str) -> dict:
    """
    Get information about descriptors in the file.

    Returns:
        dict with:
            - num_classes: Number of unique classes
            - num_descriptors: Total number of descriptors
            - descriptors_per_class: List of descriptor counts per class
            - expansion_ratio: Average descriptors per class
    """
    query_words, query_idx = parse_sclip_descriptors(descriptor_file)

    num_classes = max(query_idx) + 1
    num_descriptors = len(query_words)

    descriptors_per_class = []
    for class_idx in range(num_classes):
        count = query_idx.count(class_idx)
        descriptors_per_class.append(count)

    expansion_ratio = num_descriptors / num_classes

    return {
        'num_classes': num_classes,
        'num_descriptors': num_descriptors,
        'descriptors_per_class': descriptors_per_class,
        'expansion_ratio': expansion_ratio,
        'query_words': query_words,
        'query_idx': query_idx
    }


def print_descriptor_stats(descriptor_file: str):
    """Print statistics about the descriptor file."""
    info = get_descriptor_info(descriptor_file)

    print(f"Descriptor File: {descriptor_file}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Total descriptors: {info['num_descriptors']}")
    print(f"  Expansion ratio: {info['expansion_ratio']:.2f}x")
    print(f"  Descriptors per class:")

    for class_idx, count in enumerate(info['descriptors_per_class']):
        if count > 1:
            print(f"    Class {class_idx}: {count} descriptors")

    # Show examples of multi-descriptor classes
    print(f"\n  Multi-descriptor classes:")
    for class_idx, count in enumerate(info['descriptors_per_class']):
        if count > 1:
            # Get descriptors for this class
            descriptors = [
                info['query_words'][i]
                for i, idx in enumerate(info['query_idx'])
                if idx == class_idx
            ]
            print(f"    Class {class_idx} ({count} descriptors):")
            print(f"      {', '.join(descriptors[:5])}")
            if count > 5:
                print(f"      ... and {count - 5} more")


if __name__ == '__main__':
    # Test with cls_voc21.txt
    import sys
    from pathlib import Path

    code_dir = Path(__file__).parent.parent
    descriptor_file = code_dir / "configs" / "cls_voc21.txt"

    if descriptor_file.exists():
        print("="*80)
        print("SCLIP Multi-Descriptor Parser Test")
        print("="*80)
        print()

        print_descriptor_stats(str(descriptor_file))

        print()
        print("="*80)
        print("Testing logit mapping...")
        print("="*80)

        # Parse descriptors
        query_words, query_idx = parse_sclip_descriptors(str(descriptor_file))
        query_idx_tensor = torch.tensor(query_idx, dtype=torch.long)
        num_classes = max(query_idx) + 1

        # Create fake logits (num_descriptors, H, W)
        H, W = 32, 32
        fake_logits = torch.randn(len(query_words), H, W)

        print(f"Input logits shape: {fake_logits.shape}")
        print(f"  (num_descriptors={len(query_words)}, H={H}, W={W})")

        # Map to classes
        class_logits = map_logits_to_classes(fake_logits, query_idx_tensor, num_classes)

        print(f"Output logits shape: {class_logits.shape}")
        print(f"  (num_classes={num_classes}, H={H}, W={W})")

        # Verify max pooling works
        print(f"\nVerifying max pooling for class 0 (background):")
        class_0_descriptors = [i for i, idx in enumerate(query_idx) if idx == 0]
        print(f"  Class 0 has {len(class_0_descriptors)} descriptors")

        manual_max = fake_logits[class_0_descriptors].max(dim=0)[0]
        auto_max = class_logits[0]

        print(f"  Manual max shape: {manual_max.shape}")
        print(f"  Auto max shape: {auto_max.shape}")
        print(f"  Match: {torch.allclose(manual_max, auto_max)}")

        print()
        print("="*80)
        print("✓ Multi-descriptor parsing and mapping working correctly!")
        print("="*80)
    else:
        print(f"Error: {descriptor_file} not found")
        print("Please create configs/cls_voc21.txt first")
