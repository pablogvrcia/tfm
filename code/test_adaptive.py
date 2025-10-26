#!/usr/bin/env python3
"""
Quick test for adaptive mask selection without requiring real models.

Tests the query analysis and mask hierarchy logic.
"""

import numpy as np
from models.adaptive_selection import AdaptiveMaskSelector
from models.sam2_segmentation import MaskCandidate
from models.mask_alignment import ScoredMask


def create_mock_mask(bbox, area, score):
    """Create a mock mask for testing."""
    h, w = 480, 640
    mask = np.zeros((h, w), dtype=bool)
    x, y, mw, mh = bbox
    mask[y:y+mh, x:x+mw] = True

    return ScoredMask(
        mask_candidate=MaskCandidate(
            mask=mask,
            bbox=bbox,
            area=area,
            predicted_iou=0.9,
            stability_score=0.95
        ),
        similarity_score=score,
        background_score=0.1,
        final_score=score - 0.03,
        rank=0
    )


def test_query_analysis():
    """Test query semantic analysis."""
    print("\n" + "="*60)
    print("TEST: Query Analysis")
    print("="*60)

    selector = AdaptiveMaskSelector()

    test_cases = [
        ("car", "singular"),
        ("the red car", "singular"),
        ("tires", "parts"),
        ("wheels", "parts"),
        ("people", "instances"),
        ("mountains", "instances"),
        ("sky", "stuff"),
        ("grass", "stuff"),
        ("all the windows", "parts"),
    ]

    passed = 0
    failed = 0

    for query, expected_type in test_cases:
        semantic_type, info = selector._analyze_prompt(query)
        status = "✓" if semantic_type == expected_type else "✗"

        if semantic_type == expected_type:
            passed += 1
        else:
            failed += 1

        print(f"{status} '{query:20s}' → {semantic_type:12s} (expected: {expected_type})")

    print(f"\nPassed: {passed}/{len(test_cases)}")
    return failed == 0


def test_hierarchy_building():
    """Test mask hierarchy construction."""
    print("\n" + "="*60)
    print("TEST: Hierarchy Building")
    print("="*60)

    selector = AdaptiveMaskSelector()

    # Create mock masks with containment relationships
    # Large car mask contains smaller part masks
    masks = [
        create_mock_mask((100, 100, 400, 300), 120000, 0.75),  # 0: Entire car
        create_mock_mask((120, 120, 150, 100), 15000, 0.70),   # 1: Front section (inside car)
        create_mock_mask((350, 120, 130, 100), 13000, 0.68),   # 2: Rear section (inside car)
        create_mock_mask((120, 280, 60, 60), 3600, 0.65),      # 3: Front wheel (inside front)
        create_mock_mask((380, 280, 60, 60), 3600, 0.63),      # 4: Rear wheel (inside rear)
    ]

    hierarchy = selector._build_mask_hierarchy(masks, (480, 640))

    print("\nHierarchy structure:")
    for i, info in hierarchy.items():
        parent = info['parent']
        level = info['level']
        parent_str = f"Mask {parent}" if parent is not None else "None"
        print(f"  Mask {i}: Level {level}, Parent: {parent_str}, Children: {info['children']}")

    # Check expectations
    checks = [
        (hierarchy[0]['parent'] is None, "Mask 0 should be top-level"),
        (hierarchy[1]['parent'] == 0, "Mask 1 should be child of 0"),
        (hierarchy[3]['level'] >= 1, "Mask 3 should be at level 1+"),
    ]

    passed = sum(1 for check, _ in checks if check)
    print(f"\nChecks passed: {passed}/{len(checks)}")

    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"  {status} {desc}")

    return all(check for check, _ in checks)


def test_score_clustering():
    """Test score gap detection."""
    print("\n" + "="*60)
    print("TEST: Score Clustering")
    print("="*60)

    selector = AdaptiveMaskSelector(score_gap_threshold=0.1)

    # Create masks with clear score gaps
    masks = [
        create_mock_mask((0, 0, 100, 100), 10000, 0.75),
        create_mock_mask((0, 0, 100, 100), 10000, 0.72),
        create_mock_mask((0, 0, 100, 100), 10000, 0.70),  # Gap after this
        create_mock_mask((0, 0, 100, 100), 10000, 0.45),
        create_mock_mask((0, 0, 100, 100), 10000, 0.42),  # Gap after this
        create_mock_mask((0, 0, 100, 100), 10000, 0.15),
        create_mock_mask((0, 0, 100, 100), 10000, 0.12),
    ]

    clusters = selector._detect_score_clusters(masks)

    print(f"\nDetected {len(clusters)} clusters:")
    for i, cluster in enumerate(clusters):
        scores = [masks[idx].final_score for idx in cluster]
        print(f"  Cluster {i+1}: {len(cluster)} masks, scores: {scores}")

    # Should detect at least 2 clusters due to gaps
    success = len(clusters) >= 2
    print(f"\n{'✓' if success else '✗'} Expected ≥2 clusters, got {len(clusters)}")

    return success


def test_adaptive_selection():
    """Test adaptive selection for different query types."""
    print("\n" + "="*60)
    print("TEST: Adaptive Selection")
    print("="*60)

    selector = AdaptiveMaskSelector()

    # Create diverse set of masks
    masks = [
        create_mock_mask((100, 100, 400, 300), 120000, 0.75),  # Large mask
        create_mock_mask((120, 280, 60, 60), 3600, 0.70),      # Small mask 1
        create_mock_mask((380, 280, 60, 60), 3600, 0.68),      # Small mask 2
        create_mock_mask((250, 280, 60, 60), 3600, 0.66),      # Small mask 3
        create_mock_mask((200, 150, 80, 80), 6400, 0.45),      # Medium mask
    ]

    test_cases = [
        ("car", 1, "singular"),
        ("tires", 3, "parts"),
        ("sky", 1, "stuff"),
    ]

    all_passed = True

    for query, expected_approx, query_type in test_cases:
        selected, info = selector.select_masks_adaptive(
            masks,
            query,
            image_shape=(480, 640)
        )

        method = info['method']
        count = len(selected)

        # Allow some tolerance (±1 mask)
        success = abs(count - expected_approx) <= 1
        all_passed &= success

        status = "✓" if success else "✗"
        print(f"{status} '{query}' ({query_type}): selected {count} masks (expected ~{expected_approx}), method={method}")

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ADAPTIVE MASK SELECTION - UNIT TESTS")
    print("="*60)

    results = []

    try:
        results.append(("Query Analysis", test_query_analysis()))
    except Exception as e:
        print(f"✗ Query Analysis FAILED: {e}")
        results.append(("Query Analysis", False))

    try:
        results.append(("Hierarchy Building", test_hierarchy_building()))
    except Exception as e:
        print(f"✗ Hierarchy Building FAILED: {e}")
        results.append(("Hierarchy Building", False))

    try:
        results.append(("Score Clustering", test_score_clustering()))
    except Exception as e:
        print(f"✗ Score Clustering FAILED: {e}")
        results.append(("Score Clustering", False))

    try:
        results.append(("Adaptive Selection", test_adaptive_selection()))
    except Exception as e:
        print(f"✗ Adaptive Selection FAILED: {e}")
        results.append(("Adaptive Selection", False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - total_passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
