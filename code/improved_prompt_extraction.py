"""
Improved Prompt Extraction Strategies for CLIP-Guided SAM

Addresses tutor's feedback:
- Better exploitation of confidence maps
- Adaptive strategies per class (stuff vs thing)
- Multiple sampling strategies beyond connected components
"""

import numpy as np
from scipy.ndimage import label
from sklearn.cluster import KMeans


def is_stuff_class(class_name):
    """
    Determine if a class is "stuff" (amorphous regions) or "thing" (discrete objects).

    Stuff classes: sky, road, grass, water, etc.
    Thing classes: person, car, cat, etc.
    """
    stuff_keywords = [
        'sky', 'road', 'grass', 'ground', 'floor', 'ceiling', 'wall', 'water', 'sea',
        'building', 'tree', 'fence', 'pavement', 'sidewalk', 'snow', 'sand', 'dirt',
        'mountain', 'hill', 'field', 'fog', 'cloud', 'river', 'bridge', 'carpet',
        'wood', 'metal', 'plastic', 'fabric', 'paper', 'stone', 'mud', 'leaves',
        'background', 'other'
    ]

    class_lower = class_name.lower()
    return any(keyword in class_lower for keyword in stuff_keywords)


def extract_prompts_adaptive_threshold(seg_map, probs, vocabulary,
                                       base_confidence=0.3, min_region_size=100,
                                       confidence_boost_stuff=0.0, confidence_boost_thing=0.2):
    """
    STRATEGY 1: Adaptive thresholds per class type (EASY TO IMPLEMENT)

    Key idea: Use LOWER thresholds for "stuff" classes, HIGHER for "thing" classes.

    Args:
        base_confidence: Base minimum confidence (default 0.3 - lower than before!)
        confidence_boost_stuff: Added to base for stuff classes (default 0.0 → 0.3 total)
        confidence_boost_thing: Added to base for thing classes (default 0.2 → 0.5 total)

    Returns:
        List of prompts
    """
    H, W = seg_map.shape
    prompts = []

    print("\n[IMPROVED] Adaptive threshold strategy:")

    for class_idx, class_name in enumerate(vocabulary):
        # Adaptive threshold based on class type
        is_stuff = is_stuff_class(class_name)

        if is_stuff:
            min_confidence = base_confidence + confidence_boost_stuff  # 0.3 for stuff
            strategy = "STUFF"
        else:
            min_confidence = base_confidence + confidence_boost_thing  # 0.5 for thing
            strategy = "THING"

        # Get high-confidence regions
        class_mask = (seg_map == class_idx)
        class_confidence = probs[:, :, class_idx]
        high_conf_mask = (class_mask & (class_confidence > min_confidence))

        # Connected components
        labeled_regions, num_regions = label(high_conf_mask)

        if num_regions == 0:
            continue

        print(f"  {class_name} ({strategy}, thresh={min_confidence:.2f}): {num_regions} regions")

        # Extract prompts
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Centroid
            y_coords, x_coords = np.where(region_mask)
            centroid_x = int(x_coords.mean())
            centroid_y = int(y_coords.mean())
            confidence = class_confidence[centroid_y, centroid_x]

            prompts.append({
                'point': (centroid_x, centroid_y),
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': float(confidence),
                'region_size': int(region_size),
                'negative_points': []
            })

    print(f"[IMPROVED] Extracted {len(prompts)} prompts total")
    return prompts


def extract_prompts_confidence_weighted_sampling(seg_map, probs, vocabulary,
                                                 min_confidence=0.2, min_region_size=100,
                                                 max_prompts_per_region=5):
    """
    STRATEGY 2: Confidence-weighted sampling (MODERATE COMPLEXITY)

    Key idea: Sample MORE points in HIGH-confidence areas, especially for large regions.

    Instead of just centroid:
    1. For small regions (< 1000 pixels): 1 prompt (centroid)
    2. For medium regions (1000-5000 pixels): 2-3 prompts (confidence-weighted)
    3. For large regions (> 5000 pixels): 4-5 prompts (confidence-weighted)

    Args:
        max_prompts_per_region: Maximum prompts to extract per region

    Returns:
        List of prompts
    """
    H, W = seg_map.shape
    prompts = []

    print("\n[IMPROVED] Confidence-weighted sampling strategy:")

    for class_idx, class_name in enumerate(vocabulary):
        # Adaptive threshold
        is_stuff = is_stuff_class(class_name)
        conf_threshold = 0.2 if is_stuff else 0.4

        class_mask = (seg_map == class_idx)
        class_confidence = probs[:, :, class_idx]
        high_conf_mask = (class_mask & (class_confidence > conf_threshold))

        labeled_regions, num_regions = label(high_conf_mask)

        if num_regions == 0:
            continue

        region_prompt_counts = []

        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Determine number of prompts based on region size
            if region_size < 1000:
                num_prompts = 1  # Small region: centroid only
            elif region_size < 5000:
                num_prompts = min(3, max_prompts_per_region)  # Medium: 2-3 prompts
            else:
                num_prompts = max_prompts_per_region  # Large: up to 5 prompts

            # Get all points and their confidences
            y_coords, x_coords = np.where(region_mask)
            region_confidences = class_confidence[y_coords, x_coords]

            if num_prompts == 1:
                # Just centroid
                centroid_x = int(x_coords.mean())
                centroid_y = int(y_coords.mean())
                conf = class_confidence[centroid_y, centroid_x]

                prompts.append({
                    'point': (centroid_x, centroid_y),
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'confidence': float(conf),
                    'region_size': int(region_size),
                    'negative_points': []
                })
            else:
                # Confidence-weighted sampling
                # Sample points with probability proportional to confidence^2
                weights = region_confidences ** 2
                weights = weights / weights.sum()

                # Sample unique points
                num_points_to_sample = min(num_prompts, len(x_coords))
                sampled_indices = np.random.choice(
                    len(x_coords),
                    size=num_points_to_sample,
                    replace=False,
                    p=weights
                )

                for idx in sampled_indices:
                    px = int(x_coords[idx])
                    py = int(y_coords[idx])
                    conf = class_confidence[py, px]

                    prompts.append({
                        'point': (px, py),
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'confidence': float(conf),
                        'region_size': int(region_size),
                        'negative_points': []
                    })

            region_prompt_counts.append(num_prompts)

        if len(region_prompt_counts) > 0:
            avg_prompts = np.mean(region_prompt_counts)
            print(f"  {class_name}: {num_regions} regions, avg {avg_prompts:.1f} prompts/region")

    print(f"[IMPROVED] Extracted {len(prompts)} prompts total")
    return prompts


def extract_prompts_density_based(seg_map, probs, vocabulary,
                                   min_confidence=0.2, min_region_size=50,
                                   grid_resolution=32):
    """
    STRATEGY 3: Density-based grid sampling (ADVANCED)

    Key idea: Instead of connected components, use adaptive grid with confidence-based density.

    Algorithm:
    1. For each class, create a confidence-weighted grid
    2. Sample more densely in high-confidence areas
    3. Use confidence map to guide placement

    This is similar to FPN (Feature Pyramid Networks) multi-scale approach.

    Args:
        grid_resolution: Base grid size (32x32)

    Returns:
        List of prompts
    """
    H, W = seg_map.shape
    prompts = []

    print("\n[IMPROVED] Density-based grid sampling:")

    for class_idx, class_name in enumerate(vocabulary):
        # Adaptive threshold
        is_stuff = is_stuff_class(class_name)
        conf_threshold = 0.15 if is_stuff else 0.35  # Even lower for stuff

        class_mask = (seg_map == class_idx)
        class_confidence = probs[:, :, class_idx]

        # Get all high-confidence pixels
        high_conf_mask = (class_mask & (class_confidence > conf_threshold))

        if high_conf_mask.sum() < min_region_size:
            continue

        # Create adaptive grid based on confidence
        y_coords, x_coords = np.where(high_conf_mask)

        if len(x_coords) == 0:
            continue

        # Determine number of prompts based on class coverage
        coverage = high_conf_mask.sum()

        if coverage < 500:
            num_prompts = 1
        elif coverage < 2000:
            num_prompts = 3
        elif coverage < 10000:
            num_prompts = 5
        else:
            num_prompts = 8  # Large regions get more prompts

        # Use K-means clustering on high-confidence points
        # This groups spatially close points
        points = np.stack([x_coords, y_coords], axis=1)

        if len(points) < num_prompts:
            # Not enough points for k-means, use all
            selected_indices = range(len(points))
        else:
            # K-means clustering
            kmeans = KMeans(n_clusters=num_prompts, random_state=42, n_init=10)
            kmeans.fit(points)

            # For each cluster, find the point with highest confidence
            selected_indices = []
            for cluster_id in range(num_prompts):
                cluster_mask = (kmeans.labels_ == cluster_id)
                cluster_confidences = class_confidence[y_coords[cluster_mask], x_coords[cluster_mask]]

                if len(cluster_confidences) == 0:
                    continue

                # Find highest confidence point in this cluster
                local_best_idx = np.argmax(cluster_confidences)
                global_idx = np.where(cluster_mask)[0][local_best_idx]
                selected_indices.append(global_idx)

        # Extract prompts from selected points
        for idx in selected_indices:
            px = int(x_coords[idx])
            py = int(y_coords[idx])
            conf = class_confidence[py, px]

            prompts.append({
                'point': (px, py),
                'class_idx': class_idx,
                'class_name': class_name,
                'confidence': float(conf),
                'region_size': int(coverage),
                'negative_points': []
            })

        print(f"  {class_name}: {num_prompts} prompts, coverage={coverage} pixels")

    print(f"[IMPROVED] Extracted {len(prompts)} prompts total")
    return prompts


def extract_prompts_prob_map_exploitation(seg_map, probs, vocabulary,
                                           min_confidence=0.2, min_region_size=100,
                                           top_k_classes=3, prob_threshold_ratio=0.7):
    """
    STRATEGY 4: Full probability map exploitation (BEST - RECOMMENDED)

    Key idea: Don't just use argmax! Use the FULL probability distribution.

    Current problem:
        argmax(probs) → only the winner class
        Lost information: What if a pixel has prob=[0.4 sky, 0.35 building, 0.25 other]?
                         We only prompt for "sky" but "building" is almost as likely!

    New approach:
        1. For each pixel, consider TOP-K classes (not just argmax)
        2. Prompt for a class if prob > threshold * max_prob
        3. This finds "ambiguous regions" where multiple classes overlap

    Example:
        Pixel has probs = [0.45 sky, 0.40 building, 0.15 other]
        - Old: Only prompt for "sky"
        - New: Prompt for BOTH "sky" AND "building" (0.40 > 0.7 * 0.45)
        → SAM will generate masks for both, then we pick the best one!

    This is especially good for COCO-Stuff where boundaries are ambiguous.

    Args:
        top_k_classes: Consider top K classes per pixel (default 3)
        prob_threshold_ratio: Consider class if prob > ratio * max_prob (default 0.7)

    Returns:
        List of prompts
    """
    H, W = seg_map.shape
    prompts = []

    print("\n[IMPROVED] Full probability map exploitation:")
    print(f"  Strategy: Top-{top_k_classes} classes, threshold ratio={prob_threshold_ratio}")
    print(f"  Image shape: {H}x{W}, Num classes: {len(vocabulary)}")

    # DEBUG: Track stats per class
    debug_stats = []
    extraction_count = 0

    # For each class, find regions where it has HIGH probability
    # STRATEGY: Start with argmax winners, then expand to competitive regions
    for class_idx, class_name in enumerate(vocabulary):
        class_confidence = probs[:, :, class_idx]

        # Adaptive threshold by class type
        is_stuff = is_stuff_class(class_name)
        base_threshold = 0.15 if is_stuff else 0.3

        # HYBRID APPROACH:
        # 1. Find pixels where this class WINS argmax (traditional approach)
        # 2. ALSO find pixels where this class is in top-K with high confidence
        #    This captures "competitive" regions at class boundaries

        class_wins_argmax = (seg_map == class_idx)

        # NEW: Also consider pixels where this class is in top-K
        # even if it's not the argmax winner
        max_probs = probs.max(axis=2)  # (H, W) - max prob at each pixel

        # Get the K-th highest probability at each pixel
        k_th_probs = np.partition(probs, -top_k_classes, axis=2)[:, :, -top_k_classes]

        # Create mask: class is in top-K AND above base threshold
        in_top_k = (class_confidence >= k_th_probs)
        in_top_k_high_conf = in_top_k & (class_confidence > base_threshold)

        # COMBINE: Either wins argmax OR is competitive (top-K + high conf)
        high_conf_mask = class_wins_argmax | in_top_k_high_conf

        # DEBUG: Log stats for each class
        max_class_conf = class_confidence.max()
        argmax_pixels = class_wins_argmax.sum()
        top_k_contrib = in_top_k_high_conf.sum()
        total_high_conf = high_conf_mask.sum()

        if total_high_conf > 0 or argmax_pixels > 0:
            debug_stats.append({
                'class': class_name,
                'max_conf': max_class_conf,
                'argmax_px': argmax_pixels,
                'top_k_contrib': top_k_contrib,
                'total_px': total_high_conf,
                'threshold': base_threshold
            })

        if high_conf_mask.sum() < min_region_size:
            continue

        # Connected components on this "soft" mask
        labeled_regions, num_regions = label(high_conf_mask)

        if num_regions == 0:
            continue

        extraction_count += 1
        region_info = []

        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_regions == region_id)
            region_size = region_mask.sum()

            if region_size < min_region_size:
                continue

            # Get all points in region
            y_coords, x_coords = np.where(region_mask)

            # Calculate AVERAGE confidence over region (not just centroid!)
            avg_confidence = class_confidence[region_mask].mean()

            # Also track if this region is "pure" or "mixed"
            # Mixed regions = high prob for multiple classes
            region_max_probs = max_probs[region_mask]
            region_class_probs = class_confidence[region_mask]

            # Purity: how often is this class the clear winner?
            is_winner = (region_class_probs >= region_max_probs * 0.99)
            purity = is_winner.sum() / region_size

            # For pure regions: 1 prompt
            # For mixed regions: multiple prompts to help SAM disambiguate
            if purity > 0.8:
                num_prompts = 1  # Pure region, centroid is fine
            elif purity > 0.5:
                num_prompts = 2  # Somewhat mixed
            else:
                num_prompts = 3  # Very mixed, need more prompts

            # Confidence-weighted sampling
            weights = class_confidence[region_mask] ** 2
            weights = weights / weights.sum()

            num_points_to_sample = min(num_prompts, len(x_coords))
            sampled_indices = np.random.choice(
                len(x_coords),
                size=num_points_to_sample,
                replace=False,
                p=weights
            )

            # IMPROVED: Sample prompts only from pixels where class_idx is ACTUALLY argmax
            # This prevents sampling noisy pixels that are just "top-K" but not winners
            argmax_pixels_in_region = region_mask & class_wins_argmax

            # If we have argmax pixels, prefer those for sampling
            if argmax_pixels_in_region.any():
                argmax_y, argmax_x = np.where(argmax_pixels_in_region)
                argmax_coords = list(zip(argmax_x, argmax_y))

                # If we have enough argmax pixels, sample only from those
                if len(argmax_coords) >= num_points_to_sample:
                    # Sample from argmax pixels only (highest quality)
                    argmax_confidences = class_confidence[argmax_pixels_in_region]
                    argmax_weights = argmax_confidences / argmax_confidences.sum()
                    argmax_indices = np.random.choice(
                        len(argmax_coords),
                        size=num_points_to_sample,
                        replace=False,
                        p=argmax_weights
                    )
                    sampled_coords = [argmax_coords[i] for i in argmax_indices]
                    sampling_strategy = "argmax_only"
                else:
                    # Not enough argmax pixels, mix argmax + top-K
                    # Take ALL argmax pixels + sample remaining from top-K
                    sampled_coords = argmax_coords.copy()
                    remaining = num_points_to_sample - len(argmax_coords)

                    # Sample remaining from non-argmax region pixels
                    non_argmax_region = region_mask & ~class_wins_argmax
                    if non_argmax_region.any() and remaining > 0:
                        non_argmax_y, non_argmax_x = np.where(non_argmax_region)
                        non_argmax_conf = class_confidence[non_argmax_region]
                        non_argmax_weights = non_argmax_conf / non_argmax_conf.sum()
                        non_argmax_indices = np.random.choice(
                            len(non_argmax_y),
                            size=min(remaining, len(non_argmax_y)),
                            replace=False,
                            p=non_argmax_weights
                        )
                        sampled_coords.extend(zip(non_argmax_x[non_argmax_indices],
                                                 non_argmax_y[non_argmax_indices]))
                    sampling_strategy = f"mixed({len(argmax_coords)}argmax+{len(sampled_coords)-len(argmax_coords)}topK)"
            else:
                # No argmax pixels in region, fall back to original sampling
                sampled_coords = [(int(x_coords[i]), int(y_coords[i])) for i in sampled_indices]
                sampling_strategy = "topK_only"

            for px, py in sampled_coords:
                conf = class_confidence[py, px]
                prompts.append({
                    'point': (px, py),
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'confidence': float(conf),
                    'region_size': int(region_size),
                    'purity': float(purity),  # NEW: track region purity
                    'negative_points': []
                })

            region_info.append({
                'size': region_size,
                'purity': purity,
                'prompts': num_prompts
            })

        if len(region_info) > 0:
            avg_purity = np.mean([r['purity'] for r in region_info])
            total_prompts = sum([r['prompts'] for r in region_info])
            print(f"  {class_name}: {len(region_info)} regions, purity={avg_purity:.2f}, {total_prompts} prompts")

    print(f"[IMPROVED] Extracted {len(prompts)} prompts total")

    # DEBUG: Print classes that will be extracted (those with prompts)
    if len(debug_stats) > 0:
        print("\n[DEBUG] Classes extracted ({} total):".format(extraction_count))
        # Sort by total pixels (descending)
        sorted_stats = sorted(debug_stats, key=lambda x: x['total_px'], reverse=True)[:15]
        for stat in sorted_stats:
            print(f"  {stat['class']}: max_conf={stat['max_conf']:.3f}, "
                  f"argmax={stat['argmax_px']}, top-K+={stat['top_k_contrib']}, "
                  f"total={stat['total_px']}, thresh={stat['threshold']:.2f}")

    return prompts


# Wrapper function to easily switch strategies
def extract_prompts_improved(seg_map, probs, vocabulary, strategy='adaptive_threshold', **kwargs):
    """
    Main entry point for improved prompt extraction.

    Args:
        strategy: One of:
            - 'adaptive_threshold': Easy - just adaptive thresholds (stuff vs thing)
            - 'confidence_weighted': Moderate - sample more points in large regions
            - 'density_based': Advanced - K-means clustering for spatial coverage
            - 'prob_map': BEST - exploit full probability distribution (RECOMMENDED!)
        **kwargs: Strategy-specific parameters

    Returns:
        List of prompts
    """
    if strategy == 'adaptive_threshold':
        return extract_prompts_adaptive_threshold(seg_map, probs, vocabulary, **kwargs)
    elif strategy == 'confidence_weighted':
        return extract_prompts_confidence_weighted_sampling(seg_map, probs, vocabulary, **kwargs)
    elif strategy == 'density_based':
        return extract_prompts_density_based(seg_map, probs, vocabulary, **kwargs)
    elif strategy == 'prob_map':
        return extract_prompts_prob_map_exploitation(seg_map, probs, vocabulary, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: "
                        "['adaptive_threshold', 'confidence_weighted', 'density_based', 'prob_map']")
