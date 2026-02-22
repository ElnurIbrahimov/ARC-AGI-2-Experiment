"""
Evaluation metrics for ARC-AGI-2.

All grid inputs are numpy arrays of integers (0-9 representing colors).
The primary metric is exact grid match (pass@2), but we also compute
softer metrics for gradient signal during refinement.
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def grid_exact_match(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Exact grid match — shape and all cell values must be identical."""
    if predicted.shape != target.shape:
        return False
    return bool(np.array_equal(predicted, target))


def cell_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    """Fraction of cells that match. Returns 0.0 if shapes differ."""
    if predicted.shape != target.shape:
        return 0.0
    total = target.size
    if total == 0:
        return 1.0
    return float(np.sum(predicted == target)) / total


def shape_match(predicted: np.ndarray, target: np.ndarray) -> bool:
    """Check if output shapes match."""
    return predicted.shape == target.shape


def color_accuracy(predicted: np.ndarray, target: np.ndarray) -> float:
    """Accuracy only on non-background (non-zero) cells in the target.

    If shapes differ, returns 0.0.
    If target has no foreground cells, returns 1.0 if predicted also has none, else 0.0.
    """
    if predicted.shape != target.shape:
        return 0.0
    fg_mask = target != 0
    fg_count = int(np.sum(fg_mask))
    if fg_count == 0:
        # No foreground in target — correct iff predicted has no foreground either
        return 1.0 if int(np.sum(predicted != 0)) == 0 else 0.0
    return float(np.sum(predicted[fg_mask] == target[fg_mask])) / fg_count


# ---------------------------------------------------------------------------
# Structural similarity (soft metric for refinement signal)
# ---------------------------------------------------------------------------

def _connected_components(grid: np.ndarray) -> Tuple[np.ndarray, int]:
    """Label connected components of foreground cells (non-zero), 4-connected."""
    binary = (grid != 0).astype(np.int32)
    structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
    labeled, num_features = ndimage.label(binary, structure=structure)
    return labeled, num_features


def _object_sizes(labeled: np.ndarray, num_objects: int) -> List[int]:
    """Return sorted list of object sizes (cell counts)."""
    if num_objects == 0:
        return []
    sizes = ndimage.sum(
        np.ones_like(labeled), labeled, index=range(1, num_objects + 1)
    )
    return sorted([int(s) for s in sizes], reverse=True)


def _size_distribution_similarity(sizes_a: List[int], sizes_b: List[int]) -> float:
    """Compare two sorted-descending size distributions. Returns 0-1."""
    if not sizes_a and not sizes_b:
        return 1.0
    if not sizes_a or not sizes_b:
        return 0.0
    # Pad shorter list with zeros
    max_len = max(len(sizes_a), len(sizes_b))
    a = sizes_a + [0] * (max_len - len(sizes_a))
    b = sizes_b + [0] * (max_len - len(sizes_b))
    # Normalized L1 distance
    total = max(sum(a), sum(b), 1)
    dist = sum(abs(x - y) for x, y in zip(a, b)) / total
    return max(0.0, 1.0 - dist)


def _centroid_similarity(labeled_a: np.ndarray, num_a: int,
                         labeled_b: np.ndarray, num_b: int) -> float:
    """Compare relative positions of object centroids. Returns 0-1."""
    if num_a == 0 and num_b == 0:
        return 1.0
    if num_a == 0 or num_b == 0:
        return 0.0

    def _get_centroids(labeled, num_objs):
        centroids = ndimage.center_of_mass(
            np.ones_like(labeled), labeled, index=range(1, num_objs + 1)
        )
        return np.array(centroids)

    c_a = _get_centroids(labeled_a, num_a)
    c_b = _get_centroids(labeled_b, num_b)

    # Normalize centroids to [0, 1] using grid dimensions
    h_a, w_a = labeled_a.shape
    h_b, w_b = labeled_b.shape
    if h_a > 0 and w_a > 0:
        c_a[:, 0] /= max(h_a - 1, 1)
        c_a[:, 1] /= max(w_a - 1, 1)
    if h_b > 0 and w_b > 0:
        c_b[:, 0] /= max(h_b - 1, 1)
        c_b[:, 1] /= max(w_b - 1, 1)

    # Match centroids greedily by nearest distance
    used = set()
    total_dist = 0.0
    max_count = max(num_a, num_b)

    # Match from the smaller set to the larger
    if num_a <= num_b:
        src, tgt = c_a, c_b
    else:
        src, tgt = c_b, c_a

    for s in src:
        best_dist = float("inf")
        best_j = -1
        for j in range(len(tgt)):
            if j in used:
                continue
            d = float(np.linalg.norm(s - tgt[j]))
            if d < best_dist:
                best_dist = d
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            total_dist += best_dist

    # Unmatched objects contribute max possible distance (sqrt(2) in normalized space)
    unmatched = max_count - len(src)
    total_dist += unmatched * np.sqrt(2.0)

    # Normalize: max possible total distance
    max_total = max_count * np.sqrt(2.0)
    if max_total == 0:
        return 1.0
    return max(0.0, 1.0 - total_dist / max_total)


def _color_distribution_similarity(grid_a: np.ndarray, grid_b: np.ndarray) -> float:
    """Compare color frequency distributions. Returns 0-1."""
    def _histogram(g):
        hist = np.zeros(10, dtype=np.float64)
        for v in range(10):
            hist[v] = np.sum(g == v)
        total = g.size
        if total > 0:
            hist /= total
        return hist

    h_a = _histogram(grid_a)
    h_b = _histogram(grid_b)
    # 1 - 0.5 * L1 distance (L1 of two distributions is at most 2)
    return float(1.0 - 0.5 * np.sum(np.abs(h_a - h_b)))


def structural_similarity(predicted: np.ndarray, target: np.ndarray) -> float:
    """Structural similarity combining multiple sub-metrics.

    Compares:
    - Number of connected components (objects)
    - Size distribution of objects
    - Relative positions of object centroids
    - Color distribution

    Returns a float 0-1.
    """
    # If shapes differ, penalize but still compute what we can
    shape_penalty = 1.0
    if predicted.shape != target.shape:
        # Resize predicted to target shape for structural comparison
        # Use a simple crop/pad approach
        h_t, w_t = target.shape
        h_p, w_p = predicted.shape
        # Shape similarity penalty
        shape_penalty = min(h_t, h_p) * min(w_t, w_p) / max(h_t * w_t, h_p * w_p, 1)
        # Crop/pad predicted to target shape
        padded = np.zeros_like(target)
        copy_h = min(h_t, h_p)
        copy_w = min(w_t, w_p)
        padded[:copy_h, :copy_w] = predicted[:copy_h, :copy_w]
        predicted = padded

    # Connected components
    labeled_p, num_p = _connected_components(predicted)
    labeled_t, num_t = _connected_components(target)

    # Sub-metric 1: Object count similarity
    max_objs = max(num_p, num_t, 1)
    count_sim = 1.0 - abs(num_p - num_t) / max_objs

    # Sub-metric 2: Size distribution similarity
    sizes_p = _object_sizes(labeled_p, num_p)
    sizes_t = _object_sizes(labeled_t, num_t)
    size_sim = _size_distribution_similarity(sizes_p, sizes_t)

    # Sub-metric 3: Centroid position similarity
    pos_sim = _centroid_similarity(labeled_p, num_p, labeled_t, num_t)

    # Sub-metric 4: Color distribution similarity
    color_sim = _color_distribution_similarity(predicted, target)

    # Weighted combination
    combined = (
        0.20 * count_sim
        + 0.25 * size_sim
        + 0.25 * pos_sim
        + 0.30 * color_sim
    )
    return float(combined * shape_penalty)


# ---------------------------------------------------------------------------
# Pass@k and aggregate scoring
# ---------------------------------------------------------------------------

def pass_at_k(predictions: List[np.ndarray], target: np.ndarray, k: int = 2) -> bool:
    """True if any of the first k predictions exactly match target."""
    for pred in predictions[:k]:
        if grid_exact_match(pred, target):
            return True
    return False


def task_score(predictions: List[np.ndarray], target: np.ndarray, k: int = 2) -> Dict:
    """Full scoring dict with all metrics for one task.

    Uses the best prediction (by cell_accuracy) for soft metrics,
    and checks pass@k for the hard metric.
    """
    passed = pass_at_k(predictions, target, k=k)

    # Compute per-prediction metrics
    per_pred = []
    for pred in predictions:
        per_pred.append({
            "exact_match": grid_exact_match(pred, target),
            "cell_accuracy": cell_accuracy(pred, target),
            "shape_match": shape_match(pred, target),
            "color_accuracy": color_accuracy(pred, target),
            "structural_similarity": structural_similarity(pred, target),
        })

    # Best prediction by cell accuracy
    best_idx = max(range(len(per_pred)), key=lambda i: per_pred[i]["cell_accuracy"])
    best = per_pred[best_idx]

    return {
        "pass_at_k": passed,
        "k": k,
        "best_prediction_idx": best_idx,
        "best_exact_match": best["exact_match"],
        "best_cell_accuracy": best["cell_accuracy"],
        "best_shape_match": best["shape_match"],
        "best_color_accuracy": best["color_accuracy"],
        "best_structural_similarity": best["structural_similarity"],
        "num_predictions": len(predictions),
        "per_prediction": per_pred,
    }


def aggregate_scores(task_scores: List[Dict]) -> Dict:
    """Aggregate scores over all tasks.

    Returns pass@2 rate, mean cell accuracy, and other aggregate stats.
    """
    n = len(task_scores)
    if n == 0:
        return {
            "num_tasks": 0,
            "pass_at_k_rate": 0.0,
            "mean_cell_accuracy": 0.0,
            "mean_color_accuracy": 0.0,
            "mean_structural_similarity": 0.0,
            "shape_match_rate": 0.0,
        }

    passed = sum(1 for s in task_scores if s["pass_at_k"])
    cell_accs = [s["best_cell_accuracy"] for s in task_scores]
    color_accs = [s["best_color_accuracy"] for s in task_scores]
    struct_sims = [s["best_structural_similarity"] for s in task_scores]
    shape_matches = sum(1 for s in task_scores if s["best_shape_match"])

    return {
        "num_tasks": n,
        "pass_at_k_rate": passed / n,
        "num_passed": passed,
        "mean_cell_accuracy": float(np.mean(cell_accs)),
        "median_cell_accuracy": float(np.median(cell_accs)),
        "mean_color_accuracy": float(np.mean(color_accs)),
        "mean_structural_similarity": float(np.mean(struct_sims)),
        "shape_match_rate": shape_matches / n,
        "cell_accuracy_std": float(np.std(cell_accs)),
    }
