"""
Data augmentation for ARC grids.

Provides all 8 dihedral symmetries and color permutations.
All augmentations are applied consistently to (input, output) pairs.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from itertools import permutations
import random


# ─── Dihedral group (D4) — all 8 rigid symmetries ───────────────────

def _identity(grid: np.ndarray) -> np.ndarray:
    return grid.copy()

def _rot90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-1).copy()

def _rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=2).copy()

def _rot270(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=-3).copy()

def _hmirror(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid).copy()

def _vmirror(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid).copy()

def _rot90_hmirror(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(np.rot90(grid, k=-1)).copy()

def _rot90_vmirror(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.rot90(grid, k=-1)).copy()


DIHEDRAL_TRANSFORMS: List[Tuple[str, Callable]] = [
    ("identity",        _identity),
    ("rot90",           _rot90),
    ("rot180",          _rot180),
    ("rot270",          _rot270),
    ("hmirror",         _hmirror),
    ("vmirror",         _vmirror),
    ("rot90+hmirror",   _rot90_hmirror),   # = transpose
    ("rot90+vmirror",   _rot90_vmirror),   # = anti-transpose
]


def apply_dihedral(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
) -> List[Tuple[List[np.ndarray], List[np.ndarray], str]]:
    """
    Apply all 8 dihedral symmetries to a set of (input, output) pairs.

    Returns list of (augmented_inputs, augmented_outputs, transform_name).
    Each pair is consistently transformed.
    """
    results = []
    for name, fn in DIHEDRAL_TRANSFORMS:
        aug_inputs = [fn(np.asarray(g, dtype=int)) for g in inputs]
        aug_outputs = [fn(np.asarray(g, dtype=int)) for g in outputs]
        results.append((aug_inputs, aug_outputs, name))
    return results


# ─── Color permutation ───────────────────────────────────────────────

def random_color_permutation(
    exclude_zero: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, int]:
    """
    Generate a random permutation of colors 1-9 (keeping 0 fixed as background).
    Returns a dict mapping old_color -> new_color.
    """
    if rng is None:
        rng = np.random.default_rng()

    if exclude_zero:
        colors = list(range(1, 10))
        perm = rng.permutation(colors).tolist()
        mapping = {0: 0}
        for old, new in zip(colors, perm):
            mapping[old] = new
    else:
        colors = list(range(10))
        perm = rng.permutation(colors).tolist()
        mapping = {old: new for old, new in zip(colors, perm)}

    return mapping


def apply_color_permutation(
    grid: np.ndarray,
    mapping: Dict[int, int],
) -> np.ndarray:
    """Apply a color permutation to a grid."""
    grid = np.asarray(grid, dtype=int)
    result = np.zeros_like(grid)
    for old_c, new_c in mapping.items():
        result[grid == old_c] = new_c
    return result


def augment_with_color_permutations(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
    n_perms: int = 5,
    seed: Optional[int] = None,
) -> List[Tuple[List[np.ndarray], List[np.ndarray], Dict[int, int]]]:
    """
    Generate n_perms random color permutations and apply them.

    Returns list of (augmented_inputs, augmented_outputs, color_mapping).
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_perms):
        mapping = random_color_permutation(exclude_zero=True, rng=rng)
        aug_in = [apply_color_permutation(g, mapping) for g in inputs]
        aug_out = [apply_color_permutation(g, mapping) for g in outputs]
        results.append((aug_in, aug_out, mapping))
    return results


# ─── Combined augmentation ───────────────────────────────────────────

def augment_task(
    inputs: List[np.ndarray],
    outputs: List[np.ndarray],
    include_dihedral: bool = True,
    include_color_perm: bool = True,
    n_color_perms: int = 3,
    seed: Optional[int] = None,
) -> List[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """
    Full augmentation pipeline for an ARC task.

    Args:
        inputs: list of input grids
        outputs: list of corresponding output grids
        include_dihedral: apply all 8 dihedral symmetries
        include_color_perm: apply random color permutations
        n_color_perms: number of color permutations
        seed: random seed for reproducibility

    Returns:
        List of (augmented_inputs, augmented_outputs) pairs.
        Always includes the original as the first entry.
    """
    rng = np.random.default_rng(seed)
    results: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []

    # Start with the original
    orig_in = [np.asarray(g, dtype=int).copy() for g in inputs]
    orig_out = [np.asarray(g, dtype=int).copy() for g in outputs]
    results.append((orig_in, orig_out))

    # Collect spatial transforms
    spatial_pairs: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
    if include_dihedral:
        for aug_in, aug_out, name in apply_dihedral(inputs, outputs):
            if name == "identity":
                continue  # already added
            spatial_pairs.append((aug_in, aug_out))
    else:
        spatial_pairs.append((orig_in, orig_out))

    results.extend(spatial_pairs)

    # Apply color permutations to each spatial variant
    if include_color_perm:
        all_spatial = [(orig_in, orig_out)] + spatial_pairs
        for s_in, s_out in all_spatial:
            color_augs = augment_with_color_permutations(
                s_in, s_out, n_perms=n_color_perms, seed=rng.integers(0, 2**31),
            )
            for c_in, c_out, _ in color_augs:
                results.append((c_in, c_out))

    return results
