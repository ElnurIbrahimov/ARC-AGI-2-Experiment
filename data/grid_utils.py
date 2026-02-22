"""
Grid utility functions for ARC grids.
"""

from typing import Optional
import numpy as np


def pad_grid(grid: np.ndarray, max_h: int, max_w: int,
             pad_value: int = 0) -> np.ndarray:
    """
    Pad grid to (max_h, max_w) with pad_value.
    If grid is already larger, it is cropped.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {grid.shape}")

    h, w = grid.shape
    result = np.full((max_h, max_w), pad_value, dtype=int)
    copy_h = min(h, max_h)
    copy_w = min(w, max_w)
    result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]
    return result


def normalize_grid(grid) -> np.ndarray:
    """
    Ensure grid is a 2D int numpy array with values clamped to 0-9.
    Accepts lists, nested lists, or numpy arrays.
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    if grid.ndim != 2:
        raise ValueError(f"Cannot normalize to 2D grid: shape {grid.shape}")
    grid = np.clip(grid, 0, 9)
    return grid


def grid_to_string(grid: np.ndarray) -> str:
    """
    Pretty-print a grid. Each row is one line, values separated by spaces.
    Example:
        0 0 1
        0 2 0
        3 0 0
    """
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {grid.shape}")
    lines = []
    for row in grid:
        lines.append(" ".join(str(int(v)) for v in row))
    return "\n".join(lines)


def string_to_grid(s: str) -> np.ndarray:
    """
    Parse a grid from a string. Each line is a row, values separated by whitespace.
    """
    s = s.strip()
    if not s:
        raise ValueError("Empty string")
    rows = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            continue
        vals = [int(x) for x in line.split()]
        rows.append(vals)
    if not rows:
        raise ValueError("No rows parsed")
    # Ensure rectangular
    max_w = max(len(r) for r in rows)
    for i, row in enumerate(rows):
        if len(row) < max_w:
            rows[i] = row + [0] * (max_w - len(row))
    return np.array(rows, dtype=int)


def grids_equal(g1: np.ndarray, g2: np.ndarray) -> bool:
    """Check if two grids are identical (shape and values)."""
    g1 = np.asarray(g1, dtype=int)
    g2 = np.asarray(g2, dtype=int)
    if g1.shape != g2.shape:
        return False
    return bool(np.array_equal(g1, g2))


def grid_diff(g1: np.ndarray, g2: np.ndarray) -> Optional[np.ndarray]:
    """
    Return a boolean mask of differing cells.
    Returns None if shapes differ.
    """
    g1 = np.asarray(g1, dtype=int)
    g2 = np.asarray(g2, dtype=int)
    if g1.shape != g2.shape:
        return None
    return g1 != g2


def grid_dimensions(grid: np.ndarray) -> tuple:
    """Return (height, width) of grid."""
    grid = np.asarray(grid)
    if grid.ndim != 2:
        raise ValueError(f"Grid must be 2D, got shape {grid.shape}")
    return (grid.shape[0], grid.shape[1])


def unique_colors(grid: np.ndarray) -> set:
    """Return set of unique color values in the grid."""
    grid = np.asarray(grid, dtype=int)
    return set(np.unique(grid).tolist())


def grid_to_flat(grid: np.ndarray) -> list:
    """Flatten grid to a 1D list (row-major)."""
    return np.asarray(grid, dtype=int).flatten().tolist()


def flat_to_grid(flat: list, h: int, w: int) -> np.ndarray:
    """Reshape a flat list into a 2D grid."""
    if len(flat) != h * w:
        raise ValueError(f"Expected {h*w} values, got {len(flat)}")
    return np.array(flat, dtype=int).reshape(h, w)
