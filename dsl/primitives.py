"""
Pure Python/numpy implementations of ~50 DSL primitives for ARC grid manipulation.

All grids are 2D int numpy arrays with values 0-9 (0 = background).
Functions return copies, never views. Inputs are validated.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from collections import Counter
from scipy import ndimage


# ─── Helpers ──────────────────────────────────────────────────────────

def _validate_grid(grid: np.ndarray, name: str = "grid") -> np.ndarray:
    """Ensure grid is a 2D int array."""
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {grid.shape}")
    return grid


def _validate_color(color: int) -> int:
    if not (0 <= color <= 9):
        raise ValueError(f"Color must be 0-9, got {color}")
    return int(color)


def _connected_components_4(grid: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """
    Find 4-connected components of non-background cells.
    Returns list of (boolean_mask, color) tuples.
    """
    grid = _validate_grid(grid)
    objects = []
    visited = np.zeros_like(grid, dtype=bool)
    h, w = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and not visited[r, c]:
                color = int(grid[r, c])
                mask = np.zeros_like(grid, dtype=bool)
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (0 <= cr < h and 0 <= cc < w
                            and not visited[cr, cc]
                            and grid[cr, cc] == color):
                        visited[cr, cc] = True
                        mask[cr, cc] = True
                        for dr, dc in directions:
                            stack.append((cr + dr, cc + dc))
                objects.append((mask, color))
    return objects


# ═════════════════════════════════════════════════════════════════════
# SPATIAL
# ═════════════════════════════════════════════════════════════════════

def rot90(grid: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees clockwise."""
    grid = _validate_grid(grid)
    return np.rot90(grid, k=-1).copy()


def rot180(grid: np.ndarray) -> np.ndarray:
    """Rotate 180 degrees."""
    grid = _validate_grid(grid)
    return np.rot90(grid, k=2).copy()


def rot270(grid: np.ndarray) -> np.ndarray:
    """Rotate 270 degrees clockwise (= 90 counter-clockwise)."""
    grid = _validate_grid(grid)
    return np.rot90(grid, k=-3).copy()


def hmirror(grid: np.ndarray) -> np.ndarray:
    """Horizontal mirror — flip left-right."""
    grid = _validate_grid(grid)
    return np.fliplr(grid).copy()


def vmirror(grid: np.ndarray) -> np.ndarray:
    """Vertical mirror — flip top-bottom."""
    grid = _validate_grid(grid)
    return np.flipud(grid).copy()


def transpose(grid: np.ndarray) -> np.ndarray:
    """Transpose rows and columns."""
    grid = _validate_grid(grid)
    return grid.T.copy()


def shift(grid: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Shift grid by (dx, dy). dx = column shift, dy = row shift.
    Empty space filled with 0.
    """
    grid = _validate_grid(grid)
    result = np.zeros_like(grid)
    h, w = grid.shape
    dx, dy = int(dx), int(dy)

    # Compute source and dest slices
    src_r_start = max(0, -dy)
    src_r_end = min(h, h - dy)
    src_c_start = max(0, -dx)
    src_c_end = min(w, w - dx)

    dst_r_start = max(0, dy)
    dst_r_end = min(h, h + dy)
    dst_c_start = max(0, dx)
    dst_c_end = min(w, w + dx)

    if (src_r_end > src_r_start and src_c_end > src_c_start
            and dst_r_end > dst_r_start and dst_c_end > dst_c_start):
        result[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
            grid[src_r_start:src_r_end, src_c_start:src_c_end]
    return result


# ═════════════════════════════════════════════════════════════════════
# COLOR
# ═════════════════════════════════════════════════════════════════════

def fill(grid: np.ndarray, color: int) -> np.ndarray:
    """Fill entire grid with a single color."""
    grid = _validate_grid(grid)
    color = _validate_color(color)
    result = np.full_like(grid, color)
    return result


def recolor(grid: np.ndarray, from_color: int, to_color: int) -> np.ndarray:
    """Replace all cells of from_color with to_color."""
    grid = _validate_grid(grid)
    from_color = _validate_color(from_color)
    to_color = _validate_color(to_color)
    result = grid.copy()
    result[result == from_color] = to_color
    return result


def flood_fill(grid: np.ndarray, row: int, col: int, color: int) -> np.ndarray:
    """Flood fill from (row, col) with given color (4-connected)."""
    grid = _validate_grid(grid)
    color = _validate_color(color)
    row, col = int(row), int(col)
    h, w = grid.shape
    if not (0 <= row < h and 0 <= col < w):
        return grid.copy()

    result = grid.copy()
    old_color = int(result[row, col])
    if old_color == color:
        return result

    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < h and 0 <= c < w and result[r, c] == old_color:
            result[r, c] = color
            stack.extend([(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)])
    return result


def color_map(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """Remap colors according to mapping dict {from_color: to_color}."""
    grid = _validate_grid(grid)
    result = grid.copy()
    for from_c, to_c in mapping.items():
        result[grid == int(from_c)] = int(to_c)
    return result


def most_common_color(grid: np.ndarray) -> int:
    """Return the most common color in the grid (including 0)."""
    grid = _validate_grid(grid)
    if grid.size == 0:
        return 0
    counts = Counter(grid.flatten().tolist())
    return counts.most_common(1)[0][0]


def least_common_color(grid: np.ndarray) -> int:
    """Return the least common non-background color. Returns 0 if no non-bg colors."""
    grid = _validate_grid(grid)
    flat = grid.flatten().tolist()
    counts = Counter(c for c in flat if c != 0)
    if not counts:
        return 0
    return counts.most_common()[-1][0]


def count_color(grid: np.ndarray, color: int) -> int:
    """Count cells with the given color."""
    grid = _validate_grid(grid)
    color = _validate_color(color)
    return int(np.sum(grid == color))


# ═════════════════════════════════════════════════════════════════════
# OBJECT
# ═════════════════════════════════════════════════════════════════════

def find_objects(grid: np.ndarray) -> List[Tuple[np.ndarray, int]]:
    """
    Find all connected components (4-connected, non-background).
    Returns list of (boolean_mask, color).
    """
    return _connected_components_4(grid)


def isolate_object(grid: np.ndarray, idx: int) -> np.ndarray:
    """Return grid with only the object at the given index (0-based). Rest is 0."""
    grid = _validate_grid(grid)
    objects = _connected_components_4(grid)
    idx = int(idx)
    if idx < 0 or idx >= len(objects):
        return np.zeros_like(grid)
    mask, color = objects[idx]
    result = np.zeros_like(grid)
    result[mask] = color
    return result


def extract_largest(grid: np.ndarray) -> np.ndarray:
    """Return grid with only the largest connected object."""
    grid = _validate_grid(grid)
    objects = _connected_components_4(grid)
    if not objects:
        return np.zeros_like(grid)
    largest = max(objects, key=lambda o: o[0].sum())
    result = np.zeros_like(grid)
    result[largest[0]] = largest[1]
    return result


def extract_smallest(grid: np.ndarray) -> np.ndarray:
    """Return grid with only the smallest connected object."""
    grid = _validate_grid(grid)
    objects = _connected_components_4(grid)
    if not objects:
        return np.zeros_like(grid)
    smallest = min(objects, key=lambda o: o[0].sum())
    result = np.zeros_like(grid)
    result[smallest[0]] = smallest[1]
    return result


def bounding_box(grid: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Return (row_min, col_min, row_max, col_max) bounding box of True cells in mask.
    If mask is empty, returns (0, 0, 0, 0).
    """
    grid = _validate_grid(grid)
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))


def move_object(grid: np.ndarray, mask: np.ndarray, drow: int, dcol: int) -> np.ndarray:
    """Move object defined by mask by (drow, dcol). Object wraps are clipped."""
    grid = _validate_grid(grid)
    mask = np.asarray(mask, dtype=bool)
    drow, dcol = int(drow), int(dcol)
    h, w = grid.shape
    result = grid.copy()

    # Clear old positions
    result[mask] = 0

    # Write new positions
    rows, cols = np.where(mask)
    for r, c in zip(rows, cols):
        nr, nc = r + drow, c + dcol
        if 0 <= nr < h and 0 <= nc < w:
            result[nr, nc] = grid[r, c]
    return result


def count_objects(grid: np.ndarray) -> int:
    """Count the number of connected non-background objects."""
    return len(_connected_components_4(grid))


# ═════════════════════════════════════════════════════════════════════
# GRID
# ═════════════════════════════════════════════════════════════════════

def crop(grid: np.ndarray, r1: int, c1: int, r2: int, c2: int) -> np.ndarray:
    """Crop subgrid [r1:r2+1, c1:c2+1] (inclusive on both ends)."""
    grid = _validate_grid(grid)
    r1, c1, r2, c2 = int(r1), int(c1), int(r2), int(c2)
    h, w = grid.shape
    r1 = max(0, min(r1, h - 1))
    c1 = max(0, min(c1, w - 1))
    r2 = max(r1, min(r2, h - 1))
    c2 = max(c1, min(c2, w - 1))
    return grid[r1:r2 + 1, c1:c2 + 1].copy()


def trim(grid: np.ndarray) -> np.ndarray:
    """Remove all-zero border rows and columns."""
    grid = _validate_grid(grid)
    if grid.size == 0 or np.all(grid == 0):
        return np.zeros((1, 1), dtype=int)
    rows = np.any(grid != 0, axis=1)
    cols = np.any(grid != 0, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return grid[r_min:r_max + 1, c_min:c_max + 1].copy()


def pad(grid: np.ndarray, top: int, bottom: int, left: int, right: int,
        color: int = 0) -> np.ndarray:
    """Pad grid with given color on each side."""
    grid = _validate_grid(grid)
    top, bottom, left, right = int(top), int(bottom), int(left), int(right)
    color = int(color)
    return np.pad(grid, ((top, bottom), (left, right)),
                  mode='constant', constant_values=color).astype(int)


def concat_h(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Horizontal concatenation. Heights must match (or pad shorter with 0)."""
    grid1 = _validate_grid(grid1, "grid1")
    grid2 = _validate_grid(grid2, "grid2")
    h1, h2 = grid1.shape[0], grid2.shape[0]
    if h1 != h2:
        max_h = max(h1, h2)
        if h1 < max_h:
            grid1 = np.pad(grid1, ((0, max_h - h1), (0, 0)),
                           mode='constant', constant_values=0)
        if h2 < max_h:
            grid2 = np.pad(grid2, ((0, max_h - h2), (0, 0)),
                           mode='constant', constant_values=0)
    return np.concatenate([grid1, grid2], axis=1).astype(int)


def concat_v(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Vertical concatenation. Widths must match (or pad narrower with 0)."""
    grid1 = _validate_grid(grid1, "grid1")
    grid2 = _validate_grid(grid2, "grid2")
    w1, w2 = grid1.shape[1], grid2.shape[1]
    if w1 != w2:
        max_w = max(w1, w2)
        if w1 < max_w:
            grid1 = np.pad(grid1, ((0, 0), (0, max_w - w1)),
                           mode='constant', constant_values=0)
        if w2 < max_w:
            grid2 = np.pad(grid2, ((0, 0), (0, max_w - w2)),
                           mode='constant', constant_values=0)
    return np.concatenate([grid1, grid2], axis=0).astype(int)


def tile(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Tile grid rows x cols times."""
    grid = _validate_grid(grid)
    rows, cols = int(rows), int(cols)
    if rows < 1 or cols < 1:
        raise ValueError(f"tile rows/cols must be >= 1, got ({rows}, {cols})")
    return np.tile(grid, (rows, cols)).astype(int)


def resize(grid: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize."""
    grid = _validate_grid(grid)
    new_h, new_w = int(new_h), int(new_w)
    if new_h < 1 or new_w < 1:
        raise ValueError(f"resize dimensions must be >= 1, got ({new_h}, {new_w})")
    h, w = grid.shape
    row_idx = (np.arange(new_h) * h / new_h).astype(int)
    col_idx = (np.arange(new_w) * w / new_w).astype(int)
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    return grid[np.ix_(row_idx, col_idx)].copy()


def overlay(base: np.ndarray, top: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay top grid onto base where mask is True. Grids must be same shape."""
    base = _validate_grid(base, "base")
    top = _validate_grid(top, "top")
    mask = np.asarray(mask, dtype=bool)
    # Match shapes — take min dims
    h = min(base.shape[0], top.shape[0], mask.shape[0])
    w = min(base.shape[1], top.shape[1], mask.shape[1])
    result = base[:h, :w].copy()
    m = mask[:h, :w]
    result[m] = top[:h, :w][m]
    return result


# ═════════════════════════════════════════════════════════════════════
# SET
# ═════════════════════════════════════════════════════════════════════

def _match_shapes(g1: np.ndarray, g2: np.ndarray):
    """Ensure same shape by padding smaller grid with zeros."""
    g1 = _validate_grid(g1, "grid1")
    g2 = _validate_grid(g2, "grid2")
    h = max(g1.shape[0], g2.shape[0])
    w = max(g1.shape[1], g2.shape[1])
    if g1.shape != (h, w):
        tmp = np.zeros((h, w), dtype=int)
        tmp[:g1.shape[0], :g1.shape[1]] = g1
        g1 = tmp
    if g2.shape != (h, w):
        tmp = np.zeros((h, w), dtype=int)
        tmp[:g2.shape[0], :g2.shape[1]] = g2
        g2 = tmp
    return g1, g2


def intersection(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Non-zero where both are non-zero. Takes value from grid1."""
    g1, g2 = _match_shapes(grid1, grid2)
    result = np.where((g1 != 0) & (g2 != 0), g1, 0)
    return result.astype(int)


def union(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Non-zero from either. grid2 values override grid1 where both non-zero."""
    g1, g2 = _match_shapes(grid1, grid2)
    result = g1.copy()
    mask2 = g2 != 0
    result[mask2] = g2[mask2]
    return result.astype(int)


def difference(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Non-zero in grid1 where grid2 is zero."""
    g1, g2 = _match_shapes(grid1, grid2)
    result = np.where((g1 != 0) & (g2 == 0), g1, 0)
    return result.astype(int)


def xor(grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Non-zero in exactly one of the grids."""
    g1, g2 = _match_shapes(grid1, grid2)
    m1 = g1 != 0
    m2 = g2 != 0
    result = np.zeros_like(g1)
    result[m1 & ~m2] = g1[m1 & ~m2]
    result[m2 & ~m1] = g2[m2 & ~m1]
    return result.astype(int)


# ═════════════════════════════════════════════════════════════════════
# LOGIC
# ═════════════════════════════════════════════════════════════════════

def if_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Boolean mask where grid == color."""
    grid = _validate_grid(grid)
    color = _validate_color(color)
    return (grid == color).astype(bool)


def filter_by_size(objects: List[Tuple[np.ndarray, int]],
                   min_size: int, max_size: int) -> List[Tuple[np.ndarray, int]]:
    """Filter object list by pixel count (inclusive bounds)."""
    min_size, max_size = int(min_size), int(max_size)
    return [(mask, color) for mask, color in objects
            if min_size <= int(mask.sum()) <= max_size]


def select_by_color(grid: np.ndarray, color: int) -> np.ndarray:
    """Boolean mask of cells matching given color."""
    grid = _validate_grid(grid)
    color = _validate_color(color)
    return (grid == color).astype(bool)


# ═════════════════════════════════════════════════════════════════════
# COMPOSITION
# ═════════════════════════════════════════════════════════════════════

def sequence(funcs: list, grid: np.ndarray) -> np.ndarray:
    """Apply a list of callables in order. Each takes a grid, returns a grid."""
    grid = _validate_grid(grid)
    result = grid.copy()
    for fn in funcs:
        result = _validate_grid(fn(result))
    return result


def iterate_until_fixpoint(func, grid: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Apply func repeatedly until grid stops changing or max_iter reached."""
    grid = _validate_grid(grid)
    current = grid.copy()
    for _ in range(max_iter):
        new = _validate_grid(func(current))
        if np.array_equal(current, new):
            break
        current = new
    return current


def apply_to_each(func, objects: List[Tuple[np.ndarray, int]],
                  grid: np.ndarray) -> np.ndarray:
    """
    Apply func to each object independently on a fresh canvas,
    then merge results back.
    """
    grid = _validate_grid(grid)
    result = np.zeros_like(grid)
    for mask, color in objects:
        obj_grid = np.zeros_like(grid)
        obj_grid[mask] = color
        transformed = _validate_grid(func(obj_grid))
        # Merge: non-zero from transformed goes to result
        non_zero = transformed != 0
        h = min(result.shape[0], transformed.shape[0])
        w = min(result.shape[1], transformed.shape[1])
        nz = non_zero[:h, :w]
        result[:h, :w][nz] = transformed[:h, :w][nz]
    return result


# ═════════════════════════════════════════════════════════════════════
# PATTERN
# ═════════════════════════════════════════════════════════════════════

def detect_period(grid: np.ndarray, axis: int) -> int:
    """
    Detect repeating period along given axis (0=rows, 1=cols).
    Returns smallest period, or full dimension if no period found.
    """
    grid = _validate_grid(grid)
    axis = int(axis)
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    dim = grid.shape[axis]
    for period in range(1, dim + 1):
        if dim % period != 0:
            continue
        is_periodic = True
        for i in range(dim):
            if axis == 0:
                if not np.array_equal(grid[i, :], grid[i % period, :]):
                    is_periodic = False
                    break
            else:
                if not np.array_equal(grid[:, i], grid[:, i % period]):
                    is_periodic = False
                    break
        if is_periodic:
            return period
    return dim


def extend_pattern(grid: np.ndarray, axis: int, target_size: int) -> np.ndarray:
    """Extend repeating pattern along axis to target_size."""
    grid = _validate_grid(grid)
    axis = int(axis)
    target_size = int(target_size)
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    period = detect_period(grid, axis)
    if axis == 0:
        base = grid[:period, :]
        reps = (target_size + period - 1) // period
        extended = np.tile(base, (reps, 1))
        return extended[:target_size, :].astype(int)
    else:
        base = grid[:, :period]
        reps = (target_size + period - 1) // period
        extended = np.tile(base, (1, reps))
        return extended[:, :target_size].astype(int)


def symmetry_type(grid: np.ndarray) -> Set[str]:
    """
    Detect which symmetries the grid has.
    Returns set of: {"rot90", "rot180", "hmirror", "vmirror", "transpose", "none"}
    """
    grid = _validate_grid(grid)
    syms: Set[str] = set()

    if np.array_equal(grid, rot90(grid)):
        syms.add("rot90")
    if np.array_equal(grid, rot180(grid)):
        syms.add("rot180")
    if np.array_equal(grid, hmirror(grid)):
        syms.add("hmirror")
    if np.array_equal(grid, vmirror(grid)):
        syms.add("vmirror")
    if grid.shape[0] == grid.shape[1] and np.array_equal(grid, transpose(grid)):
        syms.add("transpose")

    if not syms:
        syms.add("none")
    return syms


# ─── Name -> function lookup ─────────────────────────────────────────

PRIMITIVE_FUNCTIONS = {
    # SPATIAL
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "hmirror": hmirror,
    "vmirror": vmirror,
    "transpose": transpose,
    "shift": shift,
    # COLOR
    "fill": fill,
    "recolor": recolor,
    "flood_fill": flood_fill,
    "color_map": color_map,
    "most_common_color": most_common_color,
    "least_common_color": least_common_color,
    "count_color": count_color,
    # OBJECT
    "find_objects": find_objects,
    "isolate_object": isolate_object,
    "extract_largest": extract_largest,
    "extract_smallest": extract_smallest,
    "bounding_box": bounding_box,
    "move_object": move_object,
    "count_objects": count_objects,
    # GRID
    "crop": crop,
    "trim": trim,
    "pad": pad,
    "concat_h": concat_h,
    "concat_v": concat_v,
    "tile": tile,
    "resize": resize,
    "overlay": overlay,
    # SET
    "intersection": intersection,
    "union": union,
    "difference": difference,
    "xor": xor,
    # LOGIC
    "if_color": if_color,
    "filter_by_size": filter_by_size,
    "select_by_color": select_by_color,
    # COMPOSITION
    "sequence": sequence,
    "iterate_until_fixpoint": iterate_until_fixpoint,
    "apply_to_each": apply_to_each,
    # PATTERN
    "detect_period": detect_period,
    "extend_pattern": extend_pattern,
    "symmetry_type": symmetry_type,
}
