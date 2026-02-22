"""
Grid visualization for ARC-AGI-2.

Uses the standard ARC color palette. Draws gridlines between cells.
Handles variable grid sizes up to 30x30.
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Rectangle


# Standard ARC color palette
ARC_COLORS = {
    0: '#000000',  # black (background)
    1: '#0074D9',  # blue
    2: '#FF4136',  # red
    3: '#2ECC40',  # green
    4: '#FFDC00',  # yellow
    5: '#AAAAAA',  # grey
    6: '#F012BE',  # magenta
    7: '#FF851B',  # orange
    8: '#7FDBFF',  # cyan
    9: '#870C25',  # maroon
}

# Build a matplotlib colormap from the ARC palette
_ARC_COLOR_LIST = [ARC_COLORS[i] for i in range(10)]
ARC_CMAP = mcolors.ListedColormap(_ARC_COLOR_LIST)
ARC_NORM = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1), ncolors=10)


def plot_grid(grid: np.ndarray, ax: Optional[plt.Axes] = None,
              title: Optional[str] = None) -> plt.Axes:
    """Plot a single ARC grid with proper colors and gridlines.

    Args:
        grid: 2D numpy array with values 0-9.
        ax: Matplotlib axes to draw on. Created if None.
        title: Optional title above the grid.

    Returns:
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(3, grid.shape[1] * 0.5),
                                                max(3, grid.shape[0] * 0.5)))
    h, w = grid.shape

    ax.imshow(grid, cmap=ARC_CMAP, norm=ARC_NORM, interpolation='nearest',
              origin='upper', aspect='equal')

    # Gridlines
    for x in range(w + 1):
        ax.axvline(x - 0.5, color='#444444', linewidth=0.5)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color='#444444', linewidth=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')

    return ax


def plot_task(inputs: List[np.ndarray], outputs: List[np.ndarray],
              predictions: Optional[List[np.ndarray]] = None,
              title: Optional[str] = None) -> plt.Figure:
    """Plot a full ARC task: demo pairs + test input + prediction vs target.

    Layout:
      Row 0: inputs  (demo_0, demo_1, ..., test)
      Row 1: outputs (demo_0, demo_1, ..., target)
      Row 2: predictions (if provided, only under test column(s))

    Args:
        inputs: List of input grids. Last one(s) are test inputs.
        outputs: List of output grids. Last one(s) are expected test outputs.
        predictions: Optional list of predicted grids for the test input(s).
        title: Optional title for the figure.

    Returns:
        The matplotlib Figure.
    """
    n = len(inputs)
    num_rows = 3 if predictions else 2
    fig, axes = plt.subplots(num_rows, n, figsize=(n * 3, num_rows * 3),
                             squeeze=False)

    for i in range(n):
        # Input row
        label = f"Demo {i+1} Input" if i < n - 1 else "Test Input"
        plot_grid(inputs[i], ax=axes[0, i], title=label)

        # Output row
        label = f"Demo {i+1} Output" if i < n - 1 else "Target"
        plot_grid(outputs[i], ax=axes[1, i], title=label)

    # Prediction row
    if predictions:
        for i in range(n):
            if i < n - 1:
                # No prediction for demo pairs — hide axes
                axes[2, i].axis('off')
            else:
                # Show prediction for test
                pred_idx = i - (n - 1)
                if pred_idx < len(predictions):
                    plot_grid(predictions[pred_idx], ax=axes[2, i],
                              title="Prediction")
                else:
                    axes[2, i].axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

    fig.tight_layout()
    return fig


def plot_refinement_progress(scores: List[float],
                             title: str = "Refinement Progress") -> plt.Figure:
    """Plot score over refinement iterations.

    Args:
        scores: List of scores, one per iteration.
        title: Plot title.

    Returns:
        The matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    iterations = list(range(len(scores)))
    ax.plot(iterations, scores, color='#0074D9', alpha=0.4, linewidth=0.8,
            label='Per-iteration')

    # Running best
    running_best = []
    best = -1.0
    for s in scores:
        best = max(best, s)
        running_best.append(best)
    ax.plot(iterations, running_best, color='#FF4136', linewidth=2.0,
            label='Running best')

    # Moving average (window=20)
    if len(scores) >= 20:
        window = 20
        ma = np.convolve(scores, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(scores)), ma, color='#2ECC40',
                linewidth=1.5, linestyle='--', label=f'MA({window})')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def save_task_visualization(inputs: List[np.ndarray], outputs: List[np.ndarray],
                            predictions: Optional[List[np.ndarray]],
                            path: str) -> None:
    """Save task visualization to file.

    Args:
        inputs: Input grids.
        outputs: Output grids.
        predictions: Optional predicted grids.
        path: File path to save to (e.g., 'task_001.png').
    """
    fig = plot_task(inputs, outputs, predictions)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_confusion_grid(predicted: np.ndarray, target: np.ndarray) -> plt.Figure:
    """Show where predictions differ from target.

    Three panels:
      1. Predicted grid
      2. Target grid
      3. Diff overlay — correct cells dimmed, incorrect cells highlighted in red

    If shapes differ, the diff panel shows the shape mismatch.
    """
    if predicted.shape != target.shape:
        # Show both grids side by side with a shape mismatch note
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        plot_grid(predicted, ax=axes[0], title=f"Predicted {predicted.shape}")
        plot_grid(target, ax=axes[1], title=f"Target {target.shape}")
        fig.suptitle("SHAPE MISMATCH", fontsize=14, fontweight='bold',
                     color='red', y=1.02)
        fig.tight_layout()
        return fig

    h, w = target.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, max(3, h * 0.5)))

    plot_grid(predicted, ax=axes[0], title="Predicted")
    plot_grid(target, ax=axes[1], title="Target")

    # Diff panel: show target with red overlay on incorrect cells
    plot_grid(target, ax=axes[2], title="Diff (red = error)")

    diff_mask = predicted != target
    for r in range(h):
        for c in range(w):
            if diff_mask[r, c]:
                rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                                 linewidth=0, facecolor='red', alpha=0.5)
                axes[2].add_patch(rect)

    error_count = int(np.sum(diff_mask))
    total = h * w
    accuracy = 1.0 - error_count / total if total > 0 else 1.0
    fig.suptitle(f"Errors: {error_count}/{total} cells  |  Accuracy: {accuracy:.1%}",
                 fontsize=12, y=1.02)

    fig.tight_layout()
    return fig
